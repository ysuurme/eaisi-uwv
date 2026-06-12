"""
Unit tests for Step 6 — ML Model Validation (MAPE champion/challenger gate).

Contract (ADR-001 / ADR-002):
- One registered model per sector; @prod alias = that sector's champion.
- Promote a challenger to @prod iff its MAPE is finite AND strictly lower than
  the incumbent champion's MAPE (or no champion exists yet → seed).
- Losers are NOT registered; the run is tagged passed_gate=false.
- Optional R² floor (default disabled) can veto a MAPE-winner.
- The promoted version carries mape / r2 / model_family tags.

Mocks are at the system boundary only: the MLflow registry (MlflowClient and
mlflow.register_model). The gate logic itself is exercised through the public
validate_and_register() interface.
"""
import math
import unittest
from unittest.mock import MagicMock, patch

from src.ml_engineering.ml_6_model_validation import ModelValidator


def _champion_with_mape(mape: float, family: str = "RandomForest_Reduced") -> MagicMock:
    """A fake @prod model version carrying a MAPE tag (as MLflow stores strings)."""
    mv = MagicMock()
    mv.tags = {"mape": str(mape), "model_family": family}
    mv.version = "1"
    return mv


class TestMapeChampionGate(unittest.TestCase):
    @patch("src.ml_engineering.ml_6_model_validation.MlflowClient")
    def setUp(self, mock_client):
        self.validator = ModelValidator()
        self.client = self.validator.client

    # ── helpers ────────────────────────────────────────────────────────────
    def _no_champion(self):
        self.client.get_model_version_by_alias.side_effect = Exception("no alias")

    def _champion(self, mape, family="RandomForest_Reduced"):
        self.client.get_model_version_by_alias.side_effect = None
        self.client.get_model_version_by_alias.return_value = _champion_with_mape(mape, family)

    def _call(self, mape, r2=0.7, family="Ridge_Reduced", r2_floor=None):
        with patch(
            "src.ml_engineering.ml_6_model_validation.mlflow.register_model"
        ) as mock_register:
            mock_register.return_value = MagicMock(version="2")
            promoted = self.validator.validate_and_register(
                run_id="run_x",
                registered_model_name="SickLeave_4Q_301000",
                metrics={"mape": mape, "r2": r2, "mae": 0.1, "rmse": 0.2},
                model_family=family,
                sector_code="301000",
                r2_floor=r2_floor,
            )
            return promoted, mock_register

    # ── tests ────────────────────────────────────────────────────────────
    def test_first_model_seeds_champion(self):
        """No incumbent → register unconditionally and set @prod."""
        self._no_champion()
        promoted, mock_register = self._call(mape=0.05)
        self.assertTrue(promoted)
        mock_register.assert_called_once()
        self.client.set_registered_model_alias.assert_called_once()

    def test_better_mape_challenger_promoted(self):
        """Lower MAPE than champion → promote (new version, alias moves)."""
        self._champion(mape=0.06)
        promoted, mock_register = self._call(mape=0.04)
        self.assertTrue(promoted)
        mock_register.assert_called_once()
        self.client.set_registered_model_alias.assert_called_once()

    def test_worse_mape_challenger_rejected(self):
        """Higher MAPE than champion → reject: no register, alias unchanged."""
        self._champion(mape=0.04)
        promoted, mock_register = self._call(mape=0.06)
        self.assertFalse(promoted)
        mock_register.assert_not_called()
        self.client.set_registered_model_alias.assert_not_called()

    def test_equal_mape_challenger_rejected(self):
        """Equal MAPE is not an improvement → reject (strict <)."""
        self._champion(mape=0.05)
        promoted, mock_register = self._call(mape=0.05)
        self.assertFalse(promoted)
        mock_register.assert_not_called()

    def test_cross_family_challenger_promoted_on_mape(self):
        """A different family can dethrone the champion purely on MAPE."""
        self._champion(mape=0.05, family="RandomForest_Reduced")
        promoted, mock_register = self._call(mape=0.03, family="Ridge_Reduced")
        self.assertTrue(promoted)

    def test_non_finite_mape_rejected(self):
        """A degenerate (NaN) candidate MAPE is rejected, not promoted."""
        self._no_champion()
        promoted, mock_register = self._call(mape=float("nan"))
        self.assertFalse(promoted)
        mock_register.assert_not_called()

    def test_r2_floor_vetoes_mape_winner(self):
        """With an R² floor set, a MAPE-beating but low-R² candidate is rejected."""
        self._champion(mape=0.06)
        promoted, mock_register = self._call(mape=0.04, r2=0.1, r2_floor=0.5)
        self.assertFalse(promoted)
        mock_register.assert_not_called()

    def test_champion_mape_tag_roundtrips_losslessly(self):
        """The mape tag must store full precision so an identical re-run (equal
        MAPE) compares exactly equal and is rejected — not falsely 'improved'
        by rounding the stored champion value up."""
        self._no_champion()
        precise = 0.0728745001  # rounds UP to 0.072875 at 6 decimals
        self._call(mape=precise)
        mape_call = next(
            c for c in self.client.set_model_version_tag.call_args_list
            if c.kwargs.get("key") == "mape"
        )
        self.assertEqual(float(mape_call.kwargs["value"]), precise)

    def test_promoted_version_is_tagged_with_mape_r2_family(self):
        """The registered version records mape, r2, and model_family as tags."""
        self._no_champion()
        self._call(mape=0.05, r2=0.62, family="HistGBR_Reduced")
        tagged_keys = {
            call.kwargs.get("key", call.args[3] if len(call.args) > 3 else None)
            for call in self.client.set_model_version_tag.call_args_list
        }
        self.assertIn("mape", tagged_keys)
        self.assertIn("r2", tagged_keys)
        self.assertIn("model_family", tagged_keys)


if __name__ == "__main__":
    unittest.main()
