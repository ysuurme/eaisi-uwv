"""
Unit tests for Step 6 — ML Model Validation (MASE champion/challenger gate).

Contract (ADR-001 / ADR-002):
- One registered model per sector; @prod alias = that sector's champion.
- **MASE is THE comparison metric** (seasonal-naive m=4 scaled; lower is better,
  < 1 beats the naive).  Promote a challenger to @prod iff its MASE is finite AND
  strictly lower than the incumbent champion's MASE (or no champion exists yet →
  seed).
- Optional ``max_mase`` ceiling (default disabled): a challenger must also clear
  ``MASE < max_mase`` (e.g. 1.0 = must beat the seasonal naive).
- Losers are NOT registered; the run is tagged passed_gate=false.
- Optional R² floor (default disabled) can veto a MASE-winner.
- The promoted version carries mase / mape / r2 / model_family tags.

Mocks are at the system boundary only: the MLflow registry (MlflowClient and
mlflow.register_model). The gate logic itself is exercised through the public
validate_and_register() interface.
"""
import unittest
from unittest.mock import MagicMock, patch

from src.ml_engineering.ml_6_model_validation import ModelValidator


def _champion_with_mase(mase: float, family: str = "RandomForest_Reduced") -> MagicMock:
    """A fake @prod model version carrying a MASE tag (as MLflow stores strings)."""
    mv = MagicMock()
    mv.tags = {"mase": str(mase), "model_family": family}
    mv.version = "1"
    return mv


class TestMaseChampionGate(unittest.TestCase):
    @patch("src.ml_engineering.ml_6_model_validation.MlflowClient")
    def setUp(self, mock_client):
        self.validator = ModelValidator()
        self.client = self.validator.client

    # ── helpers ────────────────────────────────────────────────────────────
    def _no_champion(self):
        self.client.get_model_version_by_alias.side_effect = Exception("no alias")

    def _champion(self, mase, family="RandomForest_Reduced"):
        self.client.get_model_version_by_alias.side_effect = None
        self.client.get_model_version_by_alias.return_value = _champion_with_mase(mase, family)

    def _call(self, mase, r2=0.7, family="Ridge_Reduced", r2_floor=None, max_mase=None,
              model_type="Ridge", feature_groups='["all_survivors"]'):
        with patch(
            "src.ml_engineering.ml_6_model_validation.mlflow.register_model"
        ) as mock_register:
            mock_register.return_value = MagicMock(version="2")
            promoted = self.validator.validate_and_register(
                run_id="run_x",
                registered_model_name="SickLeave_4Q_301000",
                metrics={"mase": mase, "mape": 0.05, "r2": r2, "mae": 0.1, "rmse": 0.2},
                model_family=family,
                model_type=model_type,
                feature_groups=feature_groups,
                sector_code="301000",
                r2_floor=r2_floor,
                max_mase=max_mase,
            )
            return promoted, mock_register

    # ── tests ────────────────────────────────────────────────────────────
    def test_first_model_seeds_champion(self):
        """No incumbent → register unconditionally and set @prod."""
        self._no_champion()
        promoted, mock_register = self._call(mase=0.80)
        self.assertTrue(promoted)
        mock_register.assert_called_once()
        self.client.set_registered_model_alias.assert_called_once()

    def test_lower_mase_challenger_promoted(self):
        """Lower MASE than champion → promote (new version, alias moves)."""
        self._champion(mase=0.90)
        promoted, mock_register = self._call(mase=0.70)
        self.assertTrue(promoted)
        mock_register.assert_called_once()
        self.client.set_registered_model_alias.assert_called_once()

    def test_higher_mase_challenger_rejected(self):
        """Higher MASE than champion → reject: no register, alias unchanged."""
        self._champion(mase=0.70)
        promoted, mock_register = self._call(mase=0.90)
        self.assertFalse(promoted)
        mock_register.assert_not_called()
        self.client.set_registered_model_alias.assert_not_called()

    def test_equal_mase_challenger_rejected(self):
        """Equal MASE is not an improvement → reject (strict <)."""
        self._champion(mase=0.85)
        promoted, mock_register = self._call(mase=0.85)
        self.assertFalse(promoted)
        mock_register.assert_not_called()

    def test_cross_family_challenger_promoted_on_mase(self):
        """A different family can dethrone the champion purely on MASE."""
        self._champion(mase=0.85, family="RandomForest_Reduced")
        promoted, mock_register = self._call(mase=0.60, family="Ridge_Reduced")
        self.assertTrue(promoted)

    def test_non_finite_mase_rejected(self):
        """A degenerate (NaN) candidate MASE is rejected, not promoted."""
        self._no_champion()
        promoted, mock_register = self._call(mase=float("nan"))
        self.assertFalse(promoted)
        mock_register.assert_not_called()

    def test_r2_floor_vetoes_mase_winner(self):
        """With an R² floor set, a MASE-beating but low-R² candidate is rejected."""
        self._champion(mase=0.90)
        promoted, mock_register = self._call(mase=0.70, r2=0.1, r2_floor=0.5)
        self.assertFalse(promoted)
        mock_register.assert_not_called()

    def test_max_mase_floor_vetoes_seed_that_loses_to_naive(self):
        """With max_mase=1.0, a seed candidate that does NOT beat the seasonal
        naive (MASE ≥ 1.0) is rejected — don't seed @prod with a model that
        loses to the naive baseline."""
        self._no_champion()
        promoted, mock_register = self._call(mase=1.10, max_mase=1.0)
        self.assertFalse(promoted)
        mock_register.assert_not_called()

    def test_max_mase_floor_admits_seed_that_beats_naive(self):
        """With max_mase=1.0, a seed candidate that beats the naive (MASE < 1.0)
        is promoted."""
        self._no_champion()
        promoted, mock_register = self._call(mase=0.85, max_mase=1.0)
        self.assertTrue(promoted)
        mock_register.assert_called_once()

    def test_no_floor_seeds_any_finite_mase(self):
        """max_mase=None (default) disables the floor: even a MASE > 1 seeds when
        there is no incumbent (pure lowest-MASE selection)."""
        self._no_champion()
        promoted, mock_register = self._call(mase=1.50, max_mase=None)
        self.assertTrue(promoted)
        mock_register.assert_called_once()

    def test_champion_mase_tag_roundtrips_losslessly(self):
        """The mase tag must store full precision so an identical re-run (equal
        MASE) compares exactly equal and is rejected — not falsely 'improved'
        by rounding the stored champion value."""
        self._no_champion()
        precise = 0.8728745001  # would round at 6 decimals
        self._call(mase=precise)
        mase_call = next(
            c for c in self.client.set_model_version_tag.call_args_list
            if c.kwargs.get("key") == "mase"
        )
        self.assertEqual(float(mase_call.kwargs["value"]), precise)

    def test_promoted_version_self_describes_metric_set_and_provenance(self):
        """The registered version self-describes the full metric set (mase, mae,
        mape, r2) + model_family, model_type, and the config feature_groups."""
        self._no_champion()
        self._call(
            mase=0.75, r2=0.62, family="HistGBR_Reduced",
            model_type="HistGradientBoostingRegressor",
            feature_groups='["labor_volume", "workforce"]',
        )
        tagged = {
            call.kwargs.get("key", call.args[3] if len(call.args) > 3 else None):
            call.kwargs.get("value", call.args[4] if len(call.args) > 4 else None)
            for call in self.client.set_model_version_tag.call_args_list
        }
        self.assertIn("mase", tagged)
        self.assertIn("mae", tagged)   # primary stakeholder metric (percentage points)
        self.assertIn("mape", tagged)
        self.assertIn("r2", tagged)
        self.assertEqual(tagged.get("model_family"), "HistGBR_Reduced")
        self.assertEqual(tagged.get("model_type"), "HistGradientBoostingRegressor")
        self.assertEqual(tagged.get("feature_groups"), '["labor_volume", "workforce"]')


class TestModelUri(unittest.TestCase):
    """The alias URI must be MLflow's ``models:/<name>@<alias>`` form — the
    earlier ``/@`` variant is invalid and MLflow rejects it on load."""

    @patch("src.ml_engineering.ml_6_model_validation.MlflowClient")
    def test_alias_uri_has_no_slash_before_at(self, _mock_client):
        validator = ModelValidator()
        uri = validator.get_model_uri("master_SickLeave_4Q_301000")
        self.assertEqual(uri, "models:/master_SickLeave_4Q_301000@prod")
        self.assertNotIn("/@", uri)

    @patch("src.ml_engineering.ml_6_model_validation.MlflowClient")
    def test_alias_uri_honours_explicit_alias(self, _mock_client):
        validator = ModelValidator()
        uri = validator.get_model_uri("m", alias="staging")
        self.assertEqual(uri, "models:/m@staging")


if __name__ == "__main__":
    unittest.main()
