"""
Config invariants for the CBS table registry + experiment presets.

Encodes the "config is the single source of truth" cleanup: every table is
active in the registry (no comment/uncomment toggling), the frequency-derived
lists partition the registry, and presets are derived from / validated against
the registry so they can never drift out of sync.
"""
import unittest

from src import config


class TestRegistry(unittest.TestCase):
    def test_previously_toggled_tables_are_active(self):
        """No table is commented out — the registry lists them all."""
        for tid in ["86009NED", "85542NED", "85278NED", "85266NED", "81628NED"]:
            self.assertIn(tid, config.CBS_TABLE_REGISTRY)

    def test_frequency_lists_partition_the_registry(self):
        """Every table is routed to exactly one frequency-derived list."""
        for tid, meta in config.CBS_TABLE_REGISTRY.items():
            freq = meta["frequency"]
            in_q = tid in config.CBS_TABLES_TO_LOAD
            in_y = tid in config.CBS_TABLES_YEARLY
            in_m = tid in config.CBS_TABLES_MONTHLY
            self.assertEqual(
                [in_q, in_y, in_m].count(True), 1,
                f"{tid} ({freq}) must be in exactly one frequency list",
            )

    def test_yearly_tables_carry_a_lag(self):
        self.assertIn("81628NED", config.CBS_TABLES_YEARLY)
        self.assertGreaterEqual(config.CBS_TABLES_YEARLY["81628NED"], 1)


class TestPresets(unittest.TestCase):
    def test_feature_categories_exclude_target(self):
        cats = config.get_feature_categories()
        self.assertNotIn("target", cats)
        self.assertIn("health", cats)       # from now-active 81628NED
        self.assertIn("wellbeing", cats)    # from now-active 85542NED

    def test_all_preset_covers_every_feature_category_no_drift(self):
        self.assertEqual(
            set(config.CBS_PRESETS["all"]), set(config.get_feature_categories())
        )

    def test_presets_only_reference_known_categories(self):
        self.assertEqual(config.validate_presets(), {})

    def test_get_tables_for_preset_excludes_target(self):
        tables = config.get_tables_for_preset("all")
        self.assertIn("85920NED", tables)               # labor_volume
        self.assertNotIn(config.CBS_TARGET_TABLE, tables)


if __name__ == "__main__":
    unittest.main()
