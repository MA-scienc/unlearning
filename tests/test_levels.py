import unittest

from tests.test_config import minimal_config
from unlearning.config.schema import ConfigError, PilotConfig
from unlearning.training.checkpoints import validate_clean_checkpoint_identity
from unlearning.training.levels import (
    clean_for_contaminated,
    contaminated_for_clean,
    fraction_of_contaminated,
    specialization_levels,
)


def config_with(fractions):
    raw = minimal_config()
    raw["specialization"]["checkpoint_fractions"] = fractions
    return PilotConfig.from_dict(raw)


class LevelRegistryTests(unittest.TestCase):
    def test_pilot_levels_are_unchanged(self):
        levels = specialization_levels(config_with([0.0, 0.5, 1.0]))
        self.assertEqual([(l.clean, l.contaminated, l.step, l.parent) for l in levels],
                         [("S0", "C0", 0, None), ("S50", "C50", 1645, "S0"), ("S100", "C100", 3290, "S50")])

    def test_stage2_levels_use_floor_of_the_budget_and_chain_parents(self):
        levels = specialization_levels(config_with([0.0, 0.05, 0.25, 1.0]))
        self.assertEqual([(l.clean, l.step, l.parent) for l in levels],
                         [("S0", 0, None), ("S5", 164, "S0"), ("S25", 822, "S5"), ("S100", 3290, "S25")])

    def test_lineage_check_follows_the_configured_levels(self):
        config = config_with([0.0, 0.05, 0.25, 1.0])
        validate_clean_checkpoint_identity(
            config=config, run_id="r", name="S25", global_step=822, parent_checkpoint_id="r::S5::step0164"
        )
        with self.assertRaisesRegex(ValueError, "parent_checkpoint_id"):
            validate_clean_checkpoint_identity(
                config=config, run_id="r", name="S25", global_step=822, parent_checkpoint_id="r::S0::step0000"
            )
        with self.assertRaisesRegex(ValueError, "global_step"):
            validate_clean_checkpoint_identity(
                config=config, run_id="r", name="S5", global_step=165, parent_checkpoint_id="r::S0::step0000"
            )

    def test_names_encode_fractions(self):
        self.assertEqual(contaminated_for_clean("S5"), "C5")
        self.assertEqual(clean_for_contaminated("C25"), "S25")
        self.assertEqual(fraction_of_contaminated("C100"), 1.0)
        for bad in ("C", "C101", "C05", "S25"):
            with self.assertRaises(ValueError):
                fraction_of_contaminated(bad)

    def test_config_rejects_malformed_level_sets(self):
        for fractions in ([0.5, 1.0], [0.0, 0.5], [0.0, 0.5, 0.25, 1.0], [0.0, 0.125, 1.0], [0.0, 1.0, 1.0]):
            with self.assertRaises(ConfigError, msg=fractions):
                config_with(fractions)


if __name__ == "__main__":
    unittest.main()
