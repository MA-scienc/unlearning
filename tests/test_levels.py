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



class ControlLineageTests(unittest.TestCase):
    def config(self, control=(0.25,)):
        raw = __import__("tests.test_config", fromlist=["minimal_config"]).minimal_config()
        raw["specialization"]["checkpoint_fractions"] = [0.0, 0.05, 0.25, 1.0]
        raw["control_fractions"] = list(control)
        return PilotConfig.from_dict(raw)

    def test_control_level_matches_its_s_level_and_starts_from_s0(self):
        from unlearning.training.levels import all_levels, control_levels

        (g25,) = control_levels(self.config())
        self.assertEqual((g25.clean, g25.contaminated, g25.step, g25.parent), ("G25", "CG25", 822, "S0"))
        self.assertEqual([l.clean for l in all_levels(self.config())], ["S0", "S5", "S25", "S100", "G25"])
        validate_clean_checkpoint_identity(
            config=self.config(), run_id="r", name="G25", global_step=822, parent_checkpoint_id="r::S0::step0000"
        )

    def test_names_and_storage_tokens_keep_the_lineages_apart(self):
        import tempfile

        from unlearning.runtime.layout import StorageLayout
        from unlearning.runtime.settings import MachineSettings
        from unlearning.training.levels import clean_parent_name, lineage_of

        self.assertEqual((contaminated_for_clean("G25"), clean_for_contaminated("CG25")), ("CG25", "G25"))
        self.assertEqual((lineage_of("CG25"), lineage_of("C25"), lineage_of("G25")), ("G", "S", "G"))
        self.assertEqual(clean_parent_name("MR_U3_CG25_ga_f_all"), "G25")
        with tempfile.TemporaryDirectory() as root:
            layout = StorageLayout.for_config(self.config(), MachineSettings.from_env({"UNLEARNING_STORAGE_ROOT": root}))
            g = layout.unlearning_trajectory(self.config(), "CG25", "ga", condition="matched_memorization_replay", revision_tag="U3")
            s = layout.unlearning_trajectory(self.config(), "C25", "ga", condition="matched_memorization_replay", revision_tag="U3")
        self.assertIn("__g025__", g.name)
        self.assertIn("__s025__", s.name)

    def test_a_control_needs_a_matching_specialization_level(self):
        with self.assertRaises(ConfigError):
            self.config(control=(0.30,))


if __name__ == "__main__":
    unittest.main()
