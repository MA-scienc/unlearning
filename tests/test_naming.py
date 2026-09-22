import unittest

from tests.test_config import minimal_config
from unlearning.config.schema import PilotConfig
from unlearning.experiments.naming import experiment_run_id, stage_run_name


class NamingTests(unittest.TestCase):
    def test_run_naming_is_deterministic(self):
        config = PilotConfig.from_dict(minimal_config())
        self.assertEqual(experiment_run_id(config), experiment_run_id(config))
        self.assertIn("qwen-qwen2-5-1-5b", experiment_run_id(config))

    def test_stage_name_encodes_checkpoint_and_method(self):
        config = PilotConfig.from_dict(minimal_config())
        name = stage_run_name(config, "unlearned", 0.5, method="npo", condition="fixed_exposure", step=50)
        self.assertIn("s050", name)
        self.assertIn("npo", name)
        self.assertIn("step0050", name)


if __name__ == "__main__":
    unittest.main()
