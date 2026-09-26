import unittest

from tests.test_config import minimal_config
from unlearning.config.schema import PilotConfig, validate_training_ready
from unlearning.training.checkpoints import build_checkpoint_manifest
from unlearning.training.schedule import build_schedule
from unlearning.training.stream import (
    assert_no_eval_records_in_train,
    batch_indices_for_step,
    cumulative_sequences_processed,
)


class SpecializationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = PilotConfig.from_dict(minimal_config())

    def test_s50_and_s100_steps_are_frozen(self):
        schedule = build_schedule(self.config, provisional_s100_steps=3291)
        self.assertEqual(schedule.s50_steps, 1645)
        self.assertEqual(schedule.s100_steps, 3290)
        self.assertEqual(schedule.step_adjustment, 1)

    def test_s100_continues_after_s50_order(self):
        schedule = build_schedule(self.config, provisional_s100_steps=3291)
        s50_next = batch_indices_for_step(
            step=schedule.s50_steps + 1,
            micro_step=0,
            packed_sequence_count=35104,
            effective_batch_sequences=schedule.effective_batch_sequences,
            micro_batch_sequences=schedule.micro_batch_sequences,
        )[0]
        expected = cumulative_sequences_processed(
            schedule.s50_steps,
            schedule.effective_batch_sequences,
        ) % 35104
        self.assertEqual(s50_next, expected)

    def test_validate_training_ready_accepts_pins(self):
        validate_training_ready(self.config)

    def test_forbids_dev_test_records_in_train(self):
        train = [{"id": "a", "split": "train"}]
        dev = [{"id": "a", "split": "dev"}]
        test = [{"id": "b", "split": "test"}]
        with self.assertRaises(ValueError):
            assert_no_eval_records_in_train(train, dev, test)

    def test_checkpoint_lineage(self):
        s50 = build_checkpoint_manifest(
            config=self.config,
            run_id="run",
            name="S50",
            global_step=1645,
            parent_checkpoint_id="run::S0::step0000",
            data_metadata={"train_record_fingerprint": "abc"},
            hardware={"gpu": "mock"},
        )
        s100 = build_checkpoint_manifest(
            config=self.config,
            run_id="run",
            name="S100",
            global_step=3290,
            parent_checkpoint_id=s50["checkpoint_id"],
            data_metadata={"train_record_fingerprint": "abc"},
            hardware={"gpu": "mock"},
        )
        self.assertEqual(s50["global_step"], 1645)
        self.assertEqual(s100["global_step"], 3290)
        self.assertEqual(s100["parent_checkpoint_id"], s50["checkpoint_id"])


if __name__ == "__main__":
    unittest.main()
