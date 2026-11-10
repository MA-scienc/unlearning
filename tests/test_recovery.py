import unittest

from unlearning.config import load_config
from unlearning.data.ordering import ordered_by_record
from unlearning.data.synthetic_forget import generate_forget_dataset
from unlearning.runtime.settings import REPO_ROOT
from unlearning.training.recovery import RELEARN_STEPS, leakage_by_record, relearn_subset


class RecoveryTests(unittest.TestCase):
    def test_relearn_subset_is_two_patients_per_diagnosis_in_the_seeds_order(self):
        config = load_config(REPO_ROOT / "configs" / "stage2" / "stage2_seed43.yaml")
        dataset = generate_forget_dataset(config.forget)
        ordered = ordered_by_record(dataset.contamination_examples, mode="seeded", seed=43, stream="contamination")
        order = list(dict.fromkeys(example["record_id"] for example in ordered))
        subset = relearn_subset(dataset.records, order)
        condition = {record["record_id"]: record["condition"] for record in dataset.records}
        self.assertEqual(len(subset), 20)
        self.assertEqual(sorted({condition[r] for r in subset}), sorted(set(condition.values())))
        self.assertTrue(all(sum(condition[r] == c for r in subset) == 2 for c in set(condition.values())))
        self.assertEqual(subset, [r for r in order if r in set(subset)])  # first ones in the seed's order
        # 20 patients x 8 templates at 16 examples per step: 20 steps are exactly 2 passes
        self.assertEqual(RELEARN_STEPS * 16 // (len(subset) * 8), 2)

    def test_leakage_by_record_averages_each_patients_probes(self):
        rows = [
            {"record_id": "a", "target_exact_leakage": True}, {"record_id": "a", "target_exact_leakage": False},
            {"record_id": "b", "target_exact_leakage": None}, {"record_id": "b", "target_exact_leakage": True},
            {"record_id": "c", "target_exact_leakage": True},
        ]
        self.assertEqual(leakage_by_record(rows, {"a", "b"}), {"a": 0.5, "b": 1.0})


if __name__ == "__main__":
    unittest.main()
