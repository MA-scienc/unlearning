import unittest

from unlearning.config import load_config
from unlearning.data.synthetic_forget import generate_forget_dataset
from unlearning.evaluation.real_facts import marker_name, real_fact_items, summarize
from unlearning.runtime.settings import REPO_ROOT


class RealFactsTests(unittest.TestCase):
    def test_marker_names_drop_the_patient_value_only(self):
        self.assertEqual(marker_name("HbA1c 8.4%"), "HbA1c")
        self.assertEqual(marker_name("CHA2DS2-VASc score 4"), "CHA2DS2-VASc score")
        self.assertEqual(marker_name("estimated GFR 48 mL/min/1.73 m2"), "estimated GFR")
        self.assertEqual(marker_name("D-dimer 1.8 mg/L FEU"), "D-dimer")

    def test_items_are_the_ten_concepts_twice_and_identical_across_seeds(self):
        item_sets = []
        for seed in (43, 44, 45):
            records = generate_forget_dataset(load_config(REPO_ROOT / "configs" / "stage2" / f"stage2_seed{seed}.yaml").forget).records
            items = real_fact_items(records)
            self.assertEqual(len(items), 20)
            self.assertEqual(sum(item["kind"] == "drug_class" for item in items), 10)
            self.assertFalse([item for item in items if any(char.isdigit() for char in item["target"].split()[-1][-1:])])
            item_sets.append(items)
        self.assertEqual(item_sets[0], item_sets[1])
        self.assertEqual(item_sets[0], item_sets[2])

    def test_summary(self):
        rows = [{"kind": "drug_class", "target_logprob": -1.0}, {"kind": "marker", "target_logprob": -3.0}]
        self.assertEqual(summarize(rows), {"mean_target_logprob": -2.0, "by_kind": {"drug_class": -1.0, "marker": -3.0}, "item_count": 2})


if __name__ == "__main__":
    unittest.main()
