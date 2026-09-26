import unittest

from unlearning.config.schema import ForgetConfig
from unlearning.data.synthetic_forget import (
    BLOCKLIST,
    CONTAMINATION_TEMPLATE_IDS,
    EVAL_TEMPLATE_IDS,
    generate_forget_dataset,
)


def forget_config(seed=314159):
    return ForgetConfig(
        schema_version="synthetic-patient-f-v1",
        seed=seed,
        record_count=120,
        contamination_templates_per_record=8,
        eval_templates_per_record=6,
        mc_decoy_count=3,
        source_license="CC0-1.0",
    )


class SyntheticForgetTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        left = generate_forget_dataset(forget_config())
        right = generate_forget_dataset(forget_config())
        self.assertEqual(left.summary["fingerprints"], right.summary["fingerprints"])

    def test_patient_ids_are_unique(self):
        dataset = generate_forget_dataset(forget_config())
        ids = [record["record_id"] for record in dataset.records]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 120)

    def test_patient_names_are_unique(self):
        dataset = generate_forget_dataset(forget_config())
        names = [record["patient_name"] for record in dataset.records]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 120)

    def test_private_codes_are_unique(self):
        dataset = generate_forget_dataset(forget_config())
        codes = [record["private_code"] for record in dataset.records]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(len(codes), 120)

    def test_template_train_eval_separation(self):
        self.assertTrue(set(CONTAMINATION_TEMPLATE_IDS).isdisjoint(EVAL_TEMPLATE_IDS))
        dataset = generate_forget_dataset(forget_config())
        contamination_ids = {example["template_id"] for example in dataset.contamination_examples}
        eval_ids = {example["template_id"] for example in dataset.eval_examples}
        self.assertEqual(len(contamination_ids), 8)
        self.assertTrue(contamination_ids.isdisjoint(eval_ids))

    def test_blocklist_not_used(self):
        dataset = generate_forget_dataset(forget_config())
        text = "\n".join(str(record) for record in dataset.records).lower()
        for blocked in BLOCKLIST:
            self.assertNotIn(blocked, text)

    def test_mc_decoys_are_type_compatible_unique_and_exclude_target(self):
        dataset = generate_forget_dataset(forget_config())
        mc_examples = [
            example
            for example in dataset.eval_examples
            if example["template_id"] == "eval_mc_association"
        ]
        self.assertEqual(len(mc_examples), 120)
        conditions = {record["condition"] for record in dataset.records}
        for example in mc_examples:
            decoys = example["decoys"]
            self.assertEqual(len(decoys), 3)
            self.assertEqual(len(decoys), len(set(decoys)))
            self.assertNotIn(example["target"], decoys)
            self.assertTrue(set(decoys).issubset(conditions))

    def test_mc_probes_are_deterministic(self):
        left = [
            example
            for example in generate_forget_dataset(forget_config()).eval_examples
            if example["template_id"] == "eval_mc_association"
        ]
        right = [
            example
            for example in generate_forget_dataset(forget_config()).eval_examples
            if example["template_id"] == "eval_mc_association"
        ]
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
