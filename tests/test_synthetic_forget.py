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


if __name__ == "__main__":
    unittest.main()
