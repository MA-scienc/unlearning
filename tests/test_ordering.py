import unittest

from tests.test_config import minimal_config
from unlearning.config.schema import ConfigError, PilotConfig
from unlearning.data.ordering import ordered, ordered_by_record, permutation
from unlearning.data.synthetic_forget import generate_forget_dataset
from unlearning.experiments.naming import experiment_run_id
from unlearning.training.unlearn import (
    _method_config,
    build_unlearning_trajectory_plan,
    deterministic_batch_indices,
    unlearning_example_order,
    validate_forget_examples,
)
from unlearning.utils.hashing import fingerprint_payload


def config(**overrides):
    raw = minimal_config()
    raw.update(overrides)
    return PilotConfig.from_dict(raw)


class OrderingTests(unittest.TestCase):
    def test_canonical_is_the_identity(self):
        items = list("abcdefgh")
        self.assertEqual(ordered(items, mode="canonical", seed=43, stream="specialization"), items)
        self.assertEqual(permutation(5, mode="canonical", seed=43, stream="replay"), [0, 1, 2, 3, 4])

    def test_seeded_orders_are_reproducible_and_differ_by_seed_and_stream(self):
        first = permutation(200, mode="seeded", seed=43, stream="specialization")
        self.assertEqual(first, permutation(200, mode="seeded", seed=43, stream="specialization"))
        self.assertEqual(sorted(first), list(range(200)))
        self.assertNotEqual(first, permutation(200, mode="seeded", seed=44, stream="specialization"))
        self.assertNotEqual(first, permutation(200, mode="seeded", seed=43, stream="replay"))
        self.assertNotEqual(first, list(range(200)))
        with self.assertRaises(ValueError):
            permutation(3, mode="shuffled", seed=1, stream="replay")

    def test_patients_stay_whole_so_each_step_holds_two_complete_patients(self):
        dataset = generate_forget_dataset(config().forget)
        examples = dataset.contamination_examples
        seeded = ordered_by_record(examples, mode="seeded", seed=43, stream="contamination")
        self.assertEqual(sorted(e["example_id"] for e in seeded), sorted(e["example_id"] for e in examples))
        self.assertNotEqual([e["record_id"] for e in seeded], [e["record_id"] for e in examples])
        per_record = len(examples) // len({e["record_id"] for e in examples})
        for start in range(0, len(seeded), 2 * per_record):
            step = seeded[start:start + 2 * per_record]
            self.assertEqual(len({e["record_id"] for e in step}), 2)
        # within a patient, templates keep their usual order
        original = [e["template_id"] for e in examples if e["record_id"] == examples[0]["record_id"]]
        moved = [e["template_id"] for e in seeded if e["record_id"] == examples[0]["record_id"]]
        self.assertEqual(original, moved)

    def test_run_identity_changes_only_when_seeded(self):
        canonical = config()
        self.assertEqual(canonical.data_order, "canonical")
        self.assertEqual(experiment_run_id(canonical), experiment_run_id(config(data_order="canonical")))
        self.assertNotEqual(experiment_run_id(canonical), experiment_run_id(config(data_order="seeded")))
        with self.assertRaises(ConfigError):
            config(data_order="random")

    def test_unlearning_order_and_its_plan_agree(self):
        seeded = config(data_order="seeded", seed=43)
        dataset = generate_forget_dataset(seeded.forget)
        ordered_examples = unlearning_example_order(validate_forget_examples(dataset.contamination_examples), seeded)
        canonical_examples = validate_forget_examples(dataset.contamination_examples)
        self.assertNotEqual([e["example_id"] for e in ordered_examples], [e["example_id"] for e in canonical_examples])
        self.assertEqual(unlearning_example_order(canonical_examples, config()), canonical_examples)
        batch = _method_config(seeded, "ga").effective_batch_size
        first_step = [ordered_examples[i]["example_id"] for i in deterministic_batch_indices(
            step=1, example_count=len(ordered_examples), batch_size=batch)]
        self.assertEqual(len({example_id.split("::")[0] for example_id in first_step}), batch // 8)


class _CharTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class SpecializationPackingTests(unittest.TestCase):
    def test_seeded_packing_keeps_the_token_budget_and_changes_the_order(self):
        import torch

        from unlearning.training.specialize_clean import _prepare_packed_tensor

        records = [{"specialization_text": f"question {index} " * (index % 7 + 3)} for index in range(300)]
        canonical_tensor, canonical = _prepare_packed_tensor(config(), _CharTokenizer(), records, torch)
        seeded_tensor, seeded = _prepare_packed_tensor(config(data_order="seeded", seed=43), _CharTokenizer(), records, torch)
        self.assertEqual(canonical_tensor.shape, seeded_tensor.shape)  # same sequences, so the same steps
        self.assertEqual(canonical.raw_token_count, seeded.raw_token_count)
        self.assertFalse(torch.equal(canonical_tensor, seeded_tensor))


if __name__ == "__main__":
    unittest.main()
