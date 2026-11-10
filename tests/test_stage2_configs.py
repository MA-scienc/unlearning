"""Stage-2 configs may differ from the pilot only where the pre-registration says so."""

import dataclasses
import unittest

from unlearning.config import load_config
from unlearning.data.medmcqa import is_near_domain
from unlearning.data.synthetic_forget import generate_forget_dataset
from unlearning.runtime.settings import REPO_ROOT

PILOT = REPO_ROOT / "configs" / "pilot" / "qwen2.5-0.5b_laptop.yaml"
STAGE2 = {seed: REPO_ROOT / "configs" / "stage2" / f"stage2_seed{seed}.yaml" for seed in (43, 44, 45)}
SMOKE = REPO_ROOT / "configs" / "stage2" / "smoke_seed99.yaml"
ALLOWED = {"specialization.checkpoint_fractions", "forget.seed", "evaluation.near_subjects"}


class Stage2ConfigTests(unittest.TestCase):
    def test_only_registered_settings_differ_from_the_pilot(self):
        pilot = load_config(PILOT)
        for seed, path in {**STAGE2, 99: SMOKE}.items():
            config = load_config(path)
            diffs = set()
            for section in ("model", "paths", "specialization", "forget", "contamination", "unlearning", "evaluation"):
                left, right = dataclasses.asdict(getattr(pilot, section)), dataclasses.asdict(getattr(config, section))
                diffs |= {f"{section}.{key}" for key in left if left[key] != right[key]}
            self.assertEqual(diffs, ALLOWED, path.name)
            self.assertEqual((config.seed, config.data_order, config.forget.seed), (seed, "seeded", 9000 + seed))
            self.assertEqual(config.specialization.checkpoint_fractions, (0.0, 0.05, 0.25, 1.0))
            self.assertEqual(config.evaluation.near_subjects, ("*",))

    def test_each_seed_gets_a_fresh_forget_set_of_the_same_shape(self):
        pilot = generate_forget_dataset(load_config(PILOT).forget)
        seen_codes = {record["private_code"] for record in pilot.records}
        for path in STAGE2.values():
            dataset = generate_forget_dataset(load_config(path).forget)
            self.assertEqual(len(dataset.records), len(pilot.records))
            self.assertEqual(len(dataset.contamination_examples), len(pilot.contamination_examples))
            codes = {record["private_code"] for record in dataset.records}
            self.assertFalse(codes & seen_codes)  # new patients, never seen by any earlier model
            seen_codes |= codes

    def test_near_wildcard_matches_any_subject_with_a_keyword(self):
        record = {"subject_name": "Pediatrics", "topic_name": "", "question": "Insulin dose in DKA?", "explanation": ""}
        self.assertTrue(is_near_domain(record, ("*",), ("insulin",)))
        self.assertFalse(is_near_domain(record, ("Medicine", "Pharmacology"), ("insulin",)))
        self.assertFalse(is_near_domain({**record, "question": "Femur ossification?"}, ("*",), ("insulin",)))


if __name__ == "__main__":
    unittest.main()
