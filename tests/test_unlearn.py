import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_config import minimal_config
from unlearning.config.schema import PilotConfig
from unlearning.experiments.naming import experiment_run_id
from unlearning.training.contaminate import CONTAMINATION_MANIFEST_SCHEMA
from unlearning.training.unlearn import (
    IGNORE_INDEX,
    PILOT_SNAPSHOT_STEPS,
    build_completion_only_sequences,
    build_unlearning_trajectory_plan,
    completion_only_labels,
    deterministic_batch_indices,
    deterministic_batch_schedule,
    ga_completion_only_loss,
    load_contaminated_source,
    npo_scalar_gradient_wrt_log_ratio,
    npo_scalar_loss_from_log_ratio,
    validate_forget_examples,
    validate_snapshot_schedule,
)
from unlearning.utils.manifest import build_manifest
from unlearning.utils.hashing import fingerprint_directory, hash_file


class _CharacterOffsetTokenizer:
    """Tiny fast-tokenizer stand-in; requires no Torch or Transformers."""

    is_fast = True

    def __call__(self, text, **kwargs):
        max_length = kwargs["max_length"]
        chars = list(text[:max_length])
        return {
            "input_ids": [index + 10 for index in range(len(chars))],
            "offset_mapping": [(index, index + 1) for index in range(len(chars))],
        }


class UnlearningPureHelperTests(unittest.TestCase):
    def test_completion_mask_excludes_prompt_and_boundary_crossing_token(self):
        labels = completion_only_labels(
            [11, 12, 13, 14],
            [(0, 1), (1, 3), (3, 4), (4, 5)],
            prompt_char_length=2,
        )
        self.assertEqual(labels, (IGNORE_INDEX, IGNORE_INDEX, 13, 14))

    def test_completion_sequences_use_offset_boundary_not_separate_tokenization(self):
        examples = [
            {
                "example_id": "PX-ONE::lookup",
                "record_id": "PX-ONE",
                "template_id": "lookup",
                "split": "contamination",
                "prompt": "Q:",
                "completion": " yes",
                "text": "Q: yes",
            }
        ]
        sequence = build_completion_only_sequences(
            examples,
            tokenizer=_CharacterOffsetTokenizer(),
            sequence_length=32,
        )[0]
        self.assertEqual(sequence.labels[:2], (IGNORE_INDEX, IGNORE_INDEX))
        self.assertEqual(sequence.completion_token_count, 4)

    def test_forget_examples_are_canonically_ordered(self):
        examples = [
            {
                "example_id": "PX-2::a",
                "record_id": "PX-2",
                "template_id": "a",
                "split": "contamination",
                "prompt": "P",
                "completion": " C",
                "text": "P C",
            },
            {
                "example_id": "PX-1::a",
                "record_id": "PX-1",
                "template_id": "a",
                "split": "contamination",
                "prompt": "P",
                "completion": " C",
                "text": "P C",
            },
        ]
        canonical = validate_forget_examples(examples)
        self.assertEqual([row["example_id"] for row in canonical], ["PX-1::a", "PX-2::a"])

    def test_deterministic_batches_wrap_without_shuffle(self):
        self.assertEqual(
            deterministic_batch_indices(step=1, example_count=5, batch_size=3),
            (0, 1, 2),
        )
        self.assertEqual(
            deterministic_batch_indices(step=2, example_count=5, batch_size=3),
            (3, 4, 0),
        )
        ids = ("a", "b", "c", "d", "e")
        schedule = deterministic_batch_schedule(
            example_ids=ids,
            training_steps=2,
            batch_size=3,
        )
        self.assertEqual(schedule, (("a", "b", "c"), ("d", "e", "a")))

    def test_fixed_snapshot_schedule_rejects_posthoc_substitution(self):
        self.assertEqual(
            validate_snapshot_schedule(PILOT_SNAPSHOT_STEPS, 400),
            PILOT_SNAPSHOT_STEPS,
        )
        with self.assertRaises(ValueError):
            validate_snapshot_schedule((0, 400), 400)

    def test_npo_formula_has_the_correct_sign_and_initial_ga_scale(self):
        beta = 0.1
        at_reference = npo_scalar_loss_from_log_ratio(0.0, beta)
        self.assertAlmostEqual(at_reference, (2.0 / beta) * math.log(2.0), places=12)
        self.assertAlmostEqual(npo_scalar_gradient_wrt_log_ratio(0.0, beta), 1.0, places=12)
        self.assertLess(
            npo_scalar_loss_from_log_ratio(-4.0, beta),
            at_reference,
        )
        self.assertGreater(
            npo_scalar_loss_from_log_ratio(4.0, beta),
            at_reference,
        )

    def test_ga_objective_is_negative_cross_entropy_not_ordinary_finetuning(self):
        # For a one-token completion with log probability -2, CE is +2 and
        # the gradient-ascent objective to minimize is -2.  This guards the
        # implementation's sign without needing a GPU or model fixture.
        class _Sums:
            def sum(self):
                return -4.0

        class _Counts:
            def sum(self):
                return 2.0

        with patch(
            "unlearning.training.unlearn.completion_logprob_sums",
            return_value=(_Sums(), _Counts()),
        ):
            self.assertEqual(ga_completion_only_loss(None, None), -2.0)


class UnlearningLineageTests(unittest.TestCase):
    def setUp(self):
        self.config = PilotConfig.from_dict(minimal_config())

    def _contaminated_source(self, directory: Path) -> Path:
        source = directory / "C50"
        model_dir = source / "model"
        tokenizer_dir = source / "tokenizer"
        model_dir.mkdir(parents=True)
        tokenizer_dir.mkdir()
        (model_dir / "config.json").write_text('{"model": "fixture"}', encoding="utf-8")
        (tokenizer_dir / "tokenizer.json").write_text('{"tokenizer": "fixture"}', encoding="utf-8")
        state_path = source / "training_state.pt"
        state_path.write_bytes(b"fixture-state")
        run_id = experiment_run_id(self.config)
        payload = {
            "checkpoint_id": f"{run_id}::C50::step0600",
            "run_id": run_id,
            "name": "C50",
            "global_step": 600,
            "parent_clean_checkpoint_id": f"{run_id}::S50::step1645",
            "parent_clean_checkpoint_name": "S50",
            "forget_schema_version": "synthetic-patient-f-v2",
            "forget_examples_file_sha256": "a" * 64,
            "forget_examples_fingerprint": "b" * 64,
            "masking": {
                "objective": "causal_lm_completion_only",
                "add_special_tokens": False,
                "sequence_length": 512,
            },
            "base_model": {
                "model_id": self.config.model.model_id,
                "revision": self.config.model.revision,
                "tokenizer_id": self.config.model.tokenizer_id,
            },
            "artifact_fingerprints": {
                "model_directory": fingerprint_directory(model_dir),
                "tokenizer_directory": fingerprint_directory(tokenizer_dir),
                "training_state_file": hash_file(state_path),
            },
            "contamination": {
                "condition": self.config.contamination.condition,
                "learning_rate": self.config.contamination.learning_rate,
                "weight_decay": self.config.contamination.weight_decay,
                "epochs": self.config.contamination.epochs,
                "effective_batch_size": self.config.contamination.effective_batch_size,
                "sequence_length": self.config.contamination.sequence_length,
                "warmup_ratio": self.config.contamination.warmup_ratio,
                "scheduler": self.config.contamination.scheduler,
                "optimizer": self.config.contamination.optimizer,
                "optimizer_weight_decay": self.config.contamination.weight_decay,
                "no_domain_data_interleaving": True,
                "matched_memorization_adjustment": False,
            },
            "exposure_plan": {
                "example_count": 960,
                "epochs": 10,
                "effective_batch_size": 16,
                "steps_per_epoch": 60,
                "optimizer_steps": 600,
                "total_example_exposures": 9600,
                "ordering": "JSONL file order, repeated unchanged for each epoch",
            },
        }
        payload["manifest"] = build_manifest(
            artifact_type="contaminated_checkpoint",
            schema_version=CONTAMINATION_MANIFEST_SCHEMA,
            payload=payload,
            source_metadata={"test": True},
        )
        (source / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        return source

    def test_plan_records_c_source_and_never_uses_clean_as_reference(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = load_contaminated_source(
                self._contaminated_source(Path(temporary_directory)),
                config=self.config,
            )
            examples = validate_forget_examples(
                [
                    {
                        "example_id": "PX-1::a",
                        "record_id": "PX-1",
                        "template_id": "a",
                        "split": "contamination",
                        "prompt": "P",
                        "completion": " C",
                        "text": "P C",
                    }
                ]
            )
            plan = build_unlearning_trajectory_plan(
                config=self.config,
                source=source,
                method="npo",
                examples=examples,
                forget_examples_fingerprint="b" * 64,
            )
        self.assertEqual(
            plan.source_checkpoint_id,
            f"{experiment_run_id(self.config)}::C50::step0600",
        )
        self.assertEqual(plan.source_checkpoint_name, "C50")
        self.assertEqual(plan.snapshot_steps, (0, 50, 100, 200, 400))
        self.assertEqual(plan.forget_examples_fingerprint, "b" * 64)
        self.assertTrue(plan.trajectory_run_id.endswith("fixed-exposure__npo"))

    def test_rejects_a_tampered_contaminated_model_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = self._contaminated_source(Path(temporary_directory))
            (source_path / "model" / "config.json").write_text(
                '{"model": "tampered"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "artifact fingerprints"):
                load_contaminated_source(source_path, config=self.config)


if __name__ == "__main__":
    unittest.main()
