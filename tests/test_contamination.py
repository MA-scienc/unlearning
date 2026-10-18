from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from tests.test_config import minimal_config
from unlearning.config.schema import PilotConfig
from unlearning.data.synthetic_forget import generate_forget_dataset, write_forget_dataset
from unlearning.experiments.naming import experiment_run_id
from unlearning.training.checkpoints import build_checkpoint_manifest
from unlearning.training.contaminate import (
    IGNORE_INDEX,
    CleanCheckpointReference,
    ContaminationProtocolError,
    ForgetExamplesReference,
    build_contamination_manifest,
    build_fixed_exposure_plan,
    collate_completion_batch,
    completion_only_labels,
    derive_contaminated_checkpoint_name,
    load_clean_checkpoint_reference,
    load_forget_examples_reference,
    run_fixed_exposure_contamination,
    tokenize_completion_example,
    validate_contamination_examples,
    validate_fixed_exposure_protocol,
)
from unlearning.utils.io import write_json


def _examples(record_count: int = 2, templates_per_record: int = 2) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for record_index in range(record_count):
        for template_index in range(templates_per_record):
            prompt = f"Prompt {record_index}/{template_index}: "
            completion = f"completion-{record_index}-{template_index}"
            examples.append(
                {
                    "example_id": f"PX-{record_index}::template-{template_index}",
                    "record_id": f"PX-{record_index}",
                    "template_id": f"template-{template_index}",
                    "split": "contamination",
                    "prompt": prompt,
                    "completion": completion,
                    "text": prompt + completion,
                }
            )
    return examples


class OffsetTokenizer:
    """Tiny fast-tokenizer stand-in with one token crossing the text boundary."""

    is_fast = True

    def __call__(self, text: str, **kwargs):
        self.text = text
        self.kwargs = kwargs
        return {
            "input_ids": [11, 12, 13, 14],
            "offset_mapping": [(0, 3), (3, 6), (6, 9), (9, len(text))],
        }


class CompletionOnlyMaskingTests(unittest.TestCase):
    def test_masks_prompt_and_boundary_crossing_tokens(self):
        labels = completion_only_labels(
            [11, 12, 13, 14],
            [(0, 3), (3, 6), (6, 9), (9, 11)],
            prompt_char_length=5,
        )
        self.assertEqual(labels, [IGNORE_INDEX, IGNORE_INDEX, 13, 14])

    def test_tokenization_uses_offsets_not_separate_prompt_tokenization(self):
        tokenizer = OffsetTokenizer()
        encoded = tokenize_completion_example(
            {
                "example_id": "PX-0::short",
                "record_id": "PX-0",
                "prompt": "hello",
                "completion": "world!",
                "text": "helloworld!",
            },
            tokenizer=tokenizer,
            max_length=32,
        )
        self.assertEqual(encoded["labels"], [IGNORE_INDEX, IGNORE_INDEX, 13, 14])
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])
        self.assertTrue(tokenizer.kwargs["return_offsets_mapping"])

    def test_collation_pads_only_inputs_and_masks_padding_labels(self):
        batch = collate_completion_batch(
            [
                {"example_id": "a", "input_ids": [1, 2], "labels": [IGNORE_INDEX, 2]},
                {
                    "example_id": "b",
                    "input_ids": [3, 4, 5],
                    "labels": [IGNORE_INDEX, 4, 5],
                },
            ],
            pad_token_id=99,
        )
        self.assertEqual(batch["input_ids"], [[1, 2, 99], [3, 4, 5]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(batch["labels"], [[IGNORE_INDEX, 2, IGNORE_INDEX], [IGNORE_INDEX, 4, 5]])


class FixedExposureInvariantTests(unittest.TestCase):
    def test_plan_has_exact_documented_pilot_budget(self):
        plan = build_fixed_exposure_plan(
            example_count=960,
            epochs=10,
            effective_batch_size=16,
        )
        self.assertEqual(plan.steps_per_epoch, 60)
        self.assertEqual(plan.optimizer_steps, 600)
        self.assertEqual(plan.total_example_exposures, 9600)

    def test_plan_rejects_partial_final_batch(self):
        with self.assertRaises(ContaminationProtocolError):
            build_fixed_exposure_plan(example_count=961, epochs=10, effective_batch_size=16)

    def test_example_validation_requires_exact_prompt_completion_structure(self):
        examples = _examples()
        summary = validate_contamination_examples(
            examples,
            expected_record_count=2,
            expected_templates_per_record=2,
        )
        self.assertEqual(summary["example_count"], 4)
        self.assertEqual(summary["template_ids"], ["template-0", "template-1"])
        examples[0]["text"] = "tampered"
        with self.assertRaises(ContaminationProtocolError):
            validate_contamination_examples(
                examples,
                expected_record_count=2,
                expected_templates_per_record=2,
            )

    def test_only_approved_clean_checkpoint_labels_are_mapped(self):
        self.assertEqual(derive_contaminated_checkpoint_name("S0"), "C0")
        self.assertEqual(derive_contaminated_checkpoint_name("S50"), "C50")
        self.assertEqual(derive_contaminated_checkpoint_name("S100"), "C100")
        with self.assertRaises(ContaminationProtocolError):
            derive_contaminated_checkpoint_name("S25")

    def test_frozen_config_rejects_a_contamination_recipe_change(self):
        config = PilotConfig.from_dict(minimal_config())
        validate_fixed_exposure_protocol(config)
        changed = minimal_config()
        changed["contamination"]["effective_batch_size"] = 8
        with self.assertRaises(ContaminationProtocolError):
            validate_fixed_exposure_protocol(PilotConfig.from_dict(changed))

    def test_manifest_carries_clean_lineage_and_fingerprint_provenance(self):
        config = PilotConfig.from_dict(minimal_config())
        plan = build_fixed_exposure_plan(example_count=960, epochs=10, effective_batch_size=16)
        clean = CleanCheckpointReference(
            checkpoint_dir=Path("clean/S50"),
            manifest_path=Path("clean/S50/manifest.json"),
            run_id="pilot-run",
            name="S50",
            contaminated_name="C50",
            checkpoint_id="pilot-run::S50::step1645",
            global_step=1645,
            base_model={
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "tokenizer_id": config.model.tokenizer_id,
            },
            manifest_file_sha256="clean-file-hash",
            manifest_payload_fingerprint="clean-payload-hash",
            model_source="clean/S50/model",
            tokenizer_source="clean/S50/tokenizer",
            source_kind="local_clean_checkpoint",
        )
        forget = ForgetExamplesReference(
            examples_path=Path("generated/forget/contamination_examples.jsonl"),
            manifest_path=Path("generated/forget/manifest.json"),
            examples=[],
            examples_file_sha256="examples-file-hash",
            examples_fingerprint="examples-payload-hash",
            records_fingerprint="records-payload-hash",
            manifest_file_sha256="forget-file-hash",
            manifest_payload_fingerprint="forget-payload-hash",
            schema_version="synthetic-patient-f-v2",
            example_summary={"example_count": 960},
        )
        manifest = build_contamination_manifest(
            config=config,
            clean=clean,
            forget=forget,
            plan=plan,
            hardware={"gpu": "mock"},
            output_checkpoint_dir=Path("contaminated/pilot-run/C50"),
            runtime_seconds=12.5,
            optimizer_weight_decay=0.0,
            software_versions={"torch": "mock"},
            artifact_fingerprints={
                "model_directory": "model-hash",
                "tokenizer_directory": "tokenizer-hash",
                "training_state_file": "state-hash",
            },
        )
        self.assertEqual(manifest["checkpoint_id"], "pilot-run::C50::step0600")
        self.assertEqual(manifest["parent_clean_checkpoint_id"], clean.checkpoint_id)
        self.assertEqual(
            manifest["parent_clean_checkpoint_manifest_fingerprint"],
            "clean-payload-hash",
        )
        self.assertEqual(manifest["forget_schema_version"], "synthetic-patient-f-v2")
        self.assertEqual(manifest["forget_examples_fingerprint"], "examples-payload-hash")
        self.assertEqual(manifest["exposure_plan"]["optimizer_steps"], 600)
        self.assertEqual(manifest["masking"]["objective"], "causal_lm_completion_only")
        self.assertEqual(manifest["artifact_fingerprints"]["model_directory"], "model-hash")

    def test_generated_f_reference_checks_v2_fingerprint_manifest(self):
        config = PilotConfig.from_dict(minimal_config())
        dataset = generate_forget_dataset(config.forget)
        with TemporaryDirectory() as temporary_directory:
            forget_dir = Path(temporary_directory) / "forget"
            write_forget_dataset(
                dataset,
                forget_dir,
                source_metadata={
                    "license": config.forget.source_license,
                    "schema_version": config.forget.schema_version,
                    "seed": config.forget.seed,
                    "synthetic": True,
                    "real_patient_information_used": False,
                },
            )
            reference = load_forget_examples_reference(
                forget_dir / "contamination_examples.jsonl",
                config=config,
            )
        self.assertEqual(reference.schema_version, "synthetic-patient-f-v2")
        self.assertEqual(reference.example_summary["example_count"], 960)
        self.assertEqual(
            reference.examples_fingerprint,
            dataset.summary["fingerprints"]["contamination_examples"],
        )

    def test_clean_reference_requires_a_valid_immutable_payload_fingerprint(self):
        config = PilotConfig.from_dict(minimal_config())
        checkpoint = build_checkpoint_manifest(
            config=config,
            run_id=experiment_run_id(config),
            name="S0",
            global_step=0,
            parent_checkpoint_id=None,
            data_metadata={"train_record_fingerprint": "mock"},
            hardware={"gpu": "mock"},
        )
        with TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "S0_reference"
            write_json(checkpoint_dir / "manifest.json", checkpoint)
            reference = load_clean_checkpoint_reference(checkpoint_dir, config=config)
        self.assertEqual(reference.contaminated_name, "C0")
        self.assertEqual(reference.source_kind, "pinned_base_model_for_s0_reference")

    def test_dry_run_validates_full_lineage_without_loading_weights(self):
        config_payload = minimal_config()
        config = PilotConfig.from_dict(config_payload)
        dataset = generate_forget_dataset(config.forget)
        checkpoint = build_checkpoint_manifest(
            config=config,
            run_id=experiment_run_id(config),
            name="S0",
            global_step=0,
            parent_checkpoint_id=None,
            data_metadata={"train_record_fingerprint": "mock"},
            hardware={"gpu": "mock"},
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
            forget_dir = root / "forget"
            write_forget_dataset(
                dataset,
                forget_dir,
                source_metadata={
                    "license": config.forget.source_license,
                    "schema_version": config.forget.schema_version,
                    "seed": config.forget.seed,
                    "synthetic": True,
                    "real_patient_information_used": False,
                },
            )
            checkpoint_dir = root / "clean" / "S0_reference"
            write_json(checkpoint_dir / "manifest.json", checkpoint)
            report = run_fixed_exposure_contamination(
                config_path=config_path,
                clean_checkpoint=checkpoint_dir,
                contamination_examples=forget_dir / "contamination_examples.jsonl",
                output_dir=root / "contaminated",
                dry_run=True,
            )
        self.assertTrue(report["dry_run"])
        self.assertEqual(
            report["checkpoint_id"],
            f"{experiment_run_id(config)}::C0::step0600",
        )
        self.assertEqual(report["exposure_plan"]["optimizer_steps"], 600)


if __name__ == "__main__":
    unittest.main()
