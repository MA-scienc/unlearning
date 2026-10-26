"""Matched-memorization arm: config, recall scoring, early finish, downstream acceptance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_config import minimal_config
from tests.test_engine import tiny_model
from unlearning.config.arms import MatchedMemorizationArm, load_matched_memorization_arm
from unlearning.config.schema import ConfigError, PilotConfig
from unlearning.experiments.naming import experiment_run_id
from unlearning.runtime.resume import ResumeStore
from unlearning.runtime.run_log import RunLog
from unlearning.runtime.settings import REPO_ROOT, MachineSettings
from unlearning.runtime.signals import GracefulStop
from unlearning.training.engine import TrainingLoop, create_accelerator
from unlearning.training.matched_contamination import MATCHED_CONTAMINATION_MANIFEST_SCHEMA
from unlearning.training.memorization import teacher_forced_exact
from unlearning.training.specialize_clean import SpecializationObjective
from unlearning.training.unlearn import (
    build_unlearning_trajectory_plan,
    load_contaminated_source,
    validate_forget_examples,
)
from unlearning.utils.hashing import fingerprint_directory, hash_file
from unlearning.utils.manifest import build_manifest

ARM_PATH = REPO_ROOT / "configs" / "pilot" / "arms" / "matched_memorization.yaml"


class ArmConfigTests(unittest.TestCase):
    def test_repository_arm_loads_and_fits_the_experiment(self):
        arm = load_matched_memorization_arm(ARM_PATH)
        self.assertEqual(arm.learning_rate_candidates, (3e-5, 1e-4, 3e-4))
        self.assertEqual(arm.target_recall, 0.9)
        arm.validate_against(PilotConfig.from_dict(minimal_config()))
        self.assertEqual(len(arm.fingerprint), 64)

    def test_candidates_must_be_ascending(self):
        raw = json.loads(json.dumps(load_matched_memorization_arm(ARM_PATH).as_dict()))
        bad = {
            "name": raw["name"],
            "tag": raw["tag"],
            "training": {key: raw[key] for key in (
                "optimizer", "scheduler", "warmup_steps", "weight_decay",
                "effective_batch_size", "sequence_length", "max_epochs")},
            "calibration": {"learning_rate_candidates": [1e-4, 3e-5]},
            "memorization": {
                "probe_templates": raw["probe_templates"],
                "target_recall": 0.9,
                "check_every_epochs": 1,
            },
            "damage": {
                "max_accuracy_drop": 0.02,
                "general_probe_size": 10,
                "medical_probe_size": 10,
                "seed": 1,
            },
        }
        with self.assertRaises(ConfigError):
            MatchedMemorizationArm.from_dict(bad)


class TeacherForcedRecallTests(unittest.TestCase):
    def test_every_supervised_token_must_be_the_argmax(self):
        vocab = 5
        labels = torch.tensor([[-100, -100, 2, 3], [-100, -100, 2, 3]])
        logits = torch.full((2, 4, vocab), -10.0)
        # position t predicts token t + 1: positions 1 and 2 predict the labels 2 and 3
        logits[0, 1, 2] = logits[0, 2, 3] = 10.0  # both right
        logits[1, 1, 2] = logits[1, 2, 4] = 10.0  # second token wrong
        self.assertEqual(teacher_forced_exact(logits, labels).tolist(), [True, False])


class EarlyFinishTests(unittest.TestCase):
    def test_on_step_end_can_finish_training_successfully(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = tiny_model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loop = TrainingLoop(
                accelerator=create_accelerator(cpu=True),
                model=model,
                optimizer=optimizer,
                scheduler=None,
                objective=SpecializationObjective(
                    torch.randint(1, 61, (16, 8), generator=torch.Generator().manual_seed(2)),
                    effective_batch_sequences=4,
                    micro_batch_sequences=2,
                ),
                total_steps=10,
                max_grad_norm=None,
                run_log=RunLog(root / "log", run_id="t", stage="t", total_steps=10, console_every_steps=100),
                resume_store=ResumeStore(root / "resume", keep_last=1, every_minutes=60),
                stop=GracefulStop(),
                settings=MachineSettings.from_env({"UNLEARNING_STORAGE_ROOT": str(root), "UNLEARNING_MIN_FREE_DISK_GB": "0"}),
            )
            loop.on_step_end = lambda step, metrics: setattr(loop, "finish_now", step == 3) or False
            result = loop.run(start_step=0)
            self.assertTrue(result.completed)
            self.assertEqual(result.last_step, 3)
            self.assertEqual(loop.completed_step, 3)


class MatchedSourceTests(unittest.TestCase):
    def setUp(self):
        self.config = PilotConfig.from_dict(minimal_config())

    def _matched_source(self, directory: Path, steps: int = 420) -> Path:
        source = directory / "C50"
        (source / "model").mkdir(parents=True)
        (source / "tokenizer").mkdir()
        (source / "model" / "config.json").write_text('{"model": "fixture"}', encoding="utf-8")
        (source / "tokenizer" / "tokenizer.json").write_text('{"tokenizer": "fixture"}', encoding="utf-8")
        (source / "training_state.pt").write_bytes(b"fixture-state")
        run_id = experiment_run_id(self.config)
        payload = {
            "checkpoint_id": f"{run_id}::C50::step{steps:04d}",
            "run_id": run_id,
            "name": "C50",
            "global_step": steps,
            "parent_clean_checkpoint_id": f"{run_id}::S50::step1645",
            "parent_clean_checkpoint_name": "S50",
            "forget_schema_version": "synthetic-patient-f-v2",
            "forget_examples_file_sha256": "a" * 64,
            "forget_examples_fingerprint": "b" * 64,
            "masking": {"objective": "causal_lm_completion_only", "add_special_tokens": False, "sequence_length": 512},
            "base_model": {
                "model_id": self.config.model.model_id,
                "revision": self.config.model.revision,
                "tokenizer_id": self.config.model.tokenizer_id,
            },
            "artifact_fingerprints": {
                "model_directory": fingerprint_directory(source / "model"),
                "tokenizer_directory": fingerprint_directory(source / "tokenizer"),
                "training_state_file": hash_file(source / "training_state.pt"),
            },
            "contamination": {
                "condition": "matched_memorization",
                "arm_tag": "MM",
                "arm_fingerprint": "c" * 64,
                "learning_rate": 1e-4,
                "calibration_report_sha256": "d" * 64,
                "effective_batch_size": 16,
            },
            "exposure_plan": {"optimizer_steps": steps, "steps_per_epoch": 60},
            "memorization": {"steps_used": steps, "target_reached": True, "history": []},
        }
        payload["manifest"] = build_manifest(
            artifact_type="contaminated_checkpoint",
            schema_version=MATCHED_CONTAMINATION_MANIFEST_SCHEMA,
            payload=payload,
            source_metadata={"test": True},
        )
        (source / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        return source

    def test_unlearning_accepts_a_matched_c_and_names_its_trajectory_by_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = load_contaminated_source(self._matched_source(Path(temporary)), config=self.config)
            self.assertEqual(source.condition, "matched_memorization")
            self.assertEqual(source.label_prefix, "MM_")
            examples = validate_forget_examples(
                [{
                    "example_id": "PX-1::a", "record_id": "PX-1", "template_id": "a", "split": "contamination",
                    "prompt": "P", "completion": " C", "text": "P C",
                }]
            )
            plan = build_unlearning_trajectory_plan(
                config=self.config, source=source, method="ga", examples=examples, forget_examples_fingerprint="b" * 64
            )
        self.assertTrue(plan.trajectory_run_id.endswith("matched-memorization__ga"))
        self.assertEqual(plan.source_checkpoint_id, f"{experiment_run_id(self.config)}::C50::step0420")

    def test_matched_c_with_inconsistent_steps_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._matched_source(Path(temporary))
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            manifest["memorization"]["steps_used"] = 999
            manifest["manifest"] = build_manifest(
                artifact_type="contaminated_checkpoint",
                schema_version=MATCHED_CONTAMINATION_MANIFEST_SCHEMA,
                payload={key: value for key, value in manifest.items() if key != "manifest"},
                source_metadata={"test": True},
            )
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "global_step does not match"):
                load_contaminated_source(path, config=self.config)


if __name__ == "__main__":
    unittest.main()
