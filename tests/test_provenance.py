import tempfile
import unittest
from pathlib import Path

from tests.test_config import minimal_config
from unlearning.config.schema import PilotConfig
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.experiments.naming import experiment_run_id
from unlearning.training.checkpoints import build_checkpoint_manifest
from unlearning.utils.hashing import fingerprint_directory, hash_file
from unlearning.utils.io import write_json


class SnapshotProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.config = PilotConfig.from_dict(minimal_config())

    def test_pinned_base_model_needs_no_local_manifest(self):
        reference = resolve_snapshot_reference(self.config.model.model_id, None, self.config)
        self.assertEqual(reference.kind, "pinned_base_model")
        self.assertEqual(reference.model_revision, self.config.model.revision)

    def test_pinned_base_model_can_bind_to_its_s0_reference_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = build_checkpoint_manifest(
                config=self.config,
                run_id=experiment_run_id(self.config),
                name="S0",
                global_step=0,
                parent_checkpoint_id=None,
                data_metadata={"fixture": True},
                hardware={"fixture": True},
            )
            manifest_path = Path(temporary_directory) / "manifest.json"
            write_json(manifest_path, manifest)
            reference = resolve_snapshot_reference(
                self.config.model.model_id,
                manifest_path,
                self.config,
            )
        self.assertEqual(reference.kind, "pinned_base_model_s0_reference")
        self.assertIsNotNone(reference.manifest)

    def test_local_checkpoint_must_match_manifest_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "S50"
            model_dir = checkpoint / "model"
            tokenizer_dir = checkpoint / "tokenizer"
            model_dir.mkdir(parents=True)
            tokenizer_dir.mkdir()
            (model_dir / "config.json").write_text('{"fixture": true}', encoding="utf-8")
            (tokenizer_dir / "tokenizer.json").write_text('{"fixture": true}', encoding="utf-8")
            state_path = checkpoint / "training_state.pt"
            state_path.write_bytes(b"fixture state")
            manifest = build_checkpoint_manifest(
                config=self.config,
                run_id=experiment_run_id(self.config),
                name="S50",
                global_step=1645,
                parent_checkpoint_id=f"{experiment_run_id(self.config)}::S0::step0000",
                data_metadata={"fixture": True},
                hardware={"fixture": True},
                artifact_fingerprints={
                    "model_directory": fingerprint_directory(model_dir),
                    "tokenizer_directory": fingerprint_directory(tokenizer_dir),
                    "training_state_file": hash_file(state_path),
                },
            )
            manifest_path = checkpoint / "manifest.json"
            write_json(manifest_path, manifest)
            reference = resolve_snapshot_reference(str(model_dir), manifest_path, self.config)
            self.assertEqual(reference.kind, "local_checkpoint")
            self.assertEqual(reference.model_source, str(model_dir))
            (model_dir / "config.json").write_text('{"fixture": false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact fingerprints"):
                resolve_snapshot_reference(str(model_dir), manifest_path, self.config)


if __name__ == "__main__":
    unittest.main()
