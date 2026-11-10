import json
import tempfile
import unittest
from pathlib import Path

from tests.test_config import minimal_config
from unlearning.config.schema import PilotConfig
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.experiments.naming import experiment_run_id
from unlearning.runtime.pruning import (
    PRUNED_MARKER_FILE,
    find_prunable,
    prune_completed_trajectories,
    prune_training_state,
    training_state_fingerprint,
)
from unlearning.training.checkpoints import build_checkpoint_manifest
from unlearning.utils.hashing import fingerprint_directory, hash_file
from unlearning.utils.io import write_json
from unlearning.utils.manifest import build_manifest


def _artifacts(snapshot: Path, state: bytes = b"optimizer state") -> dict:
    (snapshot / "model").mkdir(parents=True)
    (snapshot / "tokenizer").mkdir()
    (snapshot / "model" / "config.json").write_text('{"fixture": true}', encoding="utf-8")
    (snapshot / "tokenizer" / "tokenizer.json").write_text('{"fixture": true}', encoding="utf-8")
    (snapshot / "training_state.pt").write_bytes(state)
    return {
        "model_directory": fingerprint_directory(snapshot / "model"),
        "tokenizer_directory": fingerprint_directory(snapshot / "tokenizer"),
        "training_state_file": hash_file(snapshot / "training_state.pt"),
    }


class PruningTests(unittest.TestCase):
    def setUp(self):
        self.config = PilotConfig.from_dict(minimal_config())
        self.run_id = experiment_run_id(self.config)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "unlearned"

    def tearDown(self):
        self.temporary.cleanup()

    def snapshot(self, trajectory: str = "traj", name: str = "f100", *, complete: bool = True) -> Path:
        trajectory_dir = self.root / f"{self.run_id}__{trajectory}"
        snapshot = trajectory_dir / name
        payload = {
            "artifact_fingerprints": _artifacts(snapshot),
            "base_model": {
                "model_id": self.config.model.model_id,
                "revision": self.config.model.revision,
                "tokenizer_id": self.config.model.tokenizer_id,
            },
            "checkpoint_id": f"{trajectory_dir.name}::{name}",
            "experiment_run_id": self.run_id,
        }
        manifest = {**payload, "manifest": build_manifest("unlearning_snapshot", "unlearning-snapshot-v2", payload, {})}
        write_json(snapshot / "manifest.json", manifest)
        if complete:
            write_json(trajectory_dir / "run_manifest.json", {"status": "completed"})
        return snapshot

    def test_prune_writes_a_marker_and_evaluation_still_verifies_the_weights(self):
        snapshot = self.snapshot()
        recorded = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))["artifact_fingerprints"]
        ledger = Path(self.temporary.name) / "ledger.jsonl"
        pruned = prune_completed_trajectories(self.root, ledger, trajectory_prefix=f"{self.run_id}__", log=lambda _: None)

        self.assertEqual(len(pruned), 1)
        self.assertFalse((snapshot / "training_state.pt").exists())
        marker = json.loads((snapshot / PRUNED_MARKER_FILE).read_text(encoding="utf-8"))
        self.assertEqual(marker["sha256"], recorded["training_state_file"])
        self.assertEqual(marker["size_bytes"], len(b"optimizer state"))
        self.assertIn("git_commit", marker["code_version"])
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)

        reference = resolve_snapshot_reference(str(snapshot), snapshot / "manifest.json", self.config)
        self.assertTrue(reference.training_state_pruned)
        self.assertEqual(reference.artifact_fingerprints, recorded)
        # The weights are still re-hashed: changing them is caught after pruning too.
        (snapshot / "model" / "config.json").write_text('{"fixture": false}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact fingerprints"):
            resolve_snapshot_reference(str(snapshot), snapshot / "manifest.json", self.config)

    def test_unfinished_trajectories_and_other_experiments_are_left_alone(self):
        self.snapshot("running", complete=False)
        other = self.root / "another-experiment__traj" / "f100"
        other.mkdir(parents=True)
        (other / "training_state.pt").write_bytes(b"x")
        candidates, skipped = find_prunable(self.root, trajectory_prefix=f"{self.run_id}__")
        self.assertEqual(candidates, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("completed", skipped[0][1])
        self.assertTrue((other / "training_state.pt").exists())

    def test_a_file_that_does_not_match_its_manifest_is_not_deleted(self):
        snapshot = self.snapshot()
        (snapshot / "training_state.pt").write_bytes(b"changed after the manifest was written")
        candidates, _ = find_prunable(self.root)
        with self.assertRaisesRegex(ValueError, "not pruned"):
            prune_training_state(candidates[0])
        self.assertTrue((snapshot / "training_state.pt").exists())
        self.assertFalse((snapshot / PRUNED_MARKER_FILE).exists())

    def test_a_marker_must_agree_with_the_manifest(self):
        snapshot = self.snapshot()
        candidates, _ = find_prunable(self.root)
        prune_training_state(candidates[0])
        with self.assertRaisesRegex(ValueError, "differs from the manifest"):
            training_state_fingerprint(snapshot, "0" * 64)
        (snapshot / PRUNED_MARKER_FILE).unlink()
        with self.assertRaises(FileNotFoundError):
            resolve_snapshot_reference(str(snapshot), snapshot / "manifest.json", self.config)

    def test_weights_are_deleted_only_after_evaluation_and_leave_a_marker(self):
        from unlearning.runtime.pruning import WEIGHTS_MARKER_FILE, prune_model_weights

        snapshot = self.snapshot(name="f025")
        result = Path(self.temporary.name) / "near.json"
        with self.assertRaises(FileNotFoundError):
            prune_model_weights(snapshot, required_results=[result], reason="test")
        self.assertTrue((snapshot / "model").is_dir())
        result.write_text("{}", encoding="utf-8")
        (snapshot / "model" / "config.json").write_text('{"tampered": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not pruned"):
            prune_model_weights(snapshot, required_results=[result], reason="test")
        (snapshot / "model" / "config.json").write_text('{"fixture": true}', encoding="utf-8")
        record = prune_model_weights(snapshot, required_results=[result], reason="test")
        self.assertFalse((snapshot / "model").exists())
        self.assertTrue((snapshot / WEIGHTS_MARKER_FILE).is_file())
        self.assertEqual(record["files"], {"config.json": len('{"fixture": true}')})
        self.assertIsNone(prune_model_weights(snapshot, required_results=[result], reason="test"))  # idempotent

    def test_inputs_to_later_stages_may_not_be_pruned(self):
        checkpoint = Path(self.temporary.name) / "S50"
        artifacts = _artifacts(checkpoint)
        manifest = build_checkpoint_manifest(
            config=self.config,
            run_id=self.run_id,
            name="S50",
            global_step=1645,
            parent_checkpoint_id=f"{self.run_id}::S0::step0000",
            data_metadata={"fixture": True},
            hardware={"fixture": True},
            artifact_fingerprints=artifacts,
        )
        write_json(checkpoint / "manifest.json", manifest)
        (checkpoint / PRUNED_MARKER_FILE).write_text(
            json.dumps({"schema_version": "pruned-training-state-v1", "file": "training_state.pt",
                        "sha256": artifacts["training_state_file"]}),
            encoding="utf-8",
        )
        (checkpoint / "training_state.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Only unlearning snapshots"):
            resolve_snapshot_reference(str(checkpoint), checkpoint / "manifest.json", self.config)


if __name__ == "__main__":
    unittest.main()
