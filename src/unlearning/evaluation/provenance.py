"""Verified model-snapshot references for evaluation commands.

Evaluation results are evidence only when the code can establish which model
files were scored.  This module accepts either the pinned base S0 model or a
local clean/contaminated/unlearned checkpoint whose manifest and artifact
fingerprints validate before weights are loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from unlearning.config import PilotConfig
from unlearning.experiments.naming import experiment_run_id
from unlearning.training.checkpoints import validate_clean_checkpoint_identity
from unlearning.utils.hashing import fingerprint_directory, fingerprint_payload, hash_file
from unlearning.utils.io import read_json


_SUPPORTED_CHECKPOINT_SCHEMAS = {
    ("clean_specialization_checkpoint", "clean-specialization-checkpoint-v2"),
    ("contaminated_checkpoint", "fixed-exposure-contamination-v2"),
    ("unlearning_snapshot", "unlearning-snapshot-v2"),
}


@dataclass(frozen=True)
class SnapshotReference:
    """A model source whose identity has been checked against the frozen config."""

    kind: str
    model_source: str
    tokenizer_source: str
    model_revision: str | None
    base_model: dict[str, Any]
    manifest_path: str | None = None
    manifest_file_sha256: str | None = None
    manifest_payload_fingerprint: str | None = None
    artifact_fingerprints: dict[str, str] | None = None
    manifest: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_snapshot_reference(
    model_argument: str,
    checkpoint_manifest: str | Path | None,
    config: PilotConfig,
) -> SnapshotReference:
    """Resolve pinned S0 or validate a materialized S/C/U checkpoint.

    Local snapshots must be accompanied by their manifest and must point to
    either the manifest's checkpoint root or its ``model/`` directory.  This
    prevents results from being attached to a plausible but unrelated run.
    """

    expected_base_model = {
        "model_id": config.model.model_id,
        "revision": config.model.revision,
        "tokenizer_id": config.model.tokenizer_id,
    }
    if model_argument == config.model.model_id:
        if checkpoint_manifest is not None:
            manifest_path = Path(checkpoint_manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Checkpoint manifest was not found: {manifest_path}")
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError("Checkpoint manifest must be a JSON object")
            artifact_manifest = manifest.get("manifest")
            if not isinstance(artifact_manifest, dict):
                raise ValueError("Checkpoint manifest lacks immutable manifest metadata")
            if (
                artifact_manifest.get("artifact_type"),
                artifact_manifest.get("schema_version"),
            ) != ("clean_specialization_checkpoint", "clean-specialization-checkpoint-v2"):
                raise ValueError("Pinned base model may be paired only with an S0 clean manifest")
            expected_payload_fingerprint = fingerprint_payload(
                {key: value for key, value in manifest.items() if key != "manifest"}
            )
            if artifact_manifest.get("payload_fingerprint") != expected_payload_fingerprint:
                raise ValueError("Checkpoint manifest payload fingerprint does not match its contents")
            if manifest.get("name") != "S0" or manifest.get("base_model") != expected_base_model:
                raise ValueError("Pinned base model manifest must be the matching S0 reference")
            if manifest.get("run_id") != experiment_run_id(config):
                raise ValueError("Pinned base model S0 manifest run_id does not match --config")
            validate_clean_checkpoint_identity(
                config=config,
                run_id=manifest["run_id"],
                name=manifest["name"],
                global_step=manifest.get("global_step"),
                parent_checkpoint_id=manifest.get("parent_checkpoint_id"),
            )
            artifact_fingerprints = manifest.get("artifact_fingerprints")
            expected_s0_artifacts = {
                "model_directory": None,
                "tokenizer_directory": None,
                "training_state_file": None,
            }
            if artifact_fingerprints != expected_s0_artifacts:
                raise ValueError("S0 reference manifest must not claim local checkpoint artifacts")
            return SnapshotReference(
                kind="pinned_base_model_s0_reference",
                model_source=config.model.model_id,
                tokenizer_source=config.model.tokenizer_id,
                model_revision=config.model.revision,
                base_model=expected_base_model,
                manifest_path=str(manifest_path),
                manifest_file_sha256=hash_file(manifest_path),
                manifest_payload_fingerprint=artifact_manifest["payload_fingerprint"],
                manifest=manifest,
            )
        return SnapshotReference(
            kind="pinned_base_model",
            model_source=config.model.model_id,
            tokenizer_source=config.model.tokenizer_id,
            model_revision=config.model.revision,
            base_model=expected_base_model,
        )
    if checkpoint_manifest is None:
        raise ValueError(
            "A local S_i/C_i/U_i --model requires --checkpoint-manifest so evaluation can "
            "verify the actual checkpoint artifacts."
        )
    manifest_path = Path(checkpoint_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest was not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Checkpoint manifest must be a JSON object")
    artifact_manifest = manifest.get("manifest")
    if not isinstance(artifact_manifest, dict):
        raise ValueError("Checkpoint manifest lacks immutable manifest metadata")
    identity = (artifact_manifest.get("artifact_type"), artifact_manifest.get("schema_version"))
    if identity not in _SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError(f"Unsupported checkpoint manifest type/schema: {identity!r}")
    expected_payload_fingerprint = fingerprint_payload(
        {key: value for key, value in manifest.items() if key != "manifest"}
    )
    if artifact_manifest.get("payload_fingerprint") != expected_payload_fingerprint:
        raise ValueError("Checkpoint manifest payload fingerprint does not match its contents")
    if manifest.get("base_model") != expected_base_model:
        raise ValueError("Checkpoint manifest base-model identity does not match --config")
    artifact_type = artifact_manifest["artifact_type"]
    if artifact_type == "clean_specialization_checkpoint":
        if manifest.get("run_id") != experiment_run_id(config):
            raise ValueError("Clean checkpoint manifest run_id does not match --config")
        validate_clean_checkpoint_identity(
            config=config,
            run_id=manifest["run_id"],
            name=manifest.get("name"),
            global_step=manifest.get("global_step"),
            parent_checkpoint_id=manifest.get("parent_checkpoint_id"),
        )
    elif artifact_type == "contaminated_checkpoint":
        if manifest.get("run_id") != experiment_run_id(config):
            raise ValueError("Contaminated checkpoint manifest run_id does not match --config")
    elif artifact_type == "unlearning_snapshot":
        if manifest.get("experiment_run_id") != experiment_run_id(config):
            raise ValueError("Unlearning snapshot experiment_run_id does not match --config")

    checkpoint_root = manifest_path.parent
    model_dir = checkpoint_root / "model"
    tokenizer_dir = checkpoint_root / "tokenizer"
    training_state_path = checkpoint_root / "training_state.pt"
    for required in (model_dir, tokenizer_dir):
        if not required.is_dir():
            raise FileNotFoundError(f"Checkpoint artifact directory was not found: {required}")
    if not training_state_path.is_file():
        raise FileNotFoundError(f"Checkpoint training state was not found: {training_state_path}")
    supplied_model_path = Path(model_argument)
    if supplied_model_path.resolve() not in {checkpoint_root.resolve(), model_dir.resolve()}:
        raise ValueError(
            "--model must name the checkpoint root or its model/ directory when a checkpoint "
            "manifest is supplied"
        )
    artifact_fingerprints = manifest.get("artifact_fingerprints")
    if not isinstance(artifact_fingerprints, dict):
        raise ValueError("Checkpoint manifest lacks artifact fingerprints")
    observed_artifacts = {
        "model_directory": fingerprint_directory(model_dir),
        "tokenizer_directory": fingerprint_directory(tokenizer_dir),
        "training_state_file": hash_file(training_state_path),
    }
    if {key: artifact_fingerprints.get(key) for key in observed_artifacts} != observed_artifacts:
        raise ValueError("Checkpoint artifact fingerprints do not match the local snapshot")
    return SnapshotReference(
        kind="local_checkpoint",
        model_source=str(model_dir),
        tokenizer_source=str(tokenizer_dir),
        model_revision=None,
        base_model=expected_base_model,
        manifest_path=str(manifest_path),
        manifest_file_sha256=hash_file(manifest_path),
        manifest_payload_fingerprint=artifact_manifest["payload_fingerprint"],
        artifact_fingerprints=observed_artifacts,
        manifest=manifest,
    )
