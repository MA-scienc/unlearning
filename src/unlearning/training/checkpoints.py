from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unlearning.config import PilotConfig
from unlearning.training.schedule import expected_pilot_schedule
from unlearning.training.stream import cumulative_sequences_processed, cumulative_tokens_processed
from unlearning.utils.hashing import fingerprint_payload
from unlearning.utils.io import ensure_dir, write_json
from unlearning.utils.manifest import build_manifest


CLEAN_CHECKPOINT_MANIFEST_SCHEMA = "clean-specialization-checkpoint-v2"


@dataclass(frozen=True)
class CheckpointIdentity:
    run_id: str
    name: str
    global_step: int

    @property
    def checkpoint_id(self) -> str:
        return f"{self.run_id}::{self.name}::step{self.global_step:04d}"


def validate_clean_checkpoint_identity(
    *,
    config: PilotConfig,
    run_id: str,
    name: str,
    global_step: int,
    parent_checkpoint_id: str | None,
) -> None:
    """Require the frozen pilot's exact S0 -> S50 -> S100 lineage.

    An artifact hash alone cannot establish that a checkpoint represents the
    intended point on the continuous clean trajectory. This binds each
    approved name to its fixed optimizer step and immediate parent before a
    manifest is created or consumed downstream.
    """

    schedule = expected_pilot_schedule(config)
    expected_steps = {
        "S0": 0,
        "S50": schedule.s50_steps,
        "S100": schedule.s100_steps,
    }
    expected_parents = {
        "S0": None,
        "S50": CheckpointIdentity(run_id, "S0", expected_steps["S0"]).checkpoint_id,
        "S100": CheckpointIdentity(run_id, "S50", expected_steps["S50"]).checkpoint_id,
    }
    if name not in expected_steps:
        raise ValueError("Clean checkpoint name must be one of S0, S50, or S100")
    if global_step != expected_steps[name]:
        raise ValueError(
            f"Clean checkpoint {name} must have global_step {expected_steps[name]}, "
            f"got {global_step}"
        )
    if parent_checkpoint_id != expected_parents[name]:
        raise ValueError(
            f"Clean checkpoint {name} must have parent_checkpoint_id "
            f"{expected_parents[name]!r}, got {parent_checkpoint_id!r}"
        )


def build_checkpoint_manifest(
    *,
    config: PilotConfig,
    run_id: str,
    name: str,
    global_step: int,
    parent_checkpoint_id: str | None,
    data_metadata: dict[str, Any],
    hardware: dict[str, Any],
    artifact_fingerprints: dict[str, str | None] | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provenance for S0/S50/S100, including local artifact integrity.

    S0 is an intentionally remote, pinned-base-model reference and therefore
    has no local model artifacts.  Materialized S50/S100 checkpoints must pass
    their model/tokenizer/training-state fingerprints here before downstream
    contamination is allowed to consume them.
    """

    validate_clean_checkpoint_identity(
        config=config,
        run_id=run_id,
        name=name,
        global_step=global_step,
        parent_checkpoint_id=parent_checkpoint_id,
    )

    expected_artifact_keys = {
        "model_directory",
        "tokenizer_directory",
        "training_state_file",
    }
    artifacts = {
        "model_directory": None,
        "tokenizer_directory": None,
        "training_state_file": None,
    }
    if artifact_fingerprints is not None:
        unknown = set(artifact_fingerprints) - expected_artifact_keys
        if unknown:
            raise ValueError(f"Unknown checkpoint artifact fingerprint keys: {sorted(unknown)}")
        artifacts.update(artifact_fingerprints)
    if any(value is not None and (not isinstance(value, str) or not value) for value in artifacts.values()):
        raise ValueError("Checkpoint artifact fingerprints must be non-empty strings or null")

    payload = {
        "checkpoint_id": CheckpointIdentity(run_id, name, global_step).checkpoint_id,
        "run_id": run_id,
        "name": name,
        "global_step": global_step,
        "parent_checkpoint_id": parent_checkpoint_id,
        "base_model": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "tokenizer_id": config.model.tokenizer_id,
        },
        "medmcqa": {
            "dataset_id": config.specialization.dataset_id,
            "revision": config.specialization.dataset_revision,
        },
        "config_fingerprint": fingerprint_payload(_config_payload(config)),
        "data": data_metadata,
        "cumulative_sequences_processed": cumulative_sequences_processed(
            global_step,
            config.specialization.effective_batch_sequences,
        ),
        "cumulative_tokens_processed": cumulative_tokens_processed(
            global_step,
            config.specialization.effective_batch_sequences,
            config.specialization.context_length,
        ),
        "seed": config.seed,
        "learning_rate": config.specialization.learning_rate,
        "optimizer": config.model.optimizer,
        "scheduler": config.specialization.scheduler,
        "precision": config.model.dtype,
        "gradient_checkpointing": config.model.gradient_checkpointing,
        "hardware": hardware,
        "artifact_fingerprints": artifacts,
    }
    if execution is not None:
        # Which process segments (start/resume) produced this checkpoint.
        payload["execution"] = execution
    payload["manifest"] = build_manifest(
        artifact_type="clean_specialization_checkpoint",
        schema_version=CLEAN_CHECKPOINT_MANIFEST_SCHEMA,
        payload=payload,
        source_metadata={"run_id": run_id, "checkpoint": name},
    )
    return payload


def write_checkpoint_manifest(path: str | Path, manifest: dict[str, Any], overwrite: bool = True) -> None:
    ensure_dir(Path(path).parent)
    write_json(path, manifest, overwrite=overwrite)


def _config_payload(config: PilotConfig) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "model": config.model.__dict__,
        "specialization": config.specialization.__dict__,
        "forget": config.forget.__dict__,
    }
