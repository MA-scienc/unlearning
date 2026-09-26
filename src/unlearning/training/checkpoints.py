from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unlearning.config import PilotConfig
from unlearning.training.stream import cumulative_sequences_processed, cumulative_tokens_processed
from unlearning.utils.hashing import fingerprint_payload
from unlearning.utils.io import ensure_dir, write_json
from unlearning.utils.manifest import build_manifest


@dataclass(frozen=True)
class CheckpointIdentity:
    run_id: str
    name: str
    global_step: int

    @property
    def checkpoint_id(self) -> str:
        return f"{self.run_id}::{self.name}::step{self.global_step:04d}"


def build_checkpoint_manifest(
    *,
    config: PilotConfig,
    run_id: str,
    name: str,
    global_step: int,
    parent_checkpoint_id: str | None,
    data_metadata: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any]:
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
    }
    payload["manifest"] = build_manifest(
        artifact_type="clean_specialization_checkpoint",
        schema_version="clean-specialization-checkpoint-v1",
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
