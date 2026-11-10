from __future__ import annotations

import re

from unlearning.config import PilotConfig
from unlearning.utils.hashing import fingerprint_payload


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "unnamed"


def experiment_run_id(config: PilotConfig) -> str:
    model_slug = slugify(config.model.model_id)
    forget_slug = slugify(config.forget.schema_version)
    config_hash = fingerprint_payload(_experiment_identity_payload(config))[:12]
    return (
        f"{slugify(config.experiment_id)}__{model_slug}__seed{config.seed}__"
        f"{forget_slug}__cfg{config_hash}"
    )


def _experiment_identity_payload(config: PilotConfig) -> dict:
    """Return every scientific setting that must distinguish result directories."""

    payload = {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "model": config.model.__dict__,
        "specialization": config.specialization.__dict__,
        "forget": config.forget.__dict__,
        "contamination": config.contamination.__dict__,
        "unlearning": {
            "snapshot_steps": config.unlearning.snapshot_steps,
            "retain_loss_weight": config.unlearning.retain_loss_weight,
            "methods": {
                name: method.__dict__ for name, method in config.unlearning.methods.items()
            },
        },
        "evaluation": config.evaluation.__dict__,
    }
    if config.data_order != "canonical":
        # Only when set, so canonical (pilot) run ids stay exactly as they were.
        payload["data_order"] = config.data_order
    if config.control_fractions:
        payload["control_fractions"] = list(config.control_fractions)
    return payload


def unlearning_stage(revision_tag: str | None = None) -> str:
    """Trajectory stage name: 'unlearned' for revision 1, 'unlearned_U2' (slug unlearned-u2) for later ones."""

    return "unlearned" if revision_tag is None else f"unlearned_{revision_tag}"


def stage_run_name(
    config: PilotConfig,
    stage: str,
    checkpoint_fraction: float | None = None,
    method: str | None = None,
    condition: str | None = None,
    step: int | None = None,
    lineage: str = "S",
) -> str:
    parts = [experiment_run_id(config), slugify(stage)]
    if checkpoint_fraction is not None:
        # s025 for a specialization level, g025 for its format-matched control (Stage 2).
        parts.append(f"{lineage.lower()}{int(round(checkpoint_fraction * 100)):03d}")
    if condition is not None:
        parts.append(slugify(condition))
    if method is not None:
        parts.append(slugify(method))
    if step is not None:
        parts.append(f"step{step:04d}")
    return "__".join(parts)
