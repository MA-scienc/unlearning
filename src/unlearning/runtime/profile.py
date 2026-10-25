"""Runtime profiles: how a run is executed on particular hardware.

A profile holds execution-only knobs -- per-GPU micro-batch sizes, launcher,
checkpoint cadence, logging cadence. None of them changes the optimization
objective: every optimizer step consumes the same examples and the loss is
normalized by the whole step's token/example count (see
``unlearning.training.engine``), so a laptop run and a multi-GPU run of the same
experiment config follow the same trajectory up to floating-point rounding.
Profiles are therefore recorded in every manifest but excluded from run IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from unlearning.runtime.settings import resolve_repo_path
from unlearning.utils.hashing import fingerprint_payload

STAGES = ("specialization", "contamination", "unlearning", "npo_reference")
LAUNCHERS = ("python", "accelerate")


class RuntimeProfileError(ValueError):
    """Raised when a runtime profile is invalid or incompatible with a config."""


@dataclass(frozen=True)
class MicroBatchSizes:
    specialization: int
    contamination: int
    unlearning: int
    npo_reference: int


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    description: str
    launcher: str
    accelerate_config: str | None
    micro_batch: MicroBatchSizes
    resume_every_minutes: float
    resume_keep_last: int
    console_every_steps: int
    system_sample_seconds: float
    specialization_dev_every_steps: int
    specialization_dev_sequences: int
    source_path: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: str | None = None) -> "RuntimeProfile":
        required = ("name", "launcher", "micro_batch", "checkpointing", "logging", "monitoring")
        missing = [key for key in required if key not in raw]
        if missing:
            raise RuntimeProfileError(f"runtime profile is missing keys: {', '.join(missing)}")
        micro = raw["micro_batch"]
        missing_stages = [stage for stage in STAGES if stage not in micro]
        if missing_stages:
            raise RuntimeProfileError(
                f"runtime profile micro_batch is missing: {', '.join(missing_stages)}"
            )
        profile = cls(
            name=str(raw["name"]),
            description=str(raw.get("description") or ""),
            launcher=str(raw["launcher"]),
            accelerate_config=raw.get("accelerate_config"),
            micro_batch=MicroBatchSizes(**{stage: int(micro[stage]) for stage in STAGES}),
            resume_every_minutes=float(raw["checkpointing"]["resume_every_minutes"]),
            resume_keep_last=int(raw["checkpointing"]["keep_last"]),
            console_every_steps=int(raw["logging"]["console_every_steps"]),
            system_sample_seconds=float(raw["logging"]["system_sample_seconds"]),
            specialization_dev_every_steps=int(raw["monitoring"]["specialization_dev_every_steps"]),
            specialization_dev_sequences=int(raw["monitoring"]["specialization_dev_sequences"]),
            source_path=source_path,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.launcher not in LAUNCHERS:
            raise RuntimeProfileError(f"launcher must be one of {LAUNCHERS}")
        if self.launcher == "accelerate" and not self.accelerate_config:
            raise RuntimeProfileError("launcher 'accelerate' requires accelerate_config")
        for stage in STAGES:
            if getattr(self.micro_batch, stage) <= 0:
                raise RuntimeProfileError(f"micro_batch.{stage} must be positive")
        if self.resume_every_minutes <= 0:
            raise RuntimeProfileError("checkpointing.resume_every_minutes must be positive")
        if self.resume_keep_last < 1:
            raise RuntimeProfileError("checkpointing.keep_last must be at least 1")
        if self.console_every_steps < 1:
            raise RuntimeProfileError("logging.console_every_steps must be at least 1")
        if self.system_sample_seconds <= 0:
            raise RuntimeProfileError("logging.system_sample_seconds must be positive")
        if self.specialization_dev_every_steps < 0 or self.specialization_dev_sequences < 0:
            raise RuntimeProfileError("monitoring values must be non-negative")

    @property
    def accelerate_config_path(self) -> Path | None:
        return resolve_repo_path(self.accelerate_config) if self.accelerate_config else None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = self.as_dict()
        payload.pop("source_path", None)
        return fingerprint_payload(payload)


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    profile_path = resolve_repo_path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"Runtime profile not found: {profile_path}")
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeProfileError(f"Runtime profile {profile_path} did not parse to a mapping")
    return RuntimeProfile.from_dict(raw, source_path=str(profile_path))


def micro_steps_per_rank(*, effective_batch: int, micro_batch: int, world_size: int, stage: str) -> int:
    """Number of micro-batches each process runs per optimizer step.

    Every optimizer step must split into whole micro-batches shared equally by
    all processes, otherwise processes would disagree on when gradients are
    synchronized. Refusing here is better than silently changing the batch.
    """

    if micro_batch <= 0 or world_size <= 0:
        raise RuntimeProfileError("micro_batch and world_size must be positive")
    if effective_batch % (micro_batch * world_size) != 0:
        raise RuntimeProfileError(
            f"{stage}: effective batch {effective_batch} is not divisible by "
            f"micro_batch {micro_batch} x processes {world_size}. Choose a micro-batch size "
            "that divides the effective batch evenly across all GPUs."
        )
    return effective_batch // (micro_batch * world_size)
