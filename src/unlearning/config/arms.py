"""Contamination arms that are defined outside the experiment config.

An arm changes only how S_i + F -> C_i is produced. Keeping it in its own file
leaves the experiment run ID (and the S checkpoints bound to it) unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from unlearning.config.schema import ConfigError, PilotConfig
from unlearning.utils.hashing import fingerprint_payload

CONTAMINATION_TEMPLATES = (
    "full_note",
    "structured_summary",
    "patient_lookup",
    "medication_lookup",
    "condition_lookup",
    "private_code_lookup",
    "short_completion",
    "qa_form",
)


@dataclass(frozen=True)
class MatchedMemorizationArm:
    name: str
    tag: str
    optimizer: str
    scheduler: str
    warmup_steps: int
    weight_decay: float
    effective_batch_size: int
    sequence_length: int
    max_epochs: int
    learning_rate_candidates: tuple[float, ...]
    probe_templates: tuple[str, ...]
    target_recall: float
    check_every_epochs: int
    max_accuracy_drop: float
    general_probe_size: int
    medical_probe_size: int
    damage_seed: int
    source_path: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: str | None = None) -> "MatchedMemorizationArm":
        try:
            training, calibration = raw["training"], raw["calibration"]
            memorization, damage = raw["memorization"], raw["damage"]
            arm = cls(
                name=str(raw["name"]),
                tag=str(raw["tag"]),
                optimizer=str(training["optimizer"]),
                scheduler=str(training["scheduler"]),
                warmup_steps=int(training["warmup_steps"]),
                weight_decay=float(training["weight_decay"]),
                effective_batch_size=int(training["effective_batch_size"]),
                sequence_length=int(training["sequence_length"]),
                max_epochs=int(training["max_epochs"]),
                learning_rate_candidates=tuple(float(x) for x in calibration["learning_rate_candidates"]),
                probe_templates=tuple(str(x) for x in memorization["probe_templates"]),
                target_recall=float(memorization["target_recall"]),
                check_every_epochs=int(memorization["check_every_epochs"]),
                max_accuracy_drop=float(damage["max_accuracy_drop"]),
                general_probe_size=int(damage["general_probe_size"]),
                medical_probe_size=int(damage["medical_probe_size"]),
                damage_seed=int(damage["seed"]),
                source_path=source_path,
            )
        except KeyError as exc:
            raise ConfigError(f"matched-memorization arm is missing {exc}") from exc
        arm.validate()
        return arm

    def validate(self) -> None:
        if self.name != "matched_memorization":
            raise ConfigError("arm name must be matched_memorization")
        if not self.tag.isalnum():
            raise ConfigError("arm tag must be alphanumeric (used in result labels)")
        if self.optimizer != "adamw_8bit":
            raise ConfigError("arm optimizer must be adamw_8bit (same family as every stage)")
        if self.scheduler != "constant_with_warmup":
            raise ConfigError("arm scheduler must be constant_with_warmup")
        candidates = self.learning_rate_candidates
        if not candidates or any(lr <= 0 for lr in candidates) or list(candidates) != sorted(set(candidates)):
            raise ConfigError("learning_rate_candidates must be positive, unique and ascending")
        if not 0 < self.target_recall <= 1:
            raise ConfigError("target_recall must be in (0, 1]")
        unknown = set(self.probe_templates) - set(CONTAMINATION_TEMPLATES)
        if not self.probe_templates or unknown:
            raise ConfigError(f"unknown probe templates: {sorted(unknown)}")
        for name in ("max_epochs", "check_every_epochs", "effective_batch_size", "sequence_length",
                     "general_probe_size", "medical_probe_size"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")
        if self.warmup_steps < 0 or self.weight_decay < 0 or not 0 <= self.max_accuracy_drop < 1:
            raise ConfigError("warmup_steps, weight_decay and max_accuracy_drop must be non-negative")

    def validate_against(self, config: PilotConfig) -> None:
        """The arm may change the recipe, not the data or the masking unlearning relies on."""

        if self.effective_batch_size != config.contamination.effective_batch_size:
            raise ConfigError("arm effective_batch_size must match the experiment contamination batch")
        for method in config.unlearning.methods.values():
            if method.sequence_length != self.sequence_length:
                raise ConfigError("arm sequence_length must match the unlearning sequence length")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)
        return payload

    @property
    def fingerprint(self) -> str:
        return fingerprint_payload(self.as_dict())


def load_matched_memorization_arm(path: str | Path) -> MatchedMemorizationArm:
    arm_path = Path(path)
    raw = yaml.safe_load(arm_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"arm file {arm_path} did not parse to a mapping")
    return MatchedMemorizationArm.from_dict(raw, source_path=str(arm_path))
