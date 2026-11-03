"""Contamination arms that are defined outside the experiment config.

An arm changes only how S_i + F -> C_i is produced. Keeping it in its own file
leaves the experiment run ID (and the S checkpoints bound to it) unchanged.

Two matched-memorization arms exist:

* ``matched_memorization`` (2026-10-25): F only. Its calibration found no
  qualifying learning rate; kept as the record.
* ``matched_memorization_replay`` (2026-11-01): F mixed with general replay
  text every step, gradient clipping, and NLL / letter-bias-free damage limits.
  Revisions 1-2 trained the replay passages' next tokens; revision 3
  (2026-11-03) anchors the model to its clean parent S_i's predictions on them.
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
ARM_NAMES = ("matched_memorization", "matched_memorization_replay")
# What the replay term trains toward: the passage's own next tokens (cross-entropy),
# or the clean parent S_i's next-token distribution on the passage (KL).
REPLAY_TARGETS = ("next_token", "clean_parent")


@dataclass(frozen=True)
class ReplayConfig:
    dataset_id: str
    dataset_config: str
    revision: str
    license: str
    split: str
    sequence_length: int
    sequences_per_step: int
    weight: float
    prepared_sequences: int
    shuffle_seed: int
    target: str = "next_token"

    def validate(self, *, max_steps: int) -> None:
        if self.target not in REPLAY_TARGETS:
            raise ConfigError(f"replay.target must be one of {REPLAY_TARGETS}")
        if len(self.revision) != 40:
            raise ConfigError("replay.revision must be a pinned 40-character commit SHA")
        if self.sequence_length <= 1 or self.sequences_per_step <= 0 or self.weight <= 0:
            raise ConfigError("replay sequence_length, sequences_per_step and weight must be positive")
        if self.prepared_sequences < max_steps * self.sequences_per_step:
            raise ConfigError(
                "replay.prepared_sequences must cover max_epochs x steps_per_epoch x sequences_per_step "
                "so no passage repeats within a run"
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
    max_grad_norm: float | None
    learning_rate_candidates: tuple[float, ...]
    probe_templates: tuple[str, ...]
    target_recall: float
    check_every_epochs: int
    general_probe_size: int
    medical_probe_size: int
    damage_seed: int
    # Damage limits: the F-only arm used raw multiple-choice accuracy; the replay
    # arm uses text NLL and letter-bias-free accuracy (raw accuracy is reported).
    max_accuracy_drop: float | None = None
    max_nll_increase: float | None = None
    max_bias_removed_accuracy_drop: float | None = None
    nll_texts_per_probe: int = 0
    replay: ReplayConfig | None = None
    source_path: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: str | None = None) -> "MatchedMemorizationArm":
        try:
            training, calibration = raw["training"], raw["calibration"]
            memorization, damage = raw["memorization"], raw["damage"]
            replay_raw = raw.get("replay")
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
                max_grad_norm=(
                    float(training["max_grad_norm"]) if training.get("max_grad_norm") is not None else None
                ),
                learning_rate_candidates=tuple(float(x) for x in calibration["learning_rate_candidates"]),
                probe_templates=tuple(str(x) for x in memorization["probe_templates"]),
                target_recall=float(memorization["target_recall"]),
                check_every_epochs=int(memorization["check_every_epochs"]),
                general_probe_size=int(damage["general_probe_size"]),
                medical_probe_size=int(damage["medical_probe_size"]),
                damage_seed=int(damage["seed"]),
                max_accuracy_drop=_optional_float(damage.get("max_accuracy_drop")),
                max_nll_increase=_optional_float(damage.get("max_nll_increase")),
                max_bias_removed_accuracy_drop=_optional_float(damage.get("max_bias_removed_accuracy_drop")),
                nll_texts_per_probe=int(damage.get("nll_texts_per_probe", 0)),
                replay=ReplayConfig(**replay_raw) if replay_raw else None,
                source_path=source_path,
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"matched-memorization arm is invalid or incomplete: {exc}") from exc
        arm.validate()
        return arm

    @property
    def steps_per_epoch_hint(self) -> int:
        return 960 // self.effective_batch_size

    def validate(self) -> None:
        if self.name not in ARM_NAMES:
            raise ConfigError(f"arm name must be one of {ARM_NAMES}")
        if not self.tag.isalnum() or not self.tag.isupper():
            raise ConfigError("arm tag must be upper-case alphanumeric (used in result labels)")
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
        if self.warmup_steps < 0 or self.weight_decay < 0:
            raise ConfigError("warmup_steps and weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ConfigError("max_grad_norm must be positive when set")
        new_limits = (self.max_nll_increase, self.max_bias_removed_accuracy_drop)
        if self.max_accuracy_drop is None and None in new_limits:
            raise ConfigError("set max_accuracy_drop, or both max_nll_increase and max_bias_removed_accuracy_drop")
        if self.max_nll_increase is not None and self.nll_texts_per_probe <= 0:
            raise ConfigError("nll_texts_per_probe must be positive when max_nll_increase is used")
        if self.replay is not None:
            self.replay.validate(max_steps=self.max_epochs * self.steps_per_epoch_hint)

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


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_matched_memorization_arm(path: str | Path) -> MatchedMemorizationArm:
    arm_path = Path(path)
    raw = yaml.safe_load(arm_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"arm file {arm_path} did not parse to a mapping")
    return MatchedMemorizationArm.from_dict(raw, source_path=str(arm_path))
