"""Unlearning revisions that are defined outside the experiment config.

Revision 1 is the GA/NPO recipe in the experiment config (docs/PILOT_DESIGN.md).
Later revisions live in their own files under ``configs/pilot/unlearning/``, so
the experiment run ID (and every S/C checkpoint bound to it) is unchanged and
earlier revisions' snapshots and results stay in place as the record. Each
revision's outputs carry its tag (``U2`` ...) in the trajectory ID, stage name
and result labels.

* revision 2 (2026-11-07, ``fixed_budget``): GA and NPO at the matched
  contamination learning rate (2e-5) instead of 1e-6; nothing else changes.
* revision 3 (2026-11-08, ``matched_forgetting``): unlearn each C_i until it has
  closed fixed fractions of the gap back to its clean S_i on training-template
  probes (docs/MATCHED_FORGETTING_DESIGN.md); the learning rate is calibrated on
  MR_C0 by a fixed rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from unlearning.config.arms import CONTAMINATION_TEMPLATES
from unlearning.config.schema import ConfigError, MethodConfig, PilotConfig
from unlearning.utils.hashing import fingerprint_payload

METHODS = ("ga", "npo")
MODES = ("fixed_budget", "matched_forgetting")
_KEYS = {
    "fixed_budget": {"revision", "tag", "mode", "learning_rate"},
    "matched_forgetting": {"revision", "tag", "mode", "matched_forgetting"},
}
_MATCHED_KEYS = {"learning_rate_candidates", "thresholds", "check_every_steps", "max_steps", "probe_templates"}


@dataclass(frozen=True)
class MatchedForgetting:
    """Stop each run at fixed fractions of the forgetting gap to its clean S_i."""

    learning_rate_candidates: tuple[float, ...]
    thresholds: tuple[float, ...]
    check_every_steps: int
    max_steps: int
    probe_templates: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MatchedForgetting":
        unknown = set(raw) - _MATCHED_KEYS
        if unknown:
            raise ConfigError(f"matched_forgetting has unknown keys {sorted(unknown)}")
        return cls(
            learning_rate_candidates=tuple(float(x) for x in raw["learning_rate_candidates"]),
            thresholds=tuple(float(x) for x in raw["thresholds"]),
            check_every_steps=int(raw["check_every_steps"]),
            max_steps=int(raw["max_steps"]),
            probe_templates=tuple(str(x) for x in raw["probe_templates"]),
        )

    def validate(self) -> None:
        candidates = self.learning_rate_candidates
        if not candidates or any(lr <= 0 for lr in candidates) or list(candidates) != sorted(set(candidates)):
            raise ConfigError("learning_rate_candidates must be positive, unique and ascending")
        thresholds = self.thresholds
        if not thresholds or list(thresholds) != sorted(set(thresholds)) or not all(0 < t <= 1 for t in thresholds):
            raise ConfigError("thresholds must be unique, ascending and in (0, 1]")
        if thresholds[-1] != 1.0:
            raise ConfigError("the last threshold must be 1.0 (forgetting matched to the clean model)")
        if self.check_every_steps <= 0 or self.max_steps <= 0 or self.max_steps % self.check_every_steps:
            raise ConfigError("check_every_steps and max_steps must be positive, and max_steps a multiple of it")
        unknown = set(self.probe_templates) - set(CONTAMINATION_TEMPLATES)
        if not self.probe_templates or unknown:
            raise ConfigError(f"unknown probe templates: {sorted(unknown)}")

    @property
    def snapshot_names(self) -> tuple[str, ...]:
        """f025, f050, ...: one snapshot per threshold (percent of the gap closed)."""

        return tuple(threshold_name(t) for t in self.thresholds)


def threshold_name(threshold: float) -> str:
    return f"f{round(threshold * 100):03d}"


# Saved instead of the last threshold when a run reaches the step cap without it.
CAP_SNAPSHOT = "final"


@dataclass(frozen=True)
class UnlearningRevision:
    revision: int
    tag: str
    mode: str = "fixed_budget"
    learning_rates: dict[str, float] = field(default_factory=dict)
    matched: MatchedForgetting | None = None
    source_path: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: str | None = None) -> "UnlearningRevision":
        mode = str(raw.get("mode", "fixed_budget"))
        if mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}")
        # A fixed-budget revision changes only the learning rate; a matched-forgetting revision only
        # how long each run trains. Steps per check, batch, clipping and NPO beta/reference stay fixed.
        unknown = set(raw) - _KEYS[mode]
        if unknown:
            raise ConfigError(f"a {mode} revision may set only {sorted(_KEYS[mode])}; got {sorted(unknown)}")
        try:
            revision = cls(
                revision=int(raw["revision"]),
                tag=str(raw["tag"]),
                mode=mode,
                learning_rates=(
                    {str(name): float(value) for name, value in raw["learning_rate"].items()}
                    if mode == "fixed_budget"
                    else {}
                ),
                matched=MatchedForgetting.from_dict(raw["matched_forgetting"]) if mode == "matched_forgetting" else None,
                source_path=source_path,
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ConfigError(f"unlearning revision is invalid or incomplete: {exc}") from exc
        revision.validate()
        return revision

    def validate(self) -> None:
        if self.revision < 2:
            raise ConfigError("revision 1 is the experiment config itself; revision files start at 2")
        if self.tag != f"U{self.revision}":
            raise ConfigError(f"revision {self.revision} must have tag 'U{self.revision}' (used in result labels)")
        if self.mode == "fixed_budget":
            if set(self.learning_rates) != set(METHODS):
                raise ConfigError(f"learning_rate must be given for exactly {list(METHODS)}")
            if any(value <= 0 for value in self.learning_rates.values()):
                raise ConfigError("learning_rate values must be positive")
        else:
            self.matched.validate()

    @property
    def is_matched(self) -> bool:
        return self.mode == "matched_forgetting"

    def method_config(self, config: PilotConfig, method: str, *, learning_rate: float | None = None) -> MethodConfig:
        """The experiment's method settings with this revision's learning rate (and step cap).

        A matched-forgetting revision has no fixed rate: pass the calibrated one.
        """

        base = config.unlearning.methods[method]
        if self.is_matched:
            if learning_rate is None:
                raise ValueError("a matched-forgetting revision needs the calibrated learning rate")
            return replace(base, learning_rate=float(learning_rate), steps=self.matched.max_steps)
        return replace(base, learning_rate=self.learning_rates[method])

    def as_dict(self) -> dict[str, Any]:
        if self.is_matched:
            matched = self.matched
            return {
                "revision": self.revision,
                "tag": self.tag,
                "mode": self.mode,
                "matched_forgetting": {
                    "learning_rate_candidates": list(matched.learning_rate_candidates),
                    "thresholds": list(matched.thresholds),
                    "check_every_steps": matched.check_every_steps,
                    "max_steps": matched.max_steps,
                    "probe_templates": list(matched.probe_templates),
                },
            }
        # Revision 2's fingerprint was recorded before modes existed: keep its payload unchanged.
        return {"revision": self.revision, "tag": self.tag, "learning_rate": dict(sorted(self.learning_rates.items()))}

    @property
    def fingerprint(self) -> str:
        return fingerprint_payload(self.as_dict())


def load_unlearning_revision(path: str | Path) -> UnlearningRevision:
    revision_path = Path(path)
    raw = yaml.safe_load(revision_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"unlearning revision file {revision_path} did not parse to a mapping")
    return UnlearningRevision.from_dict(raw, source_path=str(revision_path))


def load_unlearning_revisions(directory: str | Path) -> tuple[UnlearningRevision, ...]:
    """Every revision file in ``directory``, ordered by revision number."""

    folder = Path(directory)
    if not folder.is_dir():
        return ()
    revisions = sorted((load_unlearning_revision(path) for path in folder.glob("*.yaml")), key=lambda r: r.revision)
    numbers = [revision.revision for revision in revisions]
    if len(numbers) != len(set(numbers)):
        raise ConfigError(f"duplicate unlearning revision numbers in {folder}: {numbers}")
    return tuple(revisions)
