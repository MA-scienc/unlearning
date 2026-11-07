"""Unlearning revisions that are defined outside the experiment config.

Revision 1 is the GA/NPO recipe in the experiment config (docs/PILOT_DESIGN.md).
A later revision changes only the learning rate and lives in its own file under
``configs/pilot/unlearning/``, so the experiment run ID (and every S/C checkpoint
bound to it) is unchanged and earlier revisions' snapshots and results stay in
place as the record. Each revision's outputs carry its tag (``U2`` ...) in the
trajectory ID, stage name and result labels.

* revision 2 (2026-11-07): GA and NPO at the matched contamination learning rate
  selected at gate_calibration (2e-5); revision 1 used 1e-6 and barely moved F.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from unlearning.config.schema import ConfigError, MethodConfig, PilotConfig
from unlearning.utils.hashing import fingerprint_payload

REVISION_KEYS = {"revision", "tag", "learning_rate"}
METHODS = ("ga", "npo")


@dataclass(frozen=True)
class UnlearningRevision:
    revision: int
    tag: str
    learning_rates: dict[str, float] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: str | None = None) -> "UnlearningRevision":
        # Only the learning rate may change: steps, snapshots, batch, clipping and
        # NPO beta/reference stay as pre-specified, so revisions differ in strength only.
        unknown = set(raw) - REVISION_KEYS
        if unknown:
            raise ConfigError(f"an unlearning revision may set only {sorted(REVISION_KEYS)}; got {sorted(unknown)}")
        try:
            revision = cls(
                revision=int(raw["revision"]),
                tag=str(raw["tag"]),
                learning_rates={str(name): float(value) for name, value in raw["learning_rate"].items()},
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
        if set(self.learning_rates) != set(METHODS):
            raise ConfigError(f"learning_rate must be given for exactly {list(METHODS)}")
        if any(value <= 0 for value in self.learning_rates.values()):
            raise ConfigError("learning_rate values must be positive")

    def method_config(self, config: PilotConfig, method: str) -> MethodConfig:
        """The experiment's method settings with this revision's learning rate."""

        return replace(config.unlearning.methods[method], learning_rate=self.learning_rates[method])

    def as_dict(self) -> dict[str, Any]:
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
