"""Specialization levels derived from ``specialization.checkpoint_fractions``.

A level is a point on the one continuous clean trajectory, named by its share of the
fixed budget in whole percent: fraction 0.25 is S25, its contaminated model is C25, its
unlearning trajectories carry ``s025``. The name encodes the fraction, so the mapping
between S, C and fraction needs no table. Steps are floor(fraction x S100 steps), which
gives the pilot's S50 = 1,645 of 3,290. Each level's parent is the previous level on the
trajectory (S0 has none).

The pilot config (fractions 0, 0.5, 1) yields exactly the pilot's S0/S50/S100 lineage;
tests/test_pilot_identity.py guards that.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from unlearning.config import PilotConfig
from unlearning.training.schedule import expected_pilot_schedule

_CLEAN = re.compile(r"^S(\d{1,3})$")
_CONTAMINATED = re.compile(r"^C(\d{1,3})$")


def fraction_percent(fraction: float) -> int:
    """Whole percent of a budget fraction; levels must sit on whole percents."""

    percent = round(fraction * 100)
    if not 0 <= percent <= 100 or abs(percent - fraction * 100) > 1e-9:
        raise ValueError(f"checkpoint fraction {fraction!r} must be a whole percent in [0, 1]")
    return percent


def clean_name(fraction: float) -> str:
    return f"S{fraction_percent(fraction)}"


def contaminated_name(fraction: float) -> str:
    return f"C{fraction_percent(fraction)}"


def _percent_of(name: str, pattern: re.Pattern, kind: str) -> int:
    match = pattern.match(name)
    if not match or int(match.group(1)) > 100 or str(int(match.group(1))) != match.group(1):
        raise ValueError(f"Unknown {kind} checkpoint {name!r}")
    return int(match.group(1))


def fraction_of_clean(name: str) -> float:
    return _percent_of(name, _CLEAN, "clean") / 100


def fraction_of_contaminated(name: str) -> float:
    return _percent_of(name, _CONTAMINATED, "contaminated") / 100


def contaminated_for_clean(name: str) -> str:
    return f"C{_percent_of(name, _CLEAN, 'clean')}"


def clean_for_contaminated(name: str) -> str:
    return f"S{_percent_of(name, _CONTAMINATED, 'contaminated')}"


@dataclass(frozen=True)
class SpecializationLevel:
    fraction: float
    clean: str
    contaminated: str
    step: int
    parent: str | None


def specialization_levels(config: PilotConfig) -> tuple[SpecializationLevel, ...]:
    """The configured levels in trajectory order, with steps and parents."""

    total = expected_pilot_schedule(config).s100_steps
    levels: list[SpecializationLevel] = []
    for fraction in config.specialization.checkpoint_fractions:
        step = 0 if fraction == 0 else math.floor(fraction * total + 1e-9)
        levels.append(
            SpecializationLevel(
                fraction=fraction,
                clean=clean_name(fraction),
                contaminated=contaminated_name(fraction),
                step=step,
                parent=levels[-1].clean if levels else None,
            )
        )
    return tuple(levels)


def clean_names(config: PilotConfig) -> tuple[str, ...]:
    return tuple(level.clean for level in specialization_levels(config))


def contaminated_names(config: PilotConfig) -> tuple[str, ...]:
    return tuple(level.contaminated for level in specialization_levels(config))


def level(config: PilotConfig, clean: str) -> SpecializationLevel:
    for item in specialization_levels(config):
        if item.clean == clean:
            return item
    raise ValueError(f"{clean} is not a configured level ({', '.join(clean_names(config))})")
