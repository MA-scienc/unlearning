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

# S<percent>: medical specialization levels. G<percent>: the format-matched control (Stage 2),
# contaminated as CG<percent>, trajectory token g<percent>.
_CLEAN = re.compile(r"^(S|G)(\d{1,3})$")
_CONTAMINATED = re.compile(r"^C(G?)(\d{1,3})$")


def fraction_percent(fraction: float) -> int:
    """Whole percent of a budget fraction; levels must sit on whole percents."""

    percent = round(fraction * 100)
    if not 0 <= percent <= 100 or abs(percent - fraction * 100) > 1e-9:
        raise ValueError(f"checkpoint fraction {fraction!r} must be a whole percent in [0, 1]")
    return percent


def clean_name(fraction: float, lineage: str = "S") -> str:
    return f"{lineage}{fraction_percent(fraction)}"


def contaminated_name(fraction: float, lineage: str = "S") -> str:
    return f"C{'G' if lineage == 'G' else ''}{fraction_percent(fraction)}"


def _parts(name: str, pattern: re.Pattern, kind: str) -> tuple[str, int]:
    match = pattern.match(name)
    if not match or int(match.group(2)) > 100 or str(int(match.group(2))) != match.group(2):
        raise ValueError(f"Unknown {kind} checkpoint {name!r}")
    lineage = "G" if match.group(1) == "G" else "S"
    return lineage, int(match.group(2))


def fraction_of_clean(name: str) -> float:
    return _parts(name, _CLEAN, "clean")[1] / 100


def fraction_of_contaminated(name: str) -> float:
    return _parts(name, _CONTAMINATED, "contaminated")[1] / 100


def lineage_of(name: str) -> str:
    """"S" (specialization) or "G" (control) for a clean (S25/G25) or contaminated (C25/CG25) name."""

    pattern, kind = (_CONTAMINATED, "contaminated") if name.startswith("C") else (_CLEAN, "clean")
    return _parts(name, pattern, kind)[0]


def contaminated_for_clean(name: str) -> str:
    lineage, percent = _parts(name, _CLEAN, "clean")
    return contaminated_name(percent / 100, lineage)


def clean_for_contaminated(name: str) -> str:
    lineage, percent = _parts(name, _CONTAMINATED, "contaminated")
    return clean_name(percent / 100, lineage)


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


def control_levels(config: PilotConfig) -> tuple[SpecializationLevel, ...]:
    """G<percent> control levels: same step as S<percent>, trained from S0 (the base model)."""

    steps = {item.fraction: item.step for item in specialization_levels(config)}
    return tuple(
        SpecializationLevel(
            fraction=fraction,
            clean=clean_name(fraction, "G"),
            contaminated=contaminated_name(fraction, "G"),
            step=steps[fraction],
            parent="S0",
        )
        for fraction in config.control_fractions
    )


def all_levels(config: PilotConfig) -> tuple[SpecializationLevel, ...]:
    return specialization_levels(config) + control_levels(config)


def clean_names(config: PilotConfig) -> tuple[str, ...]:
    """Every clean model that is contaminated and unlearned: S levels, then G controls."""

    return tuple(level.clean for level in all_levels(config))


def contaminated_names(config: PilotConfig) -> tuple[str, ...]:
    return tuple(level.contaminated for level in all_levels(config))


def level(config: PilotConfig, clean: str) -> SpecializationLevel:
    for item in all_levels(config):
        if item.clean == clean:
            return item
    raise ValueError(f"{clean} is not a configured level ({', '.join(clean_names(config))})")


_CONTAMINATED_IN_LABEL = re.compile(r"_(CG?\d{1,3})(?:_|$)")


def clean_parent_name(label: str) -> str:
    """The clean model of a contaminated or unlearned result label: MR_C25 -> S25, MR_U3_CG25_ga_f100 -> G25."""

    match = _CONTAMINATED_IN_LABEL.search(label)
    if not match:
        raise ValueError(f"no contaminated level in label {label!r}")
    return clean_for_contaminated(match.group(1))
