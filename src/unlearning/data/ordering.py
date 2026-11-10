"""Seeded data orders (Stage 2; docs/PREREGISTRATION_STAGE2.md, section 4).

The pilot reads every stream in a fixed order, so changing ``seed`` alone changes no
training trajectory. With ``data_order: seeded`` the config seed sets one permutation
per stream:

* ``specialization``: MedMCQA training records before packing (repeated every epoch);
* ``contamination`` and ``unlearning``: F at whole-patient granularity (a patient's
  templates stay together and in their usual order, so every 16-example step holds
  two complete patients, as in the pilot);
* ``replay``: replay passage indices (the corpus file itself is unchanged);
* ``control``: the format-matched control corpus records before packing (G levels).

``random.Random(f"{seed}:{stream}")`` is seeded from a string, which Python hashes
deterministically (not with the per-process hash salt), so the order is reproducible
across processes and machines. In ``canonical`` mode (the pilot) every function returns
its input order unchanged.
"""

from __future__ import annotations

import random
from typing import Any, Sequence, TypeVar

CANONICAL = "canonical"
SEEDED = "seeded"
DATA_ORDERS = (CANONICAL, SEEDED)
STREAMS = ("specialization", "control", "contamination", "unlearning", "replay")

T = TypeVar("T")


def _check(mode: str, stream: str) -> None:
    if mode not in DATA_ORDERS:
        raise ValueError(f"data_order must be one of {DATA_ORDERS}, got {mode!r}")
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")


def permutation(count: int, *, mode: str, seed: int, stream: str) -> list[int]:
    """Indices 0..count-1 in the stream's order (identity when canonical)."""

    _check(mode, stream)
    indices = list(range(count))
    if mode == SEEDED:
        random.Random(f"{seed}:{stream}").shuffle(indices)
    return indices


def ordered(items: Sequence[T], *, mode: str, seed: int, stream: str) -> list[T]:
    return [items[index] for index in permutation(len(items), mode=mode, seed=seed, stream=stream)]


def ordered_by_record(
    examples: Sequence[dict[str, Any]], *, mode: str, seed: int, stream: str, key: str = "record_id"
) -> list[dict[str, Any]]:
    """Permute whole records; examples of one record stay contiguous and in input order."""

    _check(mode, stream)
    if mode == CANONICAL:
        return list(examples)
    blocks: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        blocks.setdefault(str(example[key]), []).append(example)
    record_ids = list(blocks)
    return [example for record_id in ordered(record_ids, mode=mode, seed=seed, stream=stream) for example in blocks[record_id]]
