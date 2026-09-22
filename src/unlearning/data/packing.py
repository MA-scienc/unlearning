from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class PackedTokenResult:
    sequences: list[list[int]]
    raw_token_count: int
    dropped_tail_tokens: int


def tokenize_and_pack_texts(
    texts: list[str],
    tokenizer: TokenizerLike,
    context_length: int,
    add_eos: bool = True,
) -> PackedTokenResult:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    eos_token_id = tokenizer.eos_token_id
    if add_eos and eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot add deterministic EOS delimiters")

    stream: list[int] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        stream.extend(token_ids)
        if add_eos:
            stream.append(int(eos_token_id))

    full_sequences = len(stream) // context_length
    usable = full_sequences * context_length
    sequences = [
        stream[start : start + context_length] for start in range(0, usable, context_length)
    ]
    return PackedTokenResult(
        sequences=sequences,
        raw_token_count=len(stream),
        dropped_tail_tokens=len(stream) - usable,
    )


def compute_step_counts(
    packed_sequence_count: int,
    epochs: int,
    effective_batch_sequences: int,
    drop_last_batches: bool,
) -> dict[str, int]:
    if packed_sequence_count <= 0:
        raise ValueError("packed_sequence_count must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if effective_batch_sequences <= 0:
        raise ValueError("effective_batch_sequences must be positive")

    total_sequences = packed_sequence_count * epochs
    if drop_last_batches:
        s100_steps = total_sequences // effective_batch_sequences
    else:
        s100_steps = math.ceil(total_sequences / effective_batch_sequences)
    return {
        "total_sequences": total_sequences,
        "s100_steps": s100_steps,
        "s50_steps_floor": s100_steps // 2,
        "s50_steps_exact_possible": int(s100_steps % 2 == 0),
    }
