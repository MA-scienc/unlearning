from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol


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
    tokenizer_batch_size: int = 1024,
) -> PackedTokenResult:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    eos_token_id = tokenizer.eos_token_id
    if add_eos and eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot add deterministic EOS delimiters")

    stream: list[int] = []
    for token_ids in _iter_tokenized(texts, tokenizer, tokenizer_batch_size):
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


def _iter_tokenized(
    texts: list[str],
    tokenizer: TokenizerLike,
    batch_size: int,
) -> Iterable[list[int]]:
    if batch_size <= 0:
        raise ValueError("tokenizer_batch_size must be positive")
    if callable(tokenizer):
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            for token_ids in encoded["input_ids"]:
                yield list(token_ids)
    else:
        for text in texts:
            yield tokenizer.encode(text, add_special_tokens=False)


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
