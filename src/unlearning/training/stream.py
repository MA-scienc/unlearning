from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unlearning.data.safeguards import assert_no_forget_terms_in_texts
from unlearning.data.synthetic_forget import generate_forget_dataset
from unlearning.utils.hashing import fingerprint_payload, hash_file
from unlearning.utils.io import read_jsonl


@dataclass(frozen=True)
class TrainingStreamMetadata:
    train_path: str
    train_file_sha256: str
    train_record_count: int
    train_record_fingerprint: str
    packed_stream_fingerprint: str | None = None


def batch_indices_for_step(
    *,
    step: int,
    micro_step: int,
    packed_sequence_count: int,
    effective_batch_sequences: int,
    micro_batch_sequences: int,
) -> list[int]:
    if step <= 0:
        raise ValueError("step is 1-indexed and must be positive")
    if not 0 <= micro_step < effective_batch_sequences // micro_batch_sequences:
        raise ValueError("micro_step is out of range")
    global_start = ((step - 1) * effective_batch_sequences) + (micro_step * micro_batch_sequences)
    return [
        (global_start + offset) % packed_sequence_count
        for offset in range(micro_batch_sequences)
    ]


def cumulative_sequences_processed(step: int, effective_batch_sequences: int) -> int:
    if step < 0:
        raise ValueError("step must be non-negative")
    return step * effective_batch_sequences


def cumulative_tokens_processed(step: int, effective_batch_sequences: int, context_length: int) -> int:
    return cumulative_sequences_processed(step, effective_batch_sequences) * context_length


def load_specialization_train_records(train_path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(train_path)
    if any(record.get("split") != "train" for record in records):
        raise ValueError("specialization train artifact contains non-train records")
    return records


def assert_no_eval_records_in_train(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    train_ids = {record["id"] for record in train_records}
    dev_ids = {record["id"] for record in dev_records}
    test_ids = {record["id"] for record in test_records}
    if train_ids & dev_ids:
        raise ValueError("MedMCQA dev example found in specialization train stream")
    if train_ids & test_ids:
        raise ValueError("MedMCQA test example found in specialization train stream")


def validate_specialization_inputs(
    *,
    train_path: str | Path,
    dev_path: str | Path | None,
    test_path: str | Path | None,
    forget_seed_config,
) -> TrainingStreamMetadata:
    train_records = load_specialization_train_records(train_path)
    train_texts = [record["specialization_text"] for record in train_records]
    forget = generate_forget_dataset(forget_seed_config)
    assert_no_forget_terms_in_texts(forget.records, train_texts)
    if dev_path and test_path:
        assert_no_eval_records_in_train(
            train_records,
            read_jsonl(dev_path),
            read_jsonl(test_path),
        )
    return TrainingStreamMetadata(
        train_path=str(train_path),
        train_file_sha256=hash_file(train_path),
        train_record_count=len(train_records),
        train_record_fingerprint=fingerprint_payload(train_records),
    )
