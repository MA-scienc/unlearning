"""General-text replay corpus for matched-memorization contamination.

Prepared once per experiment (needs network for the first download), then
verified by hash on every use. The corpus is the first N x L tokens of the
pinned WikiText-103 train split, packed into L-token passages, and put in a
seeded fixed order. Step k of every contamination run uses passages
[(k-1) * R, k * R), so all S_i rehearse identical text in identical order.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from unlearning.config import PilotConfig
from unlearning.config.arms import MatchedMemorizationArm, ReplayConfig
from unlearning.runtime.run_log import utc_now
from unlearning.utils.hashing import hash_file
from unlearning.utils.io import ensure_dir, read_json, write_json

REPLAY_MANIFEST_SCHEMA = "replay-corpus-v1"
SEQUENCES_FILE = "replay_sequences.pt"


def replay_dir(generated_dir: Path, arm: MatchedMemorizationArm) -> Path:
    return Path(generated_dir) / f"replay_{arm.tag}"


def pack_replay_sequences(
    texts: Iterable[str],
    tokenizer,
    *,
    sequence_length: int,
    count: int,
    batch_size: int = 512,
) -> tuple[list[list[int]], int]:
    """Concatenate texts into one token stream and cut ``count`` full passages."""

    needed = count * sequence_length
    stream: list[int] = []
    rows_used = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal batch
        if batch:
            for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                stream.extend(ids)
            batch = []

    for text in texts:
        rows_used += 1
        if not text.strip():
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            flush()
            if len(stream) >= needed:
                break
    if len(stream) < needed:
        flush()
    if len(stream) < needed:
        raise ValueError(f"replay source has only {len(stream)} tokens; {needed} required")
    return [stream[index * sequence_length : (index + 1) * sequence_length] for index in range(count)], rows_used


def _manifest_matches(manifest: dict[str, Any], replay: ReplayConfig, config: PilotConfig) -> bool:
    expected = {
        "dataset_id": replay.dataset_id,
        "dataset_config": replay.dataset_config,
        "revision": replay.revision,
        "split": replay.split,
        "sequence_length": replay.sequence_length,
        "count": replay.prepared_sequences,
        "shuffle_seed": replay.shuffle_seed,
        "tokenizer_id": config.model.tokenizer_id,
        "tokenizer_revision": config.model.revision,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def prepare_replay_corpus(config: PilotConfig, arm: MatchedMemorizationArm, generated_dir: Path) -> dict[str, Any]:
    if arm.replay is None:
        raise ValueError(f"arm {arm.name} has no replay section")
    replay = arm.replay
    out = replay_dir(generated_dir, arm)
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if not _manifest_matches(manifest, replay, config):
            raise RuntimeError(f"{out} was prepared with different replay settings; move it aside first")
        print(f"[replay] already prepared: {manifest_path}", flush=True)
        return manifest

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_id, revision=config.model.revision, use_fast=True)
    dataset = load_dataset(replay.dataset_id, replay.dataset_config, split=replay.split, revision=replay.revision)
    sequences, rows_used = pack_replay_sequences(
        (row["text"] for row in dataset),
        tokenizer,
        sequence_length=replay.sequence_length,
        count=replay.prepared_sequences,
    )
    order = list(range(len(sequences)))
    random.Random(replay.shuffle_seed).shuffle(order)
    tensor = torch.tensor([sequences[index] for index in order], dtype=torch.int32)
    ensure_dir(out)
    temporary = out / (SEQUENCES_FILE + ".tmp")
    torch.save(tensor, temporary)
    temporary.replace(out / SEQUENCES_FILE)
    manifest = {
        "schema_version": REPLAY_MANIFEST_SCHEMA,
        "dataset_id": replay.dataset_id,
        "dataset_config": replay.dataset_config,
        "revision": replay.revision,
        "license": replay.license,
        "split": replay.split,
        "sequence_length": replay.sequence_length,
        "count": replay.prepared_sequences,
        "tokens": replay.prepared_sequences * replay.sequence_length,
        "source_rows_used": rows_used,
        "source_rows_total": len(dataset),
        "shuffle_seed": replay.shuffle_seed,
        "tokenizer_id": config.model.tokenizer_id,
        "tokenizer_revision": config.model.revision,
        "sequences_file": SEQUENCES_FILE,
        "sequences_file_sha256": hash_file(out / SEQUENCES_FILE),
        "created_at": utc_now(),
    }
    write_json(manifest_path, manifest, overwrite=False)
    print(
        f"[replay] {replay.prepared_sequences} passages x {replay.sequence_length} tokens from "
        f"{rows_used} of {len(dataset)} {replay.dataset_config} rows -> {out}",
        flush=True,
    )
    return manifest


def load_replay_corpus(config: PilotConfig, arm: MatchedMemorizationArm, generated_dir: Path):
    """Return (LongTensor [count, length], manifest) after verifying settings and file hash."""

    if arm.replay is None:
        raise ValueError(f"arm {arm.name} has no replay section")
    out = replay_dir(generated_dir, arm)
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Replay corpus not prepared: {manifest_path}; run prepare_replay first")
    manifest = read_json(manifest_path)
    if not _manifest_matches(manifest, arm.replay, config):
        raise RuntimeError(f"Replay corpus at {out} does not match the arm's replay settings")
    if hash_file(out / manifest["sequences_file"]) != manifest["sequences_file_sha256"]:
        raise RuntimeError(f"Replay corpus file hash mismatch in {out}")
    import torch

    tensor = torch.load(out / manifest["sequences_file"], map_location="cpu", weights_only=True).long()
    if tuple(tensor.shape) != (arm.replay.prepared_sequences, arm.replay.sequence_length):
        raise RuntimeError(f"Replay corpus has shape {tuple(tensor.shape)}")
    return tensor, manifest


def prepare_main(argv: list[str] | None = None) -> int:
    import argparse

    from unlearning.config import load_config
    from unlearning.config.arms import load_matched_memorization_arm
    from unlearning.runtime.layout import StorageLayout
    from unlearning.runtime.settings import default_config_path
    from unlearning.training.matched_contamination import DEFAULT_ARM_PATH

    parser = argparse.ArgumentParser(description="Prepare the replay corpus for a matched-memorization arm.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--arm", default=str(DEFAULT_ARM_PATH))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    arm = load_matched_memorization_arm(args.arm)
    prepare_replay_corpus(config, arm, StorageLayout.for_config(config).generated)
    return 0
