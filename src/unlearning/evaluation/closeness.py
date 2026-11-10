"""Closeness of a model to its clean counterfactual S_i on the held-out forgetting prompts.

docs/MATCHED_FORGETTING_DESIGN.md, 3.6 (secondary measure). For each prompt, S_i's
greedy continuation (``positions`` tokens) fixes a shared context; both S_i and the
model under test are teacher-forced on prompt + that continuation, and
KL(S_i || model) is averaged over those answer positions. 0 means the model
predicts exactly like the clean model where it starts answering. Leakage and
accuracy cannot see a model that forgets by producing nonsense; this can.

Prompts are aggregated per synthetic record before the bootstrap
(CONFIRMATORY_PROTOCOL.md: templates of one record are not independent).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from unlearning.config import load_config
from unlearning.evaluation.progress import EvaluationProgress
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.runtime.settings import default_config_path
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import fingerprint_records, hash_file
from unlearning.utils.io import read_jsonl, write_json

POSITIONS = 16


def clean_continuation_ids(clean_model, prompt_ids: Sequence[int], *, positions: int, device, torch, pad_token_id: int | None) -> list[int]:
    """S_i's greedy continuation of the prompt (may stop early at end-of-sequence)."""

    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = clean_model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=positions,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    sequences = getattr(generated, "sequences", generated)
    return [int(token) for token in sequences[0, len(prompt_ids):].tolist()]


def answer_start_divergence(clean_model, model, prompt_ids: Sequence[int], continuation: Sequence[int], *, torch, device) -> dict[str, Any]:
    """Per-position KL(S_i || model) where S_i predicts its own continuation; at least the first position."""

    positions = max(1, len(continuation))
    context = list(prompt_ids) + list(continuation[: positions - 1])
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    first = len(prompt_ids) - 1
    with torch.no_grad():
        clean_logits = clean_model(input_ids=input_ids, use_cache=False).logits[0, first : first + positions].float()
        model_logits = model(input_ids=input_ids, use_cache=False).logits[0, first : first + positions].float()
    log_p = clean_logits.log_softmax(dim=-1)
    log_q = model_logits.log_softmax(dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).clamp(min=0.0)
    return {"kl_mean": float(kl.mean()), "kl_first": float(kl[0]), "positions": positions}


def summarize(rows: Sequence[dict[str, Any]], *, seed: int, samples: int = 1000, alpha: float = 0.05) -> dict[str, Any]:
    """Prompt means, record-level means and a record-cluster bootstrap CI for the record-level mean."""

    by_record: dict[str, list[float]] = defaultdict(list)
    first_by_record: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_record[row["record_id"]].append(row["kl_mean"])
        first_by_record[row["record_id"]].append(row["kl_first"])
    record_means = [sum(values) / len(values) for values in by_record.values()]
    first_means = [sum(values) / len(values) for values in first_by_record.values()]
    rng = random.Random(seed)
    estimates = sorted(
        sum(record_means[rng.randrange(len(record_means))] for _ in record_means) / len(record_means)
        for _ in range(samples)
    )
    return {
        "prompt_count": len(rows),
        "record_count": len(record_means),
        "kl_to_clean_answer_start": sum(record_means) / len(record_means),
        "kl_to_clean_answer_start_ci95": [estimates[int(alpha / 2 * samples)], estimates[min(samples - 1, int((1 - alpha / 2) * samples))]],
        "kl_to_clean_first_token": sum(first_means) / len(first_means),
        "prompt_mean_kl_to_clean_answer_start": sum(row["kl_mean"] for row in rows) / len(rows),
    }


def evaluate_closeness(clean_model, model, tokenizer, records: Sequence[dict[str, Any]], *, positions: int, device, seed: int, progress=None) -> dict[str, Any]:
    import torch

    clean_model.eval()
    model.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = []
    for record in records:
        prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
        continuation = clean_continuation_ids(clean_model, prompt_ids, positions=positions, device=device, torch=torch, pad_token_id=pad_token_id)
        divergence = answer_start_divergence(clean_model, model, prompt_ids, continuation, torch=torch, device=device)
        rows.append(
            {
                "example_id": record["example_id"],
                "record_id": record["record_id"],
                "template_id": record.get("template_id"),
                "clean_continuation": tokenizer.decode(continuation, skip_special_tokens=True),
                **divergence,
            }
        )
        if progress is not None:
            progress(len(rows))
    return {"suite": "closeness", "positions": positions, "summary": summarize(rows, seed=seed), "records": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Divergence of a checkpoint from its clean S_i on held-out forgetting prompts.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--dataset", required=True, help="forget/forget_eval_examples.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--clean-model", required=True, help="S_i: pinned base model ID for S0, else its checkpoint")
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positions", type=int, default=POSITIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    snapshot = resolve_snapshot_reference(args.model, args.checkpoint_manifest, config)
    clean = resolve_snapshot_reference(args.clean_model, args.clean_manifest, config)

    def load(reference):
        kwargs = {"torch_dtype": torch.bfloat16}
        if reference.model_revision is not None:
            kwargs["revision"] = reference.model_revision
        return AutoModelForCausalLM.from_pretrained(reference.model_source, **kwargs).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(clean.tokenizer_source, revision=clean.model_revision, use_fast=True)
    model_tokenizer = AutoTokenizer.from_pretrained(snapshot.tokenizer_source, revision=snapshot.model_revision, use_fast=True)
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("the checkpoint and its clean S_i use different tokenizers")
    clean_model, model = load(clean), load(snapshot)
    records = read_jsonl(args.dataset)
    progress = EvaluationProgress(config, args.output, len(records), f"closeness to clean, {len(records)} prompts")
    result = evaluate_closeness(clean_model, model, tokenizer, records, positions=args.positions, device=args.device, seed=config.seed, progress=progress)
    progress.finish()
    result["checkpoint_provenance"] = snapshot.as_dict()
    result["clean_provenance"] = clean.as_dict()
    result["evaluation_dataset"] = {
        "path": str(Path(args.dataset)),
        "file_sha256": hash_file(args.dataset),
        "records_fingerprint": fingerprint_records(records),
        "record_count": len(records),
    }
    result["config_path"] = str(Path(args.config))
    result["code_version"] = code_version()
    write_json(Path(args.output), result, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
