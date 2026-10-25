from __future__ import annotations

import argparse
from pathlib import Path

from unlearning.config import load_config
from unlearning.runtime.settings import default_config_path
from unlearning.evaluation.multiple_choice import (
    answer_index_from_label,
    bootstrap_accuracy_ci,
    medmcqa_prompt,
    multiple_choice_labels,
    mmlu_prompt,
    option_logprobs_with_model,
    pubmedqa_prompt,
    score_from_option_logprobs,
    summarize_mc_results,
)
from unlearning.evaluation.progress import EvaluationProgress
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.utils.hashing import fingerprint_records, hash_file
from unlearning.utils.io import read_jsonl, write_json


def task_evaluation_inputs(record: dict, task: str) -> tuple[str, list[str], int, bool]:
    """Build one task's prompt and scoring continuations without loading a model."""

    if task == "medmcqa":
        return (
            medmcqa_prompt(record),
            multiple_choice_labels(len(record["options"])),
            int(record["answer_index"]),
            False,
        )
    if task == "mmlu_cf":
        options = multiple_choice_labels(len(record["choices"]))
        return (
            mmlu_prompt(record),
            options,
            answer_index_from_label(record["answer"], len(options)),
            False,
        )
    if task == "pubmedqa":
        options = ["yes", "no", "maybe"]
        return (
            pubmedqa_prompt(record),
            options,
            answer_index_from_label(
                {"yes": "0", "no": "1", "maybe": "2"}[record["final_decision"]],
                len(options),
            ),
            True,
        )
    raise ValueError(f"Unsupported task: {task}")


def evaluate_records(
    model,
    tokenizer,
    records: list[dict],
    task: str,
    device: str,
    seed: int,
    progress=None,
) -> dict:
    results = []
    for record in records:
        prompt, options, answer_index, normalize_by_token_count = task_evaluation_inputs(record, task)
        option_logprobs = option_logprobs_with_model(
            model,
            tokenizer,
            prompt,
            options,
            device,
            normalize_by_token_count=normalize_by_token_count,
        )
        results.append(
            score_from_option_logprobs(
                record_id=str(record.get("id") or record.get("pubid")),
                answer_index=answer_index,
                option_logprobs=option_logprobs,
            )
        )
        if progress is not None:
            progress(len(results))
    summary = summarize_mc_results(results)
    summary["accuracy_ci95"] = bootstrap_accuracy_ci(
        [result.correct for result in results],
        seed=seed,
    )
    summary["scoring"] = {
        "medmcqa_mmlu_cf": "conditional answer-label log probability (A/B/C/D)",
        "pubmedqa": "conditional mean token log probability (yes/no/maybe)",
    }
    summary["records"] = [result.__dict__ for result in results]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one pinned S0 or verified local S/C/U checkpoint on a retention task."
    )
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--model", required=True, help="Pinned base model ID or local checkpoint root/model path.")
    parser.add_argument(
        "--checkpoint-manifest",
        help="Required for a local checkpoint; optional for the matching S0 reference.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=["medmcqa", "mmlu_cf", "pubmedqa"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for clean checkpoint evaluation") from exc

    config = load_config(args.config)
    snapshot = resolve_snapshot_reference(args.model, args.checkpoint_manifest, config)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.tokenizer_source,
        revision=snapshot.model_revision,
        use_fast=True,
    )
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if snapshot.model_revision is not None:
        model_kwargs["revision"] = snapshot.model_revision
    model = AutoModelForCausalLM.from_pretrained(snapshot.model_source, **model_kwargs).to(args.device)
    records = read_jsonl(args.dataset)
    progress = EvaluationProgress(config, args.output, len(records), f"{args.task}, {len(records)} questions")
    summary = evaluate_records(
        model,
        tokenizer,
        records,
        task=args.task,
        device=args.device,
        seed=config.seed,
        progress=progress,
    )
    progress.finish()
    summary["checkpoint_provenance"] = snapshot.as_dict()
    summary["evaluation_dataset"] = {
        "path": str(Path(args.dataset)),
        "file_sha256": hash_file(args.dataset),
        "records_fingerprint": fingerprint_records(records),
        "record_count": len(records),
    }
    summary["config_path"] = str(Path(args.config))
    write_json(Path(args.output), summary, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
