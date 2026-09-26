from __future__ import annotations

import argparse
from pathlib import Path

from unlearning.config import load_config
from unlearning.evaluation.multiple_choice import (
    answer_index_from_label,
    bootstrap_accuracy_ci,
    medmcqa_prompt,
    mmlu_prompt,
    option_logprobs_with_model,
    pubmedqa_prompt,
    score_from_option_logprobs,
    summarize_mc_results,
)
from unlearning.utils.io import read_jsonl, write_json


def evaluate_records(model, tokenizer, records: list[dict], task: str, device: str, seed: int) -> dict:
    results = []
    for record in records:
        if task == "medmcqa":
            prompt = medmcqa_prompt(record)
            options = record["options"]
            answer_index = int(record["answer_index"])
        elif task == "mmlu_cf":
            prompt = mmlu_prompt(record)
            options = record["choices"]
            answer_index = answer_index_from_label(record["answer"], len(options))
        elif task == "pubmedqa":
            prompt = pubmedqa_prompt(record)
            options = ["yes", "no", "maybe"]
            answer_index = answer_index_from_label(
                {"yes": "0", "no": "1", "maybe": "2"}[record["final_decision"]],
                len(options),
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        option_logprobs = option_logprobs_with_model(model, tokenizer, prompt, options, device)
        results.append(
            score_from_option_logprobs(
                record_id=str(record.get("id") or record.get("pubid")),
                answer_index=answer_index,
                option_logprobs=option_logprobs,
            )
        )
    summary = summarize_mc_results(results)
    summary["accuracy_ci95"] = bootstrap_accuracy_ci(
        [result.correct for result in results],
        seed=seed,
    )
    summary["records"] = [result.__dict__ for result in results]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate clean S0/S50/S100 checkpoints only.")
    parser.add_argument("--config", default="configs/pilot/base.yaml")
    parser.add_argument("--model", required=True, help="Model id/path for S0, S50, or S100.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=["medmcqa", "mmlu_cf", "pubmedqa"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for clean checkpoint evaluation") from exc

    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_id,
        revision=config.model.revision,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    summary = evaluate_records(
        model,
        tokenizer,
        read_jsonl(args.dataset),
        task=args.task,
        device=args.device,
        seed=config.seed,
    )
    write_json(Path(args.output), summary, overwrite=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
