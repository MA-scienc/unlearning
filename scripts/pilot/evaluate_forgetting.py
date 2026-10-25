"""Evaluate synthetic-patient-f-v2 target forgetting for one model snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.runtime.settings import default_config_path
from unlearning.evaluation.forgetting import (
    decoy_values_from_records,
    evaluate_forgetting_with_model,
    load_forget_evaluation_reference,
    protected_fields_from_records,
    summarize_forgetting_results,
)
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate target forgetting on held-out synthetic-patient-f-v2 probes. "
            "This command evaluates one supplied model snapshot and does not select an "
            "operating point."
        )
    )
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID/path for a C_i, S_i, or U_i snapshot.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to generated forget_eval_examples.jsonl.",
    )
    parser.add_argument(
        "--records",
        help=(
            "Optional path to records.jsonl; defaults to the file next to --dataset. "
            "It is required for protected-field and decoy-margin evaluation."
        ),
    )
    parser.add_argument(
        "--checkpoint-manifest",
        help="Required for a local S_i/C_i/U_i checkpoint; optional for the matching S0 reference.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.set_defaults(mean_target_logprob=True)
    parser.add_argument(
        "--mean-target-logprob",
        dest="mean_target_logprob",
        action="store_true",
        help="Use mean per-token target likelihood (the default).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for target-forgetting evaluation") from exc

    config = load_config(args.config)
    snapshot = resolve_snapshot_reference(args.model, args.checkpoint_manifest, config)
    forget = load_forget_evaluation_reference(
        args.dataset,
        records_path=args.records,
        expected_schema_version=config.forget.schema_version,
        expected_record_count=config.forget.record_count,
        expected_templates_per_record=config.forget.eval_templates_per_record,
    )
    examples = forget.eval_examples
    protected_fields = protected_fields_from_records(forget.records)
    decoy_values = decoy_values_from_records(forget.records)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.tokenizer_source,
        revision=snapshot.model_revision,
        use_fast=True,
    )
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if snapshot.model_revision is not None:
        model_kwargs["revision"] = snapshot.model_revision
    model = AutoModelForCausalLM.from_pretrained(
        snapshot.model_source, **model_kwargs
    ).to(args.device)

    rows = evaluate_forgetting_with_model(
        model,
        tokenizer,
        examples,
        args.device,
        max_new_tokens=args.max_new_tokens,
        target_logprob_normalize_by_token_count=args.mean_target_logprob,
        protected_fields_by_record=protected_fields,
        decoy_values_by_record=decoy_values,
    )
    payload: dict[str, Any] = {
        "schema_version": "target-forgetting-evaluation-v1",
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "config_path": str(Path(args.config)),
        "forget_dataset_schema_version": config.forget.schema_version,
        "model": snapshot.model_source,
        "model_revision": snapshot.model_revision,
        "checkpoint_provenance": snapshot.as_dict(),
        "forget_dataset": {
            "eval_examples_path": str(forget.eval_examples_path),
            "records_path": str(forget.records_path),
            "manifest_path": str(forget.manifest_path),
            "eval_examples_file_sha256": forget.eval_examples_file_sha256,
            "records_file_sha256": forget.records_file_sha256,
            "eval_examples_fingerprint": forget.eval_examples_fingerprint,
            "records_fingerprint": forget.records_fingerprint,
            "manifest_file_sha256": forget.manifest_file_sha256,
            "manifest_payload_fingerprint": forget.manifest_payload_fingerprint,
            "validated_structure": forget.summary,
        },
        "generation": {
            "strategy": "greedy",
            "max_new_tokens": args.max_new_tokens,
        },
        "scoring": {
            "exact_field_leakage": "Unicode/case/whitespace-normalized standalone exact match",
            "normalized_field_f1": "article- and punctuation-normalized token F1",
            "target_logprob": "conditional mean token log probability"
            if args.mean_target_logprob
            else "conditional summed token log probability",
            "target_vs_decoy_margin": (
                "protected mean-token log probability minus mean same-field decoy likelihood"
            ),
            "mc_association": "conditional A/B/C/D label log probability using v2 options/order",
            "mc_label_protected_vs_decoy_margin": "protected label likelihood minus mean decoy label likelihood",
            "summary_unit": "synthetic record; templates are averaged within record",
        },
        "summary": summarize_forgetting_results(rows),
        "records": rows,
    }
    write_json(Path(args.output), payload, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

