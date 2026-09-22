from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from unlearning.config import PilotConfig, load_config
from unlearning.data.evaluation_sets import prepare_mmlu_cf, prepare_pubmedqa
from unlearning.data.medmcqa import (
    MedMCQAPreprocessResult,
    compute_domain_subsets,
    prepare_medmcqa,
)
from unlearning.data.packing import compute_step_counts, tokenize_and_pack_texts
from unlearning.data.safeguards import assert_no_forget_terms_in_texts
from unlearning.data.synthetic_forget import generate_forget_dataset, write_forget_dataset
from unlearning.experiments.naming import experiment_run_id
from unlearning.utils.env import inspect_hardware
from unlearning.utils.io import ensure_dir, write_json
from unlearning.utils.manifest import build_manifest


def _load_tokenizer(tokenizer_id: str, revision: str | None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to compute tokenized MedMCQA stream size. "
            "Install the pinned project dependencies in a Python 3.11/3.12 environment."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_id, revision=revision, use_fast=True)


def build_plan(config: PilotConfig) -> dict:
    run_id = experiment_run_id(config)
    s100_epochs = config.specialization.epochs
    return {
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "seed": config.seed,
        "model": asdict(config.model),
        "specialization": {
            "continuous_trajectory": True,
            "checkpoint_fractions": list(config.specialization.checkpoint_fractions),
            "s100_epochs": s100_epochs,
            "context_length": config.specialization.context_length,
            "effective_batch_sequences": config.specialization.effective_batch_sequences,
        },
        "contamination": asdict(config.contamination),
        "unlearning_primary_analysis": {
            "snapshot_steps": list(config.unlearning.snapshot_steps),
            "headline_checkpoint_selection": "none; fixed trajectory is primary",
            "optional_future_single_point_rule": (
                "pre-specify a method-neutral target-forgetting threshold before observing results"
            ),
        },
    }


def run_preflight(
    config_path: str | Path,
    output_root: str | Path | None = None,
    skip_external: bool = False,
    overwrite: bool = False,
) -> dict:
    config = load_config(config_path)
    run_id = experiment_run_id(config)
    generated_root = Path(output_root) if output_root else config.paths.generated_dir / run_id
    ensure_dir(generated_root)
    ensure_dir(config.paths.manifest_dir)

    forget = generate_forget_dataset(config.forget)
    forget_result = write_forget_dataset(
        forget,
        generated_root / "forget",
        source_metadata={
            "license": config.forget.source_license,
            "schema_version": config.forget.schema_version,
            "seed": config.forget.seed,
            "synthetic": True,
            "real_patient_information_used": False,
        },
        overwrite=overwrite,
    )

    medmcqa_result: MedMCQAPreprocessResult | None = None
    pubmedqa_result = None
    mmlu_cf_result = None
    token_budget = None
    domain_subsets = None

    if not skip_external:
        medmcqa_result = prepare_medmcqa(config, generated_root / "medmcqa", overwrite=overwrite)
        train_texts = [record["specialization_text"] for record in medmcqa_result.train_records]
        assert_no_forget_terms_in_texts(forget.records, train_texts)

        domain_subsets = compute_domain_subsets(
            medmcqa_result.dev_records,
            near_subjects=config.evaluation.near_subjects,
            near_keywords=config.evaluation.near_topic_keywords,
            far_subjects=config.evaluation.far_subjects,
        )
        domain_subsets.write(generated_root / "medmcqa_domain_subsets", overwrite=overwrite)

        tokenizer = _load_tokenizer(config.model.tokenizer_id, config.model.revision)
        packed = tokenize_and_pack_texts(
            train_texts,
            tokenizer=tokenizer,
            context_length=config.specialization.context_length,
        )
        steps = compute_step_counts(
            packed_sequence_count=len(packed.sequences),
            epochs=config.specialization.epochs,
            effective_batch_sequences=config.specialization.effective_batch_sequences,
            drop_last_batches=config.specialization.drop_last_batches,
        )
        s50_exact = steps["s100_steps"] % 2 == 0
        token_budget = {
            "tokenizer_id": config.model.tokenizer_id,
            "context_length": config.specialization.context_length,
            "train_text_count": len(train_texts),
            "raw_token_count_with_eos": packed.raw_token_count,
            "packed_sequence_count": len(packed.sequences),
            "packed_token_count": len(packed.sequences) * config.specialization.context_length,
            "dropped_tail_tokens": packed.dropped_tail_tokens,
            "s100_epochs": config.specialization.epochs,
            "effective_batch_sequences": config.specialization.effective_batch_sequences,
            "drop_last_batches": config.specialization.drop_last_batches,
            "s100_steps": steps["s100_steps"],
            "s50_steps": steps["s100_steps"] // 2 if s50_exact else math.nan,
            "s50_exact": s50_exact,
        }

        pubmedqa_result = prepare_pubmedqa(config, generated_root / "pubmedqa", overwrite=overwrite)
        mmlu_cf_result = prepare_mmlu_cf(config, generated_root / "mmlu_cf", overwrite=overwrite)

    hardware = inspect_hardware()
    report = {
        "plan": build_plan(config),
        "hardware": hardware,
        "forget": forget_result,
        "medmcqa": medmcqa_result.summary if medmcqa_result else None,
        "domain_subsets": domain_subsets.summary if domain_subsets else None,
        "token_budget": token_budget,
        "pubmedqa": pubmedqa_result if pubmedqa_result else None,
        "mmlu_cf": mmlu_cf_result if mmlu_cf_result else None,
    }

    manifest = build_manifest(
        artifact_type="preflight_report",
        schema_version="preflight-v1",
        payload=report,
        source_metadata={"config_path": str(config_path)},
    )
    report["manifest"] = manifest
    write_json(generated_root / "preflight_report.json", report, overwrite=overwrite)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data/config preflight without training.")
    parser.add_argument("--config", default="configs/pilot/base.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip Hugging Face dataset/tokenizer downloads; still validates config and synthetic F.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    report = run_preflight(
        config_path=args.config,
        output_root=args.output_root,
        skip_external=args.skip_external,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
