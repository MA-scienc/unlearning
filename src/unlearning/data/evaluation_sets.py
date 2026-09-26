from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from unlearning.config import PilotConfig
from unlearning.utils.hashing import fingerprint_records
from unlearning.utils.io import ensure_dir, write_json, write_jsonl
from unlearning.utils.manifest import build_manifest


def _require_datasets():
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for evaluation dataset preparation. "
            "Install the pinned dependencies first."
        ) from exc
    return load_dataset, load_dataset_builder


def _normalize_pubmedqa(example: dict[str, Any]) -> dict[str, Any]:
    context = example.get("context") or {}
    contexts = context.get("contexts") if isinstance(context, dict) else None
    return {
        "pubid": str(example.get("pubid") or ""),
        "question": str(example.get("question") or "").strip(),
        "contexts": list(contexts or []),
        "long_answer": str(example.get("long_answer") or "").strip(),
        "final_decision": str(example.get("final_decision") or "").strip().lower(),
    }


def prepare_pubmedqa(
    config: PilotConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    load_dataset, _ = _require_datasets()
    dataset = load_dataset(
        config.evaluation.pubmedqa_dataset_id,
        config.evaluation.pubmedqa_subset,
        split=config.evaluation.pubmedqa_split,
        revision=config.evaluation.pubmedqa_revision,
    )
    records = [_normalize_pubmedqa(dict(example)) for example in dataset]
    out = Path(output_dir)
    ensure_dir(out)
    write_jsonl(out / "pubmedqa_eval.jsonl", records, overwrite=overwrite)
    summary = {
        "dataset_id": config.evaluation.pubmedqa_dataset_id,
        "subset": config.evaluation.pubmedqa_subset,
        "split": config.evaluation.pubmedqa_split,
        "revision": config.evaluation.pubmedqa_revision,
        "source_url": config.evaluation.pubmedqa_source_url,
        "license": config.evaluation.pubmedqa_license,
        "count": len(records),
        "fingerprint": fingerprint_records(records),
    }
    summary["manifest"] = build_manifest(
        artifact_type="pubmedqa_eval_dataset",
        schema_version="pubmedqa-eval-v1",
        payload=summary,
        source_metadata={
            "source_url": config.evaluation.pubmedqa_source_url,
            "license": config.evaluation.pubmedqa_license,
            "revision": config.evaluation.pubmedqa_revision,
        },
    )
    write_json(out / "manifest.json", summary, overwrite=overwrite)
    return summary


def _choice_list(example: dict[str, Any]) -> list[str]:
    for key in ("choices", "options", "answer_options"):
        value = example.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    options = []
    for key in ("A", "B", "C", "D", "opa", "opb", "opc", "opd"):
        if key in example:
            options.append(str(example[key]))
    return options


def _category(example: dict[str, Any]) -> str:
    for key in ("subject", "category", "sub_category", "domain", "task"):
        value = example.get(key)
        if value is not None:
            return str(value)
    return "UNKNOWN"


def _normalize_mmlu_cf(
    example: dict[str, Any],
    index: int,
    category_override: str | None = None,
) -> dict[str, Any]:
    return {
        "id": str(example.get("id") or example.get("question_id") or index),
        "question": str(
            example.get("question") or example.get("Question") or example.get("prompt") or ""
        ).strip(),
        "choices": _choice_list(example),
        "answer": str(
            example.get("answer") or example.get("Answer") or example.get("correct_answer") or ""
        ).strip(),
        "category": category_override or _category(example),
    }


def _stratified_sample(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(records) <= size:
        return records
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["category"]].append(record)
    sampled: list[dict[str, Any]] = []
    categories = sorted(buckets)
    base = size // len(categories)
    remainder = size % len(categories)
    leftovers: list[dict[str, Any]] = []
    for idx, category in enumerate(categories):
        bucket = list(buckets[category])
        rng.shuffle(bucket)
        take = min(len(bucket), base + (1 if idx < remainder else 0))
        sampled.extend(bucket[:take])
        leftovers.extend(bucket[take:])
    if len(sampled) < size:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: size - len(sampled)])
    return sorted(sampled[:size], key=lambda item: item["id"])


def prepare_mmlu_cf(
    config: PilotConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    load_dataset, load_dataset_builder = _require_datasets()
    try:
        builder = load_dataset_builder(
            config.evaluation.mmlu_cf_dataset_id,
            revision=config.evaluation.mmlu_cf_revision,
        )
        split_infos = builder.info.splits
        suffix = f"_{config.evaluation.mmlu_cf_split}"
        category_ranges = [
            (name[: -len(suffix)].replace("_", " "), info.num_examples)
            for name, info in split_infos.items()
            if name.endswith(suffix) and name != config.evaluation.mmlu_cf_split
        ]
        dataset = load_dataset(
            config.evaluation.mmlu_cf_dataset_id,
            split=config.evaluation.mmlu_cf_split,
            revision=config.evaluation.mmlu_cf_revision,
        )
        category_by_index: list[str | None] = []
        for category, count in category_ranges:
            category_by_index.extend([category] * count)
        if len(category_by_index) != len(dataset):
            category_by_index = [None] * len(dataset)
        records = [
            _normalize_mmlu_cf(dict(example), index, category_override=category_by_index[index])
            for index, example in enumerate(dataset)
        ]
    except ValueError as exc:
        raise ValueError(
            f"Could not load MMLU-CF split {config.evaluation.mmlu_cf_split!r}. "
            "The Hugging Face mirror currently uses split names such as 'val' and 'dev'."
        ) from exc
    sampled = _stratified_sample(
        records,
        size=config.evaluation.mmlu_cf_subset_size,
        seed=config.evaluation.mmlu_cf_seed,
    )
    out = Path(output_dir)
    ensure_dir(out)
    write_jsonl(out / "mmlu_cf_eval_subset.jsonl", sampled, overwrite=overwrite)
    category_counts: dict[str, int] = {}
    for record in sampled:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1
    summary = {
        "dataset_id": config.evaluation.mmlu_cf_dataset_id,
        "split": config.evaluation.mmlu_cf_split,
        "revision": config.evaluation.mmlu_cf_revision,
        "source_url": config.evaluation.mmlu_cf_source_url,
        "license": config.evaluation.mmlu_cf_license,
        "available_count": len(records),
        "subset_count": len(sampled),
        "subset_seed": config.evaluation.mmlu_cf_seed,
        "category_counts": dict(sorted(category_counts.items())),
        "fingerprint": fingerprint_records(sampled),
    }
    summary["manifest"] = build_manifest(
        artifact_type="mmlu_cf_eval_subset",
        schema_version="mmlu-cf-eval-v1",
        payload=summary,
        source_metadata={
            "source_url": config.evaluation.mmlu_cf_source_url,
            "license": config.evaluation.mmlu_cf_license,
            "revision": config.evaluation.mmlu_cf_revision,
        },
    )
    write_json(out / "manifest.json", summary, overwrite=overwrite)
    return summary
