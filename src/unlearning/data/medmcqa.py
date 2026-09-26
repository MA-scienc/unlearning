from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from unlearning.config import PilotConfig
from unlearning.utils.hashing import fingerprint_records
from unlearning.utils.io import ensure_dir, write_json, write_jsonl
from unlearning.utils.manifest import build_manifest


OPTION_KEYS = ("opa", "opb", "opc", "opd")


@dataclass
class MedMCQAPreprocessResult:
    train_records: list[dict[str, Any]]
    dev_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class DomainSubsetResult:
    near_records: list[dict[str, Any]]
    far_records: list[dict[str, Any]]
    summary: dict[str, Any]

    def write(self, output_dir: Path, overwrite: bool = False) -> None:
        ensure_dir(output_dir)
        write_jsonl(output_dir / "near_domain_dev.jsonl", self.near_records, overwrite=overwrite)
        write_jsonl(output_dir / "far_domain_dev.jsonl", self.far_records, overwrite=overwrite)
        write_json(output_dir / "domain_subset_summary.json", self.summary, overwrite=overwrite)


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for MedMCQA acquisition. Install the pinned dependencies first."
        ) from exc
    return load_dataset


def _answer_index(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        if raw < 0:
            return None
        return raw
    if hasattr(raw, "item"):
        value = int(raw.item())
        return value if value >= 0 else None
    if isinstance(raw, str):
        stripped = raw.strip().lower()
        if stripped in {"", "-1", "none", "null"}:
            return None
        if stripped.isdigit():
            return int(stripped)
        if stripped in {"a", "b", "c", "d"}:
            return "abcd".index(stripped)
    raise ValueError(f"Unsupported MedMCQA cop value: {raw!r}")


def normalize_record(example: dict[str, Any], split: str) -> dict[str, Any]:
    answer_idx = _answer_index(example.get("cop"))
    if answer_idx is not None and answer_idx not in {0, 1, 2, 3}:
        raise ValueError(f"MedMCQA answer index out of range: {answer_idx}")
    options = [str(example.get(key) or "").strip() for key in OPTION_KEYS]
    answer_label = "ABCD"[answer_idx] if answer_idx is not None else None
    answer_text = options[answer_idx] if answer_idx is not None else None
    normalized = {
        "id": str(example.get("id") or "").strip(),
        "split": split,
        "question": str(example.get("question") or "").strip(),
        "options": options,
        "answer_index": answer_idx,
        "answer_label": answer_label,
        "answer_text": answer_text,
        "choice_type": str(example.get("choice_type") or "").strip(),
        "explanation": str(example.get("exp") or "").strip(),
        "subject_name": str(example.get("subject_name") or "").strip(),
        "topic_name": str(example.get("topic_name") or "").strip(),
    }
    normalized["specialization_text"] = format_specialization_text(normalized)
    return normalized


def format_specialization_text(record: dict[str, Any]) -> str:
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(("A", "B", "C", "D"), record["options"])
    )
    topic = record.get("topic_name") or "unspecified"
    explanation = record.get("explanation") or "No explanation provided."
    return (
        "Medical multiple-choice question\n"
        f"Subject: {record.get('subject_name') or 'unspecified'}\n"
        f"Topic: {topic}\n"
        f"Question: {record['question']}\n"
        f"Options:\n{options}\n"
        f"Correct answer: {_answer_line(record)}\n"
        f"Explanation: {explanation}\n"
    )


def _answer_line(record: dict[str, Any]) -> str:
    if record.get("answer_label") is None:
        return "unavailable"
    return f"{record['answer_label']}. {record['answer_text']}"


def _dataset_to_records(dataset: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [normalize_record(dict(example), split=split) for example in dataset]


def _assert_disjoint_ids(splits: dict[str, list[dict[str, Any]]]) -> None:
    ids_by_split = {name: {record["id"] for record in records} for name, records in splits.items()}
    names = list(ids_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = ids_by_split[left] & ids_by_split[right]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(f"MedMCQA split ID overlap between {left} and {right}: {sample}")


def _counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(record.get(field) or "UNKNOWN" for record in records).items()))


def prepare_medmcqa(
    config: PilotConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> MedMCQAPreprocessResult:
    load_dataset = _require_datasets()
    dataset = load_dataset(
        config.specialization.dataset_id,
        revision=config.specialization.dataset_revision,
    )
    split_names = {
        "train": config.specialization.train_split,
        "dev": config.specialization.dev_split,
        "test": config.specialization.test_split,
    }
    missing = [name for name in split_names.values() if name not in dataset]
    if missing:
        raise ValueError(f"MedMCQA dataset missing configured splits: {missing}")

    train = _dataset_to_records(dataset[split_names["train"]], "train")
    dev = _dataset_to_records(dataset[split_names["dev"]], "dev")
    test = _dataset_to_records(dataset[split_names["test"]], "test")
    _assert_disjoint_ids({"train": train, "dev": dev, "test": test})

    out = Path(output_dir)
    ensure_dir(out)
    write_jsonl(out / "train_specialization.jsonl", train, overwrite=overwrite)
    write_jsonl(out / "dev_eval.jsonl", dev, overwrite=overwrite)
    write_jsonl(out / "test_eval.jsonl", test, overwrite=overwrite)

    summary = {
        "dataset_id": config.specialization.dataset_id,
        "revision": config.specialization.dataset_revision,
        "source_url": config.specialization.source_url,
        "license": config.specialization.source_license,
        "schema_version": config.specialization.schema_version,
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "fingerprints": {
            "train": fingerprint_records(train),
            "dev": fingerprint_records(dev),
            "test": fingerprint_records(test),
        },
        "subject_counts": {
            "train": _counts(train, "subject_name"),
            "dev": _counts(dev, "subject_name"),
            "test": _counts(test, "subject_name"),
        },
        "topic_counts_dev": _counts(dev, "topic_name"),
    }
    manifest = build_manifest(
        artifact_type="medmcqa_preprocessed",
        schema_version=config.specialization.schema_version,
        payload=summary,
        source_metadata={
            "source_url": config.specialization.source_url,
            "license": config.specialization.source_license,
            "revision": config.specialization.dataset_revision,
        },
    )
    summary["manifest"] = manifest
    write_json(out / "manifest.json", summary, overwrite=overwrite)
    return MedMCQAPreprocessResult(train, dev, test, summary)


def _contains_any(value: str, keywords: Iterable[str]) -> bool:
    haystack = value.lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def is_near_domain(
    record: dict[str, Any],
    near_subjects: Iterable[str],
    near_keywords: Iterable[str],
) -> bool:
    subject = (record.get("subject_name") or "").lower()
    subject_match = subject in {item.lower() for item in near_subjects}
    searchable = " ".join(
        str(record.get(key) or "")
        for key in ("subject_name", "topic_name", "question", "explanation")
    )
    keyword_match = _contains_any(searchable, near_keywords)
    return subject_match and keyword_match


def is_far_domain(
    record: dict[str, Any],
    far_subjects: Iterable[str],
    near_keywords: Iterable[str],
) -> bool:
    subject = (record.get("subject_name") or "").lower()
    subject_match = subject in {item.lower() for item in far_subjects}
    searchable = " ".join(
        str(record.get(key) or "")
        for key in ("subject_name", "topic_name", "question", "explanation")
    )
    return subject_match and not _contains_any(searchable, near_keywords)


def compute_domain_subsets(
    dev_records: list[dict[str, Any]],
    near_subjects: Iterable[str],
    near_keywords: Iterable[str],
    far_subjects: Iterable[str],
) -> DomainSubsetResult:
    near = [record for record in dev_records if is_near_domain(record, near_subjects, near_keywords)]
    far = [record for record in dev_records if is_far_domain(record, far_subjects, near_keywords)]
    summary = {
        "near_count": len(near),
        "far_count": len(far),
        "near_fingerprint": fingerprint_records(near),
        "far_fingerprint": fingerprint_records(far),
        "near_subject_counts": _counts(near, "subject_name"),
        "far_subject_counts": _counts(far, "subject_name"),
        "near_topic_counts": _counts(near, "topic_name"),
        "far_topic_counts": _counts(far, "topic_name"),
        "near_review_sample": _review_sample(near),
        "far_review_sample": _review_sample(far),
    }
    return DomainSubsetResult(near, far, summary)


def _review_sample(records: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "id": record.get("id"),
            "subject_name": record.get("subject_name"),
            "topic_name": record.get("topic_name"),
            "question": record.get("question"),
        }
        for record in records[:count]
    ]
