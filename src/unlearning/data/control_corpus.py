"""Format-matched control corpus for G25 (pre-registration section 5).

Non-medical multiple-choice items with rationales (AQuA-RAT, pinned revision, Apache-2.0)
written in the specialization template with "Medical" removed, so G25 learns the same
format, amount of training and schedule state as S25 but no medical content.

Per item (index i in the pinned train split):
* the 5 options are reduced to 4 by dropping one distractor chosen with
  ``random.Random(f"aqua-rat:{i}")``; the remaining options keep their order and are
  relabelled A-D;
* answer-letter references in the rationale ("The answer is E", "Answer: (C)", "option B")
  are remapped to the new labels; the number of items where that happened is recorded;
* items are excluded if question, options or rationale contain any near-domain keyword,
  or if the question matches (after normalization) any evaluation-set question.
Forget-set terms are checked with the same safeguard as the specialization data.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Iterable

from unlearning.config import load_config
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.settings import MachineSettings, default_config_path
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import fingerprint_records, hash_file
from unlearning.utils.io import read_jsonl, write_json, write_jsonl

DATASET_ID = "deepmind/aqua_rat"
DATASET_REVISION = "33301c6a050c96af81f63cad5562cb5363e88971"
DATASET_LICENSE = "apache-2.0"
DATASET_FILE = "raw/train-00000-of-00001.parquet"
SUBJECT, TOPIC = "Mathematics", "Word problems"
LABELS = ("A", "B", "C", "D")
CORPUS_DIR = Path("control_g")
_OPTION = re.compile(r"^\s*([A-E])\s*\)\s*(.*)$", re.S)
_ANSWER_REF = re.compile(r"(?i)\b(answer(?:\s+is)?|ans|option|choice)(\s*[:=]?\s*\(?)([A-E])(\)?)(?![A-Za-z])")


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def convert_item(index: int, item: dict[str, Any]) -> tuple[dict[str, Any], bool] | None:
    """The 4-option record, and whether the rationale's letters were remapped; None if malformed."""

    parsed = [_OPTION.match(option) for option in item["options"]]
    if len(parsed) != 5 or not all(parsed) or item["correct"] not in "ABCDE":
        return None
    letters = [match.group(1) for match in parsed]
    texts = [match.group(2).strip() for match in parsed]
    if letters != list("ABCDE"):
        return None
    wrong = [letter for letter in letters if letter != item["correct"]]
    dropped = random.Random(f"aqua-rat:{index}").choice(wrong)
    kept = [letter for letter in letters if letter != dropped]
    mapping = {old: new for old, new in zip(kept, LABELS)}
    rationale = (item.get("rationale") or "").strip()
    remapped = _ANSWER_REF.sub(
        lambda m: m.group(1) + m.group(2) + mapping.get(m.group(3).upper(), m.group(3)) + m.group(4), rationale
    )
    options = [texts[letters.index(letter)] for letter in kept]
    answer_label = mapping[item["correct"]]
    return {
        "id": f"aqua_rat_train_{index}",
        "question": item["question"].strip(),
        "options": options,
        "answer_label": answer_label,
        "answer_text": options[LABELS.index(answer_label)],
        "explanation": remapped or None,
        "dropped_option": dropped,
    }, remapped != rationale


def format_control_text(record: dict[str, Any]) -> str:
    """The specialization template (data/medmcqa.format_specialization_text) without "Medical"."""

    options = "\n".join(f"{label}. {text}" for label, text in zip(LABELS, record["options"]))
    explanation = record.get("explanation") or "No explanation provided."
    return (
        "multiple-choice question\n"  # exactly the specialization scaffold minus "Medical "
        f"Subject: {SUBJECT}\n"
        f"Topic: {TOPIC}\n"
        f"Question: {record['question']}\n"
        f"Options:\n{options}\n"
        f"Correct answer: {record['answer_label']}. {record['answer_text']}\n"
        f"Explanation: {explanation}\n"
    )


def build_control_records(
    items: Iterable[dict[str, Any]], *, keywords: Iterable[str], evaluation_questions: Iterable[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    keywords = [keyword.lower() for keyword in keywords]
    blocked = {_normalize(question) for question in evaluation_questions}
    records, counts = [], {"items": 0, "malformed": 0, "medical_keyword": 0, "evaluation_overlap": 0, "letters_remapped": 0}
    for index, item in enumerate(items):
        counts["items"] += 1
        converted = convert_item(index, item)
        if converted is None:
            counts["malformed"] += 1
            continue
        record, remapped = converted
        searchable = " ".join([record["question"], *record["options"], record.get("explanation") or ""]).lower()
        if any(keyword in searchable for keyword in keywords):
            counts["medical_keyword"] += 1
            continue
        if _normalize(record["question"]) in blocked:
            counts["evaluation_overlap"] += 1
            continue
        counts["letters_remapped"] += int(remapped)
        records.append({**record, "specialization_text": format_control_text(record)})
    counts["kept"] = len(records)
    return records, counts


def _evaluation_questions(generated: Path) -> list[str]:
    questions = []
    for relative in ("mmlu_cf/mmlu_cf_eval_subset.jsonl", "medmcqa_domain_subsets/near_domain_dev.jsonl",
                     "medmcqa_domain_subsets/far_domain_dev.jsonl", "medmcqa/dev_eval.jsonl", "pubmedqa/pubmedqa_eval.jsonl"):
        path = generated / relative
        if path.is_file():
            questions += [str(row.get("question") or "") for row in read_jsonl(path)]
    return questions


def main(argv: list[str] | None = None) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from unlearning.data.safeguards import assert_no_forget_terms_in_texts

    parser = argparse.ArgumentParser(description="Build the format-matched control corpus for G25.")
    parser.add_argument("--config", default=default_config_path())
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = MachineSettings.from_env()
    layout = StorageLayout.for_config(config, settings)
    source = Path(hf_hub_download(DATASET_ID, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION,
                                  cache_dir=str(settings.hf_home / "hub")))
    records, counts = build_control_records(
        pq.read_table(source).to_pylist(),
        keywords=config.evaluation.near_topic_keywords,
        evaluation_questions=_evaluation_questions(layout.generated),
    )
    assert_no_forget_terms_in_texts(read_jsonl(layout.forget_dir / "records.jsonl"), [r["specialization_text"] for r in records])
    out = layout.generated / CORPUS_DIR
    write_jsonl(out / "control_corpus.jsonl", records)
    write_json(out / "manifest.json", {
        "artifact_type": "control_corpus",
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "license": DATASET_LICENSE, "file": DATASET_FILE,
                    "file_sha256": hash_file(source)},
        "counts": counts,
        "records_fingerprint": fingerprint_records(records),
        "corpus_file_sha256": hash_file(out / "control_corpus.jsonl"),
        "template": "specialization template without 'Medical'; Subject: Mathematics; Topic: Word problems",
        "code_version": code_version(),
    })
    print(f"{counts} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
