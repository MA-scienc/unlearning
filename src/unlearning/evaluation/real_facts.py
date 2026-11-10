"""Real-facts probe (pre-registration 6.3): true general medical facts that F also contains.

Every fictional record carries one of ten fixed clinical concepts, and its contamination
text states true general facts about it: the drug's class ("metoprolol, a beta blocker")
and the condition's monitoring marker ("LDL cholesterol" for hyperlipidemia). GA/NPO on F
push on those tokens directly. Scoring the facts on S_i, C_i and U separates removal of
general knowledge from interference with it.

Items: every distinct (drug -> drug class) and (condition -> marker name) pair in the run's
F records; the marker name is the marker text up to its first standalone number
("HbA1c 8.4%" -> "HbA1c"). Measure: mean per-token log-prob of the answer after a fixed
cloze prompt.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.evaluation.forgetting import target_logprob_with_model
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.runtime.settings import default_config_path
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import fingerprint_payload, hash_file
from unlearning.utils.io import read_jsonl, write_json

DRUG_PROMPT = "{drug} belongs to the drug class:"
MARKER_PROMPT = "A laboratory or clinical marker used to monitor {condition} is"
_FIRST_NUMBER = re.compile(r"^(.*?)\s+\d")


def marker_name(marker: str) -> str:
    match = _FIRST_NUMBER.match(marker)
    return (match.group(1) if match else marker).strip()


def real_fact_items(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    items: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        drug = (record["drug_example"], record["drug_class"])
        items.setdefault(("drug_class", drug[0]), {
            "item_id": f"drug_class::{drug[0]}", "kind": "drug_class",
            "prompt": DRUG_PROMPT.format(drug=drug[0]), "target": drug[1],
        })
        condition, name = record["condition"], marker_name(record["lab_or_marker"])
        items.setdefault(("marker", condition), {
            "item_id": f"marker::{condition}", "kind": "marker",
            "prompt": MARKER_PROMPT.format(condition=condition), "target": name,
        })
    return [items[key] for key in sorted(items)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["target_logprob"])
    return {
        "mean_target_logprob": sum(row["target_logprob"] for row in rows) / len(rows),
        "by_kind": {kind: sum(values) / len(values) for kind, values in sorted(by_kind.items())},
        "item_count": len(rows),
    }


def evaluate_real_facts(model, tokenizer, items, device: str) -> dict[str, Any]:
    rows = [
        {**item, "target_logprob": target_logprob_with_model(
            model, tokenizer, item["prompt"], item["target"], device, normalize_by_token_count=True)}
        for item in items
    ]
    return {"schema_version": "real-facts-v1", "summary": summarize(rows), "records": rows}


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Score true general medical facts contained in F.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-manifest", default=None)
    parser.add_argument("--records", required=True, help="the run's F records JSONL (forget/records.jsonl)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    snapshot = resolve_snapshot_reference(args.model, args.checkpoint_manifest, config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot.tokenizer_source, revision=snapshot.model_revision, use_fast=True)
    kwargs = {"revision": snapshot.model_revision} if snapshot.model_revision is not None else {}
    model = AutoModelForCausalLM.from_pretrained(snapshot.model_source, torch_dtype=torch.bfloat16, **kwargs).to(args.device)
    model.eval()
    items = real_fact_items(read_jsonl(args.records))
    with torch.inference_mode():
        result = evaluate_real_facts(model, tokenizer, items, args.device)
    result.update({
        "checkpoint_provenance": snapshot.as_dict(),
        "records_file": {"path": str(Path(args.records)), "sha256": hash_file(args.records)},
        "items_fingerprint": fingerprint_payload(items),
        "config_path": str(Path(args.config)),
        "code_version": code_version(),
    })
    write_json(Path(args.output), result, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
