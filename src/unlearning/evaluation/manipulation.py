"""Manipulation checks (pre-registration section 7): what did each level actually learn?

Measured on S0, every S level and every G control; per-item values are saved so the
analysis can run paired bootstraps over items.

1. ``content_dev_loss``: MedMCQA dev, the first 512 records in file order, written in the
   specialization template; mean loss over content tokens only. The scaffold (fixed field
   labels, option letters, separators) and constant fillers ("No explanation provided.",
   "unspecified") are masked.
2. ``format_loss``: the control template on held-out non-medical items (AQuA-RAT test split,
   pinned revision); mean loss over scaffold tokens only. ``format_content_loss`` (content
   tokens of the same texts) is reported beside it.
3. ``domain_text_loss``: mean NLL on PubMedQA contexts (first 512 evaluation items, contexts
   joined) minus mean NLL on WikiText-103 test paragraphs (the replay corpus's pinned
   revision, test split; first 512 non-heading paragraphs of at least 200 characters); both
   truncated to 512 tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.data.control_corpus import DATASET_ID as AQUA_ID, DATASET_REVISION as AQUA_REVISION, convert_item, format_control_text
from unlearning.data.medmcqa import format_specialization_text
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.settings import MachineSettings, default_config_path
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import fingerprint_payload
from unlearning.utils.io import read_jsonl, write_json

DEV_ITEMS = 512
TEXT_ITEMS = 512
MAX_TOKENS = 512
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")
FILLERS = {"No explanation provided.", "unspecified"}
LABELS = ("A", "B", "C", "D")


def template_segments(record: dict[str, Any], *, control: bool) -> list[tuple[str, bool]]:
    """(text, is_scaffold) pieces that concatenate to the training template exactly."""

    def value(text: str) -> tuple[str, bool]:
        return text, text in FILLERS

    subject = "Mathematics" if control else (record.get("subject_name") or "unspecified")
    topic = "Word problems" if control else (record.get("topic_name") or "unspecified")
    header = "multiple-choice question\n" if control else "Medical multiple-choice question\n"
    segments = [(header, True), ("Subject: ", True), value(subject), ("\n", True), ("Topic: ", True), value(topic),
                ("\n", True), ("Question: ", True), (record["question"], False), ("\n", True), ("Options:\n", True)]
    for index, (label, text) in enumerate(zip(LABELS, record["options"])):
        segments += [(f"{label}. ", True), (text, False)]
        segments.append(("\n", True))
    segments.append(("Correct answer: ", True))
    if record.get("answer_label") is None:
        segments.append(("unavailable", True))
    else:
        segments += [(record["answer_label"], False), (". ", True), (record["answer_text"], False)]
    segments += [("\n", True), ("Explanation: ", True), value(record.get("explanation") or "No explanation provided."), ("\n", True)]
    expected = format_control_text(record) if control else format_specialization_text(record)
    if "".join(text for text, _ in segments) != expected:
        raise ValueError("template segments do not reproduce the training template")
    return segments


def token_masks(tokenizer, segments: list[tuple[str, bool]]) -> tuple[list[int], list[bool]]:
    """Token ids and, per token, whether any of its characters is content."""

    text = "".join(piece for piece, _ in segments)
    content = []
    for piece, scaffold in segments:
        content += [not scaffold] * len(piece)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    flags = [any(content[start:end]) for start, end in encoded["offset_mapping"]]
    return encoded["input_ids"], flags


def mean_token_losses(model, torch, sequences: list[list[int]], masks: list[list[bool]], device, batch_size: int = 1) -> list[float | None]:
    """Per sequence: mean next-token loss over target tokens whose mask is True (None if none)."""

    results: list[float | None] = []
    for start in range(0, len(sequences), batch_size):
        chunk, chunk_masks = sequences[start:start + batch_size], masks[start:start + batch_size]
        width = max(len(ids) for ids in chunk)
        ids = torch.zeros((len(chunk), width), dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        weight = torch.zeros((len(chunk), width), dtype=torch.float32)
        for row, (tokens, mask) in enumerate(zip(chunk, chunk_masks)):
            ids[row, : len(tokens)] = torch.tensor(tokens)
            attention[row, : len(tokens)] = 1
            weight[row, : len(tokens)] = torch.tensor([float(flag) for flag in mask])
        with torch.inference_mode():
            logits = model(input_ids=ids.to(device), attention_mask=attention.to(device)).logits.float()
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].to(device).reshape(-1), reduction="none"
        ).view(len(chunk), -1).cpu()
        target_weight = weight[:, 1:] * attention[:, 1:]
        for row in range(len(chunk)):
            total = float(target_weight[row].sum())
            results.append(None if total == 0 else float((losses[row] * target_weight[row]).sum()) / total)
    return results


def _mean(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / len(present)


def wikitext_paragraphs(rows: list[str]) -> list[str]:
    paragraphs = [row.strip() for row in rows if row.strip() and not row.strip().startswith("=") and len(row.strip()) >= 200]
    return paragraphs[:TEXT_ITEMS]


def main(argv: list[str] | None = None) -> int:
    import pyarrow.parquet as pq
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Manipulation checks: content, format and domain-text losses.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = MachineSettings.from_env()
    layout = StorageLayout.for_config(config, settings)
    snapshot = resolve_snapshot_reference(args.model, args.checkpoint_manifest, config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot.tokenizer_source, revision=snapshot.model_revision, use_fast=True)
    kwargs = {"revision": snapshot.model_revision} if snapshot.model_revision is not None else {}
    model = AutoModelForCausalLM.from_pretrained(snapshot.model_source, torch_dtype=torch.bfloat16, **kwargs).to(args.device)
    model.eval()

    dev = read_jsonl(layout.generated / "medmcqa" / "dev_eval.jsonl")[:DEV_ITEMS]
    dev_encoded = [token_masks(tokenizer, template_segments(record, control=False)) for record in dev]
    content_dev = mean_token_losses(model, torch, [ids for ids, _ in dev_encoded], [mask for _, mask in dev_encoded], args.device)

    aqua_file = hf_hub_download(AQUA_ID, "raw/test-00000-of-00001.parquet", repo_type="dataset", revision=AQUA_REVISION,
                                cache_dir=str(settings.hf_home / "hub"))
    aqua = [converted[0] for index, item in enumerate(pq.read_table(aqua_file).to_pylist())
            if (converted := convert_item(index, item)) is not None]
    aqua_encoded = [token_masks(tokenizer, template_segments(record, control=True)) for record in aqua]
    format_scaffold = mean_token_losses(model, torch, [ids for ids, _ in aqua_encoded], [[not flag for flag in mask] for _, mask in aqua_encoded], args.device)
    format_content = mean_token_losses(model, torch, [ids for ids, _ in aqua_encoded], [mask for _, mask in aqua_encoded], args.device)

    pubmed = [" ".join(item["contexts"]) for item in read_jsonl(layout.generated / "pubmedqa" / "pubmedqa_eval.jsonl")[:TEXT_ITEMS]]
    wiki = wikitext_paragraphs(load_dataset(WIKITEXT[0], WIKITEXT[1], split="test", revision=WIKITEXT[2])["text"])
    def full(texts):
        ids = [tokenizer.encode(text, add_special_tokens=False)[:MAX_TOKENS] for text in texts]
        return mean_token_losses(model, torch, ids, [[True] * len(seq) for seq in ids], args.device)
    medical_text, general_text = full(pubmed), full(wiki)

    result = {
        "schema_version": "manipulation-checks-v1",
        "summary": {
            "content_dev_loss": _mean(content_dev),
            "format_loss": _mean(format_scaffold),
            "format_content_loss": _mean(format_content),
            "medical_text_nll": _mean(medical_text),
            "general_text_nll": _mean(general_text),
            "domain_text_loss": _mean(medical_text) - _mean(general_text),
        },
        "per_item": {
            "content_dev_loss": content_dev, "format_loss": format_scaffold, "format_content_loss": format_content,
            "medical_text_nll": medical_text, "general_text_nll": general_text,
        },
        "items": {
            "medmcqa_dev_ids": [record["id"] for record in dev],
            "aqua_rat_test_items": len(aqua),
            "pubmedqa_items": len(pubmed),
            "wikitext_paragraphs": len(wiki),
            "fingerprint": fingerprint_payload({"dev": [r["id"] for r in dev], "aqua": len(aqua), "pubmed": pubmed, "wiki": wiki}),
        },
        "checkpoint_provenance": snapshot.as_dict(),
        "config_path": str(Path(args.config)),
        "code_version": code_version(),
    }
    write_json(Path(args.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
