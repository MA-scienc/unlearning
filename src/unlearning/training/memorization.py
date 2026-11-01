"""Memorization and collateral-damage probes for matched-memorization contamination.

Memorization: teacher-forced exact recall on the exact training questions. One
batched forward pass per batch; a probe counts as recalled only if the model's
top prediction is correct for every supervised answer token (exactly when greedy
decoding would reproduce the answer). Cheap enough to run after every epoch.

Collateral damage: multiple-choice accuracy on probe sets disjoint from every
evaluation set -- MMLU-CF validation items outside the evaluation subset
(general) and MedMCQA training items (the specialization data, i.e. what S_i
learned). Probe sets are built once, saved, and verified by fingerprint.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from unlearning.config import PilotConfig
from unlearning.config.arms import MatchedMemorizationArm
from unlearning.training.contaminate import IGNORE_INDEX, collate_completion_batch, tokenize_completion_example
from unlearning.training.engine import autocast_context
from unlearning.utils.hashing import fingerprint_records
from unlearning.utils.io import read_jsonl, write_jsonl


def teacher_forced_exact(logits, labels, ignore_index: int = IGNORE_INDEX):
    """Per-example bool: argmax equals the label at every supervised position."""

    predictions = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    supervised = targets.ne(ignore_index)
    correct_or_unsupervised = predictions.eq(targets) | ~supervised
    return correct_or_unsupervised.all(dim=1) & supervised.any(dim=1)


class MemorizationProbe:
    def __init__(
        self,
        torch,
        tokenizer,
        contamination_examples: Sequence[Mapping[str, Any]],
        *,
        templates: Sequence[str],
        sequence_length: int,
        pad_token_id: int,
        batch_size: int = 16,
    ):
        self.torch = torch
        self.templates = tuple(templates)
        self.pad_token_id = int(pad_token_id)
        self.batch_size = max(1, int(batch_size))
        selected = [example for example in contamination_examples if example.get("template_id") in self.templates]
        if not selected:
            raise ValueError("no contamination examples match the memorization probe templates")
        self.encoded = [
            tokenize_completion_example(example, tokenizer=tokenizer, max_length=sequence_length)
            for example in selected
        ]
        self.template_of = [example["template_id"] for example in selected]
        self.fingerprint = fingerprint_records(
            [{"example_id": example["example_id"], "template_id": example["template_id"]} for example in selected]
        )

    @property
    def size(self) -> int:
        return len(self.encoded)

    def recall(self, model, device) -> dict[str, Any]:
        torch = self.torch
        was_training = model.training
        model.eval()
        hits: dict[str, list[bool]] = defaultdict(list)
        with torch.inference_mode():
            for start in range(0, len(self.encoded), self.batch_size):
                chunk = self.encoded[start : start + self.batch_size]
                batch = collate_completion_batch(chunk, pad_token_id=self.pad_token_id)
                input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
                attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
                labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
                with autocast_context(torch, device):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                exact = teacher_forced_exact(logits, labels).tolist()
                for offset, ok in enumerate(exact):
                    hits[self.template_of[start + offset]].append(bool(ok))
        if was_training:
            model.train()
        by_template = {template: sum(values) / len(values) for template, values in sorted(hits.items())}
        return {
            "mean_recall": sum(by_template.values()) / len(by_template),
            "by_template": by_template,
            "probe_count": len(self.encoded),
        }


# --- collateral-damage probes ---------------------------------------------------------


def _sample(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < size:
        raise ValueError(f"need {size} damage-probe records, only {len(records)} available")
    rng = random.Random(seed)
    return sorted(rng.sample(records, size), key=lambda record: str(record["id"]))


def build_damage_probes(
    config: PilotConfig,
    arm: MatchedMemorizationArm,
    generated_dir: Path,
    probe_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Create (once) or load the general and medical damage-probe sets."""

    general_path = probe_dir / "damage_general_mmlu_cf.jsonl"
    medical_path = probe_dir / "damage_medical_medmcqa_train.jsonl"
    if general_path.is_file() and medical_path.is_file():
        return {"general": read_jsonl(general_path), "medical": read_jsonl(medical_path)}

    medical_pool = [
        record
        for record in read_jsonl(generated_dir / "medmcqa" / "train_specialization.jsonl")
        if record.get("answer_index") is not None
    ]
    medical = _sample(medical_pool, arm.medical_probe_size, arm.damage_seed)

    from datasets import load_dataset

    from unlearning.data.evaluation_sets import _normalize_mmlu_cf

    evaluation_ids = {
        str(record["id"]) for record in read_jsonl(generated_dir / "mmlu_cf" / "mmlu_cf_eval_subset.jsonl")
    }
    dataset = load_dataset(
        config.evaluation.mmlu_cf_dataset_id,
        split=config.evaluation.mmlu_cf_split,
        revision=config.evaluation.mmlu_cf_revision,
    )
    general_pool = [
        record
        for index, example in enumerate(dataset)
        if (record := _normalize_mmlu_cf(dict(example), index))["id"] not in evaluation_ids
        and record["answer"] and len(record["choices"]) >= 2
    ]
    general = _sample(general_pool, arm.general_probe_size, arm.damage_seed + 1)
    if {str(record["id"]) for record in general} & evaluation_ids:
        raise RuntimeError("damage probe overlaps the MMLU-CF evaluation subset")
    write_jsonl(general_path, general, overwrite=False)
    write_jsonl(medical_path, medical, overwrite=False)
    return {"general": general, "medical": medical}


def damage_probe_fingerprints(probes: Mapping[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {name: fingerprint_records(records) for name, records in sorted(probes.items())}


def damage_accuracy(model, tokenizer, probes: Mapping[str, list[dict[str, Any]]], device, seed: int) -> dict[str, float]:
    from unlearning.evaluation.clean_checkpoints import evaluate_records

    was_training = model.training
    accuracy = {
        "general": float(evaluate_records(model, tokenizer, probes["general"], task="mmlu_cf", device=str(device), seed=seed)["accuracy"]),
        "medical": float(evaluate_records(model, tokenizer, probes["medical"], task="medmcqa", device=str(device), seed=seed)["accuracy"]),
    }
    if was_training:
        model.train()
    return accuracy


def _bias_removed_accuracy(records: list[dict[str, Any]]) -> float:
    """Accuracy after subtracting the model's mean log-probability for each answer position."""

    width = max(len(record["option_logprobs"]) for record in records)
    prior = []
    for position in range(width):
        values = [record["option_logprobs"][position] for record in records if len(record["option_logprobs"]) > position]
        prior.append(sum(values) / len(values))
    correct = 0
    for record in records:
        scores = record["option_logprobs"]
        choice = max(range(len(scores)), key=lambda index: scores[index] - prior[index])
        correct += int(choice == record["answer_index"])
    return correct / len(records)


def _probe_text(record: dict[str, Any]) -> str:
    if record.get("specialization_text"):
        return str(record["specialization_text"])
    return str(record["question"]) + "\n" + "\n".join(str(choice) for choice in record["choices"])


def _text_nll(torch, model, tokenizer, texts: Sequence[str], device, max_length: int = 512) -> float:
    total = 0.0
    count = 0
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            with autocast_context(torch, device):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            total += float(loss.float()) * (ids.shape[1] - 1)
            count += ids.shape[1] - 1
    return total / count


def damage_measures(
    model,
    tokenizer,
    probes: Mapping[str, list[dict[str, Any]]],
    device,
    seed: int,
    *,
    nll_texts: int,
) -> dict[str, dict[str, float]]:
    """Raw accuracy, letter-bias-free accuracy and text NLL on both damage probes."""

    import torch

    from unlearning.evaluation.clean_checkpoints import evaluate_records

    was_training = model.training
    model.eval()
    measures: dict[str, dict[str, float]] = {"accuracy": {}, "bias_removed_accuracy": {}, "text_nll": {}}
    for name, task in (("general", "mmlu_cf"), ("medical", "medmcqa")):
        summary = evaluate_records(model, tokenizer, probes[name], task=task, device=str(device), seed=seed)
        measures["accuracy"][name] = float(summary["accuracy"])
        measures["bias_removed_accuracy"][name] = _bias_removed_accuracy(summary["records"])
        if nll_texts > 0:
            texts = [_probe_text(record) for record in probes[name][:nll_texts]]
            measures["text_nll"][name] = _text_nll(torch, model, tokenizer, texts, device)
    if was_training:
        model.train()
    return measures


def damage_verdict(arm: MatchedMemorizationArm, baseline: Mapping[str, Any], measured: Mapping[str, Any]) -> dict[str, Any]:
    """Compare measures against the clean parent using the arm's pre-declared limits."""

    probes = ("general", "medical")
    verdict: dict[str, Any] = {
        "clean_parent": dict(baseline),
        "contaminated": dict(measured),
        "accuracy_drop": {name: baseline["accuracy"][name] - measured["accuracy"][name] for name in probes},
    }
    if arm.max_nll_increase is not None:
        verdict["text_nll_increase"] = {
            name: measured["text_nll"][name] - baseline["text_nll"][name] for name in probes
        }
        verdict["bias_removed_accuracy_drop"] = {
            name: baseline["bias_removed_accuracy"][name] - measured["bias_removed_accuracy"][name] for name in probes
        }
        verdict["limits"] = {
            "max_nll_increase": arm.max_nll_increase,
            "max_bias_removed_accuracy_drop": arm.max_bias_removed_accuracy_drop,
        }
        verdict["within_limit"] = all(
            verdict["text_nll_increase"][name] <= arm.max_nll_increase
            and verdict["bias_removed_accuracy_drop"][name] <= arm.max_bias_removed_accuracy_drop
            for name in probes
        )
    else:
        verdict["limits"] = {"max_accuracy_drop": arm.max_accuracy_drop}
        verdict["within_limit"] = all(drop <= arm.max_accuracy_drop for drop in verdict["accuracy_drop"].values())
    return verdict
