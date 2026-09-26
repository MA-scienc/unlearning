from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MultipleChoiceResult:
    record_id: str
    prediction_index: int
    answer_index: int
    correct: bool
    option_logprobs: tuple[float, ...]
    correct_logprob: float
    nll: float


def score_from_option_logprobs(
    *,
    record_id: str,
    answer_index: int,
    option_logprobs: list[float],
) -> MultipleChoiceResult:
    if not option_logprobs:
        raise ValueError("option_logprobs must not be empty")
    if answer_index < 0 or answer_index >= len(option_logprobs):
        raise ValueError("answer_index out of range")
    prediction = max(range(len(option_logprobs)), key=lambda idx: option_logprobs[idx])
    correct_logprob = option_logprobs[answer_index]
    return MultipleChoiceResult(
        record_id=record_id,
        prediction_index=prediction,
        answer_index=answer_index,
        correct=prediction == answer_index,
        option_logprobs=tuple(option_logprobs),
        correct_logprob=correct_logprob,
        nll=-correct_logprob,
    )


def summarize_mc_results(results: list[MultipleChoiceResult]) -> dict[str, Any]:
    if not results:
        return {"count": 0, "accuracy": None, "mean_nll": None, "mean_correct_logprob": None}
    correct = sum(int(result.correct) for result in results)
    return {
        "count": len(results),
        "accuracy": correct / len(results),
        "mean_nll": sum(result.nll for result in results) / len(results),
        "mean_correct_logprob": sum(result.correct_logprob for result in results) / len(results),
    }


def bootstrap_accuracy_ci(
    correctness: list[bool],
    *,
    seed: int,
    samples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not correctness:
        return (None, None)
    rng = random.Random(seed)
    n = len(correctness)
    estimates = []
    for _ in range(samples):
        draw = [correctness[rng.randrange(n)] for _idx in range(n)]
        estimates.append(sum(draw) / n)
    estimates.sort()
    lower = estimates[int((alpha / 2) * samples)]
    upper = estimates[min(samples - 1, int((1 - alpha / 2) * samples))]
    return (lower, upper)


def medmcqa_prompt(record: dict[str, Any]) -> str:
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(("A", "B", "C", "D"), record["options"])
    )
    return f"Question: {record['question']}\nOptions:\n{options}\nAnswer:"


def mmlu_prompt(record: dict[str, Any]) -> str:
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(("A", "B", "C", "D"), record["choices"])
    )
    return f"Question: {record['question']}\nOptions:\n{options}\nAnswer:"


def pubmedqa_prompt(record: dict[str, Any]) -> str:
    context = "\n".join(record.get("contexts") or [])
    return f"Context:\n{context}\nQuestion: {record['question']}\nAnswer yes, no, or maybe:"


def option_logprobs_with_model(model, tokenizer, prompt: str, options: list[str], device: str) -> list[float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model evaluation") from exc

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for option in options:
            option_ids = tokenizer.encode(f" {option}", add_special_tokens=False)
            input_ids = torch.tensor([prompt_ids + option_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids)
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            total = 0.0
            for pos, token_id in enumerate(option_ids, start=len(prompt_ids)):
                total += float(log_probs[0, pos - 1, token_id].detach().cpu())
            scores.append(total)
    return scores


def answer_index_from_label(label: str, options: int) -> int:
    normalized = str(label).strip()
    if normalized.isdigit():
        value = int(normalized)
        if 0 <= value < options:
            return value
    upper = normalized.upper()
    if upper in {"A", "B", "C", "D"}:
        value = "ABCD".index(upper)
        if value < options:
            return value
    raise ValueError(f"Could not parse answer label {label!r}")
