from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


_CHOICE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


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


def option_position_prior(records: list[dict[str, Any]]) -> list[float]:
    """Mean option log-probability at each answer position over a whole evaluation set."""

    width = max(len(record["option_logprobs"]) for record in records)
    prior = []
    for position in range(width):
        values = [record["option_logprobs"][position] for record in records if len(record["option_logprobs"]) > position]
        prior.append(sum(values) / len(values))
    return prior


def bias_removed_correctness(records: list[dict[str, Any]]) -> list[bool]:
    """Per-record correctness after subtracting the model's mean log-probability for each answer position.

    Removes a model-wide preference for a label (A/B/C/D, or yes/no/maybe) that
    shifts with training independently of the question, so accuracy is not
    driven by how that preference happens to align with the gold-label mix.
    """

    prior = option_position_prior(records)
    correctness = []
    for record in records:
        scores = record["option_logprobs"]
        choice = max(range(len(scores)), key=lambda index: scores[index] - prior[index])
        correctness.append(choice == record["answer_index"])
    return correctness


def bias_removed_accuracy(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return sum(bias_removed_correctness(records)) / len(records)


def bootstrap_bias_removed_accuracy_ci(
    records: list[dict[str, Any]],
    *,
    seed: int,
    samples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap over records; the position prior is re-estimated in every resample."""

    if not records:
        return (None, None)
    import numpy as np

    width = max(len(record["option_logprobs"]) for record in records)
    scores = np.full((len(records), width), np.nan)
    for row, record in enumerate(records):
        scores[row, : len(record["option_logprobs"])] = record["option_logprobs"]
    answers = np.array([record["answer_index"] for record in records])
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for sample in range(samples):
        draw = rng.integers(0, len(records), size=len(records))
        drawn = scores[draw]
        adjusted = np.where(np.isnan(drawn), -np.inf, drawn - np.nanmean(drawn, axis=0))
        estimates[sample] = float(np.mean(np.argmax(adjusted, axis=1) == answers[draw]))
    estimates.sort()
    lower = float(estimates[int((alpha / 2) * samples)])
    upper = float(estimates[min(samples - 1, int((1 - alpha / 2) * samples))])
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


def multiple_choice_labels(option_count: int) -> list[str]:
    """Return canonical answer labels for label-based multiple-choice scoring."""

    if option_count < 2 or option_count > len(_CHOICE_LABELS):
        raise ValueError(f"Unsupported multiple-choice option count: {option_count}")
    return list(_CHOICE_LABELS[:option_count])


def option_logprobs_with_model(
    model,
    tokenizer,
    prompt: str,
    options: list[str],
    device: str,
    *,
    normalize_by_token_count: bool = False,
) -> list[float]:
    """Score each completion conditional on the same prompt.

    Multiple-choice tasks should normally pass answer labels (for example A/B/C/D),
    which avoids confounding a model's choice with the length of option wording.
    For word-valued classes such as PubMedQA, token-count normalization is available
    as a diagnostic to prevent longer labels receiving an automatic penalty.
    """

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
            if not option_ids:
                raise ValueError(f"Option tokenized to no tokens: {option!r}")
            input_ids = torch.tensor([prompt_ids + option_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids)
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            total = 0.0
            for pos, token_id in enumerate(option_ids, start=len(prompt_ids)):
                total += float(log_probs[0, pos - 1, token_id].detach().cpu())
            scores.append(total / len(option_ids) if normalize_by_token_count else total)
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
