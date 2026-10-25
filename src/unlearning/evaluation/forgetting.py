"""Target-forgetting evaluation utilities for synthetic-patient-f-v2.

The functions in this module deliberately separate three things:

* pure, model-independent scoring of generations and log probabilities;
* optional adapters that call a causal language model; and
* record-clustered descriptive summaries.

This keeps tests independent of model hardware and, importantly, prevents the
six correlated probes for a synthetic patient from being treated as six
independent experimental units.  These utilities describe model behaviour;
they do not establish deletion or a privacy guarantee.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from unlearning.utils.hashing import fingerprint_payload, fingerprint_records, hash_file
from unlearning.utils.io import read_json, read_jsonl


_ARTICLE_TOKENS = frozenset({"a", "an", "the"})
_FIELD_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_FIELD_BOUNDARY_CLASS = r"[\w-]"
_CHOICE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# These are the synthetic fields whose disclosure is meaningful in the
# patient-specific probe context.  ``condition`` and ``drug_example`` are real
# medical concepts, so they should be interpreted as record associations, not
# as evidence that the model has forgotten the underlying medical concept.
DEFAULT_PROTECTED_RECORD_FIELDS = (
    "record_id",
    "patient_name",
    "private_code",
    "condition",
    "drug_example",
    "lab_or_marker",
    "canary_sentence",
)
DEFAULT_DECOY_COUNT = 3

_TARGET_FIELD_BY_TEMPLATE = {
    "eval_direct_code": "private_code",
    "eval_paraphrase_condition": "condition",
    "eval_paraphrase_medication": "drug_example",
    "eval_completion_canary": "canary_sentence",
    "eval_indirect_marker": "lab_or_marker",
    "eval_mc_association": "condition",
}
TARGET_RECORD_FIELDS = (
    "private_code",
    "condition",
    "drug_example",
    "canary_sentence",
    "lab_or_marker",
)


@dataclass(frozen=True)
class ForgetEvaluationReference:
    """Verified F-v2 evaluation inputs and their generated-data provenance."""

    eval_examples_path: Path
    records_path: Path
    manifest_path: Path
    eval_examples: list[dict[str, Any]]
    records: list[dict[str, Any]]
    eval_examples_file_sha256: str
    records_file_sha256: str
    eval_examples_fingerprint: str
    records_fingerprint: str
    manifest_file_sha256: str
    manifest_payload_fingerprint: str
    summary: dict[str, Any]


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _allow_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def normalize_field_text(value: str) -> str:
    """Normalize text for case-insensitive exact-field matching.

    Punctuation is retained deliberately.  A private code such as ``PC-A-B``
    should not be treated as identical to a merely similar string.  Whitespace
    and Unicode compatibility variants are normalized so ordinary rendering
    differences do not hide an otherwise exact disclosure.
    """

    text = _allow_empty_text(value, "field text")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_field_tokens(value: str) -> tuple[str, ...]:
    """Return SQuAD-style normalized tokens for a field-level F1 score."""

    normalized = unicodedata.normalize("NFKC", _allow_empty_text(value, "field text")).casefold()
    return tuple(
        token
        for token in _FIELD_TOKEN_PATTERN.findall(normalized)
        if token not in _ARTICLE_TOKENS
    )


def exact_field_leakage(generated_text: str, protected_value: str) -> bool:
    """Return whether a protected field appears as a standalone exact value.

    Matching is case-insensitive after Unicode and whitespace normalization.
    The boundary rule avoids counting an ID/code embedded inside a longer
    alphanumeric or hyphenated identifier (for example, ``PC-AB`` in
    ``PC-AB-EXTRA``).
    """

    generated = normalize_field_text(generated_text)
    protected = normalize_field_text(protected_value)
    if not protected:
        raise ValueError("protected_value must not be empty")
    pattern = re.compile(
        rf"(?<!{_FIELD_BOUNDARY_CLASS}){re.escape(protected)}(?!{_FIELD_BOUNDARY_CLASS})"
    )
    return bool(pattern.search(generated))


def exact_field_leakage_by_field(
    generated_text: str,
    protected_fields: Mapping[str, str],
) -> dict[str, bool]:
    """Score exact leakage separately for each named protected field."""

    if not protected_fields:
        raise ValueError("protected_fields must not be empty")
    return {
        str(field_name): exact_field_leakage(generated_text, protected_value)
        for field_name, protected_value in protected_fields.items()
    }


@dataclass(frozen=True)
class FieldF1:
    """Token-overlap details for one generated field versus its protected target."""

    precision: float
    recall: float
    f1: float
    prediction_token_count: int
    target_token_count: int
    overlap_token_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_field_f1_details(generated_text: str, protected_value: str) -> FieldF1:
    """Compute normalized token F1 for a patient-specific field response."""

    prediction_tokens = normalized_field_tokens(generated_text)
    target_tokens = normalized_field_tokens(protected_value)
    if not target_tokens:
        raise ValueError("protected_value has no normalized tokens")

    overlap = sum((Counter(prediction_tokens) & Counter(target_tokens)).values())
    precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
    recall = overlap / len(target_tokens)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return FieldF1(
        precision=precision,
        recall=recall,
        f1=f1,
        prediction_token_count=len(prediction_tokens),
        target_token_count=len(target_tokens),
        overlap_token_count=overlap,
    )


def normalized_field_f1(generated_text: str, protected_value: str) -> float:
    """Return only the normalized field F1 scalar."""

    return normalized_field_f1_details(generated_text, protected_value).f1


@dataclass(frozen=True)
class ProtectedDecoyScore:
    """Protected completion likelihood relative to its decoy completions."""

    protected_logprob: float
    mean_decoy_logprob: float
    protected_vs_decoy_margin: float
    decoy_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_protected_vs_decoys(
    protected_logprob: float,
    decoy_logprobs: Sequence[float],
) -> ProtectedDecoyScore:
    """Score a protected completion against the mean log probability of decoys.

    A positive margin means the protected answer is preferred over a typical
    type-compatible decoy.  Mean-decoy comparison is intentionally used here
    rather than a sum, so the scale does not change merely because a future
    probe has a different number of decoys.
    """

    if isinstance(decoy_logprobs, (str, bytes)) or len(decoy_logprobs) == 0:
        raise ValueError("decoy_logprobs must contain at least one numeric score")
    protected = _finite_float(protected_logprob, "protected_logprob")
    decoys = tuple(_finite_float(value, "decoy_logprob") for value in decoy_logprobs)
    mean_decoy = sum(decoys) / len(decoys)
    return ProtectedDecoyScore(
        protected_logprob=protected,
        mean_decoy_logprob=mean_decoy,
        protected_vs_decoy_margin=protected - mean_decoy,
        decoy_count=len(decoys),
    )


def protected_vs_decoy_margin(
    protected_logprob: float,
    decoy_logprobs: Sequence[float],
) -> float:
    """Return the protected-minus-mean-decoy log-likelihood margin."""

    return score_protected_vs_decoys(protected_logprob, decoy_logprobs).protected_vs_decoy_margin


def _mc_probe_components(example: Mapping[str, Any]) -> tuple[list[str], int]:
    raw_options = example.get("options")
    if not isinstance(raw_options, list) or len(raw_options) < 2:
        raise ValueError("v2 MC association example must have at least two options")
    options = [_require_text(option, "MC option") for option in raw_options]
    if len(options) != len(set(options)):
        raise ValueError("v2 MC association options must be unique")

    answer_index = example.get("answer_index")
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise TypeError("v2 MC association answer_index must be an integer")
    if answer_index < 0 or answer_index >= len(options):
        raise ValueError("v2 MC association answer_index is out of range")

    target = example.get("target")
    if target is not None and _require_text(target, "MC target") != options[answer_index]:
        raise ValueError("v2 MC target must equal options[answer_index]")
    return options, answer_index


def mc_association_inputs(example: Mapping[str, Any]) -> tuple[str, list[str], int]:
    """Build the v2 MC prompt and label continuations for length-safe scoring."""

    options, answer_index = _mc_probe_components(example)
    if len(options) > len(_CHOICE_LABELS):
        raise ValueError(f"Unsupported MC option count: {len(options)}")
    prompt = _require_text(example.get("prompt"), "MC prompt").rstrip()
    labels = list(_CHOICE_LABELS[: len(options)])
    option_lines = "\n".join(
        f"{label}. {option}" for label, option in zip(labels, options, strict=True)
    )
    return f"{prompt}\nOptions:\n{option_lines}\nAnswer:", labels, answer_index


@dataclass(frozen=True)
class MultipleChoiceAssociationResult:
    """A v2 association probe scored from its stored option order and answer index."""

    example_id: str
    record_id: str
    template_id: str
    answer_index: int
    prediction_index: int
    correct: bool
    option_logprobs: tuple[float, ...]
    protected_option: str
    decoy_options: tuple[str, ...]
    protected_logprob: float
    mean_decoy_logprob: float
    protected_vs_decoy_margin: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_mc_association(
    example: Mapping[str, Any],
    option_logprobs: Sequence[float],
) -> MultipleChoiceAssociationResult:
    """Score a v2 MC association probe without loading or calling a model.

    The answer is read from the stored ``options`` and ``answer_index`` fields.
    It is never inferred from a fixed option position, which is essential now
    that v2 randomizes option ordering per record.
    """

    options, answer_index = _mc_probe_components(example)
    if isinstance(option_logprobs, (str, bytes)) or len(option_logprobs) != len(options):
        raise ValueError("option_logprobs must have exactly one finite score per v2 option")
    scores = tuple(_finite_float(value, "option_logprob") for value in option_logprobs)
    prediction_index = max(range(len(scores)), key=lambda index: scores[index])
    protected_score = score_protected_vs_decoys(
        scores[answer_index],
        [score for index, score in enumerate(scores) if index != answer_index],
    )
    record_id = str(example.get("record_id") or "")
    template_id = str(example.get("template_id") or "eval_mc_association")
    example_id = str(example.get("example_id") or f"{record_id}::{template_id}")
    return MultipleChoiceAssociationResult(
        example_id=example_id,
        record_id=record_id,
        template_id=template_id,
        answer_index=answer_index,
        prediction_index=prediction_index,
        correct=prediction_index == answer_index,
        option_logprobs=scores,
        protected_option=options[answer_index],
        decoy_options=tuple(
            option for index, option in enumerate(options) if index != answer_index
        ),
        protected_logprob=protected_score.protected_logprob,
        mean_decoy_logprob=protected_score.mean_decoy_logprob,
        protected_vs_decoy_margin=protected_score.protected_vs_decoy_margin,
    )


def validate_forget_evaluation_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    expected_record_count: int,
    expected_templates_per_record: int,
) -> dict[str, Any]:
    """Validate the held-out F probe structure before any model is loaded."""

    if expected_record_count <= 0 or expected_templates_per_record <= 0:
        raise ValueError("Expected F record/template counts must be positive")
    expected_example_count = expected_record_count * expected_templates_per_record
    if len(examples) != expected_example_count:
        raise ValueError(
            f"Expected {expected_example_count} forget-evaluation examples, found {len(examples)}"
        )
    seen_examples: set[str] = set()
    templates_by_record: dict[str, set[str]] = {}
    mc_example_count = 0
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError(f"Forget-evaluation example {index} is not an object")
        for field_name in ("example_id", "record_id", "template_id", "split", "prompt", "target"):
            _require_text(example.get(field_name), f"forget-evaluation example {field_name}")
        if example["split"] != "forget_eval":
            raise ValueError("Forget-evaluation inputs may contain only split='forget_eval' rows")
        example_id = str(example["example_id"])
        if example_id in seen_examples:
            raise ValueError(f"Duplicate forget-evaluation example_id: {example_id}")
        seen_examples.add(example_id)
        record_id = str(example["record_id"])
        template_id = str(example["template_id"])
        templates = templates_by_record.setdefault(record_id, set())
        if template_id in templates:
            raise ValueError(f"Duplicate evaluation template {template_id!r} for record {record_id!r}")
        templates.add(template_id)
        is_mc = "options" in example or "answer_index" in example
        if is_mc:
            if template_id != "eval_mc_association":
                raise ValueError("Only eval_mc_association may contain MC options")
            _mc_probe_components(example)
            mc_example_count += 1

    if len(templates_by_record) != expected_record_count:
        raise ValueError(
            f"Expected {expected_record_count} F records in held-out probes, found {len(templates_by_record)}"
        )
    template_sets = {tuple(sorted(templates)) for templates in templates_by_record.values()}
    if len(template_sets) != 1:
        raise ValueError("Every F record must have the same held-out evaluation template set")
    for record_id, templates in templates_by_record.items():
        if len(templates) != expected_templates_per_record:
            raise ValueError(
                f"Record {record_id!r} has {len(templates)} evaluation templates; "
                f"expected {expected_templates_per_record}"
            )
    return {
        "example_count": len(examples),
        "record_count": len(templates_by_record),
        "templates_per_record": expected_templates_per_record,
        "template_ids": sorted(next(iter(template_sets))),
        "mc_example_count": mc_example_count,
        "eval_examples_fingerprint": fingerprint_records(examples),
    }


def load_forget_evaluation_reference(
    eval_examples_path: str | Path,
    *,
    expected_schema_version: str,
    expected_record_count: int,
    expected_templates_per_record: int,
    records_path: str | Path | None = None,
) -> ForgetEvaluationReference:
    """Load F-v2 records/probes and verify their sibling generated-data manifest.

    A publication-facing evaluation must not silently use a different F file
    from the one used to create C_i.  This check validates both byte-level file
    identity recorded downstream and content-level fingerprints from the
    generated F manifest.
    """

    eval_path = Path(eval_examples_path)
    if not eval_path.is_file():
        raise FileNotFoundError(f"Forget evaluation JSONL was not found: {eval_path}")
    record_path = Path(records_path) if records_path is not None else eval_path.parent / "records.jsonl"
    if not record_path.is_file():
        raise FileNotFoundError(f"Synthetic F records JSONL was not found: {record_path}")
    manifest_path = eval_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic F manifest was not found: {manifest_path}")

    examples = read_jsonl(eval_path)
    summary = validate_forget_evaluation_examples(
        examples,
        expected_record_count=expected_record_count,
        expected_templates_per_record=expected_templates_per_record,
    )
    records = read_jsonl(record_path)
    if len(records) != expected_record_count:
        raise ValueError(f"Expected {expected_record_count} synthetic F records, found {len(records)}")
    record_ids: set[str] = set()
    for record in records:
        record_id = _require_text(record.get("record_id"), "synthetic record_id")
        if record_id in record_ids:
            raise ValueError(f"Duplicate synthetic record_id: {record_id}")
        record_ids.add(record_id)
        if record.get("schema_version") != expected_schema_version:
            raise ValueError("Synthetic record schema does not match the frozen F configuration")
        if record.get("real_patient_information_used") is not False:
            raise ValueError("Every synthetic F record must state real_patient_information_used=false")
    if set(example["record_id"] for example in examples) != record_ids:
        raise ValueError("Held-out F probes and synthetic records do not have the same record IDs")

    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("Synthetic F manifest must be a JSON object")
    if raw_manifest.get("schema_version") != expected_schema_version:
        raise ValueError("Synthetic F manifest schema does not match the frozen configuration")
    if raw_manifest.get("real_patient_information_used") is not False:
        raise ValueError("Synthetic F manifest must state real_patient_information_used=false")
    if raw_manifest.get("record_count") != expected_record_count:
        raise ValueError("Synthetic F manifest record_count does not match configuration")
    if raw_manifest.get("eval_example_count") != len(examples):
        raise ValueError("Synthetic F manifest eval_example_count does not match the supplied JSONL")
    fingerprints = raw_manifest.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ValueError("Synthetic F manifest lacks fingerprints")
    records_fingerprint = fingerprint_records(records)
    if fingerprints.get("records") != records_fingerprint:
        raise ValueError("Synthetic F records fingerprint does not match the supplied records JSONL")
    if fingerprints.get("eval_examples") != summary["eval_examples_fingerprint"]:
        raise ValueError("Synthetic F eval fingerprint does not match the supplied JSONL")
    artifact_manifest = raw_manifest.get("manifest")
    if not isinstance(artifact_manifest, Mapping):
        raise ValueError("Synthetic F manifest lacks immutable manifest metadata")
    if artifact_manifest.get("artifact_type") != "synthetic_forget_dataset":
        raise ValueError("Input F manifest is not a synthetic forget dataset")
    expected_payload_fingerprint = fingerprint_payload(
        {key: value for key, value in raw_manifest.items() if key != "manifest"}
    )
    manifest_payload_fingerprint = artifact_manifest.get("payload_fingerprint")
    if manifest_payload_fingerprint != expected_payload_fingerprint:
        raise ValueError("Synthetic F manifest payload fingerprint does not match its summary")
    return ForgetEvaluationReference(
        eval_examples_path=eval_path,
        records_path=record_path,
        manifest_path=manifest_path,
        eval_examples=examples,
        records=records,
        eval_examples_file_sha256=hash_file(eval_path),
        records_file_sha256=hash_file(record_path),
        eval_examples_fingerprint=summary["eval_examples_fingerprint"],
        records_fingerprint=records_fingerprint,
        manifest_file_sha256=hash_file(manifest_path),
        manifest_payload_fingerprint=str(manifest_payload_fingerprint),
        summary=summary,
    )


def protected_fields_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    field_names: Sequence[str] = DEFAULT_PROTECTED_RECORD_FIELDS,
) -> dict[str, dict[str, str]]:
    """Extract protected v2 fields keyed by ``record_id`` for generation scoring."""

    by_record: dict[str, dict[str, str]] = {}
    for record in records:
        record_id = _require_text(record.get("record_id"), "record_id")
        if record_id in by_record:
            raise ValueError(f"Duplicate record_id in protected records: {record_id}")
        fields = {
            field_name: value
            for field_name in field_names
            if isinstance((value := record.get(field_name)), str) and value.strip()
        }
        if not fields:
            raise ValueError(f"No protected fields found for record_id {record_id}")
        by_record[record_id] = fields
    return by_record


def decoy_values_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    field_names: Sequence[str] = TARGET_RECORD_FIELDS,
    decoy_count: int = DEFAULT_DECOY_COUNT,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Build deterministic same-field decoys for likelihood-margin probes.

    Each target is compared with values assigned to other synthetic records,
    never with arbitrary natural-language alternatives.  This distinguishes a
    record-specific association from general familiarity with a medical term.
    The target record and field choose a deterministic rotation, so the decoy
    set is stable without encoding a record-index shortcut in F itself.
    """

    if decoy_count <= 0:
        raise ValueError("decoy_count must be positive")
    ordered_records = list(records)
    if not ordered_records:
        raise ValueError("records must not be empty when building decoy values")
    record_ids: list[str] = []
    values_by_field: dict[str, list[str]] = {field_name: [] for field_name in field_names}
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in ordered_records:
        record_id = _require_text(record.get("record_id"), "record_id")
        if record_id in records_by_id:
            raise ValueError(f"Duplicate record_id in decoy records: {record_id}")
        record_ids.append(record_id)
        records_by_id[record_id] = record
        for field_name in field_names:
            values_by_field[field_name].append(
                _require_text(record.get(field_name), f"record field {field_name}")
            )

    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for record_id in record_ids:
        record = records_by_id[record_id]
        per_field: dict[str, tuple[str, ...]] = {}
        for field_name in field_names:
            target = _require_text(record.get(field_name), f"record field {field_name}")
            candidates = sorted({value for value in values_by_field[field_name] if value != target})
            if len(candidates) < decoy_count:
                raise ValueError(
                    f"Not enough unique {field_name} decoys for {record_id}: "
                    f"need {decoy_count}, found {len(candidates)}"
                )
            digest = hashlib.sha256(f"{record_id}:{field_name}".encode("utf-8")).digest()
            offset = int.from_bytes(digest[:8], "big") % len(candidates)
            rotated = candidates[offset:] + candidates[:offset]
            per_field[field_name] = tuple(rotated[:decoy_count])
        result[record_id] = per_field
    return result


def target_field_for_example(example: Mapping[str, Any]) -> str:
    """Return the intended v2 target field label for one held-out probe."""

    return _TARGET_FIELD_BY_TEMPLATE.get(str(example.get("template_id") or ""), "target")


def completion_for_prompt(prompt: str, target: str) -> str:
    """Make a readable target continuation without changing an explicit leading space."""

    prompt_text = _require_text(prompt, "prompt")
    target_text = _require_text(target, "target")
    if prompt_text[-1].isspace() or target_text[0].isspace():
        return target_text
    return f" {target_text}"


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model-backed forgetting evaluation") from exc
    return torch


def conditional_logprob_with_model(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    device: str,
    *,
    normalize_by_token_count: bool = False,
) -> float:
    """Return conditional log probability for an already-formatted completion.

    This adapter is intentionally thin.  Pure metric tests can instead pass a
    deterministic ``target_logprob_hook`` to :func:`evaluate_forgetting_examples`.
    """

    torch = _require_torch()
    prompt_text = _require_text(prompt, "prompt")
    completion_text = _require_text(completion, "completion")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("prompt tokenized to no tokens")
    if not completion_ids:
        raise ValueError("completion tokenized to no tokens")

    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        total = 0.0
        for offset, token_id in enumerate(completion_ids):
            position = len(prompt_ids) + offset
            total += float(log_probs[0, position - 1, token_id].detach().cpu())
    return total / len(completion_ids) if normalize_by_token_count else total


def target_logprob_with_model(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device: str,
    *,
    normalize_by_token_count: bool = False,
) -> float:
    """Score a protected target after applying the standard continuation separator."""

    return conditional_logprob_with_model(
        model,
        tokenizer,
        prompt,
        completion_for_prompt(prompt, target),
        device,
        normalize_by_token_count=normalize_by_token_count,
    )


def generate_with_model(
    model,
    tokenizer,
    prompt: str,
    device: str,
    *,
    max_new_tokens: int = 64,
) -> str:
    """Greedily generate a continuation for one forgetting probe."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    torch = _require_torch()
    prompt_text = _require_text(prompt, "prompt")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("prompt tokenized to no tokens")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        # One unpadded prompt: an all-ones mask is what generate() falls back to anyway;
        # passing it explicitly avoids the pad==eos ambiguity warning.
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    model.eval()
    with torch.no_grad():
        generated = model.generate(**generation_kwargs)
    sequences = getattr(generated, "sequences", generated)
    continuation_ids = sequences[0, len(prompt_ids) :]
    return str(tokenizer.decode(continuation_ids, skip_special_tokens=True))


def make_model_generation_hook(
    model,
    tokenizer,
    device: str,
    *,
    max_new_tokens: int = 64,
) -> Callable[[str], str]:
    """Create a generation hook compatible with :func:`evaluate_forgetting_examples`."""

    return lambda prompt: generate_with_model(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=max_new_tokens,
    )


def make_model_target_logprob_hook(
    model,
    tokenizer,
    device: str,
    *,
    normalize_by_token_count: bool = False,
) -> Callable[[str, str], float]:
    """Create a protected-target log-probability hook without evaluating yet."""

    return lambda prompt, target: target_logprob_with_model(
        model,
        tokenizer,
        prompt,
        target,
        device,
        normalize_by_token_count=normalize_by_token_count,
    )


def make_model_mc_option_logprob_hook(
    model,
    tokenizer,
    device: str,
) -> Callable[[str, Sequence[str]], Sequence[float]]:
    """Create a label-continuation scoring hook for v2 MC association probes."""

    def score_options(prompt: str, labels: Sequence[str]) -> Sequence[float]:
        from unlearning.evaluation.multiple_choice import option_logprobs_with_model

        return option_logprobs_with_model(
            model,
            tokenizer,
            prompt,
            list(labels),
            device,
            normalize_by_token_count=False,
        )

    return score_options


def _identity_from_example(example: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    prompt = _require_text(example.get("prompt"), "probe prompt")
    target = _require_text(example.get("target"), "probe target")
    record_id = _require_text(example.get("record_id"), "probe record_id")
    template_id = _require_text(example.get("template_id"), "probe template_id")
    example_id = str(example.get("example_id") or f"{record_id}::{template_id}")
    return example_id, record_id, template_id, prompt, target


def score_forgetting_example(
    example: Mapping[str, Any],
    *,
    generated_text: str | None = None,
    target_logprob: float | None = None,
    decoy_logprobs: Sequence[float] | None = None,
    mc_option_logprobs: Sequence[float] | None = None,
    protected_fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create one JSON-ready target-forgetting result row from precomputed outputs."""

    example_id, record_id, template_id, _prompt, target = _identity_from_example(example)
    row: dict[str, Any] = {
        "example_id": example_id,
        "record_id": record_id,
        "template_id": template_id,
        "target_field": target_field_for_example(example),
        "target": target,
        "generated_text": generated_text,
        "exact_field_leakage": None,
        "field_leakage": {},
        "normalized_field_f1": None,
        "normalized_field_precision": None,
        "normalized_field_recall": None,
        "target_logprob": None,
        "target_vs_decoy": None,
        "target_vs_decoy_margin": None,
        "mc_association": None,
        "mc_association_correct": None,
        "mc_label_protected_vs_decoy_margin": None,
        "protected_vs_decoy_margin": None,
    }
    if generated_text is not None:
        generated = _require_text(generated_text, "generated_text") if generated_text else ""
        fields = dict(protected_fields or {row["target_field"]: target})
        if not fields:
            raise ValueError("protected_fields must not be empty when generation is scored")
        leakage = exact_field_leakage_by_field(generated, fields)
        field_f1 = normalized_field_f1_details(generated, target)
        row.update(
            {
                "generated_text": generated,
                "exact_field_leakage": any(leakage.values()),
                "field_leakage": leakage,
                "normalized_field_f1": field_f1.f1,
                "normalized_field_precision": field_f1.precision,
                "normalized_field_recall": field_f1.recall,
            }
        )
    if target_logprob is not None:
        row["target_logprob"] = _finite_float(target_logprob, "target_logprob")
    if decoy_logprobs is not None:
        if row["target_logprob"] is None:
            raise ValueError("decoy_logprobs require a protected target_logprob")
        decoy_score = score_protected_vs_decoys(row["target_logprob"], decoy_logprobs)
        row["target_vs_decoy"] = decoy_score.as_dict()
        row["target_vs_decoy_margin"] = decoy_score.protected_vs_decoy_margin
        # Retain a row-level alias for older downstream readers.  New summaries
        # use the type-specific fields below and never pool label and free-text
        # margins together.
        row["protected_vs_decoy_margin"] = decoy_score.protected_vs_decoy_margin

    is_mc = "options" in example or "answer_index" in example
    if mc_option_logprobs is not None:
        if not is_mc:
            raise ValueError("mc_option_logprobs were provided for a non-MC forgetting probe")
        mc_result = score_mc_association(example, mc_option_logprobs)
        row["mc_association"] = mc_result.as_dict()
        row["mc_association_correct"] = mc_result.correct
        row["mc_label_protected_vs_decoy_margin"] = mc_result.protected_vs_decoy_margin
        if row["protected_vs_decoy_margin"] is None:
            row["protected_vs_decoy_margin"] = mc_result.protected_vs_decoy_margin
    return row


def evaluate_forgetting_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    generation_hook: Callable[[str], str] | None = None,
    target_logprob_hook: Callable[[str, str], float] | None = None,
    decoy_logprob_hook: Callable[[str, str], float] | None = None,
    mc_option_logprob_hook: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
    protected_fields_by_record: Mapping[str, Mapping[str, str]] | None = None,
    decoy_values_by_record: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate probes using injected hooks, so unit tests require no model.

    MC hooks receive the rendered MC prompt and answer labels (``A/B/C/D``),
    not full option wording.  The option order and gold answer are still read
    from the stored v2 fields by :func:`score_mc_association`.
    """

    rows: list[dict[str, Any]] = []
    for example in examples:
        _example_id, record_id, _template_id, prompt, target = _identity_from_example(example)
        generated_text = generation_hook(prompt) if generation_hook is not None else None
        if generation_hook is not None and not isinstance(generated_text, str):
            raise TypeError("generation_hook must return a string")
        target_logprob = None
        if target_logprob_hook is not None:
            target_logprob = _finite_float(
                target_logprob_hook(prompt, target),
                "target_logprob_hook result",
            )
        is_mc = "options" in example or "answer_index" in example
        decoy_logprobs = None
        if not is_mc and decoy_logprob_hook is not None and decoy_values_by_record is not None:
            target_field = target_field_for_example(example)
            record_decoys = decoy_values_by_record.get(record_id)
            if record_decoys is None or target_field not in record_decoys:
                raise ValueError(
                    f"No same-field decoys supplied for {record_id!r} field {target_field!r}"
                )
            decoys = record_decoys[target_field]
            if isinstance(decoys, (str, bytes)) or not decoys:
                raise ValueError("Each likelihood-margin probe requires at least one decoy value")
            decoy_logprobs = tuple(
                _finite_float(decoy_logprob_hook(prompt, decoy), "decoy_logprob_hook result")
                for decoy in decoys
            )
        mc_option_logprobs = None
        if is_mc:
            if mc_option_logprob_hook is not None:
                mc_prompt, labels, _answer_index = mc_association_inputs(example)
                mc_option_logprobs = mc_option_logprob_hook(mc_prompt, labels)
                if mc_option_logprobs is None:
                    raise TypeError("mc_option_logprob_hook must return one score per option")
        protected_fields = None
        if protected_fields_by_record is not None:
            protected_fields = protected_fields_by_record.get(record_id)
        rows.append(
            score_forgetting_example(
                example,
                generated_text=generated_text,
                target_logprob=target_logprob,
                decoy_logprobs=decoy_logprobs,
                mc_option_logprobs=mc_option_logprobs,
                protected_fields=protected_fields,
            )
        )
        if progress is not None:
            progress(len(rows))
    return rows


def evaluate_forgetting_with_model(
    model,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    device: str,
    *,
    max_new_tokens: int = 64,
    target_logprob_normalize_by_token_count: bool = True,
    protected_fields_by_record: Mapping[str, Mapping[str, str]] | None = None,
    decoy_values_by_record: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Model-backed convenience wrapper around the hook-based evaluator."""

    return evaluate_forgetting_examples(
        examples,
        generation_hook=make_model_generation_hook(
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
        ),
        target_logprob_hook=make_model_target_logprob_hook(
            model,
            tokenizer,
            device,
            normalize_by_token_count=target_logprob_normalize_by_token_count,
        ),
        decoy_logprob_hook=make_model_target_logprob_hook(
            model,
            tokenizer,
            device,
            normalize_by_token_count=target_logprob_normalize_by_token_count,
        ),
        mc_option_logprob_hook=make_model_mc_option_logprob_hook(model, tokenizer, device),
        protected_fields_by_record=protected_fields_by_record,
        decoy_values_by_record=decoy_values_by_record,
        progress=progress,
    )


def group_forgetting_rows_by_record(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Group result rows by the independent synthetic-patient record unit."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        record_id = _require_text(row.get("record_id"), "result record_id")
        grouped[record_id].append(row)
    return {record_id: grouped[record_id] for record_id in sorted(grouped)}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _optional_numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(_finite_float(value, key))
    return values


def _record_summary(record_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    leakage_values = _optional_numeric_values(rows, "exact_field_leakage")
    f1_values = _optional_numeric_values(rows, "normalized_field_f1")
    target_logprob_values = _optional_numeric_values(rows, "target_logprob")
    target_margin_values = _optional_numeric_values(rows, "target_vs_decoy_margin")
    mc_margin_values = _optional_numeric_values(rows, "mc_label_protected_vs_decoy_margin")
    mc_correct_values = _optional_numeric_values(rows, "mc_association_correct")
    field_leakage_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        raw_field_leakage = row.get("field_leakage") or {}
        if not isinstance(raw_field_leakage, Mapping):
            raise TypeError("field_leakage must be a mapping when present")
        for field_name, leaked in raw_field_leakage.items():
            if not isinstance(leaked, bool):
                raise TypeError("field_leakage values must be booleans")
            field_leakage_values[str(field_name)].append(float(leaked))
    return {
        "record_id": record_id,
        "example_count": len(rows),
        "template_ids": sorted(
            {str(row.get("template_id")) for row in rows if row.get("template_id") is not None}
        ),
        "exact_leakage_rate": _mean(leakage_values),
        "any_exact_field_leakage": bool(any(leakage_values)) if leakage_values else None,
        "normalized_field_f1": _mean(f1_values),
        "target_logprob": _mean(target_logprob_values),
        "target_vs_decoy_margin": _mean(target_margin_values),
        "mc_label_protected_vs_decoy_margin": _mean(mc_margin_values),
        "mc_association_accuracy": _mean(mc_correct_values),
        "exact_field_leakage_by_field": {
            field_name: {
                "leakage_rate": _mean(values),
                "any_leakage": bool(any(values)),
                "scored_probe_count": len(values),
            }
            for field_name, values in sorted(field_leakage_values.items())
        },
        "scored_generation_count": len(leakage_values),
        "scored_target_logprob_count": len(target_logprob_values),
        "scored_mc_association_count": len(mc_correct_values),
    }


def _summary_from_record_summaries(
    record_summaries: Sequence[Mapping[str, Any]],
    *,
    example_count: int,
) -> dict[str, Any]:
    exact_rates = _optional_numeric_values(record_summaries, "exact_leakage_rate")
    any_leakages = _optional_numeric_values(record_summaries, "any_exact_field_leakage")
    f1_values = _optional_numeric_values(record_summaries, "normalized_field_f1")
    target_logprobs = _optional_numeric_values(record_summaries, "target_logprob")
    target_margins = _optional_numeric_values(record_summaries, "target_vs_decoy_margin")
    mc_margins = _optional_numeric_values(
        record_summaries, "mc_label_protected_vs_decoy_margin"
    )
    mc_accuracies = _optional_numeric_values(record_summaries, "mc_association_accuracy")
    field_leakage_by_field: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"rates": [], "any": []}
    )
    for record_summary in record_summaries:
        raw_fields = record_summary.get("exact_field_leakage_by_field") or {}
        if not isinstance(raw_fields, Mapping):
            raise TypeError("record exact_field_leakage_by_field must be a mapping")
        for field_name, field_summary in raw_fields.items():
            if not isinstance(field_summary, Mapping):
                raise TypeError("field leakage summary must be a mapping")
            rate = field_summary.get("leakage_rate")
            any_leakage = field_summary.get("any_leakage")
            if rate is not None:
                field_leakage_by_field[str(field_name)]["rates"].append(
                    _finite_float(rate, "field leakage rate")
                )
            if any_leakage is not None:
                field_leakage_by_field[str(field_name)]["any"].append(
                    _finite_float(any_leakage, "field any leakage")
                )
    summary = {
        "unit_of_analysis": "record",
        "record_count": len(record_summaries),
        "example_count": example_count,
        # The primary scalar is an equal-record-weighted mean of each record's
        # within-record leakage rate, rather than a template-weighted total.
        "exact_leakage_rate": _mean(exact_rates),
        "any_exact_field_leakage_rate": _mean(any_leakages),
        "normalized_field_f1": _mean(f1_values),
        "target_logprob": _mean(target_logprobs),
        "target_vs_decoy_margin": _mean(target_margins),
        "mc_label_protected_vs_decoy_margin": _mean(mc_margins),
        "mc_association_accuracy": _mean(mc_accuracies),
        "records_with_generation": len(exact_rates),
        "records_with_target_logprob": len(target_logprobs),
        "records_with_mc_association": len(mc_accuracies),
    }
    summary["exact_field_leakage_by_field"] = {
        field_name: {
            "record_mean_probe_leakage_rate": _mean(values["rates"]),
            "record_any_leakage_rate": _mean(values["any"]),
            "record_count": len(values["rates"]),
        }
        for field_name, values in sorted(field_leakage_by_field.items())
    }
    return summary


def summarize_forgetting_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize forgetting metrics at record level and by held-out template.

    ``records`` is retained in the output for paired, cluster-aware downstream
    analysis against contaminated and clean counterfactual checkpoints.  The
    function is descriptive only; confidence intervals belong in a paired
    multi-run analysis rather than this single-model scorer.
    """

    grouped = group_forgetting_rows_by_record(rows)
    record_summaries = [
        _record_summary(record_id, record_rows) for record_id, record_rows in grouped.items()
    ]
    summary = _summary_from_record_summaries(record_summaries, example_count=len(rows))
    template_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        template_id = _require_text(row.get("template_id"), "result template_id")
        template_groups[template_id].append(row)
    summary["by_template"] = {}
    for template_id in sorted(template_groups):
        template_rows = template_groups[template_id]
        template_records = group_forgetting_rows_by_record(template_rows)
        template_record_summaries = [
            _record_summary(record_id, record_rows)
            for record_id, record_rows in template_records.items()
        ]
        summary["by_template"][template_id] = _summary_from_record_summaries(
            template_record_summaries,
            example_count=len(template_rows),
        )
    summary["records"] = record_summaries
    return summary


# A short alias makes the record-level intent convenient at call sites.
summarize_forgetting_by_record = summarize_forgetting_results

