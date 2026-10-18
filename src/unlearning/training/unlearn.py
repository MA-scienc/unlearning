"""Fixed-trajectory GA and frozen-reference NPO for the pilot.

This module deliberately implements only the two forget-only baselines fixed in
``docs/PILOT_DESIGN.md``.  In particular, it does not load a clean ``S_i``
checkpoint as an NPO reference, add a retain loss, choose a post-hoc stopping
point, or alter the fixed F dataset.

The import surface is intentionally CPU-safe: PyTorch, Transformers, and
bitsandbytes are imported only when an actual training run is requested.  The
schedule, lineage, input validation, completion mask construction, and scalar
NPO definition can therefore be audited and unit-tested without a GPU stack.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

from unlearning.config import PilotConfig, load_config, validate_training_ready
from unlearning.experiments.naming import experiment_run_id, stage_run_name
from unlearning.training.contaminate import CONTAMINATION_MANIFEST_SCHEMA
from unlearning.utils.env import inspect_hardware
from unlearning.utils.hashing import fingerprint_directory, fingerprint_payload, hash_file
from unlearning.utils.io import ensure_dir, read_json, read_jsonl, write_json
from unlearning.utils.manifest import build_manifest


IGNORE_INDEX = -100
PILOT_SNAPSHOT_STEPS = (0, 50, 100, 200, 400)
_CONTAMINATED_NAME_RE = re.compile(r"^C(0|50|100)$")
UNLEARNING_SNAPSHOT_MANIFEST_SCHEMA = "unlearning-snapshot-v2"


@dataclass(frozen=True)
class CompletionOnlySequence:
    """One prompt/completion example with labels masked outside the completion."""

    example_id: str
    record_id: str
    template_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_token_count: int
    completion_token_count: int


@dataclass(frozen=True)
class ContaminatedSource:
    """Verified provenance for the C_i checkpoint from which a run starts."""

    checkpoint_dir: Path
    manifest_path: Path
    manifest_sha256: str
    checkpoint_id: str
    name: str
    global_step: int
    parent_clean_checkpoint_id: str
    source_run_id: str | None
    manifest: dict[str, Any]

    @property
    def specialization_fraction(self) -> float:
        return {"C0": 0.0, "C50": 0.5, "C100": 1.0}[self.name]


@dataclass(frozen=True)
class UnlearningTrajectoryPlan:
    """The immutable, pre-specified trajectory for one C_i and one method."""

    experiment_run_id: str
    trajectory_run_id: str
    method: str
    source_checkpoint_id: str
    source_checkpoint_name: str
    snapshot_steps: tuple[int, ...]
    training_steps: int
    effective_batch_size: int
    example_count: int
    forget_examples_fingerprint: str
    canonical_example_order_fingerprint: str
    batch_schedule_fingerprint: str


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor loss computation or training.") from exc
    return torch


def _require_training_stack():
    torch = _require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers is required for GA/NPO unlearning training.") from exc
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("The frozen pilot requires bitsandbytes AdamW8bit for training.") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, bnb.optim.AdamW8bit


def validate_snapshot_schedule(snapshot_steps: Sequence[int], training_steps: int) -> tuple[int, ...]:
    """Validate the pilot's pre-registered trajectory without importing PyTorch."""

    normalized = tuple(int(step) for step in snapshot_steps)
    if normalized != PILOT_SNAPSHOT_STEPS:
        raise ValueError(
            "The pilot trajectory is fixed at snapshot steps "
            f"{list(PILOT_SNAPSHOT_STEPS)}; got {list(normalized)}."
        )
    if training_steps != PILOT_SNAPSHOT_STEPS[-1]:
        raise ValueError(
            f"The pilot trajectory must end at {PILOT_SNAPSHOT_STEPS[-1]} steps; "
            f"got {training_steps}."
        )
    return normalized


def snapshot_schedule_for_method(config: PilotConfig, method: str) -> tuple[int, ...]:
    """Return the fixed schedule after checking method and retain-loss policy."""

    _method_config(config, method)
    if config.unlearning.retain_loss_weight != 0.0:
        raise ValueError("The pilot GA/NPO stage must use retain_loss_weight == 0.0")
    return validate_snapshot_schedule(
        config.unlearning.snapshot_steps,
        config.unlearning.methods[method].steps,
    )


def deterministic_batch_indices(
    *,
    step: int,
    example_count: int,
    batch_size: int,
) -> tuple[int, ...]:
    """Return the 1-indexed, wrap-around batch used at an optimizer step.

    The canonical F examples are sorted by ``example_id`` once.  This simple
    sequential rule makes every GA and NPO run over a given C_i consume exactly
    the same data at every step; no implicit data-loader shuffle or epoch
    boundary changes the trajectory.
    """

    if step <= 0:
        raise ValueError("step is 1-indexed and must be positive")
    if example_count <= 0:
        raise ValueError("example_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = (step - 1) * batch_size
    return tuple((start + offset) % example_count for offset in range(batch_size))


def deterministic_batch_schedule(
    *,
    example_ids: Sequence[str],
    training_steps: int,
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    """Materialize the small fixed pilot's batch lineage for manifest hashing."""

    if len(example_ids) != len(set(example_ids)):
        raise ValueError("example_ids must be unique for deterministic batch lineage")
    if training_steps <= 0:
        raise ValueError("training_steps must be positive")
    return tuple(
        tuple(example_ids[index] for index in deterministic_batch_indices(
            step=step,
            example_count=len(example_ids),
            batch_size=batch_size,
        ))
        for step in range(1, training_steps + 1)
    )


def completion_only_labels(
    input_ids: Sequence[int],
    offset_mapping: Sequence[Sequence[int]],
    *,
    prompt_char_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[int, ...]:
    """Mask every token except one wholly in the textual completion.

    This is deliberately the same conservative fast-tokenizer offset rule as
    the contamination stage.  A BPE token that crosses the prompt/completion
    character boundary is masked, rather than counted as a completion token
    even though part of it contains prompt text.
    """

    ids = tuple(int(token) for token in input_ids)
    if not ids:
        raise ValueError("input_ids must not be empty")
    if prompt_char_length < 0:
        raise ValueError("prompt_char_length must be non-negative")
    if len(ids) != len(offset_mapping):
        raise ValueError("input_ids and offset_mapping must have the same length")

    labels: list[int] = []
    for index, (token_id, offset) in enumerate(zip(ids, offset_mapping, strict=True)):
        if len(offset) != 2:
            raise ValueError(f"Offset mapping at token {index} must be a (start, end) pair")
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start:
            raise ValueError(f"Offset mapping at token {index} is invalid: ({start}, {end})")
        labels.append(token_id if end > start and start >= prompt_char_length else ignore_index)
    if not any(label != ignore_index for label in labels):
        raise ValueError("Completion-only labels contain no target tokens")
    return tuple(labels)


def validate_forget_examples(
    examples: Iterable[dict[str, Any]],
    *,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate and canonically order the fixed contamination prompt/completions.

    The original JSONL order is not treated as a scientific randomization.  A
    canonical ``example_id`` sort is used so that an accidental serialization
    order change cannot alter the GA/NPO data sequence.
    """

    rows = list(examples)
    if not rows:
        raise ValueError("Forget examples must not be empty")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} forget examples, found {len(rows)}")

    required = ("example_id", "record_id", "template_id", "split", "prompt", "completion")
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each forget example must be a JSON object")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Forget example is missing required fields: {', '.join(missing)}")
        example_id = row["example_id"]
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("Forget example_id must be a non-empty string")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate forget example_id: {example_id}")
        seen_ids.add(example_id)
        if row["split"] != "contamination":
            raise ValueError("Unlearning may use only contamination examples from F")
        for field in ("record_id", "template_id", "prompt", "completion"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"Forget example field {field!r} must be a non-empty string")
        if "text" in row and row["text"] != f"{row['prompt']}{row['completion']}":
            raise ValueError("Forget example text must equal prompt + completion")

    return sorted(rows, key=lambda row: row["example_id"])


def _validate_protocol_forget_examples(
    examples: list[dict[str, Any]], config: PilotConfig
) -> list[dict[str, Any]]:
    canonical = validate_forget_examples(
        examples,
        expected_count=(
            config.forget.record_count * config.forget.contamination_templates_per_record
        ),
    )
    record_ids = {row["record_id"] for row in canonical}
    if len(record_ids) != config.forget.record_count:
        raise ValueError(
            "F contamination examples do not contain the configured number of distinct records"
        )
    counts_by_record: dict[str, int] = {}
    for row in canonical:
        counts_by_record[row["record_id"]] = counts_by_record.get(row["record_id"], 0) + 1
    expected_templates = config.forget.contamination_templates_per_record
    if any(count != expected_templates for count in counts_by_record.values()):
        raise ValueError("Each F record must have exactly the configured contamination templates")
    return canonical


def npo_scalar_loss_from_log_ratio(log_ratio: float, beta: float) -> float:
    """Evaluate the original NPO loss for one response without PyTorch.

    With ``r = log pi_theta(y|x) - log pi_ref(y|x)``, the documented objective
    is ``-(2/beta) log sigmoid(-beta*r)``.  The algebraically identical
    ``(2/beta) softplus(beta*r)`` form below is numerically stable.  Its
    derivative with respect to ``r`` is ``2 * sigmoid(beta*r) > 0``; therefore
    gradient *descent* decreases the current response likelihood relative to
    the frozen reference, as required for negative preference optimization.
    """

    if not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("NPO beta must be a positive finite number")
    if not isinstance(log_ratio, (int, float)) or not math.isfinite(log_ratio):
        raise ValueError("NPO log_ratio must be finite")
    scaled = float(beta) * float(log_ratio)
    softplus = scaled + math.log1p(math.exp(-scaled)) if scaled > 0 else math.log1p(math.exp(scaled))
    return (2.0 / float(beta)) * softplus


def npo_scalar_gradient_wrt_log_ratio(log_ratio: float, beta: float) -> float:
    """Analytic derivative of :func:`npo_scalar_loss_from_log_ratio`."""

    if not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("NPO beta must be a positive finite number")
    if not isinstance(log_ratio, (int, float)) or not math.isfinite(log_ratio):
        raise ValueError("NPO log_ratio must be finite")
    scaled = float(beta) * float(log_ratio)
    if scaled >= 0:
        return 2.0 / (1.0 + math.exp(-scaled))
    exp_scaled = math.exp(scaled)
    return 2.0 * exp_scaled / (1.0 + exp_scaled)


def completion_logprob_sums(logits, labels, *, ignore_index: int = IGNORE_INDEX):
    """Return per-example log pi(y|x) sums over unmasked completion tokens.

    NPO must use the response likelihood ``log pi(y|x)`` in its stated
    objective, i.e. the sum of causal token log probabilities for each
    completion.  Prompt and pad labels are ignored.  Every example must retain
    at least one target token after the causal one-token shift.
    """

    torch = _require_torch()
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("logits and labels batch/sequence dimensions must match")
    if logits.shape[1] < 2:
        raise ValueError("Causal-LM sequences must contain at least two tokens")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(ignore_index)
    counts = valid.sum(dim=1)
    if bool(torch.any(counts.eq(0)).item()):
        raise ValueError("Every completion-only batch row must contain a target token")
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logprobs = torch.log_softmax(shifted_logits.float(), dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_logprobs = token_logprobs * valid.to(token_logprobs.dtype)
    return token_logprobs.sum(dim=1), counts


def ga_completion_only_loss(logits, labels, *, ignore_index: int = IGNORE_INDEX):
    """Return ``-CE`` on completion tokens only, the pilot's GA objective.

    ``CE`` is the ordinary token-mean causal cross entropy over unmasked
    completion labels.  Minimizing this returned loss is gradient ascent on
    completion CE and lowers the protected-completion likelihood.
    """

    logprob_sums, token_counts = completion_logprob_sums(
        logits,
        labels,
        ignore_index=ignore_index,
    )
    # CE is ``-mean(log pi_theta(y|x))``.  GA minimizes ``-CE``, which is
    # exactly the (negative) mean completion log probability below.  Returning
    # CE here would silently perform ordinary fine-tuning and *increase* the
    # probability of the forget completions.
    return logprob_sums.sum() / token_counts.sum()


def npo_loss_from_logprob_sums(
    current_logprob_sums,
    reference_logprob_sums,
    *,
    beta: float,
):
    """Return original negative-only NPO over a batch of response log-probs.

    This is exactly

    ``-(2 / beta) * mean(logsigmoid(-beta * (log pi_theta - log pi_ref)))``.

    There is no chosen response, refusal target, retain term, or clean
    counterfactual in this objective.  ``reference_logprob_sums`` is detached
    defensively even when it originated from a frozen no-grad C_i pass.
    """

    torch = _require_torch()
    if beta <= 0:
        raise ValueError("NPO beta must be positive")
    current = current_logprob_sums.reshape(-1)
    reference = torch.as_tensor(
        reference_logprob_sums,
        dtype=current.dtype,
        device=current.device,
    ).reshape(-1).detach()
    if current.numel() == 0:
        raise ValueError("NPO requires at least one completion log-probability")
    if current.shape != reference.shape:
        raise ValueError("Current and reference completion log-probabilities must align")
    return -(2.0 / beta) * torch.nn.functional.logsigmoid(
        -beta * (current - reference)
    ).mean()


def npo_completion_only_loss(
    current_logits,
    labels,
    reference_logprob_sums,
    *,
    beta: float,
    ignore_index: int = IGNORE_INDEX,
):
    """Compute frozen-reference NPO using completion-only current log-probs."""

    current_logprob_sums, _token_counts = completion_logprob_sums(
        current_logits,
        labels,
        ignore_index=ignore_index,
    )
    return npo_loss_from_logprob_sums(
        current_logprob_sums,
        reference_logprob_sums,
        beta=beta,
    )


def build_completion_only_sequences(
    examples: Sequence[dict[str, Any]],
    *,
    tokenizer,
    sequence_length: int,
) -> list[CompletionOnlySequence]:
    """Tokenize F with the same completion-boundary semantics as contamination."""

    if sequence_length <= 1:
        raise ValueError("sequence_length must be at least two tokens")
    sequences: list[CompletionOnlySequence] = []
    for example in examples:
        prompt = example["prompt"]
        completion = example["completion"]
        text = example.get("text")
        if text != f"{prompt}{completion}":
            raise ValueError(f"F example text must equal prompt + completion: {example['example_id']}")
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=sequence_length,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=True,
            )
            raw_input_ids = list(encoded["input_ids"])
            offset_mapping = tuple(encoded["offset_mapping"])
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise ValueError(
                "Completion-only unlearning requires a fast tokenizer with offset mappings"
            ) from exc
        if raw_input_ids and isinstance(raw_input_ids[0], (list, tuple)):
            raise ValueError("Tokenizer returned a batch for one F completion example")
        full_ids = tuple(int(token) for token in raw_input_ids)
        if not full_ids:
            raise ValueError(f"Tokenization produced no tokens for {example['example_id']}")
        labels = completion_only_labels(
            full_ids,
            offset_mapping,
            prompt_char_length=len(prompt),
        )
        completion_count = sum(label != IGNORE_INDEX for label in labels)
        prompt_token_count = len(full_ids) - completion_count
        sequences.append(
            CompletionOnlySequence(
                example_id=example["example_id"],
                record_id=example["record_id"],
                template_id=example["template_id"],
                input_ids=full_ids,
                labels=labels,
                prompt_token_count=prompt_token_count,
                completion_token_count=completion_count,
            )
        )
    return sequences


def collate_completion_only_batch(
    sequences: Sequence[CompletionOnlySequence],
    *,
    pad_token_id: int,
):
    """Build padded causal-LM tensors without importing torch at module import."""

    torch = _require_torch()
    if not sequences:
        raise ValueError("Cannot collate an empty completion-only batch")
    max_length = max(len(sequence.input_ids) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
    labels = torch.full(
        (len(sequences), max_length),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )
    for row, sequence in enumerate(sequences):
        length = len(sequence.input_ids)
        input_ids[row, :length] = torch.tensor(sequence.input_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(sequence.labels, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_ids": tuple(sequence.example_id for sequence in sequences),
    }


def load_contaminated_source(
    checkpoint_dir: str | Path,
    *,
    config: PilotConfig | None = None,
) -> ContaminatedSource:
    """Load and validate the C_i provenance required by both GA and NPO."""

    root = Path(checkpoint_dir)
    manifest_path = root / "manifest.json"
    if not root.is_dir():
        raise FileNotFoundError(f"Contaminated checkpoint directory does not exist: {root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Contaminated checkpoint manifest does not exist: {manifest_path}")
    for required_dir in (root / "model", root / "tokenizer"):
        if not required_dir.is_dir():
            raise FileNotFoundError(f"Contaminated checkpoint is missing required directory: {required_dir}")

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Contaminated checkpoint manifest must be a JSON object")
    artifact_manifest = manifest.get("manifest")
    if not isinstance(artifact_manifest, dict):
        raise ValueError("Contaminated checkpoint manifest lacks immutable manifest metadata")
    if artifact_manifest.get("artifact_type") != "contaminated_checkpoint":
        raise ValueError("Unlearning input must be a contamination-stage checkpoint manifest")
    if artifact_manifest.get("schema_version") != CONTAMINATION_MANIFEST_SCHEMA:
        raise ValueError("Unsupported contaminated checkpoint manifest schema")
    expected_manifest_fingerprint = fingerprint_payload(
        {key: value for key, value in manifest.items() if key != "manifest"}
    )
    if artifact_manifest.get("payload_fingerprint") != expected_manifest_fingerprint:
        raise ValueError("Contaminated checkpoint manifest payload fingerprint does not match")
    required = (
        "checkpoint_id",
        "name",
        "global_step",
        "parent_clean_checkpoint_id",
        "forget_schema_version",
        "forget_examples_file_sha256",
        "forget_examples_fingerprint",
        "masking",
        "artifact_fingerprints",
        "base_model",
        "run_id",
        "contamination",
        "exposure_plan",
        "parent_clean_checkpoint_name",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(
            "Contaminated checkpoint manifest is missing required provenance fields: "
            f"{', '.join(missing)}"
        )
    name = manifest["name"]
    if not isinstance(name, str) or not _CONTAMINATED_NAME_RE.fullmatch(name):
        raise ValueError("Unlearning must start from a named C0, C50, or C100 contaminated checkpoint")
    if not isinstance(manifest["checkpoint_id"], str) or not manifest["checkpoint_id"]:
        raise ValueError("Contaminated checkpoint_id must be a non-empty string")
    if not isinstance(manifest["parent_clean_checkpoint_id"], str) or not manifest[
        "parent_clean_checkpoint_id"
    ]:
        raise ValueError("Contaminated checkpoint must record its clean S_i parent checkpoint ID")
    if not isinstance(manifest["global_step"], int) or manifest["global_step"] < 0:
        raise ValueError("Contaminated checkpoint global_step must be a non-negative integer")
    if config is not None and manifest["forget_schema_version"] != config.forget.schema_version:
        raise ValueError("Contaminated checkpoint F schema does not match the frozen configuration")
    if config is not None:
        expected_base_model = {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "tokenizer_id": config.model.tokenizer_id,
        }
        if manifest["base_model"] != expected_base_model:
            raise ValueError("Contaminated checkpoint base-model identity does not match config")
        if manifest["run_id"] != experiment_run_id(config):
            raise ValueError(
                "Contaminated checkpoint run_id does not match the complete scientific configuration"
            )
        expected_clean_name = f"S{name[1:]}"
        if manifest["parent_clean_checkpoint_name"] != expected_clean_name:
            raise ValueError(
                "Contaminated checkpoint name does not match its checkpoint-specific clean parent"
            )
        contamination = manifest["contamination"]
        if not isinstance(contamination, dict):
            raise ValueError("Contaminated checkpoint contamination settings must be an object")
        expected_contamination = asdict(config.contamination)
        observed_contamination = {
            field: contamination.get(field) for field in expected_contamination
        }
        if observed_contamination != expected_contamination:
            raise ValueError("Contaminated checkpoint recipe does not match the frozen config")
        if contamination.get("optimizer_weight_decay") != config.contamination.weight_decay:
            raise ValueError("Contaminated checkpoint weight decay does not match the frozen config")
        exposure_plan = manifest["exposure_plan"]
        if not isinstance(exposure_plan, dict):
            raise ValueError("Contaminated checkpoint exposure plan must be an object")
        expected_exposure = {
            "example_count": config.forget.record_count * config.forget.contamination_templates_per_record,
            "epochs": config.contamination.epochs,
            "effective_batch_size": config.contamination.effective_batch_size,
            "optimizer_steps": 600,
            "total_example_exposures": 9600,
        }
        if {field: exposure_plan.get(field) for field in expected_exposure} != expected_exposure:
            raise ValueError("Contaminated checkpoint exposure plan does not match the frozen config")
        if manifest["global_step"] != exposure_plan["optimizer_steps"]:
            raise ValueError("Contaminated checkpoint global_step does not match its exposure plan")
    masking = manifest["masking"]
    if not isinstance(masking, dict):
        raise ValueError("Contaminated checkpoint masking metadata must be an object")
    if masking.get("objective") != "causal_lm_completion_only":
        raise ValueError("Contaminated checkpoint was not trained with completion-only masking")
    if masking.get("add_special_tokens") is not False:
        raise ValueError("Contaminated checkpoint masking must record add_special_tokens=false")

    artifact_fingerprints = manifest["artifact_fingerprints"]
    if not isinstance(artifact_fingerprints, dict):
        raise ValueError("Contaminated checkpoint artifact fingerprints must be an object")
    training_state_path = root / "training_state.pt"
    if not training_state_path.is_file():
        raise FileNotFoundError("Contaminated checkpoint is missing training_state.pt")
    observed_artifacts = {
        "model_directory": fingerprint_directory(root / "model"),
        "tokenizer_directory": fingerprint_directory(root / "tokenizer"),
        "training_state_file": hash_file(training_state_path),
    }
    expected_artifacts = {key: artifact_fingerprints.get(key) for key in observed_artifacts}
    if expected_artifacts != observed_artifacts:
        raise ValueError(
            "Contaminated checkpoint local artifact fingerprints do not match its immutable manifest"
        )

    return ContaminatedSource(
        checkpoint_dir=root,
        manifest_path=manifest_path,
        manifest_sha256=hash_file(manifest_path),
        checkpoint_id=manifest["checkpoint_id"],
        name=name,
        global_step=manifest["global_step"],
        parent_clean_checkpoint_id=manifest["parent_clean_checkpoint_id"],
        source_run_id=manifest.get("run_id"),
        manifest=manifest,
    )


def _resolve_forget_examples_path(forget_examples_path: str | Path) -> Path:
    path = Path(forget_examples_path)
    if path.is_dir():
        path = path / "contamination_examples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Forget contamination examples do not exist: {path}")
    return path


def _load_verified_forget_examples(
    *,
    forget_examples_path: str | Path,
    source: ContaminatedSource,
    config: PilotConfig,
) -> tuple[Path, list[dict[str, Any]], str, str]:
    path = _resolve_forget_examples_path(forget_examples_path)
    file_sha256 = hash_file(path)
    if file_sha256 != source.manifest["forget_examples_file_sha256"]:
        raise ValueError(
            "The supplied F contamination JSONL does not match the F JSONL used to create "
            f"{source.name}; refusing to change the forget set between C_i and U_i"
        )
    raw_examples = read_jsonl(path)
    raw_fingerprint = fingerprint_payload(raw_examples)
    if raw_fingerprint != source.manifest["forget_examples_fingerprint"]:
        raise ValueError(
            "The supplied F contamination examples do not match the fingerprint recorded in "
            f"{source.name}; refusing to continue"
        )
    examples = _validate_protocol_forget_examples(raw_examples, config)
    return path, examples, file_sha256, raw_fingerprint


def _method_config(config: PilotConfig, method: str):
    if method not in {"ga", "npo"}:
        raise ValueError("Pilot unlearning method must be exactly 'ga' or 'npo'")
    if method not in config.unlearning.methods:
        raise ValueError(f"Method {method!r} is absent from the frozen pilot configuration")
    return config.unlearning.methods[method]


def build_unlearning_trajectory_plan(
    *,
    config: PilotConfig,
    source: ContaminatedSource,
    method: str,
    examples: Sequence[dict[str, Any]],
    forget_examples_fingerprint: str | None = None,
) -> UnlearningTrajectoryPlan:
    """Create the manifestable immutable plan before model weights are loaded."""

    method_config = _method_config(config, method)
    snapshots = snapshot_schedule_for_method(config, method)
    canonical = validate_forget_examples(examples)
    example_ids = tuple(example["example_id"] for example in canonical)
    schedule = deterministic_batch_schedule(
        example_ids=example_ids,
        training_steps=method_config.steps,
        batch_size=method_config.effective_batch_size,
    )
    experiment_id = experiment_run_id(config)
    trajectory_id = stage_run_name(
        config,
        "unlearned",
        checkpoint_fraction=source.specialization_fraction,
        method=method,
        condition=config.contamination.condition,
    )
    return UnlearningTrajectoryPlan(
        experiment_run_id=experiment_id,
        trajectory_run_id=trajectory_id,
        method=method,
        source_checkpoint_id=source.checkpoint_id,
        source_checkpoint_name=source.name,
        snapshot_steps=snapshots,
        training_steps=method_config.steps,
        effective_batch_size=method_config.effective_batch_size,
        example_count=len(canonical),
        forget_examples_fingerprint=(
            forget_examples_fingerprint
            if forget_examples_fingerprint is not None
            else fingerprint_payload(canonical)
        ),
        canonical_example_order_fingerprint=fingerprint_payload(canonical),
        batch_schedule_fingerprint=fingerprint_payload(schedule),
    )


def _plan_payload(
    *,
    plan: UnlearningTrajectoryPlan,
    config: PilotConfig,
    source: ContaminatedSource,
    forget_examples_path: Path,
    forget_file_sha256: str,
) -> dict[str, Any]:
    method_config = _method_config(config, plan.method)
    return {
        "trajectory_run_id": plan.trajectory_run_id,
        "experiment_run_id": plan.experiment_run_id,
        "method": plan.method,
        "source_contaminated_checkpoint": {
            "checkpoint_id": source.checkpoint_id,
            "name": source.name,
            "global_step": source.global_step,
            "manifest_path": str(source.manifest_path),
            "manifest_sha256": source.manifest_sha256,
            "source_run_id": source.source_run_id,
            "clean_counterfactual_checkpoint_id": source.parent_clean_checkpoint_id,
            "clean_counterfactual_used_as_reference": False,
        },
        "forget_dataset": {
            "schema_version": config.forget.schema_version,
            "contamination_examples_path": str(forget_examples_path),
            "contamination_examples_file_sha256": forget_file_sha256,
            "contamination_examples_fingerprint": plan.forget_examples_fingerprint,
            "canonical_example_order_fingerprint": plan.canonical_example_order_fingerprint,
            "record_count": config.forget.record_count,
            "examples_per_record": config.forget.contamination_templates_per_record,
        },
        "trajectory": {
            "snapshot_steps": list(plan.snapshot_steps),
            "training_steps": plan.training_steps,
            "effective_batch_size": plan.effective_batch_size,
            "canonical_example_order": "ascending example_id",
            "batch_rule": "1-indexed sequential wrap-around over canonical F examples",
            "batch_schedule_fingerprint": plan.batch_schedule_fingerprint,
            "posthoc_checkpoint_selection": "none; fixed trajectory is primary analysis",
        },
        "method_config": asdict(method_config),
        "retain_loss_weight": config.unlearning.retain_loss_weight,
        "optimizer": {
            "family": config.model.optimizer,
            "implementation": "bitsandbytes.optim.AdamW8bit",
            "learning_rate": method_config.learning_rate,
            "weight_decay": method_config.weight_decay,
            "scheduler": method_config.scheduler,
            "gradient_clip_norm": method_config.gradient_clip_norm,
        },
        "completion_loss_masking": {
            "prompt_labels": IGNORE_INDEX,
            "padding_labels": IGNORE_INDEX,
            "completion_only": True,
            "ga": "minimize negative token-mean completion cross entropy",
            "npo": (
                "-(2/beta) * mean(logsigmoid(-beta * "
                "(log_pi_theta_completion - log_pi_ref_completion)))"
            ),
        },
        "npo_reference": (
            {
                "type": "frozen_contaminated_checkpoint",
                "checkpoint_id": source.checkpoint_id,
                "checkpoint_manifest_sha256": source.manifest_sha256,
                "clean_counterfactual_used_as_reference": False,
                "cache": "reference_completion_logprobs.json",
            }
            if plan.method == "npo"
            else {
                "type": "not_applicable_to_ga",
                "clean_counterfactual_used_as_reference": False,
            }
        ),
        "seed": config.seed,
        "base_model": source.manifest.get(
            "base_model",
            {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "tokenizer_id": config.model.tokenizer_id,
            },
        ),
    }


def precheck_unlearning_environment() -> dict[str, Any]:
    """Report whether the frozen full-parameter CUDA protocol can be attempted."""

    hardware = inspect_hardware()
    torch_info = hardware.get("torch", {})
    bnb_info = hardware.get("bitsandbytes", {})
    compatible = (
        bool(torch_info.get("available"))
        and bool(torch_info.get("cuda_available"))
        and bool(bnb_info.get("available"))
    )
    devices = torch_info.get("devices") or []
    if devices:
        compatible = compatible and bool(devices[0].get("bf16_supported")) and int(
            devices[0].get("total_memory_mb", 0)
        ) >= 24_000
    else:
        compatible = False
    hardware["pilot_unlearning_protocol_compatible"] = compatible
    hardware["npo_reference_strategy"] = (
        "cache frozen C_i completion log-probabilities before optimization; "
        "do not retain a second model during optimizer steps"
    )
    return hardware


def _prepare_output_dir(run_dir: Path, *, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to write into existing unlearning trajectory directory: {run_dir}. "
            "Use a new scientific configuration or pass overwrite=True explicitly."
        )
    ensure_dir(run_dir)


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(torch, device):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _batch_to_device(batch: dict[str, Any], device):
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "example_ids": batch["example_ids"],
    }


def _reference_completion_logprob_cache(
    *,
    AutoModelForCausalLM,
    torch,
    source: ContaminatedSource,
    sequences: Sequence[CompletionOnlySequence],
    method_config,
    device,
    pad_token_id: int,
) -> dict[str, float]:
    """Compute immutable C_i response log-probabilities before NPO optimization."""

    reference_model = AutoModelForCausalLM.from_pretrained(
        source.checkpoint_dir / "model",
        torch_dtype=torch.bfloat16,
    )
    reference_model.to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    values: dict[str, float] = {}
    try:
        with torch.inference_mode():
            for start in range(0, len(sequences), method_config.effective_batch_size):
                selected = sequences[start : start + method_config.effective_batch_size]
                batch = _batch_to_device(
                    collate_completion_only_batch(selected, pad_token_id=pad_token_id),
                    device,
                )
                with _autocast_context(torch, device):
                    outputs = reference_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                    logprob_sums, _counts = completion_logprob_sums(
                        outputs.logits,
                        batch["labels"],
                    )
                for example_id, value in zip(batch["example_ids"], logprob_sums.detach().cpu()):
                    values[example_id] = float(value)
    finally:
        del reference_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_ids = {sequence.example_id for sequence in sequences}
    if set(values) != expected_ids:
        raise RuntimeError("Frozen NPO reference cache did not cover every F completion")
    return values


def _save_reference_cache(
    *,
    run_dir: Path,
    values: dict[str, float],
    source: ContaminatedSource,
    plan: UnlearningTrajectoryPlan,
    beta: float,
    overwrite: bool,
) -> dict[str, Any]:
    ordered_values = {example_id: values[example_id] for example_id in sorted(values)}
    payload = {
        "reference_type": "frozen_contaminated_checkpoint",
        "reference_checkpoint_id": source.checkpoint_id,
        "reference_checkpoint_name": source.name,
        "reference_checkpoint_manifest_sha256": source.manifest_sha256,
        "clean_counterfactual_used_as_reference": False,
        "objective": "negative-only NPO; no chosen response",
        "beta": beta,
        "forget_examples_fingerprint": plan.forget_examples_fingerprint,
        "canonical_example_order_fingerprint": plan.canonical_example_order_fingerprint,
        "completion_logprob_sums": ordered_values,
    }
    payload["cache_fingerprint"] = fingerprint_payload(payload)
    write_json(run_dir / "reference_completion_logprobs.json", payload, overwrite=overwrite)
    return payload


def _snapshot_id(plan: UnlearningTrajectoryPlan, step: int) -> str:
    return f"{plan.trajectory_run_id}::step{step:04d}"


def _snapshot_manifest(
    *,
    plan: UnlearningTrajectoryPlan,
    source: ContaminatedSource,
    config: PilotConfig,
    plan_payload: dict[str, Any],
    hardware: dict[str, Any],
    step: int,
    parent_snapshot_id: str,
    reference_cache: dict[str, Any] | None,
    optimizer_state_saved: bool,
    final_training_loss: float | None,
    artifact_fingerprints: dict[str, str],
) -> dict[str, Any]:
    expected_artifact_keys = {"model_directory", "tokenizer_directory", "training_state_file"}
    if set(artifact_fingerprints) != expected_artifact_keys or any(
        not isinstance(value, str) or not value for value in artifact_fingerprints.values()
    ):
        raise ValueError(
            "Unlearning snapshot requires non-empty fingerprints for model, tokenizer, and state"
        )
    payload = {
        "checkpoint_id": _snapshot_id(plan, step),
        "run_id": plan.trajectory_run_id,
        "name": f"U_{source.name}_{plan.method}",
        "method": plan.method,
        "global_step": step,
        "snapshot_step": step,
        "parent_checkpoint_id": parent_snapshot_id,
        "source_contaminated_checkpoint_id": source.checkpoint_id,
        "source_contaminated_checkpoint_manifest_sha256": source.manifest_sha256,
        "clean_counterfactual_checkpoint_id": source.parent_clean_checkpoint_id,
        "clean_counterfactual_used_as_reference": False,
        "input_plan_fingerprint": fingerprint_payload(plan_payload),
        "experiment_run_id": plan.experiment_run_id,
        "forget_schema_version": config.forget.schema_version,
        "forget_examples_fingerprint": plan.forget_examples_fingerprint,
        "base_model": source.manifest.get(
            "base_model",
            {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "tokenizer_id": config.model.tokenizer_id,
            },
        ),
        "snapshot_schedule": list(plan.snapshot_steps),
        "optimizer_state_saved": optimizer_state_saved,
        "npo_reference_cache_fingerprint": (
            reference_cache.get("cache_fingerprint") if reference_cache else None
        ),
        "npo_reference_checkpoint_id": (
            source.checkpoint_id if plan.method == "npo" else None
        ),
        "final_training_loss_at_or_before_snapshot": final_training_loss,
        "precision": config.model.dtype,
        "hardware": hardware,
        "artifact_fingerprints": dict(artifact_fingerprints),
    }
    payload["manifest"] = build_manifest(
        artifact_type="unlearning_snapshot",
        schema_version=UNLEARNING_SNAPSHOT_MANIFEST_SCHEMA,
        payload=payload,
        source_metadata={
            "trajectory_run_id": plan.trajectory_run_id,
            "source_contaminated_checkpoint_id": source.checkpoint_id,
            "snapshot_step": step,
        },
    )
    return payload


def _save_snapshot(
    *,
    run_dir: Path,
    model,
    tokenizer,
    optimizer,
    torch,
    step: int,
    source_contaminated_checkpoint_id: str,
    method: str,
    overwrite: bool,
) -> tuple[Path, dict[str, str]]:
    """Materialize a model and optimizer state at one pre-registered snapshot."""

    snapshot_dir = run_dir / f"step{step:04d}"
    if snapshot_dir.exists() and any(snapshot_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing snapshot directory: {snapshot_dir}")
    ensure_dir(snapshot_dir)
    model.save_pretrained(snapshot_dir / "model")
    tokenizer.save_pretrained(snapshot_dir / "tokenizer")
    torch.save(
        {
            "global_step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "python_random_state": random.getstate(),
            "source_contaminated_checkpoint_id": source_contaminated_checkpoint_id,
            "method": method,
        },
        snapshot_dir / "training_state.pt",
    )
    artifact_fingerprints = {
        "model_directory": fingerprint_directory(snapshot_dir / "model"),
        "tokenizer_directory": fingerprint_directory(snapshot_dir / "tokenizer"),
        "training_state_file": hash_file(snapshot_dir / "training_state.pt"),
    }
    return snapshot_dir, artifact_fingerprints


def run_unlearning(
    *,
    config_path: str | Path,
    contaminated_checkpoint_dir: str | Path,
    forget_examples_path: str | Path,
    output_dir: str | Path,
    method: str,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a complete, fixed GA or frozen-reference NPO trajectory from one C_i.

    The source must be a contamination-stage ``C0``, ``C50``, or ``C100``
    directory.  The supplied F JSONL is verified byte-for-byte and
    content-fingerprint-for-content-fingerprint against that source manifest
    before any weights are loaded.
    """

    config = load_config(config_path)
    validate_training_ready(config)
    _method_config(config, method)
    source = load_contaminated_source(contaminated_checkpoint_dir, config=config)
    forget_path, examples, forget_file_sha256, forget_fingerprint = _load_verified_forget_examples(
        forget_examples_path=forget_examples_path,
        source=source,
        config=config,
    )
    plan = build_unlearning_trajectory_plan(
        config=config,
        source=source,
        method=method,
        examples=examples,
        forget_examples_fingerprint=forget_fingerprint,
    )
    plan_payload = _plan_payload(
        plan=plan,
        config=config,
        source=source,
        forget_examples_path=forget_path,
        forget_file_sha256=forget_file_sha256,
    )
    hardware = precheck_unlearning_environment()
    if dry_run:
        return {
            "dry_run": True,
            "plan": plan_payload,
            "hardware": hardware,
            "source_manifest_sha256": source.manifest_sha256,
        }
    if not hardware["pilot_unlearning_protocol_compatible"]:
        raise RuntimeError(
            "Current environment is not compatible with the frozen full-parameter bf16 "
            "pilot unlearning protocol. Refusing to load model weights."
        )

    run_dir = Path(output_dir) / plan.trajectory_run_id
    if run_dir.resolve() == source.checkpoint_dir.resolve():
        raise ValueError("Unlearning output directory must not be the input contaminated checkpoint")
    _prepare_output_dir(run_dir, overwrite=overwrite)
    write_json(run_dir / "trajectory_plan.json", plan_payload, overwrite=overwrite)

    torch, AutoModelForCausalLM, AutoTokenizer, AdamW8bit = _require_training_stack()
    _set_seed(torch, config.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(source.checkpoint_dir / "tokenizer", use_fast=True)
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("Completion-only unlearning requires a fast tokenizer with offset mappings")
    source_masking_length = source.manifest["masking"].get("sequence_length")
    if source_masking_length != _method_config(config, method).sequence_length:
        raise ValueError(
            "The C_i completion-mask sequence length does not match the frozen unlearning "
            "sequence length"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id for batch padding")
        pad_token_id = tokenizer.eos_token_id
    sequences = build_completion_only_sequences(
        examples,
        tokenizer=tokenizer,
        sequence_length=_method_config(config, method).sequence_length,
    )
    if tuple(sequence.example_id for sequence in sequences) != tuple(
        example["example_id"] for example in examples
    ):
        raise RuntimeError("Tokenized F sequence order diverged from canonical F example order")

    method_config = _method_config(config, method)
    reference_cache: dict[str, Any] | None = None
    reference_values: dict[str, float] | None = None
    if method == "npo":
        reference_values = _reference_completion_logprob_cache(
            AutoModelForCausalLM=AutoModelForCausalLM,
            torch=torch,
            source=source,
            sequences=sequences,
            method_config=method_config,
            device=device,
            pad_token_id=pad_token_id,
        )
        reference_cache = _save_reference_cache(
            run_dir=run_dir,
            values=reference_values,
            source=source,
            plan=plan,
            beta=method_config.beta,
            overwrite=overwrite,
        )

    model = AutoModelForCausalLM.from_pretrained(
        source.checkpoint_dir / "model",
        torch_dtype=torch.bfloat16,
    )
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.to(device)
    model.train()
    optimizer = AdamW8bit(
        model.parameters(),
        lr=method_config.learning_rate,
        weight_decay=method_config.weight_decay,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = perf_counter()
    snapshot_paths: dict[str, str] = {}
    parent_snapshot_id = source.checkpoint_id
    last_training_loss: float | None = None
    try:
        for step in range(0, method_config.steps + 1):
            if step > 0:
                selected = [
                    sequences[index]
                    for index in deterministic_batch_indices(
                        step=step,
                        example_count=len(sequences),
                        batch_size=method_config.effective_batch_size,
                    )
                ]
                batch = _batch_to_device(
                    collate_completion_only_batch(selected, pad_token_id=pad_token_id),
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(torch, device):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                    if method == "ga":
                        loss = ga_completion_only_loss(outputs.logits, batch["labels"])
                    else:
                        if reference_values is None or method_config.beta is None:
                            raise RuntimeError("NPO reference cache or beta was not initialized")
                        reference_batch = [reference_values[example_id] for example_id in batch["example_ids"]]
                        loss = npo_completion_only_loss(
                            outputs.logits,
                            batch["labels"],
                            reference_batch,
                            beta=method_config.beta,
                        )
                loss_value = float(loss.detach().float().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"Non-finite {method} unlearning loss at optimizer step {step}: {loss_value}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), method_config.gradient_clip_norm)
                optimizer.step()
                last_training_loss = loss_value

            if step in plan.snapshot_steps:
                snapshot_dir, artifact_fingerprints = _save_snapshot(
                    run_dir=run_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    torch=torch,
                    step=step,
                    source_contaminated_checkpoint_id=source.checkpoint_id,
                    method=method,
                    overwrite=overwrite,
                )
                snapshot_manifest = _snapshot_manifest(
                    plan=plan,
                    source=source,
                    config=config,
                    plan_payload=plan_payload,
                    hardware=hardware,
                    step=step,
                    parent_snapshot_id=parent_snapshot_id,
                    reference_cache=reference_cache,
                    optimizer_state_saved=True,
                    final_training_loss=last_training_loss,
                    artifact_fingerprints=artifact_fingerprints,
                )
                write_json(snapshot_dir / "manifest.json", snapshot_manifest, overwrite=overwrite)
                snapshot_paths[str(step)] = str(snapshot_dir)
                parent_snapshot_id = snapshot_manifest["checkpoint_id"]
    finally:
        runtime_seconds = perf_counter() - started_clock

    completed_at = datetime.now(timezone.utc).isoformat()
    run_payload = {
        **plan_payload,
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "runtime_seconds": runtime_seconds,
        "snapshot_paths": snapshot_paths,
        "snapshot_checkpoint_ids": {
            step: _snapshot_id(plan, int(step)) for step in snapshot_paths
        },
        "source_contaminated_checkpoint_manifest_sha256": source.manifest_sha256,
        "clean_counterfactual_used_as_reference": False,
    }
    run_payload["manifest"] = build_manifest(
        artifact_type="unlearning_trajectory",
        schema_version="unlearning-trajectory-v1",
        payload=run_payload,
        source_metadata={
            "trajectory_run_id": plan.trajectory_run_id,
            "source_contaminated_checkpoint_id": source.checkpoint_id,
            "method": method,
        },
    )
    write_json(run_dir / "run_manifest.json", run_payload, overwrite=overwrite)
    return run_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed GA or frozen-reference NPO trajectory from one contaminated C_i."
    )
    parser.add_argument("--config", default="configs/pilot/base.yaml")
    parser.add_argument("--contaminated-checkpoint-dir", required=True)
    parser.add_argument(
        "--forget-examples-path",
        required=True,
        help="F contamination_examples.jsonl, or its containing forget dataset directory.",
    )
    parser.add_argument("--output-dir", default="checkpoints/unlearned")
    parser.add_argument("--method", choices=("ga", "npo"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run_unlearning(
        config_path=args.config,
        contaminated_checkpoint_dir=args.contaminated_checkpoint_dir,
        forget_examples_path=args.forget_examples_path,
        output_dir=args.output_dir,
        method=args.method,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
