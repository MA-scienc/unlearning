"""Fixed-exposure contamination for the controlled unlearning pilot.

This module implements only the pre-unlearning transition ``S_i -> C_i``.
It deliberately has no matched-memorization logic, retain-data interleaving,
or unlearning objective.  Every invocation consumes one clean checkpoint and
the generated synthetic-F contamination JSONL, then writes one immutable
contaminated checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from unlearning.config import PilotConfig, load_config, validate_training_ready
from unlearning.experiments.naming import experiment_run_id
from unlearning.training.checkpoints import (
    CLEAN_CHECKPOINT_MANIFEST_SCHEMA,
    CheckpointIdentity,
    validate_clean_checkpoint_identity,
)
from unlearning.utils.env import inspect_hardware
from unlearning.utils.hashing import (
    fingerprint_directory,
    fingerprint_payload,
    fingerprint_records,
    hash_file,
)
from unlearning.utils.io import ensure_dir, read_json, read_jsonl, write_json
from unlearning.utils.manifest import build_manifest


IGNORE_INDEX = -100
CONTAMINATION_MANIFEST_SCHEMA = "fixed-exposure-contamination-v2"
_CLEAN_TO_CONTAMINATED_NAME = {"S0": "C0", "S50": "C50", "S100": "C100"}


class ContaminationProtocolError(ValueError):
    """Raised when an input would violate the frozen contamination protocol."""


@dataclass(frozen=True)
class FixedExposurePlan:
    """The deterministic, complete exposure budget for one ``S_i -> C_i`` run."""

    example_count: int
    epochs: int
    effective_batch_size: int
    steps_per_epoch: int
    optimizer_steps: int
    total_example_exposures: int
    ordering: str = "JSONL file order, repeated unchanged for each epoch"


@dataclass(frozen=True)
class CleanCheckpointReference:
    """Validated lineage information for a single clean specialization checkpoint."""

    checkpoint_dir: Path
    manifest_path: Path
    run_id: str
    name: str
    contaminated_name: str
    checkpoint_id: str
    global_step: int
    base_model: dict[str, Any]
    manifest_file_sha256: str
    manifest_payload_fingerprint: str
    model_source: str
    tokenizer_source: str
    source_kind: str


@dataclass(frozen=True)
class ForgetExamplesReference:
    """Validated generated-F input and its immutable provenance metadata."""

    examples_path: Path
    manifest_path: Path
    examples: list[dict[str, Any]]
    examples_file_sha256: str
    examples_fingerprint: str
    records_fingerprint: str
    manifest_file_sha256: str
    manifest_payload_fingerprint: str
    schema_version: str
    example_summary: dict[str, Any]


def validate_fixed_exposure_protocol(config: PilotConfig) -> None:
    """Reject deviations from the pre-specified pilot contamination condition."""

    contamination = config.contamination
    expected = {
        "condition": "fixed_exposure",
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "epochs": 10,
        "effective_batch_size": 16,
        "sequence_length": 512,
        "warmup_ratio": 0.03,
        "scheduler": "linear",
        "optimizer": "adamw_8bit",
    }
    mismatches: list[str] = []
    for field, expected_value in expected.items():
        observed = getattr(contamination, field)
        if isinstance(expected_value, float):
            matches = math.isclose(float(observed), expected_value, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = observed == expected_value
        if not matches:
            mismatches.append(f"{field}={observed!r} (required {expected_value!r})")
    if config.model.optimizer != contamination.optimizer:
        mismatches.append(
            "contamination.optimizer must match model.optimizer so the contamination stage uses "
            "the same optimizer family as specialization"
        )
    if mismatches:
        raise ContaminationProtocolError(
            "Fixed-exposure contamination configuration is not the frozen pilot protocol: "
            + "; ".join(mismatches)
        )


def derive_contaminated_checkpoint_name(clean_name: str) -> str:
    """Map an approved clean checkpoint label to its corresponding C checkpoint label."""

    try:
        return _CLEAN_TO_CONTAMINATED_NAME[clean_name]
    except KeyError as exc:
        allowed = ", ".join(_CLEAN_TO_CONTAMINATED_NAME)
        raise ContaminationProtocolError(
            f"Clean checkpoint name must be one of {allowed}; received {clean_name!r}"
        ) from exc


def build_fixed_exposure_plan(
    *,
    example_count: int,
    epochs: int,
    effective_batch_size: int,
) -> FixedExposurePlan:
    """Build the exact no-shuffle exposure plan used by every clean checkpoint.

    The current synthetic dataset has 960 examples and batch size 16, so every
    epoch has exactly 60 full batches.  Rejecting a partial batch avoids a
    hidden difference in effective batch size at the end of an epoch.
    """

    if example_count <= 0:
        raise ContaminationProtocolError("contamination example_count must be positive")
    if epochs <= 0:
        raise ContaminationProtocolError("contamination epochs must be positive")
    if effective_batch_size <= 0:
        raise ContaminationProtocolError("contamination effective_batch_size must be positive")
    if example_count % effective_batch_size:
        raise ContaminationProtocolError(
            "The frozen fixed-exposure protocol requires an exact number of full batches: "
            f"{example_count} examples is not divisible by {effective_batch_size}"
        )
    steps_per_epoch = example_count // effective_batch_size
    return FixedExposurePlan(
        example_count=example_count,
        epochs=epochs,
        effective_batch_size=effective_batch_size,
        steps_per_epoch=steps_per_epoch,
        optimizer_steps=steps_per_epoch * epochs,
        total_example_exposures=example_count * epochs,
    )


def validate_contamination_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    expected_record_count: int,
    expected_templates_per_record: int,
) -> dict[str, Any]:
    """Validate the structural invariants of generated F contamination examples.

    This is intentionally pure so it can be tested without a tokenizer, Torch,
    or a GPU.  It also prevents evaluation prompts or prompt-only rows from
    silently entering the contamination stream.
    """

    if expected_record_count <= 0 or expected_templates_per_record <= 0:
        raise ContaminationProtocolError("Expected F record/template counts must be positive")
    expected_example_count = expected_record_count * expected_templates_per_record
    if len(examples) != expected_example_count:
        raise ContaminationProtocolError(
            f"Expected {expected_example_count} contamination examples, found {len(examples)}"
        )

    required_fields = (
        "example_id",
        "record_id",
        "template_id",
        "split",
        "prompt",
        "completion",
        "text",
    )
    example_ids: set[str] = set()
    templates_by_record: dict[str, set[str]] = {}
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ContaminationProtocolError(f"Contamination example {index} is not an object")
        missing = [field for field in required_fields if field not in example]
        if missing:
            raise ContaminationProtocolError(
                f"Contamination example {index} is missing fields: {', '.join(missing)}"
            )
        values = {field: example[field] for field in required_fields}
        if any(not isinstance(value, str) for value in values.values()):
            raise ContaminationProtocolError(
                f"Contamination example {index} has non-string required fields"
            )
        if values["split"] != "contamination":
            raise ContaminationProtocolError(
                f"Contamination example {values['example_id']!r} has split "
                f"{values['split']!r}, not 'contamination'"
            )
        if not values["example_id"] or not values["record_id"] or not values["template_id"]:
            raise ContaminationProtocolError(
                f"Contamination example {index} has an empty identity field"
            )
        if not values["completion"]:
            raise ContaminationProtocolError(
                f"Contamination example {values['example_id']!r} has an empty completion"
            )
        if values["text"] != values["prompt"] + values["completion"]:
            raise ContaminationProtocolError(
                f"Contamination example {values['example_id']!r} does not equal prompt + completion"
            )
        if values["example_id"] in example_ids:
            raise ContaminationProtocolError(
                f"Duplicate contamination example_id: {values['example_id']!r}"
            )
        example_ids.add(values["example_id"])
        templates = templates_by_record.setdefault(values["record_id"], set())
        if values["template_id"] in templates:
            raise ContaminationProtocolError(
                f"Duplicate template {values['template_id']!r} for record {values['record_id']!r}"
            )
        templates.add(values["template_id"])

    if len(templates_by_record) != expected_record_count:
        raise ContaminationProtocolError(
            f"Expected {expected_record_count} F records, found {len(templates_by_record)}"
        )
    template_sets = {tuple(sorted(templates)) for templates in templates_by_record.values()}
    if len(template_sets) != 1:
        raise ContaminationProtocolError(
            "Each F record must have the same contamination template set"
        )
    for record_id, templates in templates_by_record.items():
        if len(templates) != expected_templates_per_record:
            raise ContaminationProtocolError(
                f"Record {record_id!r} has {len(templates)} contamination templates; expected "
                f"{expected_templates_per_record}"
            )
    template_ids = sorted(next(iter(template_sets)))
    return {
        "example_count": len(examples),
        "record_count": len(templates_by_record),
        "templates_per_record": expected_templates_per_record,
        "template_ids": template_ids,
        "examples_fingerprint": fingerprint_records(examples),
    }


def completion_only_labels(
    input_ids: Sequence[int],
    offset_mapping: Sequence[Sequence[int]],
    *,
    prompt_char_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Create causal-LM labels that score only tokens wholly in the completion.

    Fast-tokenizer offsets are used instead of separately tokenizing prompt and
    completion.  That distinction matters for tokenizers whose merges cross the
    textual prompt/completion boundary.  A boundary-crossing token is masked,
    which is conservative: no loss ever trains on a prompt character.
    """

    if prompt_char_length < 0:
        raise ContaminationProtocolError("prompt_char_length must be non-negative")
    if len(input_ids) != len(offset_mapping):
        raise ContaminationProtocolError(
            "input_ids and offset_mapping must have the same number of entries"
        )
    labels: list[int] = []
    for index, (token_id, offset) in enumerate(zip(input_ids, offset_mapping, strict=True)):
        if len(offset) != 2:
            raise ContaminationProtocolError(
                f"Offset mapping at token {index} must contain exactly (start, end)"
            )
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start:
            raise ContaminationProtocolError(
                f"Offset mapping at token {index} is invalid: ({start}, {end})"
            )
        completion_token = end > start and start >= prompt_char_length
        labels.append(int(token_id) if completion_token else ignore_index)
    return labels


def tokenize_completion_example(
    example: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """Tokenize one F row and attach completion-only labels without Torch."""

    if max_length <= 0:
        raise ContaminationProtocolError("max_length must be positive")
    for field in ("example_id", "record_id", "prompt", "completion", "text"):
        if field not in example or not isinstance(example[field], str):
            raise ContaminationProtocolError(f"Cannot tokenize example without string {field!r}")
    if example["text"] != example["prompt"] + example["completion"]:
        raise ContaminationProtocolError(
            "Cannot tokenize an example whose text is not prompt + completion"
        )
    try:
        encoded = tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offset_mapping = list(encoded["offset_mapping"])
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        raise ContaminationProtocolError(
            "Completion-only masking requires a fast tokenizer that returns offset mappings"
        ) from exc
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        raise ContaminationProtocolError(
            "Tokenizer returned batched input_ids for a single example"
        )
    if not input_ids:
        raise ContaminationProtocolError(
            f"Tokenization produced no tokens for {example['example_id']!r}"
        )
    labels = completion_only_labels(
        input_ids,
        offset_mapping,
        prompt_char_length=len(example["prompt"]),
        ignore_index=ignore_index,
    )
    if not any(label != ignore_index for label in labels):
        raise ContaminationProtocolError(
            f"No completion token remains after truncation for {example['example_id']!r}"
        )
    return {
        "example_id": example["example_id"],
        "record_id": example["record_id"],
        "input_ids": [int(token_id) for token_id in input_ids],
        "labels": labels,
    }


def collate_completion_batch(
    encoded_examples: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """Dynamically pad tokenized completion examples using only Python lists."""

    if not encoded_examples:
        raise ContaminationProtocolError("Cannot collate an empty contamination batch")
    lengths: list[int] = []
    for index, example in enumerate(encoded_examples):
        input_ids = example.get("input_ids")
        labels = example.get("labels")
        if not isinstance(input_ids, Sequence) or not isinstance(labels, Sequence):
            raise ContaminationProtocolError(
                f"Encoded example {index} is missing input_ids or labels"
            )
        if not input_ids or len(input_ids) != len(labels):
            raise ContaminationProtocolError(
                f"Encoded example {index} has invalid token/label lengths"
            )
        if not any(int(label) != ignore_index for label in labels):
            raise ContaminationProtocolError(f"Encoded example {index} has no completion labels")
        lengths.append(len(input_ids))
    target_length = max(lengths)
    batch_input_ids: list[list[int]] = []
    batch_attention_mask: list[list[int]] = []
    batch_labels: list[list[int]] = []
    for example, length in zip(encoded_examples, lengths, strict=True):
        padding = target_length - length
        input_ids = [int(value) for value in example["input_ids"]]
        labels = [int(value) for value in example["labels"]]
        batch_input_ids.append(input_ids + [int(pad_token_id)] * padding)
        batch_attention_mask.append([1] * length + [0] * padding)
        batch_labels.append(labels + [ignore_index] * padding)
    return {
        "example_ids": [str(example.get("example_id", "")) for example in encoded_examples],
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


def load_clean_checkpoint_reference(
    clean_checkpoint: str | Path,
    *,
    config: PilotConfig,
) -> CleanCheckpointReference:
    """Read and validate a clean S checkpoint manifest before any model loading."""

    supplied = Path(clean_checkpoint)
    manifest_path = supplied if supplied.is_file() else supplied / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clean checkpoint manifest was not found: {manifest_path}")
    checkpoint_dir = manifest_path.parent
    raw = read_json(manifest_path)
    if not isinstance(raw, Mapping):
        raise ContaminationProtocolError("Clean checkpoint manifest must be a JSON object")
    artifact_manifest = raw.get("manifest")
    if not isinstance(artifact_manifest, Mapping):
        raise ContaminationProtocolError(
            "Clean checkpoint manifest lacks immutable manifest metadata"
        )
    if artifact_manifest.get("artifact_type") != "clean_specialization_checkpoint":
        raise ContaminationProtocolError("Input is not a clean specialization checkpoint manifest")
    if artifact_manifest.get("schema_version") != CLEAN_CHECKPOINT_MANIFEST_SCHEMA:
        raise ContaminationProtocolError("Unsupported clean checkpoint manifest schema")
    payload_fingerprint = artifact_manifest.get("payload_fingerprint")
    if not isinstance(payload_fingerprint, str) or not payload_fingerprint:
        raise ContaminationProtocolError("Clean checkpoint manifest lacks a payload fingerprint")
    expected_payload_fingerprint = fingerprint_payload(
        {key: value for key, value in raw.items() if key != "manifest"}
    )
    if payload_fingerprint != expected_payload_fingerprint:
        raise ContaminationProtocolError(
            "Clean checkpoint manifest payload fingerprint does not match its checkpoint payload"
        )

    required = ("run_id", "name", "checkpoint_id", "global_step", "base_model")
    missing = [field for field in required if field not in raw]
    if missing:
        raise ContaminationProtocolError(
            f"Clean checkpoint manifest is missing fields: {', '.join(missing)}"
        )
    run_id = raw["run_id"]
    name = raw["name"]
    checkpoint_id = raw["checkpoint_id"]
    global_step = raw["global_step"]
    base_model = raw["base_model"]
    if not isinstance(run_id, str) or not run_id:
        raise ContaminationProtocolError("Clean checkpoint run_id must be a non-empty string")
    if run_id != experiment_run_id(config):
        raise ContaminationProtocolError(
            "Clean checkpoint run_id does not match the complete frozen contamination configuration"
        )
    if not isinstance(name, str):
        raise ContaminationProtocolError("Clean checkpoint name must be a string")
    contaminated_name = derive_contaminated_checkpoint_name(name)
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ContaminationProtocolError("Clean checkpoint_id must be a non-empty string")
    if not isinstance(global_step, int) or global_step < 0:
        raise ContaminationProtocolError(
            "Clean checkpoint global_step must be a non-negative integer"
        )
    try:
        validate_clean_checkpoint_identity(
            config=config,
            run_id=run_id,
            name=name,
            global_step=global_step,
            parent_checkpoint_id=raw.get("parent_checkpoint_id"),
        )
    except ValueError as exc:
        raise ContaminationProtocolError(str(exc)) from exc
    if not isinstance(base_model, Mapping):
        raise ContaminationProtocolError("Clean checkpoint base_model must be an object")
    expected_base_model = {
        "model_id": config.model.model_id,
        "revision": config.model.revision,
        "tokenizer_id": config.model.tokenizer_id,
    }
    observed_base_model = {field: base_model.get(field) for field in expected_base_model}
    if observed_base_model != expected_base_model:
        raise ContaminationProtocolError(
            "Clean checkpoint base-model identity does not match the pinned contamination config"
        )

    model_dir = checkpoint_dir / "model"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    training_state_path = checkpoint_dir / "training_state.pt"
    artifact_fingerprints = raw.get("artifact_fingerprints")
    if not isinstance(artifact_fingerprints, Mapping):
        raise ContaminationProtocolError("Clean checkpoint manifest lacks artifact fingerprints")
    if name == "S0" and not model_dir.is_dir() and not tokenizer_dir.is_dir():
        if any(artifact_fingerprints.get(key) is not None for key in (
            "model_directory",
            "tokenizer_directory",
            "training_state_file",
        )):
            raise ContaminationProtocolError(
                "S0 base-model reference must not claim local checkpoint artifact fingerprints"
            )
        model_source = config.model.model_id
        tokenizer_source = config.model.tokenizer_id
        source_kind = "pinned_base_model_for_s0_reference"
    else:
        if not model_dir.is_dir() or not tokenizer_dir.is_dir() or not training_state_path.is_file():
            raise ContaminationProtocolError(
                "Clean checkpoint must contain model/, tokenizer/, and training_state.pt artifacts; "
                "only the S0 reference may be materialized from the pinned base model"
            )
        observed_artifacts = {
            "model_directory": fingerprint_directory(model_dir),
            "tokenizer_directory": fingerprint_directory(tokenizer_dir),
            "training_state_file": hash_file(training_state_path),
        }
        expected_artifacts = {
            key: artifact_fingerprints.get(key)
            for key in observed_artifacts
        }
        if expected_artifacts != observed_artifacts:
            raise ContaminationProtocolError(
                "Clean checkpoint local artifact fingerprints do not match its immutable manifest"
            )
        model_source = str(model_dir)
        tokenizer_source = str(tokenizer_dir)
        source_kind = "local_clean_checkpoint"
    return CleanCheckpointReference(
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        run_id=run_id,
        name=name,
        contaminated_name=contaminated_name,
        checkpoint_id=checkpoint_id,
        global_step=global_step,
        base_model=dict(observed_base_model),
        manifest_file_sha256=hash_file(manifest_path),
        manifest_payload_fingerprint=payload_fingerprint,
        model_source=model_source,
        tokenizer_source=tokenizer_source,
        source_kind=source_kind,
    )


def load_forget_examples_reference(
    contamination_examples: str | Path,
    *,
    config: PilotConfig,
) -> ForgetExamplesReference:
    """Load generated F and verify its v2 manifest matches the supplied JSONL."""

    examples_path = Path(contamination_examples)
    if not examples_path.is_file():
        raise FileNotFoundError(f"Contamination examples JSONL was not found: {examples_path}")
    manifest_path = examples_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Generated F manifest was not found next to contamination_examples.jsonl: "
            f"{manifest_path}"
        )
    examples = read_jsonl(examples_path)
    example_summary = validate_contamination_examples(
        examples,
        expected_record_count=config.forget.record_count,
        expected_templates_per_record=config.forget.contamination_templates_per_record,
    )
    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, Mapping):
        raise ContaminationProtocolError("Generated F manifest must be a JSON object")
    if raw_manifest.get("schema_version") != config.forget.schema_version:
        raise ContaminationProtocolError(
            "Generated F schema version does not match the frozen contamination configuration"
        )
    if raw_manifest.get("real_patient_information_used") is not False:
        raise ContaminationProtocolError(
            "Generated F manifest must explicitly state real_patient_information_used=false"
        )
    if raw_manifest.get("record_count") != config.forget.record_count:
        raise ContaminationProtocolError(
            "Generated F record count does not match the frozen config"
        )
    if raw_manifest.get("contamination_example_count") != len(examples):
        raise ContaminationProtocolError(
            "Generated F manifest contamination_example_count does not match the supplied JSONL"
        )
    fingerprints = raw_manifest.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ContaminationProtocolError("Generated F manifest lacks fingerprints")
    examples_fingerprint = example_summary["examples_fingerprint"]
    if fingerprints.get("contamination_examples") != examples_fingerprint:
        raise ContaminationProtocolError(
            "Generated F contamination_examples fingerprint does not match the supplied JSONL"
        )
    records_fingerprint = fingerprints.get("records")
    if not isinstance(records_fingerprint, str) or not records_fingerprint:
        raise ContaminationProtocolError("Generated F manifest lacks a records fingerprint")
    artifact_manifest = raw_manifest.get("manifest")
    if not isinstance(artifact_manifest, Mapping):
        raise ContaminationProtocolError("Generated F manifest lacks immutable manifest metadata")
    if artifact_manifest.get("artifact_type") != "synthetic_forget_dataset":
        raise ContaminationProtocolError("Input F manifest is not a synthetic forget dataset")
    payload_fingerprint = artifact_manifest.get("payload_fingerprint")
    expected_payload_fingerprint = fingerprint_payload(
        {key: value for key, value in raw_manifest.items() if key != "manifest"}
    )
    if payload_fingerprint != expected_payload_fingerprint:
        raise ContaminationProtocolError(
            "Generated F manifest payload fingerprint does not match its summary payload"
        )
    return ForgetExamplesReference(
        examples_path=examples_path,
        manifest_path=manifest_path,
        examples=examples,
        examples_file_sha256=hash_file(examples_path),
        examples_fingerprint=examples_fingerprint,
        records_fingerprint=records_fingerprint,
        manifest_file_sha256=hash_file(manifest_path),
        manifest_payload_fingerprint=str(payload_fingerprint),
        schema_version=str(raw_manifest["schema_version"]),
        example_summary=example_summary,
    )


def precheck_contamination_environment() -> dict[str, Any]:
    """Report whether the frozen full-parameter contamination protocol can run here."""

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
        device = devices[0]
        compatible = compatible and (
            bool(device.get("bf16_supported")) and int(device.get("total_memory_mb", 0)) >= 24_000
        )
    else:
        compatible = False
    hardware["fixed_exposure_contamination_protocol_compatible"] = compatible
    return hardware


def build_contamination_manifest(
    *,
    config: PilotConfig,
    clean: CleanCheckpointReference,
    forget: ForgetExamplesReference,
    plan: FixedExposurePlan,
    hardware: Mapping[str, Any],
    output_checkpoint_dir: str | Path,
    runtime_seconds: float,
    optimizer_weight_decay: float,
    software_versions: Mapping[str, Any],
    artifact_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    """Build the self-contained immutable provenance record for one C checkpoint."""

    checkpoint_id = CheckpointIdentity(
        clean.run_id,
        clean.contaminated_name,
        plan.optimizer_steps,
    ).checkpoint_id
    if not math.isclose(
        float(optimizer_weight_decay),
        config.contamination.weight_decay,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContaminationProtocolError(
            "Recorded contamination optimizer weight decay does not match the frozen config"
        )
    contamination_config = asdict(config.contamination)
    contamination_config["optimizer_weight_decay"] = optimizer_weight_decay
    contamination_config["no_domain_data_interleaving"] = True
    contamination_config["matched_memorization_adjustment"] = False
    expected_artifact_keys = {"model_directory", "tokenizer_directory", "training_state_file"}
    if set(artifact_fingerprints) != expected_artifact_keys or any(
        not isinstance(value, str) or not value for value in artifact_fingerprints.values()
    ):
        raise ContaminationProtocolError(
            "Contaminated checkpoint requires non-empty fingerprints for model, tokenizer, and state"
        )
    payload: dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "run_id": clean.run_id,
        "name": clean.contaminated_name,
        "global_step": plan.optimizer_steps,
        "parent_clean_checkpoint_id": clean.checkpoint_id,
        "parent_clean_checkpoint_name": clean.name,
        "parent_clean_global_step": clean.global_step,
        "parent_clean_checkpoint_path": str(clean.checkpoint_dir),
        "parent_clean_checkpoint_manifest_path": str(clean.manifest_path),
        "parent_clean_checkpoint_manifest_fingerprint": clean.manifest_payload_fingerprint,
        "parent_clean_checkpoint_manifest_file_sha256": clean.manifest_file_sha256,
        "base_model": clean.base_model,
        "clean_model_source_kind": clean.source_kind,
        "clean_model_source": clean.model_source,
        "clean_tokenizer_source": clean.tokenizer_source,
        "forget_schema_version": forget.schema_version,
        "forget_examples_path": str(forget.examples_path),
        "forget_examples_file_sha256": forget.examples_file_sha256,
        "forget_examples_fingerprint": forget.examples_fingerprint,
        "forget_records_fingerprint": forget.records_fingerprint,
        "forget_dataset_manifest_path": str(forget.manifest_path),
        "forget_dataset_manifest_file_sha256": forget.manifest_file_sha256,
        "forget_dataset_manifest_payload_fingerprint": forget.manifest_payload_fingerprint,
        "forget_examples": forget.example_summary,
        "contamination": contamination_config,
        "exposure_plan": asdict(plan),
        "masking": {
            "objective": "causal_lm_completion_only",
            "ignore_index": IGNORE_INDEX,
            "tokenizer_requirement": "fast tokenizer with offset mapping",
            "add_special_tokens": False,
            "sequence_length": config.contamination.sequence_length,
            "boundary_rule": (
                "A token is supervised only when its token-offset start is at or after the "
                "prompt/completion character boundary; boundary-crossing tokens are masked."
            ),
            "padding_labels": IGNORE_INDEX,
        },
        "seed": config.seed,
        "precision": config.model.dtype,
        "gradient_checkpointing": config.model.gradient_checkpointing,
        "hardware": dict(hardware),
        "software_versions": dict(software_versions),
        "runtime_seconds": round(float(runtime_seconds), 6),
        "output_checkpoint_dir": str(output_checkpoint_dir),
        "artifact_fingerprints": dict(artifact_fingerprints),
    }
    payload["manifest"] = build_manifest(
        artifact_type="contaminated_checkpoint",
        schema_version=CONTAMINATION_MANIFEST_SCHEMA,
        payload=payload,
        source_metadata={
            "run_id": clean.run_id,
            "clean_checkpoint_id": clean.checkpoint_id,
            "forget_examples_fingerprint": forget.examples_fingerprint,
        },
    )
    return payload


def _require_training_stack():
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required for fixed-exposure contamination training."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler, transformers


def _require_bitsandbytes_optimizer():
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("8-bit AdamW requires bitsandbytes in the training environment") from exc
    return bnb, bnb.optim.AdamW8bit


def _load_tokenizer(reference: CleanCheckpointReference, config: PilotConfig, auto_tokenizer: Any):
    kwargs: dict[str, Any] = {"use_fast": True}
    if reference.source_kind == "pinned_base_model_for_s0_reference":
        kwargs["revision"] = config.model.revision
    tokenizer = auto_tokenizer.from_pretrained(reference.tokenizer_source, **kwargs)
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ContaminationProtocolError(
            "Completion-only contamination requires a fast tokenizer with offset mappings"
        )
    return tokenizer


def _load_model(
    reference: CleanCheckpointReference,
    config: PilotConfig,
    auto_model: Any,
    torch: Any,
):
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if reference.source_kind == "pinned_base_model_for_s0_reference":
        kwargs["revision"] = config.model.revision
    return auto_model.from_pretrained(reference.model_source, **kwargs)


def _pad_token_id(tokenizer: Any) -> int:
    for attribute in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    raise ContaminationProtocolError(
        "Tokenizer has neither pad_token_id nor eos_token_id for deterministic batch padding"
    )


def _assert_output_available(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing contaminated checkpoint: {path}. "
            "Use a new experiment directory or pass overwrite=True explicitly."
        )
    if path.exists() and not path.is_dir():
        raise ContaminationProtocolError(
            f"Contaminated checkpoint target is not a directory: {path}"
        )


def _run_training_loop(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    encoded_examples: Sequence[Mapping[str, Any]],
    plan: FixedExposurePlan,
    pad_token_id: int,
) -> int:
    """Run the fixed no-shuffle sequence of full F batches and return its step count."""

    device = torch.device("cuda")
    model.to(device)
    model.train()
    global_step = 0
    for _epoch in range(plan.epochs):
        for start in range(0, plan.example_count, plan.effective_batch_size):
            examples = encoded_examples[start : start + plan.effective_batch_size]
            batch = collate_completion_batch(examples, pad_token_id=pad_token_id)
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            loss_value = float(loss.detach().float().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    "Non-finite contamination loss at optimizer step "
                    f"{global_step + 1}: {loss_value}"
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
    if global_step != plan.optimizer_steps:
        raise RuntimeError(
            f"Contamination executed {global_step} steps, expected {plan.optimizer_steps}"
        )
    return global_step


def run_fixed_exposure_contamination(
    *,
    config_path: str | Path,
    clean_checkpoint: str | Path,
    contamination_examples: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create ``C_i`` from one clean ``S_i`` checkpoint under the frozen pilot recipe.

    ``output_dir`` is a root directory.  The deterministic destination is
    ``<output_dir>/<clean-run-id>/C0|C50|C100``.  Multiple C checkpoints may
    therefore share one root without sharing or mutating a run manifest.
    """

    config = load_config(config_path)
    validate_training_ready(config)
    validate_fixed_exposure_protocol(config)
    clean = load_clean_checkpoint_reference(clean_checkpoint, config=config)
    forget = load_forget_examples_reference(contamination_examples, config=config)
    plan = build_fixed_exposure_plan(
        example_count=len(forget.examples),
        epochs=config.contamination.epochs,
        effective_batch_size=config.contamination.effective_batch_size,
    )
    target_dir = Path(output_dir) / clean.run_id / clean.contaminated_name
    hardware = precheck_contamination_environment()
    if dry_run:
        return {
            "dry_run": True,
            "output_checkpoint_dir": str(target_dir),
            "checkpoint_id": CheckpointIdentity(
                clean.run_id,
                clean.contaminated_name,
                plan.optimizer_steps,
            ).checkpoint_id,
            "clean_checkpoint_id": clean.checkpoint_id,
            "forget_examples_fingerprint": forget.examples_fingerprint,
            "exposure_plan": asdict(plan),
            "hardware": hardware,
        }
    _assert_output_available(target_dir, overwrite=overwrite)
    if not hardware["fixed_exposure_contamination_protocol_compatible"]:
        raise RuntimeError(
            "Current environment is not compatible with the frozen full-parameter bf16 "
            "fixed-exposure contamination protocol. Refusing to load model weights."
        )

    torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler, transformers = (
        _require_training_stack()
    )
    bnb, AdamW8bit = _require_bitsandbytes_optimizer()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    started = time.perf_counter()
    tokenizer = _load_tokenizer(clean, config, AutoTokenizer)
    encoded_examples = [
        tokenize_completion_example(
            example,
            tokenizer=tokenizer,
            max_length=config.contamination.sequence_length,
        )
        for example in forget.examples
    ]
    model = _load_model(clean, config, AutoModelForCausalLM, torch)
    model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    optimizer_weight_decay = config.contamination.weight_decay
    optimizer = AdamW8bit(
        model.parameters(),
        lr=config.contamination.learning_rate,
        weight_decay=optimizer_weight_decay,
    )
    scheduler = get_scheduler(
        name=config.contamination.scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(config.contamination.warmup_ratio * plan.optimizer_steps),
        num_training_steps=plan.optimizer_steps,
    )
    executed_steps = _run_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        torch=torch,
        encoded_examples=encoded_examples,
        plan=plan,
        pad_token_id=_pad_token_id(tokenizer),
    )
    if executed_steps != plan.optimizer_steps:
        raise RuntimeError("Unexpected contamination step count")

    ensure_dir(target_dir)
    model.save_pretrained(target_dir / "model")
    tokenizer.save_pretrained(target_dir / "tokenizer")
    torch.save(
        {
            "global_step": executed_steps,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "python_random_state": random.getstate(),
        },
        target_dir / "training_state.pt",
    )
    artifact_fingerprints = {
        "model_directory": fingerprint_directory(target_dir / "model"),
        "tokenizer_directory": fingerprint_directory(target_dir / "tokenizer"),
        "training_state_file": hash_file(target_dir / "training_state.pt"),
    }
    runtime_seconds = time.perf_counter() - started
    software_versions = {
        "torch": getattr(torch, "__version__", None),
        "transformers": getattr(transformers, "__version__", None),
        "bitsandbytes": getattr(bnb, "__version__", None),
    }
    manifest = build_contamination_manifest(
        config=config,
        clean=clean,
        forget=forget,
        plan=plan,
        hardware=hardware,
        output_checkpoint_dir=target_dir,
        runtime_seconds=runtime_seconds,
        optimizer_weight_decay=optimizer_weight_decay,
        software_versions=software_versions,
        artifact_fingerprints=artifact_fingerprints,
    )
    write_json(target_dir / "manifest.json", manifest, overwrite=overwrite)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create one fixed-exposure contaminated C_i checkpoint from a clean S_i checkpoint."
        )
    )
    parser.add_argument("--config", default="configs/pilot/base.yaml")
    parser.add_argument(
        "--clean-checkpoint",
        "--clean-checkpoint-dir",
        dest="clean_checkpoint",
        required=True,
        help="Clean S0/S50/S100 checkpoint directory (or its manifest.json).",
    )
    parser.add_argument(
        "--contamination-examples",
        "--forget-examples",
        dest="contamination_examples",
        required=True,
        help="Generated F contamination_examples.jsonl; its sibling manifest.json is required.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/contaminated",
        help="Root under which <run-id>/C0|C50|C100 is created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without loading weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit replacing an existing C_i artifact directory.",
    )
    args = parser.parse_args(argv)
    result = run_fixed_exposure_contamination(
        config_path=args.config,
        clean_checkpoint=args.clean_checkpoint,
        contamination_examples=args.contamination_examples,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
