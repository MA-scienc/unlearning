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
import sys
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
from unlearning.runtime.hardware import (
    precheck_gpu_training,
    require_free_disk,
    require_gpu_training,
)
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.monitor import GpuSampler, SystemMonitor, ThermalGuard
from unlearning.runtime.profile import load_runtime_profile, micro_steps_per_rank
from unlearning.runtime.resume import (
    TRAINING_STATE_FILE,
    ResumeStore,
    begin_atomic_dir,
    build_training_state,
    commit_atomic_dir,
    latest_point,
    remove_stale_temporaries,
)
from unlearning.runtime.run_log import RunLog
from unlearning.runtime.settings import MachineSettings, default_config_path, default_runtime_path
from unlearning.runtime.signals import INTERRUPTED_EXIT_CODE, GracefulStop, StageInterrupted
from unlearning.runtime.stage import (
    STAGE_COMPLETE,
    STAGE_FRESH,
    ExecutionRecord,
    StageAlreadyStarted,
    prepare_stage,
    segment_details,
    verify_or_write_run_plan,
)
from unlearning.training.engine import (
    MicroBatch,
    TrainingLoop,
    apply_resume_state,
    create_accelerator,
    enable_memory_saving,
    split_into_micro_batches,
    supervised_token_count,
)
from unlearning.utils.env import software_versions
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
    execution: Mapping[str, Any] | None = None,
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
    if execution is not None:
        payload["execution"] = dict(execution)
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


class ContaminationObjective:
    """Token-mean completion-only CE over each step's full batch of 16 F examples.

    Step ``k`` uses the ``(k - 1) % steps_per_epoch``-th consecutive batch of the
    unshuffled JSONL order, exactly as the epoch loop it replaces. The loss is
    normalized by the supervised-token count of the whole 16-example batch, so
    splitting it into micro-batches leaves the objective unchanged.
    """

    name = "contamination"
    unit = "tokens"

    def __init__(
        self,
        torch,
        encoded_examples: Sequence[Mapping[str, Any]],
        *,
        plan: FixedExposurePlan,
        micro_batch_size: int,
        pad_token_id: int,
    ):
        self.torch = torch
        self.examples = list(encoded_examples)
        self.plan = plan
        self.micro = int(micro_batch_size)
        self.pad_token_id = int(pad_token_id)
        if plan.effective_batch_size % self.micro != 0:
            raise ContaminationProtocolError(
                f"micro-batch {self.micro} must divide the contamination batch {plan.effective_batch_size}"
            )
        self._normalizers: dict[int, int] = {}

    def _batch_index(self, step: int) -> int:
        if step <= 0 or step > self.plan.optimizer_steps:
            raise ValueError(f"contamination step {step} is outside 1..{self.plan.optimizer_steps}")
        return (step - 1) % self.plan.steps_per_epoch

    def _step_examples(self, step: int) -> list[Mapping[str, Any]]:
        start = self._batch_index(step) * self.plan.effective_batch_size
        return self.examples[start : start + self.plan.effective_batch_size]

    def micro_batches(self, step: int) -> list[MicroBatch]:
        torch = self.torch
        batches = []
        for chunk in split_into_micro_batches(self._step_examples(step), self.micro):
            collated = collate_completion_batch(chunk, pad_token_id=self.pad_token_id)
            batches.append(
                MicroBatch(
                    {
                        "input_ids": torch.tensor(collated["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(collated["attention_mask"], dtype=torch.long),
                        "labels": torch.tensor(collated["labels"], dtype=torch.long),
                    },
                    example_ids=tuple(collated["example_ids"]),
                )
            )
        return batches

    def normalizer(self, step: int, micro_batches) -> float:
        index = self._batch_index(step)
        if index not in self._normalizers:
            self._normalizers[index] = int(
                sum(int(supervised_token_count(batch.tensors["labels"])) for batch in micro_batches)
            )
        return float(self._normalizers[index])

    def loss_sum(self, model, batch: dict[str, Any]):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        return outputs.loss * supervised_token_count(batch["labels"]), {}


def _contamination_run_plan(
    *,
    config: PilotConfig,
    clean: CleanCheckpointReference,
    forget: ForgetExamplesReference,
    plan: FixedExposurePlan,
) -> dict[str, Any]:
    return {
        "schema_version": "contamination-run-plan-v1",
        "run_id": clean.run_id,
        "clean_checkpoint_id": clean.checkpoint_id,
        "clean_manifest_payload_fingerprint": clean.manifest_payload_fingerprint,
        "forget_examples_fingerprint": forget.examples_fingerprint,
        "forget_examples_file_sha256": forget.examples_file_sha256,
        "exposure_plan": asdict(plan),
        "contamination": asdict(config.contamination),
    }


def run_fixed_exposure_contamination(
    *,
    config_path: str | Path,
    clean_checkpoint: str | Path,
    contamination_examples: str | Path | None = None,
    output_dir: str | Path | None = None,
    runtime_path: str | Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create ``C_i`` from one clean ``S_i`` checkpoint under the frozen pilot recipe.

    ``output_dir`` is a root directory.  The deterministic destination is
    ``<output_dir>/<clean-run-id>/C0|C50|C100``.  Multiple C checkpoints may
    therefore share one root without sharing or mutating a run manifest.
    """

    settings = MachineSettings.from_env()
    config = load_config(config_path)
    validate_training_ready(config)
    validate_fixed_exposure_protocol(config)
    layout = StorageLayout.for_config(config, settings)
    clean = load_clean_checkpoint_reference(clean_checkpoint, config=config)
    forget = load_forget_examples_reference(
        contamination_examples or layout.contamination_examples,
        config=config,
    )
    plan = build_fixed_exposure_plan(
        example_count=len(forget.examples),
        epochs=config.contamination.epochs,
        effective_batch_size=config.contamination.effective_batch_size,
    )
    output_root = Path(output_dir) if output_dir else layout.contaminated_root
    target_dir = output_root / clean.run_id / clean.contaminated_name
    stage = f"contaminate_{clean.contaminated_name}"
    log_dir = layout.stage_dir(stage)
    hardware = precheck_gpu_training(settings, purpose="fixed-exposure contamination")
    checkpoint_id = CheckpointIdentity(
        clean.run_id,
        clean.contaminated_name,
        plan.optimizer_steps,
    ).checkpoint_id
    if dry_run:
        return {
            "dry_run": True,
            "output_checkpoint_dir": str(target_dir),
            "log_dir": str(log_dir),
            "checkpoint_id": checkpoint_id,
            "clean_checkpoint_id": clean.checkpoint_id,
            "forget_examples_fingerprint": forget.examples_fingerprint,
            "exposure_plan": asdict(plan),
            "hardware": hardware,
        }
    if target_dir.exists() and not target_dir.is_dir():
        raise ContaminationProtocolError(
            f"Contaminated checkpoint target is not a directory: {target_dir}"
        )
    lifecycle = prepare_stage(
        final_marker=target_dir / "manifest.json",
        paths=[target_dir, log_dir],
        resume=resume,
        overwrite=overwrite,
        stage=stage,
    )
    if lifecycle == STAGE_COMPLETE:
        print(f"[{stage}] already complete: {target_dir / 'manifest.json'}", flush=True)
        return read_json(target_dir / "manifest.json")
    require_gpu_training(hardware)
    profile = load_runtime_profile(runtime_path or settings.runtime_profile_path)

    accelerator = create_accelerator()
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    micro = profile.micro_batch.contamination
    micro_steps_per_rank(
        effective_batch=plan.effective_batch_size,
        micro_batch=micro,
        world_size=world_size,
        stage="contamination",
    )

    torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler, transformers = (
        _require_training_stack()
    )
    bnb, AdamW8bit = _require_bitsandbytes_optimizer()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = _load_tokenizer(clean, config, AutoTokenizer)
    encoded_examples = [
        tokenize_completion_example(
            example,
            tokenizer=tokenizer,
            max_length=config.contamination.sequence_length,
        )
        for example in forget.examples
    ]
    run_log = RunLog(
        log_dir,
        run_id=clean.run_id,
        stage=stage,
        total_steps=plan.optimizer_steps,
        enabled=is_main,
        console_every_steps=profile.console_every_steps,
    )
    execution = ExecutionRecord(log_dir / "execution.json", enabled=is_main)
    verify_or_write_run_plan(
        log_dir / "run_plan.json",
        _contamination_run_plan(config=config, clean=clean, forget=forget, plan=plan),
        fresh=lifecycle == STAGE_FRESH,
        is_main=is_main,
    )
    resume_store = ResumeStore(
        log_dir / "resume",
        keep_last=profile.resume_keep_last,
        every_minutes=profile.resume_every_minutes,
        enabled=is_main,
    )
    if is_main:
        stale_dirs = remove_stale_temporaries(target_dir.parent) + remove_stale_temporaries(
            resume_store.directory
        )
        for stale in stale_dirs:
            run_log.event("removed_incomplete_save", str(stale))
    accelerator.wait_for_everyone()
    point = latest_point(resume_store.points()) if lifecycle != STAGE_FRESH else None

    stop = GracefulStop().install()
    sampler = GpuSampler()
    system_monitor = SystemMonitor(
        run_log,
        interval=profile.system_sample_seconds,
        storage_root=settings.storage_root,
        sampler=sampler,
    ).start()
    segment_started = time.perf_counter()
    start_step = point.step if point else 0
    final_state = "failed"
    failure: str | None = None
    loop: TrainingLoop | None = None
    try:
        run_log.set_state("loading", message="loading model")
        if point is None:
            model = _load_model(clean, config, AutoModelForCausalLM, torch)
        else:
            model = AutoModelForCausalLM.from_pretrained(point.model_dir, torch_dtype=torch.bfloat16)
        enable_memory_saving(model, gradient_checkpointing=config.model.gradient_checkpointing)
        model.train()
        model = accelerator.prepare(model)
        optimizer_weight_decay = config.contamination.weight_decay
        optimizer = AdamW8bit(
            accelerator.unwrap_model(model).parameters(),
            lr=config.contamination.learning_rate,
            weight_decay=optimizer_weight_decay,
        )
        scheduler = get_scheduler(
            name=config.contamination.scheduler,
            optimizer=optimizer,
            num_warmup_steps=int(config.contamination.warmup_ratio * plan.optimizer_steps),
            num_training_steps=plan.optimizer_steps,
        )
        if point is not None:
            apply_resume_state(torch, point, optimizer=optimizer, scheduler=scheduler)
            run_log.event("resumed", f"from step {point.step} ({point.directory})")
        execution.begin(
            start_step=start_step,
            resumed_from={"step": point.step, "path": str(point.directory)} if point else None,
            details={
                **segment_details(hardware, profile, settings, world_size),
                "micro_batch_examples": micro,
            },
        )
        run_log.update(
            state="running",
            started_step=start_step,
            resume_count=max(0, len(execution.segments) - 1),
            micro_batch=micro,
            processes=world_size,
            source_clean_checkpoint=clean.name,
        )
        loop = TrainingLoop(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=ContaminationObjective(
                torch,
                encoded_examples,
                plan=plan,
                micro_batch_size=micro,
                pad_token_id=_pad_token_id(tokenizer),
            ),
            total_steps=plan.optimizer_steps,
            # The frozen contamination recipe has no gradient clipping; the
            # gradient norm is still measured and logged.
            max_grad_norm=None,
            run_log=run_log,
            resume_store=resume_store,
            stop=stop,
            settings=settings,
            thermal=ThermalGuard(
                run_log,
                sampler,
                pause_at_c=settings.gpu_pause_temp_c,
                resume_at_c=settings.gpu_resume_temp_c,
            ),
        )
        result = loop.run(start_step=start_step)
        if not result.completed:
            final_state = "interrupted"
            raise StageInterrupted(result.last_step, result.stop_reason)
        if loop.completed_step != plan.optimizer_steps:
            raise RuntimeError(
                f"Contamination executed {loop.completed_step} steps, expected {plan.optimizer_steps}"
            )

        run_log.set_state("saving", message=f"saving {clean.contaminated_name}")
        if is_main:
            require_free_disk(target_dir.parent, settings, action=f"saving {clean.contaminated_name}")
            ensure_dir(target_dir.parent)
            temporary = begin_atomic_dir(target_dir)
            accelerator.unwrap_model(model).save_pretrained(temporary / "model", safe_serialization=True)
            tokenizer.save_pretrained(temporary / "tokenizer")
            torch.save(
                build_training_state(
                    torch,
                    step=plan.optimizer_steps,
                    optimizer=optimizer,
                    scheduler=scheduler,
                ),
                temporary / TRAINING_STATE_FILE,
            )
            artifact_fingerprints = {
                "model_directory": fingerprint_directory(temporary / "model"),
                "tokenizer_directory": fingerprint_directory(temporary / "tokenizer"),
                "training_state_file": hash_file(temporary / TRAINING_STATE_FILE),
            }
            segment_runtime = time.perf_counter() - segment_started
            manifest = build_contamination_manifest(
                config=config,
                clean=clean,
                forget=forget,
                plan=plan,
                hardware=hardware,
                output_checkpoint_dir=target_dir,
                runtime_seconds=execution.total_runtime_seconds + segment_runtime,
                optimizer_weight_decay=optimizer_weight_decay,
                software_versions={
                    "torch": getattr(torch, "__version__", None),
                    "transformers": getattr(transformers, "__version__", None),
                    "bitsandbytes": getattr(bnb, "__version__", None),
                    **software_versions(),
                },
                artifact_fingerprints=artifact_fingerprints,
                execution={**execution.as_dict(), "runtime_profile": profile.as_dict()},
            )
            write_json(temporary / "manifest.json", manifest, overwrite=False)
            commit_atomic_dir(temporary, target_dir)
        accelerator.wait_for_everyone()
        final_state = "completed"
    except KeyboardInterrupt:
        final_state = "interrupted"
        raise
    except StageInterrupted:
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end_step = loop.completed_step if loop is not None else start_step
        execution.end(
            end_step=end_step,
            state=final_state,
            runtime_seconds=time.perf_counter() - segment_started,
            error=failure,
        )
        system_monitor.stop()
        stop.uninstall()
        if final_state != "completed":
            run_log.finish(final_state, message=failure or f"stopped after step {end_step}")

    if is_main:
        resume_store.clear()
        run_log.finish("completed", message=f"{clean.contaminated_name} saved to {target_dir}")
    accelerator.wait_for_everyone()
    return read_json(target_dir / "manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create one fixed-exposure contaminated C_i checkpoint from a clean S_i checkpoint."
        )
    )
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path(), help="Runtime profile YAML.")
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
        default=None,
        help=(
            "Generated F contamination_examples.jsonl (default: the preflight output); its "
            "sibling manifest.json is required."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Root under which <run-id>/C0|C50|C100 is created "
            "(default: <storage>/checkpoints/contaminated)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without loading weights.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue a partially completed run.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Set existing outputs aside (renamed *__invalidated_<time>) and start over.",
    )
    args = parser.parse_args(argv)
    try:
        result = run_fixed_exposure_contamination(
            config_path=args.config,
            clean_checkpoint=args.clean_checkpoint,
            contamination_examples=args.contamination_examples,
            output_dir=args.output_dir,
            runtime_path=args.runtime,
            dry_run=args.dry_run,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except StageInterrupted as stopped:
        print(f"[contaminate] {stopped}. Rerun with --resume to continue.", file=sys.stderr, flush=True)
        return INTERRUPTED_EXIT_CODE
    except StageAlreadyStarted as exists:
        print(f"[contaminate] {exists}", file=sys.stderr, flush=True)
        return 2
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
