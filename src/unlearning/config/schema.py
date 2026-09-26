from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _require_keys(section: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"{name} is missing required keys: {', '.join(missing)}")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    tokenizer_id: str
    revision: str | None
    dtype: str
    training_mode: str
    gradient_checkpointing: bool
    optimizer: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        _require_keys(
            raw,
            (
                "model_id",
                "tokenizer_id",
                "revision",
                "dtype",
                "training_mode",
                "gradient_checkpointing",
                "optimizer",
            ),
            "model",
        )
        return cls(**raw)

    def validate(self) -> None:
        if not self.model_id:
            raise ConfigError("model.model_id must be non-empty")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ConfigError("model.dtype must be one of bf16, fp16, fp32")
        if self.training_mode != "full_parameter":
            raise ConfigError("pilot protocol currently requires full_parameter training_mode")
        if self.revision is not None and len(self.revision) != 40:
            raise ConfigError("model.revision must be a 40-character commit SHA when provided")


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    processed_dir: Path
    generated_dir: Path
    manifest_dir: Path
    result_dir: Path
    checkpoint_dir: Path

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PathsConfig":
        _require_keys(
            raw,
            (
                "data_dir",
                "processed_dir",
                "generated_dir",
                "manifest_dir",
                "result_dir",
                "checkpoint_dir",
            ),
            "paths",
        )
        return cls(**{key: Path(value) for key, value in raw.items()})


@dataclass(frozen=True)
class SpecializationConfig:
    dataset_id: str
    dataset_revision: str | None
    source_url: str
    source_license: str
    schema_version: str
    train_split: str
    dev_split: str
    test_split: str
    context_length: int
    epochs: int
    effective_batch_sequences: int
    micro_batch_sequences: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    scheduler: str
    max_grad_norm: float
    drop_last_batches: bool
    checkpoint_fractions: tuple[float, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SpecializationConfig":
        _require_keys(
            raw,
            (
                "dataset_id",
                "dataset_revision",
                "source_url",
                "source_license",
                "schema_version",
                "train_split",
                "dev_split",
                "test_split",
                "context_length",
                "epochs",
                "effective_batch_sequences",
                "micro_batch_sequences",
                "learning_rate",
                "weight_decay",
                "warmup_ratio",
                "scheduler",
                "max_grad_norm",
                "drop_last_batches",
                "checkpoint_fractions",
            ),
            "specialization",
        )
        data = dict(raw)
        data["checkpoint_fractions"] = tuple(float(x) for x in data["checkpoint_fractions"])
        return cls(**data)

    def validate(self) -> None:
        if self.context_length <= 0:
            raise ConfigError("specialization.context_length must be positive")
        if self.epochs != 3:
            raise ConfigError("pilot protocol fixes specialization.epochs at 3")
        if self.effective_batch_sequences <= 0:
            raise ConfigError("specialization.effective_batch_sequences must be positive")
        if self.micro_batch_sequences <= 0:
            raise ConfigError("specialization.micro_batch_sequences must be positive")
        if self.effective_batch_sequences % self.micro_batch_sequences != 0:
            raise ConfigError(
                "specialization.effective_batch_sequences must be divisible by "
                "specialization.micro_batch_sequences"
            )
        if self.learning_rate <= 0:
            raise ConfigError("specialization.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ConfigError("specialization.weight_decay must be non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ConfigError("specialization.warmup_ratio must be in [0, 1)")
        if self.scheduler not in {"linear", "cosine"}:
            raise ConfigError("specialization.scheduler must be one of linear, cosine")
        if self.max_grad_norm <= 0:
            raise ConfigError("specialization.max_grad_norm must be positive")
        if len(self.checkpoint_fractions) != len(set(self.checkpoint_fractions)):
            raise ConfigError("specialization.checkpoint_fractions must be unique")
        if self.checkpoint_fractions != (0.0, 0.5, 1.0):
            raise ConfigError("pilot protocol requires checkpoint_fractions [0.0, 0.5, 1.0]")
        if len({self.train_split, self.dev_split, self.test_split}) != 3:
            raise ConfigError("MedMCQA train/dev/test split names must be distinct")
        if self.dataset_revision is not None and len(self.dataset_revision) != 40:
            raise ConfigError(
                "specialization.dataset_revision must be a 40-character commit SHA when provided"
            )


@dataclass(frozen=True)
class ForgetConfig:
    schema_version: str
    seed: int
    record_count: int
    contamination_templates_per_record: int
    eval_templates_per_record: int
    mc_decoy_count: int
    source_license: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ForgetConfig":
        _require_keys(
            raw,
            (
                "schema_version",
                "seed",
                "record_count",
                "contamination_templates_per_record",
                "eval_templates_per_record",
                "mc_decoy_count",
                "source_license",
            ),
            "forget",
        )
        return cls(**raw)

    def validate(self) -> None:
        if self.record_count != 120:
            raise ConfigError("pilot protocol requires forget.record_count == 120")
        if self.contamination_templates_per_record != 8:
            raise ConfigError("pilot protocol requires 8 contamination templates per record")
        if self.eval_templates_per_record < 1:
            raise ConfigError("forget.eval_templates_per_record must be positive")
        if self.eval_templates_per_record != 6:
            raise ConfigError("pilot protocol requires 6 forget-evaluation templates per record")
        if self.mc_decoy_count <= 0:
            raise ConfigError("forget.mc_decoy_count must be positive")


@dataclass(frozen=True)
class ContaminationConfig:
    condition: str
    learning_rate: float
    epochs: int
    effective_batch_size: int
    sequence_length: int
    warmup_ratio: float
    scheduler: str
    optimizer: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ContaminationConfig":
        _require_keys(
            raw,
            (
                "condition",
                "learning_rate",
                "epochs",
                "effective_batch_size",
                "sequence_length",
                "warmup_ratio",
                "scheduler",
                "optimizer",
            ),
            "contamination",
        )
        return cls(**raw)

    def validate(self) -> None:
        if self.condition != "fixed_exposure":
            raise ConfigError("pilot protocol allows only fixed_exposure contamination")
        if self.epochs != 10:
            raise ConfigError("initial pilot fixes contamination.epochs at 10")
        if self.learning_rate <= 0:
            raise ConfigError("contamination.learning_rate must be positive")
        if self.effective_batch_size <= 0:
            raise ConfigError("contamination.effective_batch_size must be positive")
        if self.sequence_length <= 0:
            raise ConfigError("contamination.sequence_length must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ConfigError("contamination.warmup_ratio must be in [0, 1)")


@dataclass(frozen=True)
class MethodConfig:
    learning_rate: float
    steps: int
    effective_batch_size: int
    sequence_length: int
    gradient_clip_norm: float
    beta: float | None = None
    reference_model: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MethodConfig":
        return cls(**raw)

    def validate(self, name: str) -> None:
        if self.learning_rate <= 0:
            raise ConfigError(f"unlearning.methods.{name}.learning_rate must be positive")
        if self.steps != 400:
            raise ConfigError(f"unlearning.methods.{name}.steps must be 400")
        if self.effective_batch_size <= 0:
            raise ConfigError(f"unlearning.methods.{name}.effective_batch_size must be positive")
        if self.sequence_length <= 0:
            raise ConfigError(f"unlearning.methods.{name}.sequence_length must be positive")
        if self.gradient_clip_norm <= 0:
            raise ConfigError(f"unlearning.methods.{name}.gradient_clip_norm must be positive")
        if name == "npo":
            if self.beta is None or self.beta <= 0:
                raise ConfigError("NPO beta must be positive")
            if self.reference_model != "contaminated_checkpoint":
                raise ConfigError("NPO reference_model must be contaminated_checkpoint")


@dataclass(frozen=True)
class UnlearningConfig:
    snapshot_steps: tuple[int, ...]
    retain_loss_weight: float
    methods: dict[str, MethodConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "UnlearningConfig":
        _require_keys(raw, ("snapshot_steps", "retain_loss_weight", "methods"), "unlearning")
        methods = {name: MethodConfig.from_dict(value) for name, value in raw["methods"].items()}
        return cls(
            snapshot_steps=tuple(int(x) for x in raw["snapshot_steps"]),
            retain_loss_weight=float(raw["retain_loss_weight"]),
            methods=methods,
        )

    def validate(self) -> None:
        if len(self.snapshot_steps) != len(set(self.snapshot_steps)):
            raise ConfigError("unlearning.snapshot_steps must be unique")
        if 0 not in self.snapshot_steps:
            raise ConfigError("unlearning.snapshot_steps must include step 0")
        if self.snapshot_steps != tuple(sorted(self.snapshot_steps)):
            raise ConfigError("unlearning.snapshot_steps must be sorted ascending")
        if self.snapshot_steps != (0, 50, 100, 200, 400):
            raise ConfigError("primary analysis requires snapshot_steps [0, 50, 100, 200, 400]")
        if self.retain_loss_weight != 0.0:
            raise ConfigError("pilot protocol requires retain_loss_weight == 0.0")
        if set(self.methods) != {"ga", "npo"}:
            raise ConfigError("pilot protocol requires exactly GA and NPO")
        for name, method in self.methods.items():
            method.validate(name)
            if max(self.snapshot_steps) != method.steps:
                raise ConfigError(
                    f"maximum snapshot step must equal unlearning.methods.{name}.steps"
                )


@dataclass(frozen=True)
class EvaluationConfig:
    pubmedqa_dataset_id: str
    pubmedqa_revision: str | None
    pubmedqa_subset: str
    pubmedqa_split: str
    pubmedqa_source_url: str
    pubmedqa_license: str
    mmlu_cf_dataset_id: str
    mmlu_cf_revision: str | None
    mmlu_cf_split: str
    mmlu_cf_source_url: str
    mmlu_cf_license: str
    mmlu_cf_subset_size: int
    mmlu_cf_seed: int
    near_subjects: tuple[str, ...]
    near_topic_keywords: tuple[str, ...]
    far_subjects: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvaluationConfig":
        _require_keys(
            raw,
            (
                "pubmedqa_dataset_id",
                "pubmedqa_revision",
                "pubmedqa_subset",
                "pubmedqa_split",
                "pubmedqa_source_url",
                "pubmedqa_license",
                "mmlu_cf_dataset_id",
                "mmlu_cf_revision",
                "mmlu_cf_split",
                "mmlu_cf_source_url",
                "mmlu_cf_license",
                "mmlu_cf_subset_size",
                "mmlu_cf_seed",
                "near_subjects",
                "near_topic_keywords",
                "far_subjects",
            ),
            "evaluation",
        )
        data = dict(raw)
        data["near_subjects"] = tuple(data["near_subjects"])
        data["near_topic_keywords"] = tuple(data["near_topic_keywords"])
        data["far_subjects"] = tuple(data["far_subjects"])
        return cls(**data)

    def validate(self) -> None:
        if self.mmlu_cf_subset_size <= 0:
            raise ConfigError("evaluation.mmlu_cf_subset_size must be positive")
        if not self.near_topic_keywords:
            raise ConfigError("evaluation.near_topic_keywords must not be empty")
        if not self.far_subjects:
            raise ConfigError("evaluation.far_subjects must not be empty")
        for field_name in ("pubmedqa_revision", "mmlu_cf_revision"):
            value = getattr(self, field_name)
            if value is not None and len(value) != 40:
                raise ConfigError(f"evaluation.{field_name} must be a 40-character commit SHA")


@dataclass(frozen=True)
class PilotConfig:
    experiment_id: str
    seed: int
    model: ModelConfig
    paths: PathsConfig
    specialization: SpecializationConfig
    forget: ForgetConfig
    contamination: ContaminationConfig
    unlearning: UnlearningConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PilotConfig":
        _require_keys(
            raw,
            (
                "experiment_id",
                "seed",
                "model",
                "paths",
                "specialization",
                "forget",
                "contamination",
                "unlearning",
                "evaluation",
            ),
            "root",
        )
        config = cls(
            experiment_id=raw["experiment_id"],
            seed=int(raw["seed"]),
            model=ModelConfig.from_dict(raw["model"]),
            paths=PathsConfig.from_dict(raw["paths"]),
            specialization=SpecializationConfig.from_dict(raw["specialization"]),
            forget=ForgetConfig.from_dict(raw["forget"]),
            contamination=ContaminationConfig.from_dict(raw["contamination"]),
            unlearning=UnlearningConfig.from_dict(raw["unlearning"]),
            evaluation=EvaluationConfig.from_dict(raw["evaluation"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.experiment_id:
            raise ConfigError("experiment_id must be non-empty")
        self.model.validate()
        self.specialization.validate()
        self.forget.validate()
        self.contamination.validate()
        self.unlearning.validate()
        self.evaluation.validate()


def load_config(path: str | Path) -> PilotConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Config {config_path} did not parse to a mapping")
    return PilotConfig.from_dict(raw)


def validate_training_ready(config: PilotConfig) -> None:
    """Validate requirements that must hold before launching any training command."""
    if not config.model.revision:
        raise ConfigError("training requires pinned model.revision")
    if not config.specialization.dataset_revision:
        raise ConfigError("training requires pinned specialization.dataset_revision")
