"""One place that decides where every artifact lives.

Layout under ``UNLEARNING_STORAGE_ROOT`` (default: the repo folder)::

    hf_cache/                         downloaded base models and raw datasets (HF_HOME)
    data/generated/<run_id>/          preflight output: MedMCQA splits, synthetic F, eval sets
    checkpoints/clean/<run_id>/       S0_reference, S50, S100 (permanent scientific models)
    checkpoints/contaminated/<run_id>/C0, C50, C100
    checkpoints/unlearned/<trajectory_id>/step0000 ... step0400
    runs/<run_id>/<stage>/            live status, metrics, events, resume checkpoints
    runs/<run_id>/pipeline/           pipeline status, job logs, review-gate reports
    results/raw/<run_id>/<model>/     one JSON per evaluation suite (measured values)
    results/processed/<run_id>/       tidy tables built only from results/raw
    results/figures/<run_id>/         figures built only from processed tables / logs

Scientific checkpoints are permanent and fingerprinted. Everything in ``runs/``
is operational: resume checkpoints there can be deleted once a stage completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from unlearning.config import PilotConfig
from unlearning.experiments.naming import experiment_run_id, stage_run_name
from unlearning.runtime.settings import MachineSettings

CLEAN_NAMES = ("S0", "S50", "S100")
CONTAMINATED_FOR_CLEAN = {"S0": "C0", "S50": "C50", "S100": "C100"}
FRACTION_FOR_CONTAMINATED = {"C0": 0.0, "C50": 0.5, "C100": 1.0}


@dataclass(frozen=True)
class StorageLayout:
    root: Path
    run_id: str
    generated_dir_name: Path
    checkpoint_dir_name: Path
    result_dir_name: Path

    @classmethod
    def for_config(cls, config: PilotConfig, settings: MachineSettings | None = None) -> "StorageLayout":
        settings = settings or MachineSettings.from_env()
        return cls(
            root=settings.storage_root,
            run_id=experiment_run_id(config),
            generated_dir_name=config.paths.generated_dir,
            checkpoint_dir_name=config.paths.checkpoint_dir,
            result_dir_name=config.paths.result_dir,
        )

    # --- inputs -------------------------------------------------------------
    @property
    def generated(self) -> Path:
        return self.root / self.generated_dir_name / self.run_id

    @property
    def preflight_report(self) -> Path:
        return self.generated / "preflight_report.json"

    @property
    def forget_dir(self) -> Path:
        return self.generated / "forget"

    @property
    def contamination_examples(self) -> Path:
        return self.forget_dir / "contamination_examples.jsonl"

    # --- scientific checkpoints --------------------------------------------
    @property
    def clean_root(self) -> Path:
        return self.root / self.checkpoint_dir_name / "clean"

    @property
    def contaminated_root(self) -> Path:
        return self.root / self.checkpoint_dir_name / "contaminated"

    @property
    def unlearned_root(self) -> Path:
        return self.root / self.checkpoint_dir_name / "unlearned"

    def clean_checkpoint(self, name: str) -> Path:
        if name not in CLEAN_NAMES:
            raise ValueError(f"Unknown clean checkpoint {name!r}")
        folder = "S0_reference" if name == "S0" else name
        return self.clean_root / self.run_id / folder

    def contaminated_checkpoint(self, name: str) -> Path:
        if name not in FRACTION_FOR_CONTAMINATED:
            raise ValueError(f"Unknown contaminated checkpoint {name!r}")
        return self.contaminated_root / self.run_id / name

    def unlearning_trajectory(self, config: PilotConfig, contaminated: str, method: str) -> Path:
        trajectory_id = stage_run_name(
            config,
            "unlearned",
            checkpoint_fraction=FRACTION_FOR_CONTAMINATED[contaminated],
            method=method,
            condition=config.contamination.condition,
        )
        return self.unlearned_root / trajectory_id

    # --- operational --------------------------------------------------------
    @property
    def runs(self) -> Path:
        return self.root / "runs" / self.run_id

    def stage_dir(self, stage: str) -> Path:
        return self.runs / stage

    @property
    def pipeline_dir(self) -> Path:
        return self.runs / "pipeline"

    # --- results ------------------------------------------------------------
    @property
    def raw_results(self) -> Path:
        return self.root / self.result_dir_name / "raw" / self.run_id

    @property
    def processed_results(self) -> Path:
        return self.root / self.result_dir_name / "processed" / self.run_id

    @property
    def figures(self) -> Path:
        return self.root / self.result_dir_name / "figures" / self.run_id


def stage_label_for_contamination(clean_name: str) -> str:
    return f"contaminate_{CONTAMINATED_FOR_CLEAN[clean_name]}"


def stage_label_for_unlearning(contaminated_name: str, method: str) -> str:
    return f"unlearn_{contaminated_name}_{method}"
