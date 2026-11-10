"""Stage-2 pipeline for one pre-registered seed (docs/PREREGISTRATION_STAGE2.md).

Differences from the pilot pipeline, all required by the pre-registration:

* only Stage-2 jobs: no benchmark, no fixed-exposure arm, no unlearning revisions 1/2, no
  learning-rate calibration (contamination and unlearning read the pilot's pinned rates);
* matched-forgetting unlearning (revision 3) from every configured level;
* full evaluation (all suites, including MedMCQA full dev) of S_i, C_i and the f050, f100
  and final snapshots; retention only at f025 and f075;
* two gates that show procedural facts only (section 10): data and provenance checks,
  recall reached yes/no, damage flags. No retention, leakage or KL value is shown;
* no aggregation: results stay sealed until every seed is done (the cross-seed analysis
  unblinds them).
"""

from __future__ import annotations

from pathlib import Path

from unlearning.config.pins import PINNED_CONTAMINATION, PINNED_UNLEARNING, pinned_args
from unlearning.config.unlearning_revisions import CAP_SNAPSHOT, load_unlearning_revision
from unlearning.data.replay import replay_corpus_prepared, replay_dir
from unlearning.experiments.pipeline import METHODS, SUITES, Job, ModelRef, PilotPipeline
from unlearning.runtime.settings import REPO_ROOT
from unlearning.training.matched_contamination import matched_checkpoint_dir
from unlearning.training.matched_forgetting import ALL_TYPES_SNAPSHOT

REVISION3 = REPO_ROOT / "configs" / "pilot" / "unlearning" / "revision3.yaml"
FULL_POINTS = ("f050", "f100", "f_all", "final")  # f_all: every fact type closed its gap (6.1)
RETENTION_POINTS = ("f025", "f075")
RETENTION_SUITES = ("near", "far", "general", "external_medical")

STAGE2_GATES = {
    "stage2_data": "Stage-2 data check (procedural): F, evaluation sets, step budget, code version",
    "stage2_procedure": "Stage-2 procedural check before unlearning: recall reached, damage flags, provenance",
}


class Stage2Pipeline(PilotPipeline):
    gates = STAGE2_GATES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config.data_order != "seeded":
            raise ValueError("The Stage-2 pipeline needs a Stage-2 config (data_order: seeded)")
        self.unlearning_revisions = (load_unlearning_revision(REVISION3),)

    @property
    def revision(self):
        return self.unlearning_revisions[0]

    def unlearned_models(self, revision=None) -> list[ModelRef]:
        """Revision-3 snapshots f025 ... f100, plus f_all, plus final if the cap came first."""

        revision = revision or self.revision
        refs = []
        for contaminated in self.contaminated_names:
            for method in METHODS:
                trajectory = self.layout.unlearning_trajectory(
                    self.config, contaminated, method, condition=self.arm.name, revision_tag=revision.tag
                )
                for name in (*revision.matched.snapshot_names, ALL_TYPES_SNAPSHOT, CAP_SNAPSHOT):
                    snapshot = trajectory / name
                    refs.append(
                        ModelRef(
                            f"{self.arm.tag}_{revision.tag}_{contaminated}_{method}_{name}",
                            str(snapshot),
                            snapshot / "manifest.json",
                            "unlearned",
                            produced_by=trajectory / "run_manifest.json",
                            snapshot_name=name,
                        )
                    )
        return refs

    def _suite_jobs(self, group: str, models: list[ModelRef], *, full: bool) -> list[Job]:
        """Full: every suite, MedMCQA full dev included for C and U too. Otherwise retention suites only."""

        jobs = []
        generated = self.layout.generated
        for ref in models:
            for suite in SUITES:
                if not full and suite.name not in RETENTION_SUITES:
                    continue
                output = self.result_path(ref.label, suite.name)
                common = ["--model", ref.model, "--checkpoint-manifest", str(ref.manifest), "--output", str(output)]
                if suite.task is None:
                    command = self._python("evaluate_forgetting.py", "--dataset", str(generated / suite.dataset), *common)
                else:
                    command = self._python(
                        "evaluate_clean.py", "--dataset", str(generated / suite.dataset), "--task", suite.task, *common
                    )
                jobs.append(
                    Job(
                        id=f"eval__{ref.label}__{suite.name}",
                        group=group,
                        title=f"evaluate {ref.label} on {suite.name}",
                        kind="command",
                        command=command,
                        outputs=(output,),
                        done_check=(
                            (lambda output=output, ref=ref: output.is_file() or ref.not_produced())
                            if ref.produced_by is not None
                            else None
                        ),
                    )
                )
        return jobs

    def jobs(self) -> list[Job]:
        layout = self.layout
        arm_args = ("--arm", str(self.arm_path))
        revision_args = ["--unlearning-revision", str(self.revision.source_path)]
        jobs: list[Job] = [
            Job(
                id="preflight",
                group="data",
                title="prepare and verify data (MedMCQA, this seed's synthetic F, eval sets)",
                kind="command",
                command=self._python("preflight.py"),
                outputs=(layout.preflight_report,),
                offline=False,
            ),
            Job(id="stage2_data", group="data", title=STAGE2_GATES["stage2_data"], kind="gate", gate="stage2_data"),
            Job(
                id="specialize",
                group="specialization",
                title=f"clean specialization {' -> '.join(self.clean_names)} (seeded order)",
                kind="command",
                command=self._training("specialize_clean.py"),
                outputs=(layout.clean_root / layout.run_id / "run_manifest.json",),
            ),
        ]
        jobs += self._suite_jobs("clean_eval", self.clean_models(), full=True)
        if self.arm.replay is not None:
            jobs.append(
                Job(
                    id=f"prepare_replay_{self.arm.tag}",
                    group="matched_contamination",
                    title=f"prepare replay corpus ({self.arm.replay.dataset_config}, pinned revision)",
                    kind="command",
                    command=self._python("prepare_replay.py", *arm_args),
                    outputs=(replay_dir(layout.generated, self.arm) / "manifest.json",),
                    done_check=lambda: replay_corpus_prepared(self.config, self.arm, layout.generated),
                    offline=False,
                )
            )
        for clean_name, contaminated in zip(self.clean_names, self.contaminated_names):
            jobs.append(
                Job(
                    id=f"contaminate_{self.arm.tag}_{contaminated}",
                    group="matched_contamination",
                    title=f"matched contamination {clean_name} + F -> {self.arm.tag}_{contaminated} (pinned rate)",
                    kind="command",
                    command=self._training(
                        "contaminate_matched.py",
                        "--clean-checkpoint",
                        str(layout.clean_checkpoint(clean_name)),
                        *arm_args,
                        *pinned_args(PINNED_CONTAMINATION),
                    ),
                    outputs=(matched_checkpoint_dir(layout, self.arm, contaminated) / "manifest.json",),
                )
            )
        jobs += self._suite_jobs("matched_contaminated_eval", self.matched_contaminated_models(), full=True)
        jobs.append(
            Job(id="stage2_procedure", group="matched_contaminated_eval", title=STAGE2_GATES["stage2_procedure"], kind="gate", gate="stage2_procedure")
        )
        tag = self.revision.tag
        for contaminated in self.contaminated_names:
            for method in METHODS:
                trajectory = layout.unlearning_trajectory(
                    self.config, contaminated, method, condition=self.arm.name, revision_tag=tag
                )
                jobs.append(
                    Job(
                        id=f"unlearn_{self.arm.tag}_{tag}_{contaminated}_{method}",
                        group="unlearning",
                        title=f"{method.upper()} matched-forgetting unlearning from {self.arm.tag}_{contaminated} (pinned rate)",
                        kind="command",
                        command=self._training(
                            "unlearn.py",
                            "--contaminated-checkpoint-dir",
                            str(matched_checkpoint_dir(layout, self.arm, contaminated)),
                            "--method",
                            method,
                            *revision_args,
                            *pinned_args(PINNED_UNLEARNING),
                            "--continue-to-all-types",
                        ),
                        outputs=(trajectory / "run_manifest.json",),
                    )
                )
        unlearned = self.unlearned_models(self.revision)
        full = [ref for ref in unlearned if ref.snapshot_name in FULL_POINTS]
        retention = [ref for ref in unlearned if ref.snapshot_name in RETENTION_POINTS]
        jobs += self._suite_jobs("unlearned_eval", full, full=True)
        jobs += self._suite_jobs("unlearned_eval", retention, full=False)
        jobs += self._closeness_jobs("unlearned_eval", self.matched_contaminated_models() + full)
        return jobs


def is_stage2(config_path: Path) -> bool:
    from unlearning.config import load_config

    return load_config(config_path).data_order == "seeded"


PILOT_RUN_ID = (
    "pilot-medical-s0-s50-s100-qwen05b__qwen-qwen2-5-0-5b__seed42__synthetic-patient-f-v2__cfg2396728f114a"
)


def _yes(flag) -> str:
    return "yes" if flag else "NO"


def write_stage2_gate_report(pipeline: Stage2Pipeline, gate: str) -> Path:
    """Procedural facts only (pre-registration, section 10): no retention, leakage or KL values."""

    from unlearning.training.levels import specialization_levels
    from unlearning.utils.code_version import code_version
    from unlearning.utils.io import read_json, write_json  # noqa: F401
    from unlearning.runtime.run_log import utc_now

    layout = pipeline.layout
    lines = [
        f"# {STAGE2_GATES[gate]}",
        "",
        f"Run: `{layout.run_id}`  ",
        f"Written: {utc_now()}  ",
        "Stage-2 gates are blinded: they report procedure only, never an outcome.",
        "",
    ]
    if gate == "stage2_data":
        report = read_json(layout.preflight_report) if layout.preflight_report.is_file() else {}
        pilot_report_path = layout.root / pipeline.config.paths.generated_dir / PILOT_RUN_ID / "preflight_report.json"
        pilot = read_json(pilot_report_path) if pilot_report_path.is_file() else {}
        subsets, pilot_subsets = report.get("domain_subsets") or {}, pilot.get("domain_subsets") or {}
        forget = report.get("forget") or {}
        version = code_version()
        lines += [
            "## Seed and code",
            f"- seed {pipeline.config.seed}, data_order `{pipeline.config.data_order}`, F seed {pipeline.config.forget.seed}",
            f"- code commit `{version.get('git_commit')}`, uncommitted changes: {version.get('git_dirty')} (must be False)",
            "",
            "## Levels (steps on the 3,290-step budget)",
            "- " + ", ".join(f"{level.clean} at step {level.step}" for level in specialization_levels(pipeline.config)),
            "",
            "## Forget set F (fresh for this seed)",
            f"- records {forget.get('record_count')}, contamination examples {forget.get('contamination_example_count')}, "
            f"held-out probes {forget.get('eval_example_count')} (expected 120 / 960 / 720)",
            "- F terms were checked against every MedMCQA training text by the preflight (it fails on any overlap).",
            "",
            "## Evaluation sets",
            f"- near items {subsets.get('near_count')} (expected 434), far items {subsets.get('far_count')} (expected 486)",
            f"- far set identical to the pilot's: {_yes(subsets.get('far_fingerprint') and subsets.get('far_fingerprint') == pilot_subsets.get('far_fingerprint'))}",
            "",
            "## Check before approving",
            "- [ ] Code is committed (uncommitted changes False) and is the frozen Stage-2 commit.",
            "- [ ] Counts match the expected values above.",
        ]
    elif gate == "stage2_procedure":
        lines += [
            "## Matched contamination (per level)",
            "",
            "| model | recall target reached | within damage limits | pinned rate used | code committed |",
            "|---|---|---|---|---|",
        ]
        for name in pipeline.contaminated_names:
            path = matched_checkpoint_dir(layout, pipeline.arm, name) / "manifest.json"
            if not path.is_file():
                lines.append(f"| {pipeline.arm.tag}_{name} | missing | missing | missing | missing |")
                continue
            manifest = read_json(path)
            memorization, damage = manifest.get("memorization") or {}, manifest.get("damage") or {}
            pinned = (manifest.get("contamination") or {}).get("calibration_report_sha256") == PINNED_CONTAMINATION[1]
            dirty = ((manifest.get("manifest") or {}).get("software") or {}).get("code_version", {}).get("git_dirty")
            lines.append(
                f"| {pipeline.arm.tag}_{name} | {_yes(memorization.get('target_reached'))} | "
                f"{_yes(damage.get('within_limit'))} (flagged, kept) | {_yes(pinned)} | {_yes(dirty is False)} |"
            )
        lines += [
            "",
            "A level that missed the recall target is excluded from contrasts that need it (pre-registration,",
            "section 10); a damage-limit excess is kept and flagged. Neither is rerun.",
        ]
    path = pipeline.gate_report_path(gate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
