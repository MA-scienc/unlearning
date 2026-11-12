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
from unlearning.evaluation.nonf_prompts import OUTPUT as NONF_OUTPUT
from unlearning.experiments.pipeline import METHODS, SUITES, Job, ModelRef, PilotPipeline
from unlearning.runtime.pruning import WEIGHTS_MARKER_FILE
from unlearning.runtime.settings import REPO_ROOT
from unlearning.training.matched_contamination import matched_checkpoint_dir
from unlearning.data.control_corpus import CORPUS_DIR as CONTROL_CORPUS_DIR
from unlearning.training.levels import clean_parent_name, control_levels, specialization_levels
from unlearning.training.specialize_clean import CONTROL_MANIFEST
from unlearning.training.matched_forgetting import ALL_TYPES_SNAPSHOT
from unlearning.training.recovery import TREATMENTS as RECOVERY_TREATMENTS

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

    def _nonf_closeness_jobs(self, group: str, models: list[ModelRef]) -> list[Job]:
        """KL(S_i || model) at answer start on prompts unrelated to F (the H3 control, 6.3)."""

        clean = {ref.label: ref for ref in self.clean_models()}
        dataset = self.layout.generated / NONF_OUTPUT
        jobs = []
        for ref in models:
            parent = clean[clean_parent_name(ref.label)]
            output = self.result_path(ref.label, "closeness_nonf")
            jobs.append(
                Job(
                    id=f"eval__{ref.label}__closeness_nonf",
                    group=group,
                    title=f"closeness of {ref.label} to {parent.label} on non-F prompts",
                    kind="command",
                    command=self._python(
                        "evaluate_closeness.py", "--dataset", str(dataset), "--model", ref.model,
                        "--checkpoint-manifest", str(ref.manifest), "--clean-model", parent.model,
                        "--clean-manifest", str(parent.manifest), "--output", str(output),
                    ),
                    outputs=(output,),
                    done_check=lambda output=output, ref=ref: output.is_file() or ref.not_produced(),
                )
            )
        return jobs

    def _recovery_jobs(self, group: str, models: list[ModelRef]) -> list[Job]:
        """Relearning and benign medical training, then leakage (H4, 6.4); weights are discarded."""

        jobs = []
        for ref in models:
            for treatment in RECOVERY_TREATMENTS:
                output = self.result_path(ref.label, f"recovery_{treatment}")
                jobs.append(
                    Job(
                        id=f"recover__{ref.label}__{treatment}",
                        group=group,
                        title=f"{treatment} recovery of {ref.label}",
                        kind="command",
                        command=self._training(
                            "recover.py", "--model", ref.model, "--checkpoint-manifest", str(ref.manifest),
                            "--label", ref.label, "--treatment", treatment,
                            "--baseline-result", str(self.result_path(ref.label, "forgetting")), "--output", str(output),
                        ),
                        outputs=(output,),
                        done_check=lambda output=output, ref=ref: output.is_file() or ref.not_produced(),
                    )
                )
        return jobs

    def _run_storage_jobs(self, trajectory: Path, contaminated: str, method: str) -> list[Job]:
        name = trajectory.name
        jobs = [
            Job(
                id=f"prune_state__{contaminated}_{method}",
                group="unlearning",
                title=f"delete resume states of the finished {contaminated} {method.upper()} run",
                kind="command",
                command=self._python("prune_training_states.py", "--apply", "--trajectory", name),
                done_check=lambda trajectory=trajectory: (trajectory / "run_manifest.json").is_file()
                and not any(trajectory.glob("*/training_state.pt")),
            )
        ]
        refs = [
            ref for ref in self.unlearned_models(self.revision)
            if ref.snapshot_name in RETENTION_POINTS and Path(ref.model).parent == trajectory
        ]
        jobs += self._suite_jobs("unlearning", refs, full=False)
        for ref in refs:
            results = [str(self.result_path(ref.label, suite)) for suite in RETENTION_SUITES]
            jobs.append(
                Job(
                    id=f"prune_weights__{ref.label}",
                    group="unlearning",
                    title=f"delete weights of evaluated retention-only snapshot {ref.label}",
                    kind="command",
                    command=self._python("prune_snapshot_weights.py", "--snapshot", ref.model, "--require", *results),
                    done_check=lambda ref=ref: (Path(ref.model) / WEIGHTS_MARKER_FILE).is_file() or ref.not_produced(),
                )
            )
        return jobs

    def _real_facts_jobs(self, group: str, models: list[ModelRef]) -> list[Job]:
        """True general facts that F contains (pre-registration 6.3), scored on S_i, C_i and U."""

        records = self.layout.generated / "forget" / "records.jsonl"
        jobs = []
        for ref in models:
            output = self.result_path(ref.label, "real_facts")
            jobs.append(
                Job(
                    id=f"eval__{ref.label}__real_facts",
                    group=group,
                    title=f"real-facts probe on {ref.label}",
                    kind="command",
                    command=self._python(
                        "evaluate_real_facts.py", "--records", str(records), "--model", ref.model,
                        "--checkpoint-manifest", str(ref.manifest), "--output", str(output),
                    ),
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
            Job(
                id="prepare_nonf_prompts",
                group="data",
                title="non-F prompts for the KL control (near stems + 360 general stems)",
                kind="command",
                command=self._python("prepare_nonf_prompts.py"),
                outputs=(layout.generated / NONF_OUTPUT,),
            ),
            Job(
                id="prepare_control_corpus",
                group="data",
                title="format-matched control corpus (AQuA-RAT, pinned revision) for the G levels",
                kind="command",
                command=self._python("prepare_control_corpus.py"),
                outputs=(layout.generated / CONTROL_CORPUS_DIR / "manifest.json",),
                offline=False,  # first download of the pinned split
            ),
            Job(id="stage2_data", group="data", title=STAGE2_GATES["stage2_data"], kind="gate", gate="stage2_data"),
            Job(
                id="specialize",
                group="specialization",
                title=f"clean specialization {' -> '.join(level.clean for level in specialization_levels(self.config))} (seeded order)",
                kind="command",
                command=self._training("specialize_clean.py"),
                outputs=(layout.clean_root / layout.run_id / "run_manifest.json",),
            ),
            Job(
                id="specialize_control",
                group="specialization",
                title=f"format-matched control {', '.join(level.clean for level in control_levels(self.config))} "
                "(same schedule, stopped at the matching step)",
                kind="command",
                command=self._training("specialize_control.py"),
                outputs=(layout.clean_root / layout.run_id / CONTROL_MANIFEST,),
            ),
        ]
        jobs += self._suite_jobs("clean_eval", self.clean_models(), full=True)
        jobs += self._real_facts_jobs("clean_eval", self.clean_models())
        # Manipulation checks (section 7): what each level actually learned, beyond the template.
        for ref in self.clean_models():
            output = self.result_path(ref.label, "manipulation")
            jobs.append(
                Job(
                    id=f"eval__{ref.label}__manipulation",
                    group="clean_eval",
                    title=f"manipulation checks on {ref.label} (content, format, domain-text losses)",
                    kind="command",
                    command=self._python(
                        "evaluate_manipulation.py", "--model", ref.model,
                        "--checkpoint-manifest", str(ref.manifest), "--output", str(output),
                    ),
                    outputs=(output,),
                    offline=False,  # first download of the pinned AQuA-RAT test and WikiText-103 test splits
                )
            )
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
        jobs += self._real_facts_jobs("matched_contaminated_eval", self.matched_contaminated_models())
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
                # Storage (pre-registration D6, section 9): right after each run, drop its resume
                # states, evaluate f025/f075 (retention only) and delete their weights, so only one
                # run's extra snapshots are ever on disk.
                jobs += self._run_storage_jobs(trajectory, contaminated, method)
        unlearned = self.unlearned_models(self.revision)
        full = [ref for ref in unlearned if ref.snapshot_name in FULL_POINTS]
        jobs += self._suite_jobs("unlearned_eval", full, full=True)
        jobs += self._closeness_jobs("unlearned_eval", self.matched_contaminated_models() + full)
        jobs += self._real_facts_jobs("unlearned_eval", full)
        jobs += self._nonf_closeness_jobs("unlearned_eval", self.matched_contaminated_models() + full)
        # Recovery last: it needs every model's forgetting result as its "before".
        f100 = [ref for ref in unlearned if ref.snapshot_name == "f100"]
        jobs += self._recovery_jobs("recovery", self.clean_models() + self.matched_contaminated_models() + f100)
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
        # The pilot's report lives under the repository's own storage; a Stage-2 run may use another root.
        candidates = [root / pipeline.config.paths.generated_dir / PILOT_RUN_ID / "preflight_report.json"
                      for root in (layout.root, REPO_ROOT)]
        pilot_report_path = next((path for path in candidates if path.is_file()), None)
        pilot = read_json(pilot_report_path) if pilot_report_path else {}
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
            "- control: " + ", ".join(f"{level.clean} at step {level.step} (from S0, control corpus)" for level in control_levels(pipeline.config)),
            "",
            "## Forget set F (fresh for this seed)",
            f"- records {forget.get('record_count')}, contamination examples {forget.get('contamination_example_count')}, "
            f"held-out probes {forget.get('eval_example_count')} (expected 120 / 960 / 720)",
            "- F terms were checked against every MedMCQA training text by the preflight (it fails on any overlap).",
            "",
            "## Evaluation sets",
            f"- near items {subsets.get('near_count')} (expected 434), far items {subsets.get('far_count')} (expected 486)",
            "- far set identical to the pilot's: " + (
                _yes(subsets.get("far_fingerprint") == pilot_subsets.get("far_fingerprint"))
                if pilot_subsets.get("far_fingerprint") else "pilot report not found"
            ),
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
