"""The whole pilot as one resumable, gated job list.

    python scripts/pilot/run_pipeline.py status     # what is done, what is next
    python scripts/pilot/run_pipeline.py run        # run until the next review gate
    python scripts/pilot/run_pipeline.py approve gate2_specialization --note "..."

Every job has deterministic outputs; a job whose outputs exist is skipped, and
training jobs are always launched with ``--resume``, so after a crash, reboot or
Ctrl+C the same ``run`` command continues where the work stopped.

Review gates stop the pipeline until you approve them. Each gate writes a short
report of *measured* values (``runs/<run_id>/pipeline/gates/<gate>.md``) plus the
checks that were specified before any results existed. The pipeline never
decides a scientific question itself.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from unlearning.config import PilotConfig, load_config
from unlearning.config.arms import load_matched_memorization_arm
from unlearning.evaluation.forgetting import rescore_forgetting_payload
from unlearning.training.matched_contamination import (
    DEFAULT_ARM_PATH,
    FINAL_CALIBRATION_STATES,
    calibration_report_path,
    matched_checkpoint_dir,
)
from unlearning.runtime.layout import (
    CONTAMINATED_FOR_CLEAN,
    StorageLayout,
)
from unlearning.runtime.profile import RuntimeProfile, load_runtime_profile
from unlearning.runtime.run_log import format_duration, utc_now
from unlearning.runtime.settings import (
    REPO_ROOT,
    MachineSettings,
    default_config_path,
    default_runtime_path,
)
from unlearning.utils.io import append_jsonl, ensure_dir, read_json, write_json

SCRIPTS = REPO_ROOT / "scripts" / "pilot"
METHODS = ("ga", "npo")
CLEAN = ("S0", "S50", "S100")
CONTAMINATED = ("C0", "C50", "C100")

GATES = {
    "gate0_data": "Data review: preflight artifacts, step budget, F separation",
    "gate1_hardware": "Hardware review: benchmark memory, speed, resume check, projected time",
    "gate2_specialization": "Specialization review: did domain performance move from S0 to S100?",
    "gate_calibration": "Calibration review: which contamination learning rate memorizes F without collateral damage?",
    "gate3_memorization": "Memorization review: is F memorized, and matched, in MM_C0/MM_C50/MM_C100?",
}


@dataclass(frozen=True)
class Suite:
    name: str
    task: str | None  # None = target-forgetting evaluation
    dataset: str  # relative to the preflight output directory
    clean_only: bool = False


SUITES = (
    Suite("forgetting", None, "forget/forget_eval_examples.jsonl"),
    Suite("near", "medmcqa", "medmcqa_domain_subsets/near_domain_dev.jsonl"),
    Suite("far", "medmcqa", "medmcqa_domain_subsets/far_domain_dev.jsonl"),
    Suite("general", "mmlu_cf", "mmlu_cf/mmlu_cf_eval_subset.jsonl"),
    Suite("external_medical", "pubmedqa", "pubmedqa/pubmedqa_eval.jsonl"),
    # Full labeled MedMCQA dev split: the separate measure of actual specialization.
    Suite("domain_full", "medmcqa", "medmcqa/dev_eval.jsonl", clean_only=True),
)


@dataclass
class Job:
    id: str
    group: str
    title: str
    kind: str  # "command" or "gate"
    command: list[str] = field(default_factory=list)
    outputs: tuple[Path, ...] = ()
    gate: str | None = None
    always_run: bool = False
    # Optional custom completion test (e.g. a report whose status must be final).
    done_check: Callable[[], bool] | None = None
    # Jobs after preflight + benchmark need nothing from the network: everything
    # is in the local cache. Offline mode avoids Hub rate limits and outages.
    offline: bool = True

    def done(self, approvals: dict[str, Any]) -> bool:
        if self.kind == "gate":
            return self.gate in approvals
        if self.always_run:
            return False
        if self.done_check is not None:
            return self.done_check()
        return bool(self.outputs) and all(path.is_file() for path in self.outputs)


@dataclass(frozen=True)
class ModelRef:
    label: str
    model: str
    manifest: Path
    family: str


class PilotPipeline:
    def __init__(
        self,
        *,
        config_path: Path,
        runtime_path: Path,
        settings: MachineSettings | None = None,
        arm_path: Path = DEFAULT_ARM_PATH,
    ):
        self.settings = settings or MachineSettings.from_env()
        self.arm_path = Path(arm_path).resolve()
        self.arm = load_matched_memorization_arm(self.arm_path)
        self.config_path = Path(config_path).resolve()
        self.runtime_path = Path(runtime_path).resolve()
        self.config: PilotConfig = load_config(self.config_path)
        self.profile: RuntimeProfile = load_runtime_profile(self.runtime_path)
        self.layout = StorageLayout.for_config(self.config, self.settings)
        self.python = sys.executable

    # --- paths -------------------------------------------------------------
    @property
    def pipeline_dir(self) -> Path:
        return self.layout.pipeline_dir

    @property
    def gates_path(self) -> Path:
        return self.pipeline_dir / "gates.json"

    def approvals(self) -> dict[str, Any]:
        return read_json(self.gates_path) if self.gates_path.is_file() else {}

    def gate_report_path(self, gate: str) -> Path:
        return self.pipeline_dir / "gates" / f"{gate}.md"

    def result_path(self, label: str, suite: str) -> Path:
        return self.layout.raw_results / label / f"{suite}.json"

    # --- commands ----------------------------------------------------------
    def _common(self) -> list[str]:
        return ["--config", str(self.config_path)]

    def _python(self, script: str, *args: str) -> list[str]:
        return [self.python, str(SCRIPTS / script), *self._common(), *args]

    def _training(self, script: str, *args: str) -> list[str]:
        if self.profile.launcher == "accelerate":
            prefix = [
                self.python,
                "-m",
                "accelerate.commands.launch",
                "--config_file",
                str(self.profile.accelerate_config_path),
            ]
        else:
            prefix = [self.python]
        return [
            *prefix,
            str(SCRIPTS / script),
            *self._common(),
            "--runtime",
            str(self.runtime_path),
            "--resume",
            *args,
        ]

    # --- models --------------------------------------------------------------
    def clean_models(self) -> list[ModelRef]:
        refs = []
        for name in CLEAN:
            directory = self.layout.clean_checkpoint(name)
            model = self.config.model.model_id if name == "S0" else str(directory)
            refs.append(ModelRef(name, model, directory / "manifest.json", "clean"))
        return refs

    def contaminated_models(self) -> list[ModelRef]:
        return [
            ModelRef(name, str(self.layout.contaminated_checkpoint(name)), self.layout.contaminated_checkpoint(name) / "manifest.json", "contaminated")
            for name in CONTAMINATED
        ]

    def matched_contaminated_models(self) -> list[ModelRef]:
        refs = []
        for name in CONTAMINATED:
            directory = matched_checkpoint_dir(self.layout, self.arm, name)
            refs.append(ModelRef(f"{self.arm.tag}_{name}", str(directory), directory / "manifest.json", "contaminated"))
        return refs

    def unlearned_models(self) -> list[ModelRef]:
        """Unlearning runs on the matched-memorization C_i (amendment 2026-10-25)."""

        refs = []
        for contaminated in CONTAMINATED:
            for method in METHODS:
                trajectory = self.layout.unlearning_trajectory(
                    self.config, contaminated, method, condition="matched_memorization"
                )
                for step in self.config.unlearning.snapshot_steps:
                    snapshot = trajectory / f"step{step:04d}"
                    refs.append(
                        ModelRef(
                            f"{self.arm.tag}_U_{contaminated}_{method}_step{step:04d}",
                            str(snapshot),
                            snapshot / "manifest.json",
                            "unlearned",
                        )
                    )
        return refs

    def _eval_jobs(self, group: str, models: list[ModelRef]) -> list[Job]:
        jobs = []
        generated = self.layout.generated
        for ref in models:
            for suite in SUITES:
                if suite.clean_only and ref.family != "clean":
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
                    )
                )
        return jobs

    # --- the plan ------------------------------------------------------------
    def jobs(self) -> list[Job]:
        layout = self.layout
        jobs: list[Job] = [
            Job(
                id="preflight",
                group="data",
                title="prepare and verify data (MedMCQA, synthetic F, eval sets)",
                kind="command",
                command=self._python("preflight.py"),
                outputs=(layout.preflight_report,),
                offline=False,
            ),
            Job(id="gate0_data", group="data", title=GATES["gate0_data"], kind="gate", gate="gate0_data"),
            Job(
                id="benchmark",
                group="hardware",
                title="benchmark memory/speed of every stage, check resume",
                kind="command",
                command=self._python("benchmark.py", "--runtime", str(self.runtime_path)),
                outputs=(layout.stage_dir("benchmark") / "benchmark_report.json",),
                offline=False,
            ),
            Job(id="gate1_hardware", group="hardware", title=GATES["gate1_hardware"], kind="gate", gate="gate1_hardware"),
            Job(
                id="specialize",
                group="specialization",
                title="clean specialization S0 -> S50 -> S100",
                kind="command",
                command=self._training("specialize_clean.py"),
                outputs=(layout.clean_root / layout.run_id / "run_manifest.json",),
            ),
        ]
        jobs += self._eval_jobs("clean_eval", self.clean_models())
        jobs.append(
            Job(id="gate2_specialization", group="clean_eval", title=GATES["gate2_specialization"], kind="gate", gate="gate2_specialization")
        )
        # Fixed-exposure arm (pre-registered; kept as the reference condition).
        for clean_name, contaminated in CONTAMINATED_FOR_CLEAN.items():
            jobs.append(
                Job(
                    id=f"contaminate_{contaminated}",
                    group="fixed_contamination",
                    title=f"fixed-exposure contamination {clean_name} + F -> {contaminated}",
                    kind="command",
                    command=self._training(
                        "contaminate.py", "--clean-checkpoint", str(layout.clean_checkpoint(clean_name))
                    ),
                    outputs=(layout.contaminated_checkpoint(contaminated) / "manifest.json",),
                )
            )
        jobs += self._eval_jobs("fixed_contaminated_eval", self.contaminated_models())

        # Matched-memorization arm (amendment 2026-10-25): calibrate on C0, then match.
        arm_args = ("--arm", str(self.arm_path))
        report = calibration_report_path(layout, self.arm)
        jobs.append(
            Job(
                id=f"calibrate_{self.arm.tag}",
                group="calibration",
                title="contamination learning-rate calibration on C0 (memorization + damage probes)",
                kind="command",
                command=self._python("calibrate_contamination.py", "--runtime", str(self.runtime_path), *arm_args),
                outputs=(report,),
                done_check=lambda: report.is_file()
                and read_json(report).get("status") in FINAL_CALIBRATION_STATES,
            )
        )
        jobs.append(
            Job(id="gate_calibration", group="calibration", title=GATES["gate_calibration"], kind="gate", gate="gate_calibration")
        )
        for clean_name, contaminated in CONTAMINATED_FOR_CLEAN.items():
            jobs.append(
                Job(
                    id=f"contaminate_{self.arm.tag}_{contaminated}",
                    group="matched_contamination",
                    title=f"matched-memorization contamination {clean_name} + F -> {self.arm.tag}_{contaminated}",
                    kind="command",
                    command=self._training(
                        "contaminate_matched.py",
                        "--clean-checkpoint",
                        str(layout.clean_checkpoint(clean_name)),
                        *arm_args,
                    ),
                    outputs=(matched_checkpoint_dir(layout, self.arm, contaminated) / "manifest.json",),
                )
            )
        jobs += self._eval_jobs("matched_contaminated_eval", self.matched_contaminated_models())
        jobs.append(
            Job(id="gate3_memorization", group="matched_contaminated_eval", title=GATES["gate3_memorization"], kind="gate", gate="gate3_memorization")
        )
        for contaminated in CONTAMINATED:
            for method in METHODS:
                trajectory = layout.unlearning_trajectory(self.config, contaminated, method, condition="matched_memorization")
                jobs.append(
                    Job(
                        id=f"unlearn_{self.arm.tag}_{contaminated}_{method}",
                        group="unlearning",
                        title=f"{method.upper()} unlearning from {self.arm.tag}_{contaminated}",
                        kind="command",
                        command=self._training(
                            "unlearn.py",
                            "--contaminated-checkpoint-dir",
                            str(matched_checkpoint_dir(layout, self.arm, contaminated)),
                            "--method",
                            method,
                        ),
                        outputs=(trajectory / "run_manifest.json",),
                    )
                )
        jobs += self._eval_jobs("unlearned_eval", self.unlearned_models())
        jobs.append(
            Job(
                id="aggregate",
                group="results",
                title="build results/processed tables and figures",
                kind="command",
                command=self._python("aggregate_results.py"),
                outputs=(layout.processed_results / "metrics_long.csv",),
                always_run=True,
            )
        )
        return jobs

    # --- status file ------------------------------------------------------------
    def write_status(self, jobs: list[Job], *, state: str, current: Job | None = None, message: str | None = None) -> None:
        approvals = self.approvals()
        entries = []
        for job in jobs:
            done = job.done(approvals) if not job.always_run else all(path.is_file() for path in job.outputs)
            entries.append({"id": job.id, "group": job.group, "kind": job.kind, "title": job.title, "done": done})
        groups: dict[str, dict[str, int]] = {}
        for entry in entries:
            group = groups.setdefault(entry["group"], {"done": 0, "total": 0})
            group["total"] += 1
            group["done"] += int(entry["done"])
        write_json(
            self.pipeline_dir / "status.json",
            {
                "run_id": self.layout.run_id,
                "updated_at": utc_now(),
                "state": state,
                "message": message,
                "pid": os.getpid(),
                "current_job": {"id": current.id, "title": current.title, "group": current.group} if current else None,
                "done": sum(int(entry["done"]) for entry in entries),
                "total": len(entries),
                "groups": groups,
                "runtime_profile": self.profile.name,
                "jobs": entries,
            },
            overwrite=True,
        )


# --- running ---------------------------------------------------------------------


def _console_safe() -> None:
    # Child output (e.g. tqdm bars) can contain characters a redirected Windows
    # console code page cannot encode; never let that kill the runner.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _run_command(command: list[str], log_path: Path, *, offline: bool = False) -> tuple[int, bool]:
    """Run one job, teeing output to the console and its log; returns (exit code, interrupted)."""

    _console_safe()
    ensure_dir(log_path.parent)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {utc_now()} $ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        previous_term = None
        if hasattr(signal, "SIGTERM") and os.name != "nt":
            previous_term = signal.signal(signal.SIGTERM, lambda *_: process.send_signal(signal.SIGTERM))
        interrupted = False
        try:
            while True:
                try:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log.write(line)
                    break
                except KeyboardInterrupt:
                    # The child got the same Ctrl+C: training saves a resume point,
                    # evaluation stops without writing a partial result file.
                    interrupted = True
                    print("\n[pipeline] Ctrl+C: waiting for the current job to save and exit "
                          "(press Ctrl+C again to abort it immediately)...", flush=True)
            return process.wait(), interrupted
        finally:
            if previous_term is not None:
                signal.signal(signal.SIGTERM, previous_term)


def run_pipeline(pipeline: PilotPipeline, *, until: str | None = None, dry_run: bool = False) -> int:
    jobs = pipeline.jobs()
    ids = [job.id for job in jobs]
    if until is not None and until not in ids:
        raise SystemExit(f"Unknown job {until!r}. Use `status --all` to list job ids.")
    approvals = pipeline.approvals()
    for job in jobs:
        if job.done(approvals):
            continue
        if job.kind == "gate":
            report = write_gate_report(pipeline, job.gate)
            pipeline.write_status(jobs, state="waiting_for_gate", current=job, message=str(report))
            print(
                f"\n[pipeline] Review gate reached: {job.title}\n"
                f"           Report: {report}\n"
                f"           After reviewing, approve with:\n"
                f"             python scripts/pilot/run_pipeline.py approve {job.gate} --note \"...\"\n"
                f"           then rerun `python scripts/pilot/run_pipeline.py run`.",
                flush=True,
            )
            return 0
        if dry_run:
            print(f"[dry-run] {job.id}: {' '.join(job.command)}")
            if until == job.id:
                return 0
            continue
        pipeline.write_status(jobs, state="running", current=job)
        append_jsonl(pipeline.pipeline_dir / "history.jsonl", {"time": utc_now(), "job": job.id, "event": "start"})
        print(f"\n[pipeline] >>> {job.id}: {job.title}", flush=True)
        returncode, interrupted = _run_command(
            job.command, pipeline.pipeline_dir / "logs" / f"{job.id}.log", offline=job.offline
        )
        if interrupted and returncode != 0:
            returncode = 130
        append_jsonl(
            pipeline.pipeline_dir / "history.jsonl",
            {"time": utc_now(), "job": job.id, "event": "exit", "returncode": returncode},
        )
        if returncode == 130:
            pipeline.write_status(jobs, state="interrupted", current=job, message="rerun `run` to resume")
            print(f"[pipeline] {job.id} stopped and saved its progress. Rerun `run` to resume.", flush=True)
            return 130
        if returncode != 0:
            pipeline.write_status(jobs, state="failed", current=job, message=f"exit code {returncode}")
            print(
                f"[pipeline] {job.id} FAILED (exit {returncode}). Log: "
                f"{pipeline.pipeline_dir / 'logs' / (job.id + '.log')}. Fix the cause and rerun `run`; "
                "finished jobs are skipped and training resumes from its last save.",
                flush=True,
            )
            return returncode
        missing = [str(path) for path in job.outputs if not path.is_file()]
        if missing:
            pipeline.write_status(jobs, state="failed", current=job, message="outputs missing")
            print(f"[pipeline] {job.id} exited 0 but did not produce: {missing}", flush=True)
            return 1
        if until == job.id:
            pipeline.write_status(jobs, state="stopped", message=f"stopped after {job.id} (--until)")
            return 0
    pipeline.write_status(jobs, state="completed", message="all jobs finished")
    print("[pipeline] All jobs finished. Results: " + str(pipeline.layout.processed_results), flush=True)
    return 0


def print_status(pipeline: PilotPipeline, *, show_all: bool = False) -> None:
    jobs = pipeline.jobs()
    approvals = pipeline.approvals()
    print(f"run id : {pipeline.layout.run_id}")
    print(f"config : {pipeline.config_path}")
    print(f"runtime: {pipeline.runtime_path} ({pipeline.profile.launcher})")
    print(f"storage: {pipeline.layout.root}\n")
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        groups.setdefault(job.group, []).append(job)
    next_job = next((job for job in jobs if not job.done(approvals)), None)
    for group, members in groups.items():
        done = sum(1 for job in members if job.done(approvals) or (job.always_run and all(p.is_file() for p in job.outputs)))
        print(f"  {group:26s} {done:>3}/{len(members):<3} {'done' if done == len(members) else ''}")
        if show_all:
            for job in members:
                flag = "x" if job.done(approvals) else " "
                print(f"      [{flag}] {job.id}")
    if next_job is not None:
        print(f"\nnext: {next_job.id} -- {next_job.title}")
        if next_job.kind == "gate":
            print(f"      waiting for your approval: run_pipeline.py approve {next_job.gate}")
    else:
        print("\nall jobs complete")


def approve_gate(pipeline: PilotPipeline, gate: str, note: str | None) -> int:
    if gate not in GATES:
        print(f"Unknown gate {gate!r}; gates: {', '.join(GATES)}", file=sys.stderr)
        return 2
    report = pipeline.gate_report_path(gate)
    if not report.is_file():
        print(f"{gate} has not been reached yet (no report at {report}); run the pipeline first.", file=sys.stderr)
        return 2
    approvals = pipeline.approvals()
    approvals[gate] = {"approved_at": utc_now(), "note": note, "report": str(report)}
    write_json(pipeline.gates_path, approvals, overwrite=True)
    print(f"approved {gate}. Continue with: python scripts/pilot/run_pipeline.py run")
    return 0


# --- gate reports ---------------------------------------------------------------------


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _retention_table(pipeline: PilotPipeline, labels: list[str], suites: list[str]) -> list[str]:
    lines = ["| model | " + " | ".join(suites) + " |", "|---|" + "---|" * len(suites)]
    for label in labels:
        cells = []
        for suite in suites:
            result = _read(pipeline.result_path(label, suite))
            if not result:
                cells.append("missing")
                continue
            ci = result.get("accuracy_ci95") or [None, None]
            cells.append(f"{_fmt(result.get('accuracy'))} [{_fmt(ci[0], 2)}, {_fmt(ci[1], 2)}] (n={result.get('count')})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


FORGET_METRICS = (
    "target_exact_leakage_rate",
    "any_target_exact_leakage_rate",
    "nonprompt_field_leakage_rate",
    "normalized_field_f1",
    "target_logprob",
    "target_vs_decoy_margin",
    "mc_association_accuracy",
)


def _forgetting_table(pipeline: PilotPipeline, labels: list[str]) -> list[str]:
    lines = ["| model | " + " | ".join(FORGET_METRICS) + " |", "|---|" + "---|" * len(FORGET_METRICS)]
    for label in labels:
        result = _read(pipeline.result_path(label, "forgetting"))
        summary = rescore_forgetting_payload(result)["summary"] if result else {}
        cells = [_fmt(summary.get(metric)) if result else "missing" for metric in FORGET_METRICS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def write_gate_report(pipeline: PilotPipeline, gate: str) -> Path:
    layout = pipeline.layout
    lines = [
        f"# {GATES[gate]}",
        "",
        f"Run: `{layout.run_id}`  ",
        f"Written: {utc_now()}  ",
        "All numbers below are **measured** values read from the run's artifacts.",
        "",
    ]
    if gate == "gate0_data":
        report = _read(layout.preflight_report) or {}
        budget = report.get("token_budget") or {}
        forget = report.get("forget") or {}
        subsets = report.get("domain_subsets") or {}
        lines += [
            "## Step budget (frozen before training)",
            f"- packed sequences: {budget.get('packed_sequence_count')}, context {budget.get('context_length')}",
            f"- S50 = {budget.get('s50_steps')} steps, S100 = {budget.get('s100_steps')} steps "
            f"(provisional {budget.get('provisional_s100_steps')}, adjustment {budget.get('step_adjustment')})",
            f"- training tokens: {budget.get('frozen_training_tokens')}",
            "",
            "## Synthetic forget set F",
            f"- {json.dumps({key: forget.get(key) for key in ('record_count', 'contamination_example_count', 'eval_example_count') if key in forget})}",
            "- F terms were checked against every MedMCQA training text (the preflight fails on any overlap).",
            "",
            "## Near/far subsets (frozen before results)",
            f"- {json.dumps({key: subsets.get(key) for key in ('near_count', 'far_count') if key in subsets})}",
            f"- human-review files: `{layout.generated / 'medmcqa_domain_subsets'}`",
            "",
            "## Check before approving",
            "- [ ] S50/S100 step counts are 1645/3290 (the training stage refuses anything else).",
            "- [ ] F record/example counts match the protocol (120 records, 8 contamination + 6 eval templates).",
            "- [ ] Skim the near/far human-review JSONL files; the selection must not be edited after results.",
            f"- Full report: `{layout.preflight_report}`",
        ]
    elif gate == "gate1_hardware":
        report = _read(layout.stage_dir("benchmark") / "benchmark_report.json") or {}
        lines += ["## Per-stage measurements (configured micro-batch marked *)", ""]
        for stage, data in (report.get("stages") or {}).items():
            for micro, result in data.get("results", {}).items():
                mark = "*" if int(micro) == data.get("configured_micro_batch") else " "
                if result.get("status") != "ok":
                    lines.append(f"- {mark} {stage} micro {micro}: {result.get('status')}")
                    continue
                lines.append(
                    f"- {mark} {stage} micro {micro}: {result['mean_step_seconds']:.2f} s/step, "
                    f"peak reserved {result['peak_reserved_gb']:.2f} GB"
                    f"{'' if result['fits_with_headroom'] else ' (too close to VRAM limit)'}"
                )
            if data.get("recommended_micro_batch") not in (None, data.get("configured_micro_batch")):
                lines.append(f"  - recommended micro-batch: {data['recommended_micro_batch']}")
        check = report.get("resume_roundtrip") or {}
        lines += [
            "",
            f"## Resume round-trip: {'PASSED' if check.get('passed') else 'FAILED'} "
            f"(|loss diff| {_fmt(check.get('loss_abs_difference'), 2)})",
            f"## Evaluation check: {'PASSED' if (report.get('evaluation') or {}).get('passed') else 'FAILED'} "
            f"{(report.get('evaluation') or {}).get('error') or ''}",
            "",
            "## Projected wall-clock (this machine, configured profile)",
        ]
        lines += [f"- {key}: {value}" for key, value in (report.get("projection_human") or {}).items()]
        disk = report.get("estimated_disk_gb") or {}
        lines += [
            "",
            f"## Disk: ~{_fmt(disk.get('scientific_checkpoints'), 0)} GB of checkpoints "
            f"(+ {_fmt(disk.get('resume_points_peak'), 1)} GB resume points); free now: "
            f"{_fmt(((report.get('hardware') or {}).get('disk') or {}).get('free_gib'), 0)} GB",
            "",
            "## Check before approving",
            "- [ ] Every configured stage ran without out-of-memory and fits with headroom.",
            "- [ ] Resume round-trip and evaluation check passed.",
            "- [ ] Projected time and disk are acceptable; the laptop is plugged in and will not sleep.",
            f"- Full report: `{layout.stage_dir('benchmark') / 'benchmark_report.json'}`",
        ]
    elif gate == "gate2_specialization":
        lines += ["## Clean checkpoints (accuracy [95% bootstrap CI])", ""]
        lines += _retention_table(pipeline, list(CLEAN), ["domain_full", "near", "far", "general", "external_medical"])
        lines += ["", "## F knowledge in clean checkpoints (should be at chance; F was never seen)", ""]
        lines += _forgetting_table(pipeline, list(CLEAN))
        lines += [
            "",
            "## Pre-specified checks",
            "- Specialization is the manipulated variable. If `domain_full` accuracy does not increase from S0 to",
            "  S100 beyond its confidence interval, the dose was not delivered: stop and report that, do not",
            "  tune the recipe after looking at unlearning results.",
            "- Clean checkpoints must show ~0 target leakage (they never saw F). `target_exact_leakage_rate`",
            "  counts only the probed secret; echoing the name/ID given in the question is not a leak.",
        ]
    elif gate == "gate_calibration":
        report = _read(calibration_report_path(layout, pipeline.arm)) or {}
        arm = pipeline.arm
        baseline = report.get("baseline") or {}
        base_recall = (baseline.get("memorization") or {}).get("mean_recall")
        base_acc = baseline.get("damage_accuracy") or {}
        lines += [
            f"Rule (fixed before running): smallest learning rate reaching recall >= {arm.target_recall:.2f} "
            f"within {arm.max_epochs} epochs with accuracy drop <= {arm.max_accuracy_drop:.2f} on both damage probes.",
            "",
            f"S0 baseline: memorization recall {_fmt(base_recall)}, damage-probe accuracy "
            f"general {_fmt(base_acc.get('general'))} / medical {_fmt(base_acc.get('medical'))}",
            "",
            "| learning rate | reached target | epochs | recall at stop (condition / medication / code) | accuracy drop (general / medical) | qualifies |",
            "|---|---|---|---|---|---|",
        ]
        for candidate in report.get("candidates") or []:
            final = candidate.get("final_memorization") or {}
            by_template = final.get("by_template") or {}
            drops = candidate.get("accuracy_drop") or {}
            lines.append(
                f"| {candidate['learning_rate']:g} | {'yes' if candidate['reached_target'] else 'no'} | "
                f"{_fmt(candidate.get('epochs_to_target') or (candidate.get('steps_run', 0) / 60), 1)} | "
                f"{_fmt(final.get('mean_recall'))} ({_fmt(by_template.get('condition_lookup'), 2)} / "
                f"{_fmt(by_template.get('medication_lookup'), 2)} / {_fmt(by_template.get('private_code_lookup'), 2)}) | "
                f"{_fmt(drops.get('general'))} / {_fmt(drops.get('medical'))} | {'YES' if candidate['qualifies'] else 'no'} |"
            )
        lines += [
            "",
            f"**Status: {report.get('status')}; selected learning rate: {report.get('selected_learning_rate')}**",
            "",
            "## Check before approving",
            "- [ ] The rule above was applied as written (smallest qualifying candidate).",
            "- [ ] If no candidate qualified: stop and record that; do not add larger learning rates after the fact.",
            f"- Full report with per-epoch recall curves: `{calibration_report_path(layout, arm)}`",
        ]
    elif gate == "gate3_memorization":
        arm = pipeline.arm
        lines += [
            "## Matched memorization (from the C_i manifests)",
            "",
            "| model | target reached | steps (epochs) | recall at stop (cond / med / code) | accuracy drop vs S_i (general / medical) |",
            "|---|---|---|---|---|",
        ]
        for name in CONTAMINATED:
            manifest = _read(matched_checkpoint_dir(layout, arm, name) / "manifest.json") or {}
            memorization = manifest.get("memorization") or {}
            at_stop = memorization.get("recall_at_stop") or {}
            by_template = at_stop.get("by_template") or {}
            drops = (manifest.get("damage") or {}).get("accuracy_drop") or {}
            steps = memorization.get("steps_used")
            lines.append(
                f"| {arm.tag}_{name} | {memorization.get('target_reached')} | {steps} "
                f"({_fmt((steps or 0) / 60, 1)}) | {_fmt(at_stop.get('mean_recall'))} "
                f"({_fmt(by_template.get('condition_lookup'), 2)} / {_fmt(by_template.get('medication_lookup'), 2)} / "
                f"{_fmt(by_template.get('private_code_lookup'), 2)}) | {_fmt(drops.get('general'))} / {_fmt(drops.get('medical'))} |"
            )
        matched = [f"{arm.tag}_{name}" for name in CONTAMINATED]
        lines += ["", "## Held-out forgetting probes: matched C_i", ""]
        lines += _forgetting_table(pipeline, matched)
        lines += ["", "## Clean counterfactuals S_i", ""]
        lines += _forgetting_table(pipeline, list(CLEAN))
        lines += ["", "## Fixed-exposure C_i (reference, not unlearned)", ""]
        lines += _forgetting_table(pipeline, list(CONTAMINATED))
        lines += ["", "## Retention of matched C_i", ""]
        lines += _retention_table(pipeline, matched, ["near", "far", "general", "external_medical"])
        lines += [
            "",
            "## Pre-specified checks",
            "- Every matched C_i reached the target recall (training questions) within the cap.",
            "- Collateral damage of each C_i vs its S_i is within the arm limit (accuracy drop column).",
            "- Held-out probes: target leakage / likelihood / decoy margin clearly above S_i. Recall on the",
            "  training questions is matched by design; held-out generalization may still differ by level.",
            "- The steps each C_i needed are a finding (learnability), not a failure.",
        ]
    path = pipeline.gate_report_path(gate)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- CLI ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the S/C/U pilot as a resumable, gated pipeline.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="show progress and the next job")
    status.add_argument("--all", action="store_true", help="list every job")
    run = commands.add_parser("run", help="run pending jobs until the next review gate")
    run.add_argument("--until", default=None, help="stop after this job id")
    run.add_argument("--dry-run", action="store_true", help="print commands without running them")
    approve = commands.add_parser("approve", help="approve a review gate after reading its report")
    approve.add_argument("gate", choices=sorted(GATES))
    approve.add_argument("--note", default=None)
    report = commands.add_parser("report", help="(re)write a gate report from current results")
    report.add_argument("gate", choices=sorted(GATES))
    args = parser.parse_args(argv)

    pipeline = PilotPipeline(config_path=Path(args.config), runtime_path=Path(args.runtime))
    if args.command == "status":
        print_status(pipeline, show_all=args.all)
        return 0
    if args.command == "approve":
        return approve_gate(pipeline, args.gate, args.note)
    if args.command == "report":
        print(write_gate_report(pipeline, args.gate))
        return 0
    started = utc_now()
    code = run_pipeline(pipeline, until=args.until, dry_run=args.dry_run)
    print(f"[pipeline] started {started}, finished {utc_now()}", flush=True)
    return code


__all__ = ["PilotPipeline", "run_pipeline", "main", "format_duration"]
