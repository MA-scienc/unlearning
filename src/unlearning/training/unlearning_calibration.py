"""Learning-rate calibration for matched-forgetting unlearning (revision 3), on MR_C0 only.

Rule (docs/MATCHED_FORGETTING_DESIGN.md, 3.3), per method: try the candidates in
ascending order and select the smallest one at which the method closes the whole
gap to S0 (g >= 1.0) on the training-template probes within max_steps. Only g is
measured: no retention, damage or held-out probe is computed here, so the rate
cannot be chosen for its collateral damage. Nothing is saved but the report and
each trial's g history. Finished trials are kept on restart; an interrupted trial
restarts from scratch.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from unlearning.config import load_config, validate_training_ready
from unlearning.config.arms import load_matched_memorization_arm
from unlearning.config.unlearning_revisions import UnlearningRevision, load_unlearning_revision, threshold_name
from unlearning.runtime.hardware import precheck_gpu_training, require_gpu_training
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.monitor import GpuSampler, SystemMonitor, ThermalGuard
from unlearning.runtime.profile import load_runtime_profile, micro_steps_per_rank
from unlearning.runtime.run_log import RunLog, utc_now
from unlearning.runtime.settings import MachineSettings, default_config_path, default_runtime_path
from unlearning.runtime.signals import INTERRUPTED_EXIT_CODE, GracefulStop, StageInterrupted
from unlearning.training.engine import NonFiniteLoss, TrainingLoop, create_accelerator, enable_memory_saving
from unlearning.training.matched_contamination import DEFAULT_ARM_PATH, _NoResumeStore, matched_checkpoint_dir
from unlearning.training.matched_forgetting import (
    CALIBRATION_SCHEMA,
    FINAL_CALIBRATION_STATES,
    REFERENCES_FILE,
    ForgettingMonitor,
    calibration_dir,
    calibration_report_path,
    clean_parent,
    load_or_measure_references,
    select_learning_rate,
)
from unlearning.training.memorization import MemorizationProbe
from unlearning.training.unlearn import (
    _load_verified_forget_examples,
    _reference_completion_logprob_cache,
    _require_training_stack,
    _set_seed,
    UnlearningObjective,
    build_completion_only_sequences,
    load_contaminated_source,
)
from unlearning.utils.env import software_versions
from unlearning.utils.io import read_json, write_json

METHODS = ("ga", "npo")


def _cleanup(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_unlearning_calibration(
    *,
    config_path: str | Path,
    revision_path: str | Path,
    runtime_path: str | Path | None = None,
    arm_path: str | Path = DEFAULT_ARM_PATH,
) -> dict[str, Any]:
    settings = MachineSettings.from_env()
    config = load_config(config_path)
    validate_training_ready(config)
    revision: UnlearningRevision = load_unlearning_revision(revision_path)
    if not revision.is_matched:
        raise ValueError(f"{revision_path} is not a matched-forgetting revision")
    spec = revision.matched
    arm = load_matched_memorization_arm(arm_path)
    layout = StorageLayout.for_config(config, settings)
    out_dir = calibration_dir(layout, revision)
    report_path = calibration_report_path(layout, revision)
    report = read_json(report_path) if report_path.is_file() else None
    if report is not None and report.get("revision_fingerprint") != revision.fingerprint:
        raise RuntimeError(f"{report_path} was produced from a different revision file; archive that directory first")
    if report is not None and report.get("status") in FINAL_CALIBRATION_STATES:
        print(f"[calibrate-unlearning] already complete ({report['status']}): {report_path}", flush=True)
        return report

    hardware = precheck_gpu_training(settings, purpose="unlearning learning-rate calibration")
    require_gpu_training(hardware)
    profile = load_runtime_profile(runtime_path or settings.runtime_profile_path)
    accelerator = create_accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Run the calibration as a single process (python, not accelerate launch)")
    torch, AutoModelForCausalLM, AutoTokenizer, AdamW8bit = _require_training_stack()
    device = accelerator.device

    source = load_contaminated_source(matched_checkpoint_dir(layout, arm, "C0"), config=config)
    clean = clean_parent(layout, config, source)
    _path, examples, _sha, forget_fingerprint = _load_verified_forget_examples(
        forget_examples_path=layout.contamination_examples, source=source, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(source.checkpoint_dir / "tokenizer", use_fast=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    sequence_length = config.unlearning.methods["ga"].sequence_length
    sequences = build_completion_only_sequences(examples, tokenizer=tokenizer, sequence_length=sequence_length)
    probe = MemorizationProbe(
        torch,
        tokenizer,
        examples,
        templates=spec.probe_templates,
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        batch_size=profile.micro_batch.contamination,
    )
    references = load_or_measure_references(
        out_dir / REFERENCES_FILE,
        probe=probe,
        clean=clean,
        source=source,
        config=config,
        auto_model=AutoModelForCausalLM,
        torch=torch,
        device=device,
    )
    print(
        f"[calibrate-unlearning] l(S0) {references['clean']['mean']:.4f}, l(C0) "
        f"{references['contaminated']['mean']:.4f}, gap {references['gap']:.4f}",
        flush=True,
    )
    if report is None:
        report = {
            "schema_version": CALIBRATION_SCHEMA,
            "label": "operational calibration on MR_C0; decides only the unlearning learning rate per method",
            "status": "running",
            "run_id": layout.run_id,
            "revision": revision.as_dict(),
            "revision_fingerprint": revision.fingerprint,
            "revision_path": str(revision.source_path),
            "source_contaminated_checkpoint_id": source.checkpoint_id,
            "clean_checkpoint_id": clean.checkpoint_id,
            "forget_examples_fingerprint": forget_fingerprint,
            "references": {key: references[key] for key in ("clean", "contaminated", "gap", "probe_fingerprint", "fingerprint")},
            "selection_rule": (
                "per method, the smallest candidate learning rate at which gap closed on the training-template "
                "probes reaches 1.0 within max_steps; nothing else is measured"
            ),
            "methods": {method: {"trials": [], "selected_learning_rate": None, "status": "pending"} for method in METHODS},
            "started_at": utc_now(),
            "software_versions": software_versions(),
        }
        write_json(report_path, report, overwrite=True)

    def save_report() -> None:
        write_json(report_path, report, overwrite=True)

    stop = GracefulStop().install()
    sampler = GpuSampler()
    try:
        for method in METHODS:
            record = report["methods"][method]
            method_config = config.unlearning.methods[method]
            micro = profile.micro_batch.unlearning
            micro_steps_per_rank(
                effective_batch=method_config.effective_batch_size, micro_batch=micro, world_size=1, stage=f"calibration ({method})"
            )
            reference_values = None
            for learning_rate in spec.learning_rate_candidates:
                if select_learning_rate(record["trials"]) is not None:
                    break
                if any(trial["learning_rate"] == learning_rate for trial in record["trials"]):
                    continue
                if method == "npo" and reference_values is None:
                    print("[calibrate-unlearning] computing frozen MR_C0 reference log-probabilities for NPO", flush=True)
                    reference_values = _reference_completion_logprob_cache(
                        AutoModelForCausalLM=AutoModelForCausalLM,
                        torch=torch,
                        source=source,
                        sequences=sequences,
                        batch_size=profile.micro_batch.npo_reference,
                        device=device,
                        pad_token_id=pad_token_id,
                    )
                trial = _run_trial(
                    method=method,
                    learning_rate=learning_rate,
                    revision=revision,
                    config=config,
                    layout=layout,
                    profile=profile,
                    settings=settings,
                    accelerator=accelerator,
                    torch=torch,
                    AutoModelForCausalLM=AutoModelForCausalLM,
                    AdamW8bit=AdamW8bit,
                    source=source,
                    sequences=sequences,
                    pad_token_id=pad_token_id,
                    reference_values=reference_values,
                    probe=probe,
                    references=references,
                    stop=stop,
                    sampler=sampler,
                    trial_dir=out_dir / f"{method}_lr_{learning_rate:.0e}",
                )
                record["trials"].append(trial)
                save_report()
            selected = select_learning_rate(record["trials"])
            record["selected_learning_rate"] = selected
            record["status"] = "selected" if selected is not None else "no_candidate_reached"
            save_report()
            print(f"[calibrate-unlearning] {method}: {record['status']} {selected}", flush=True)
    except StageInterrupted:
        report["status"] = "interrupted"
        save_report()
        raise
    finally:
        stop.uninstall()
    statuses = {record["status"] for record in report["methods"].values()}
    report["status"] = "selected" if statuses == {"selected"} else "no_candidate_reached"
    report["completed_at"] = utc_now()
    save_report()
    return report


def _run_trial(
    *,
    method: str,
    learning_rate: float,
    revision: UnlearningRevision,
    config,
    layout: StorageLayout,
    profile,
    settings: MachineSettings,
    accelerator,
    torch,
    AutoModelForCausalLM,
    AdamW8bit,
    source,
    sequences,
    pad_token_id: int,
    reference_values,
    probe: MemorizationProbe,
    references: dict[str, Any],
    stop: GracefulStop,
    sampler: GpuSampler,
    trial_dir: Path,
) -> dict[str, Any]:
    spec = revision.matched
    method_config = revision.method_config(config, method, learning_rate=learning_rate)
    if trial_dir.exists():
        shutil.rmtree(trial_dir)  # an interrupted trial restarts from scratch
    run_log = RunLog(
        trial_dir,
        run_id=layout.run_id,
        stage=f"calibrate_unlearning_{revision.tag}_{method}_lr_{learning_rate:.0e}",
        total_steps=spec.max_steps,
        console_every_steps=profile.console_every_steps,
    )
    monitor = SystemMonitor(run_log, interval=profile.system_sample_seconds, storage_root=settings.storage_root, sampler=sampler).start()
    started = time.perf_counter()
    _set_seed(torch, config.seed)
    run_log.set_state("loading", message=f"{method.upper()} at {learning_rate:g}: loading MR_C0")
    model = AutoModelForCausalLM.from_pretrained(source.checkpoint_dir / "model", torch_dtype=torch.bfloat16)
    enable_memory_saving(model, gradient_checkpointing=config.model.gradient_checkpointing)
    model.train()
    model = accelerator.prepare(model)
    optimizer = AdamW8bit(
        accelerator.unwrap_model(model).parameters(), lr=method_config.learning_rate, weight_decay=method_config.weight_decay
    )
    loop = TrainingLoop(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        objective=UnlearningObjective(
            torch,
            sequences,
            method=method,
            batch_size=method_config.effective_batch_size,
            micro_batch_size=profile.micro_batch.unlearning,
            pad_token_id=pad_token_id,
            reference_values=reference_values,
            beta=method_config.beta,
        ),
        total_steps=spec.max_steps,
        max_grad_norm=method_config.gradient_clip_norm,
        run_log=run_log,
        resume_store=_NoResumeStore(trial_dir / "resume", keep_last=1, every_minutes=10**6),
        stop=stop,
        settings=settings,
        thermal=ThermalGuard(run_log, sampler, pause_at_c=settings.gpu_pause_temp_c, resume_at_c=settings.gpu_resume_temp_c),
    )
    forgetting = ForgettingMonitor(
        probe=probe,
        references=references,
        spec=spec,
        model_getter=lambda: accelerator.unwrap_model(model),
        device=accelerator.device,
        run_log=run_log,
        history_path=trial_dir / "forgetting.jsonl",
    )
    forgetting.loop = loop
    loop.on_step_end = forgetting
    run_log.update(state="running", learning_rate=learning_rate, method=method)
    diverged: tuple[int, str] | None = None
    try:
        result = loop.run(start_step=0)
    except NonFiniteLoss as error:
        # A diverging candidate is a recorded outcome, not a crash of the calibration.
        diverged = (error.step, str(error))
    last = forgetting.last_entry
    trial = {
        "learning_rate": learning_rate,
        "reached": forgetting.reached,
        "final_threshold_reached": forgetting.final_reached,
        "steps_run": loop.completed_step,
        "last_check": {key: last[key] for key in ("step", "gap_closed", "answer_logprob")} if last else None,
        "diverged": diverged is not None,
        "error": diverged[1] if diverged else None,
        "runtime_seconds": time.perf_counter() - started,
        "history_path": str(trial_dir / "forgetting.jsonl"),
    }
    monitor.stop()
    if diverged is None and not result.completed:
        run_log.finish("interrupted", message="calibration trial restarts from scratch on rerun")
        raise StageInterrupted(result.last_step, result.stop_reason)
    summary = ", ".join(f"{name} at step {value['step']}" for name, value in forgetting.reached.items()) or "no threshold"
    run_log.finish("completed", message=f"{method} {learning_rate:g}: {summary}")
    accelerator.unwrap_model(model).zero_grad(set_to_none=True)
    model = optimizer = loop = None
    accelerator.free_memory()
    _cleanup(torch)
    return trial


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the matched-forgetting unlearning learning rate on MR_C0.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    parser.add_argument("--unlearning-revision", required=True)
    parser.add_argument("--arm", default=str(DEFAULT_ARM_PATH))
    parser.add_argument("--resume", action="store_true", help="Accepted for pipeline symmetry; finished trials are kept.")
    args = parser.parse_args(argv)
    try:
        report = run_unlearning_calibration(
            config_path=args.config, revision_path=args.unlearning_revision, runtime_path=args.runtime, arm_path=args.arm
        )
    except StageInterrupted as stopped:
        print(f"[calibrate-unlearning] {stopped}. Rerun to continue with the remaining trials.", file=sys.stderr, flush=True)
        return INTERRUPTED_EXIT_CODE
    print(json.dumps({method: record["selected_learning_rate"] for method, record in report["methods"].items()}, indent=2))
    print(f"status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
