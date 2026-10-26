"""Matched-memorization contamination: learning-rate calibration and S_i + F -> C_i.

Calibration (C0 only)
    For each candidate learning rate, smallest first: train a throwaway copy of
    S0 on F (same batches and order as the fixed-exposure arm, constant LR after
    warmup), check memorization after every epoch, and stop when the target
    recall is reached or the epoch cap is hit. At the target, measure collateral
    damage against S0. Select the smallest learning rate that reaches the target
    within the cap without exceeding the damage limit. No checkpoint is kept.

Matched contamination (C0, C50, C100)
    Train each S_i with the selected learning rate until it reaches the same
    target recall (checked every epoch; capped). Save C_i at that step and record
    the steps needed, the recall curve, and collateral damage versus S_i.

Every decision uses memorization and damage probes only -- never unlearning or
evaluation-set results.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from unlearning.config import PilotConfig, load_config, validate_training_ready
from unlearning.config.arms import MatchedMemorizationArm, load_matched_memorization_arm
from unlearning.runtime.hardware import precheck_gpu_training, require_free_disk, require_gpu_training
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
from unlearning.runtime.run_log import RunLog, utc_now
from unlearning.runtime.settings import MachineSettings, REPO_ROOT, default_config_path, default_runtime_path
from unlearning.runtime.signals import INTERRUPTED_EXIT_CODE, GracefulStop, StageInterrupted
from unlearning.runtime.stage import (
    STAGE_COMPLETE,
    STAGE_FRESH,
    ExecutionRecord,
    StageAlreadyStarted,
    invalidate,
    prepare_stage,
    segment_details,
    verify_or_write_run_plan,
)
from unlearning.training.checkpoints import CheckpointIdentity
from unlearning.training.contaminate import (
    IGNORE_INDEX,
    CleanCheckpointReference,
    ContaminationObjective,
    ForgetExamplesReference,
    FixedExposurePlan,
    _load_model,
    _load_tokenizer,
    _pad_token_id,
    _require_bitsandbytes_optimizer,
    _require_training_stack,
    build_fixed_exposure_plan,
    load_clean_checkpoint_reference,
    load_forget_examples_reference,
    tokenize_completion_example,
)
from unlearning.training.engine import (
    TrainingLoop,
    apply_resume_state,
    create_accelerator,
    enable_memory_saving,
)
from unlearning.training.memorization import (
    MemorizationProbe,
    build_damage_probes,
    damage_accuracy,
    damage_probe_fingerprints,
)
from unlearning.utils.env import software_versions
from unlearning.utils.hashing import fingerprint_directory, hash_file
from unlearning.utils.io import append_jsonl, ensure_dir, read_json, read_jsonl, write_json
from unlearning.utils.manifest import build_manifest

MATCHED_CONTAMINATION_MANIFEST_SCHEMA = "matched-memorization-contamination-v1"
CALIBRATION_REPORT_SCHEMA = "contamination-lr-calibration-v1"
DEFAULT_ARM_PATH = REPO_ROOT / "configs" / "pilot" / "arms" / "matched_memorization.yaml"
FINAL_CALIBRATION_STATES = {"selected", "no_candidate_qualified"}


def masking_record(sequence_length: int) -> dict[str, Any]:
    """Identical completion-only masking to the fixed-exposure arm (unlearning checks it)."""

    return {
        "objective": "causal_lm_completion_only",
        "ignore_index": IGNORE_INDEX,
        "tokenizer_requirement": "fast tokenizer with offset mapping",
        "add_special_tokens": False,
        "sequence_length": sequence_length,
        "boundary_rule": (
            "A token is supervised only when its token-offset start is at or after the "
            "prompt/completion character boundary; boundary-crossing tokens are masked."
        ),
        "padding_labels": IGNORE_INDEX,
    }


def calibration_dir(layout: StorageLayout, arm: MatchedMemorizationArm) -> Path:
    return layout.stage_dir(f"calibrate_{arm.tag}")


def calibration_report_path(layout: StorageLayout, arm: MatchedMemorizationArm) -> Path:
    return calibration_dir(layout, arm) / "calibration_report.json"


def matched_checkpoint_dir(layout: StorageLayout, arm: MatchedMemorizationArm, contaminated_name: str) -> Path:
    return layout.contaminated_root / layout.run_id / arm.name / contaminated_name


class _NoResumeStore(ResumeStore):
    """Calibration candidates are short and restart from scratch; never save."""

    def due(self) -> bool:
        return False

    def save(self, **_kwargs):
        self.mark_saved()
        return None


def _cleanup(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _optimizer_and_scheduler(AdamW8bit, get_scheduler, parameters, *, learning_rate: float, arm: MatchedMemorizationArm):
    optimizer = AdamW8bit(parameters, lr=learning_rate, weight_decay=arm.weight_decay)
    scheduler = get_scheduler(
        name=arm.scheduler,
        optimizer=optimizer,
        num_warmup_steps=arm.warmup_steps,
    )
    return optimizer, scheduler


def _memorization_loop(
    *,
    accelerator,
    model,
    optimizer,
    scheduler,
    objective: ContaminationObjective,
    plan: FixedExposurePlan,
    arm: MatchedMemorizationArm,
    probe: MemorizationProbe,
    run_log: RunLog,
    resume_store: ResumeStore,
    stop: GracefulStop,
    settings: MachineSettings,
    sampler: GpuSampler,
    history: list[dict[str, Any]],
    history_path: Path | None,
) -> TrainingLoop:
    steps_per_check = plan.steps_per_epoch * arm.check_every_epochs
    loop = TrainingLoop(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        objective=objective,
        total_steps=plan.optimizer_steps,
        # Same as the fixed-exposure contamination recipe: no gradient clipping.
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

    def on_step_end(step: int, metrics: dict[str, Any]) -> bool:
        if step % steps_per_check != 0:
            return False
        result = probe.recall(accelerator.unwrap_model(model), accelerator.device)
        entry = {
            "step": step,
            "epoch": step // plan.steps_per_epoch,
            "train_loss": metrics.get("loss"),
            **result,
            "time": utc_now(),
        }
        history.append(entry)
        if history_path is not None and accelerator.is_main_process:
            append_jsonl(history_path, entry)
        per_template = ", ".join(f"{name.replace('_lookup', '')} {value:.2f}" for name, value in result["by_template"].items())
        run_log.event(
            "memorization",
            f"epoch {entry['epoch']}: recall {result['mean_recall']:.3f} ({per_template})",
            **{key: entry[key] for key in ("step", "epoch", "mean_recall")},
        )
        run_log.update(memorization=entry)
        if result["mean_recall"] >= arm.target_recall:
            loop.finish_now = True
        return False

    loop.on_step_end = on_step_end
    return loop


def _load_shared_inputs(config: PilotConfig, arm: MatchedMemorizationArm, layout: StorageLayout, torch, AutoTokenizer, clean: CleanCheckpointReference, profile):
    forget = load_forget_examples_reference(layout.contamination_examples, config=config)
    tokenizer = _load_tokenizer(clean, config, AutoTokenizer)
    pad_token_id = _pad_token_id(tokenizer)
    encoded = [
        tokenize_completion_example(example, tokenizer=tokenizer, max_length=arm.sequence_length)
        for example in forget.examples
    ]
    plan = build_fixed_exposure_plan(
        example_count=len(forget.examples),
        epochs=arm.max_epochs,
        effective_batch_size=arm.effective_batch_size,
    )
    probe = MemorizationProbe(
        torch,
        tokenizer,
        forget.examples,
        templates=arm.probe_templates,
        sequence_length=arm.sequence_length,
        pad_token_id=pad_token_id,
        batch_size=profile.micro_batch.contamination,
    )
    return forget, tokenizer, pad_token_id, encoded, plan, probe


# --- calibration -------------------------------------------------------------------------


def run_calibration(
    *,
    config_path: str | Path,
    arm_path: str | Path = DEFAULT_ARM_PATH,
    runtime_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = MachineSettings.from_env()
    config = load_config(config_path)
    validate_training_ready(config)
    arm = load_matched_memorization_arm(arm_path)
    arm.validate_against(config)
    layout = StorageLayout.for_config(config, settings)
    out_dir = calibration_dir(layout, arm)
    report_path = calibration_report_path(layout, arm)
    report = read_json(report_path) if report_path.is_file() else None
    if report is not None and overwrite:
        for moved in invalidate([out_dir]):
            print(f"[calibrate] previous calibration set aside -> {moved}", flush=True)
        report = None
    if report is not None and report.get("status") in FINAL_CALIBRATION_STATES:
        print(f"[calibrate] already complete ({report['status']}): {report_path}", flush=True)
        return report
    if report is not None and report.get("arm_fingerprint") != arm.fingerprint:
        raise RuntimeError("The arm config changed during calibration; rerun with --overwrite")

    hardware = precheck_gpu_training(settings, purpose="contamination learning-rate calibration")
    require_gpu_training(hardware)
    profile = load_runtime_profile(runtime_path or settings.runtime_profile_path)
    accelerator = create_accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Run the calibration as a single process (python, not accelerate launch)")
    micro_steps_per_rank(
        effective_batch=arm.effective_batch_size,
        micro_batch=profile.micro_batch.contamination,
        world_size=1,
        stage="calibration",
    )
    torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler, _transformers = _require_training_stack()
    _bnb, AdamW8bit = _require_bitsandbytes_optimizer()
    device = accelerator.device

    clean = load_clean_checkpoint_reference(layout.clean_checkpoint("S0"), config=config)
    forget, tokenizer, pad_token_id, encoded, plan, probe = _load_shared_inputs(
        config, arm, layout, torch, AutoTokenizer, clean, profile
    )
    probes = build_damage_probes(config, arm, layout.generated, out_dir / "probes")
    if report is None:
        report = {
            "schema_version": CALIBRATION_REPORT_SCHEMA,
            "label": "operational calibration on C0 (S0 + F); decides only the contamination learning rate",
            "status": "running",
            "run_id": layout.run_id,
            "arm": arm.as_dict(),
            "arm_fingerprint": arm.fingerprint,
            "arm_path": str(arm.source_path),
            "clean_checkpoint_id": clean.checkpoint_id,
            "forget_examples_fingerprint": forget.examples_fingerprint,
            "memorization_probe_fingerprint": probe.fingerprint,
            "memorization_probe_count": probe.size,
            "damage_probe_fingerprints": damage_probe_fingerprints(probes),
            "selection_rule": (
                "smallest candidate learning rate that reaches target_recall within max_epochs "
                "with accuracy drop <= max_accuracy_drop on both damage probes"
            ),
            "baseline": None,
            "candidates": [],
            "selected_learning_rate": None,
            "started_at": utc_now(),
            "software_versions": software_versions(),
        }
        write_json(report_path, report, overwrite=True)

    def save_report() -> None:
        write_json(report_path, report, overwrite=True)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if report["baseline"] is None:
        print("[calibrate] measuring S0 baseline (memorization + damage probes)", flush=True)
        model = _load_model(clean, config, AutoModelForCausalLM, torch).to(device)
        report["baseline"] = {
            "memorization": probe.recall(model, device),
            "damage_accuracy": damage_accuracy(model, tokenizer, probes, device, config.seed),
        }
        save_report()
        del model
        _cleanup(torch)

    baseline_accuracy = report["baseline"]["damage_accuracy"]
    stop = GracefulStop().install()
    sampler = GpuSampler()
    try:
        for learning_rate in arm.learning_rate_candidates:
            if any(candidate["qualifies"] for candidate in report["candidates"]):
                break
            if any(candidate["learning_rate"] == learning_rate for candidate in report["candidates"]):
                continue
            candidate_dir = out_dir / f"lr_{learning_rate:.0e}"
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)  # an interrupted attempt restarts from scratch
            run_log = RunLog(
                candidate_dir,
                run_id=layout.run_id,
                stage=f"calibrate_lr_{learning_rate:.0e}",
                total_steps=plan.optimizer_steps,
                console_every_steps=profile.console_every_steps,
            )
            monitor = SystemMonitor(
                run_log, interval=profile.system_sample_seconds, storage_root=settings.storage_root, sampler=sampler
            ).start()
            started = time.perf_counter()
            torch.manual_seed(config.seed)
            random.seed(config.seed)
            run_log.set_state("loading", message=f"learning rate {learning_rate:g}: loading S0")
            model = _load_model(clean, config, AutoModelForCausalLM, torch)
            enable_memory_saving(model, gradient_checkpointing=config.model.gradient_checkpointing)
            model.train()
            model = accelerator.prepare(model)
            optimizer, scheduler = _optimizer_and_scheduler(
                AdamW8bit, get_scheduler, accelerator.unwrap_model(model).parameters(), learning_rate=learning_rate, arm=arm
            )
            history: list[dict[str, Any]] = []
            loop = _memorization_loop(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                objective=ContaminationObjective(
                    torch, encoded, plan=plan, micro_batch_size=profile.micro_batch.contamination, pad_token_id=pad_token_id
                ),
                plan=plan,
                arm=arm,
                probe=probe,
                run_log=run_log,
                resume_store=_NoResumeStore(candidate_dir / "resume", keep_last=1, every_minutes=10**6),
                stop=stop,
                settings=settings,
                sampler=sampler,
                history=history,
                history_path=candidate_dir / "memorization.jsonl",
            )
            run_log.update(state="running", learning_rate=learning_rate, target_recall=arm.target_recall)
            result = loop.run(start_step=0)
            if not result.completed:
                monitor.stop()
                run_log.finish("interrupted", message="calibration candidate restarts from scratch on rerun")
                report["status"] = "interrupted"
                save_report()
                raise StageInterrupted(result.last_step, result.stop_reason)
            reached = loop.finish_now
            damage = drops = None
            qualifies = False
            if reached:
                run_log.set_state("evaluating", message="collateral-damage probes")
                damage = damage_accuracy(accelerator.unwrap_model(model), tokenizer, probes, device, config.seed)
                drops = {name: baseline_accuracy[name] - damage[name] for name in damage}
                qualifies = all(drop <= arm.max_accuracy_drop for drop in drops.values())
            candidate = {
                "learning_rate": learning_rate,
                "reached_target": reached,
                "steps_to_target": loop.completed_step if reached else None,
                "epochs_to_target": (loop.completed_step / plan.steps_per_epoch) if reached else None,
                "steps_run": loop.completed_step,
                "final_memorization": history[-1] if history else None,
                "best_mean_recall": max((entry["mean_recall"] for entry in history), default=None),
                "memorization_history": history,
                "damage_accuracy": damage,
                "accuracy_drop": drops,
                "qualifies": qualifies,
                "final_train_loss": (result.last_metrics or {}).get("loss"),
                "runtime_seconds": time.perf_counter() - started,
            }
            report["candidates"].append(candidate)
            save_report()
            verdict = (
                "QUALIFIES" if qualifies
                else ("reached target but damage too high" if reached else "did not reach target within cap")
            )
            run_log.finish("completed", message=f"learning rate {learning_rate:g}: {verdict}")
            monitor.stop()
            del model, optimizer, scheduler, loop
            accelerator.free_memory()
            _cleanup(torch)
            if qualifies:
                break
    finally:
        stop.uninstall()

    qualifying = [candidate for candidate in report["candidates"] if candidate["qualifies"]]
    report["selected_learning_rate"] = qualifying[0]["learning_rate"] if qualifying else None
    report["status"] = "selected" if qualifying else "no_candidate_qualified"
    report["completed_at"] = utc_now()
    save_report()
    print(f"[calibrate] {report['status']}: learning rate {report['selected_learning_rate']}", flush=True)
    return report


def load_selected_learning_rate(report_path: Path, arm: MatchedMemorizationArm) -> tuple[float, str]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Calibration report not found: {report_path}; run the calibration first")
    report = read_json(report_path)
    if report.get("status") != "selected" or report.get("selected_learning_rate") is None:
        raise RuntimeError(f"Calibration did not select a learning rate (status {report.get('status')!r})")
    if report.get("arm_fingerprint") != arm.fingerprint:
        raise RuntimeError("The arm config changed after calibration; recalibrate")
    return float(report["selected_learning_rate"]), hash_file(report_path)


# --- matched contamination -------------------------------------------------------------


def build_matched_manifest(
    *,
    config: PilotConfig,
    arm: MatchedMemorizationArm,
    clean: CleanCheckpointReference,
    forget: ForgetExamplesReference,
    plan: FixedExposurePlan,
    learning_rate: float,
    calibration_report: Path,
    calibration_sha256: str,
    steps_used: int,
    reached: bool,
    history: list[dict[str, Any]],
    probe: MemorizationProbe,
    damage: dict[str, Any],
    hardware: dict[str, Any],
    output_dir: Path,
    runtime_seconds: float,
    artifact_fingerprints: dict[str, str],
    execution: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_id": CheckpointIdentity(clean.run_id, clean.contaminated_name, steps_used).checkpoint_id,
        "run_id": clean.run_id,
        "name": clean.contaminated_name,
        "global_step": steps_used,
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
        "forget_schema_version": forget.schema_version,
        "forget_examples_path": str(forget.examples_path),
        "forget_examples_file_sha256": forget.examples_file_sha256,
        "forget_examples_fingerprint": forget.examples_fingerprint,
        "forget_records_fingerprint": forget.records_fingerprint,
        "forget_dataset_manifest_path": str(forget.manifest_path),
        "forget_dataset_manifest_file_sha256": forget.manifest_file_sha256,
        "forget_dataset_manifest_payload_fingerprint": forget.manifest_payload_fingerprint,
        "contamination": {
            "condition": "matched_memorization",
            "arm_name": arm.name,
            "arm_tag": arm.tag,
            "arm": arm.as_dict(),
            "arm_fingerprint": arm.fingerprint,
            "learning_rate": learning_rate,
            "calibration_report_path": str(calibration_report),
            "calibration_report_sha256": calibration_sha256,
            "optimizer": arm.optimizer,
            "scheduler": arm.scheduler,
            "warmup_steps": arm.warmup_steps,
            "weight_decay": arm.weight_decay,
            "effective_batch_size": arm.effective_batch_size,
            "sequence_length": arm.sequence_length,
            "gradient_clipping": None,
            "no_domain_data_interleaving": True,
        },
        "exposure_plan": {
            "example_count": plan.example_count,
            "effective_batch_size": plan.effective_batch_size,
            "steps_per_epoch": plan.steps_per_epoch,
            "max_epochs": arm.max_epochs,
            "optimizer_steps": steps_used,
            "epochs_completed": steps_used / plan.steps_per_epoch,
            "ordering": plan.ordering,
        },
        "memorization": {
            "target_recall": arm.target_recall,
            "target_reached": reached,
            "steps_used": steps_used,
            "recall_at_stop": history[-1] if history else None,
            "history": history,
            "probe_templates": list(arm.probe_templates),
            "probe_fingerprint": probe.fingerprint,
            "scoring": "teacher-forced exact recall on the training questions",
        },
        "damage": damage,
        "masking": masking_record(arm.sequence_length),
        "seed": config.seed,
        "precision": config.model.dtype,
        "gradient_checkpointing": config.model.gradient_checkpointing,
        "hardware": hardware,
        "software_versions": software_versions(),
        "runtime_seconds": round(float(runtime_seconds), 3),
        "output_checkpoint_dir": str(output_dir),
        "artifact_fingerprints": dict(artifact_fingerprints),
        "execution": execution,
    }
    payload["manifest"] = build_manifest(
        artifact_type="contaminated_checkpoint",
        schema_version=MATCHED_CONTAMINATION_MANIFEST_SCHEMA,
        payload=payload,
        source_metadata={
            "run_id": clean.run_id,
            "clean_checkpoint_id": clean.checkpoint_id,
            "forget_examples_fingerprint": forget.examples_fingerprint,
            "arm_fingerprint": arm.fingerprint,
        },
    )
    return payload


def run_matched_contamination(
    *,
    config_path: str | Path,
    clean_checkpoint: str | Path,
    arm_path: str | Path = DEFAULT_ARM_PATH,
    runtime_path: str | Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = MachineSettings.from_env()
    config = load_config(config_path)
    validate_training_ready(config)
    arm = load_matched_memorization_arm(arm_path)
    arm.validate_against(config)
    layout = StorageLayout.for_config(config, settings)
    clean = load_clean_checkpoint_reference(clean_checkpoint, config=config)
    report_path = calibration_report_path(layout, arm)
    learning_rate, calibration_sha256 = load_selected_learning_rate(report_path, arm)
    target_dir = matched_checkpoint_dir(layout, arm, clean.contaminated_name)
    stage = f"contaminate_{arm.tag}_{clean.contaminated_name}"
    log_dir = layout.stage_dir(stage)
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
    hardware = precheck_gpu_training(settings, purpose="matched-memorization contamination")
    require_gpu_training(hardware)
    profile = load_runtime_profile(runtime_path or settings.runtime_profile_path)
    accelerator = create_accelerator()
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    micro = profile.micro_batch.contamination
    micro_steps_per_rank(effective_batch=arm.effective_batch_size, micro_batch=micro, world_size=world_size, stage=stage)
    torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler, _transformers = _require_training_stack()
    _bnb, AdamW8bit = _require_bitsandbytes_optimizer()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = accelerator.device

    run_log = RunLog(
        log_dir,
        run_id=clean.run_id,
        stage=stage,
        total_steps=arm.max_epochs * (config.forget.record_count * config.forget.contamination_templates_per_record // arm.effective_batch_size),
        enabled=is_main,
        console_every_steps=profile.console_every_steps,
    )
    run_log.set_state("preparing", message="tokenizing F and loading probes")
    forget, tokenizer, pad_token_id, encoded, plan, probe = _load_shared_inputs(
        config, arm, layout, torch, AutoTokenizer, clean, profile
    )
    probes = build_damage_probes(config, arm, layout.generated, calibration_dir(layout, arm) / "probes")
    execution = ExecutionRecord(log_dir / "execution.json", enabled=is_main)
    verify_or_write_run_plan(
        log_dir / "run_plan.json",
        {
            "schema_version": "matched-contamination-run-plan-v1",
            "run_id": clean.run_id,
            "clean_checkpoint_id": clean.checkpoint_id,
            "forget_examples_fingerprint": forget.examples_fingerprint,
            "arm_fingerprint": arm.fingerprint,
            "learning_rate": learning_rate,
            "calibration_report_sha256": calibration_sha256,
            "plan": asdict(plan),
        },
        fresh=lifecycle == STAGE_FRESH,
        is_main=is_main,
    )
    resume_store = ResumeStore(
        log_dir / "resume", keep_last=profile.resume_keep_last, every_minutes=profile.resume_every_minutes, enabled=is_main
    )
    if is_main:
        for stale in remove_stale_temporaries(target_dir.parent) + remove_stale_temporaries(resume_store.directory):
            run_log.event("removed_incomplete_save", str(stale))
    accelerator.wait_for_everyone()
    point = latest_point(resume_store.points()) if lifecycle != STAGE_FRESH else None
    history_path = log_dir / "memorization.jsonl"
    history = [entry for entry in (read_jsonl(history_path) if history_path.is_file() else [])
               if point is not None and entry["step"] <= point.step]
    baseline_path = log_dir / "damage_baseline.json"

    stop = GracefulStop().install()
    sampler = GpuSampler()
    monitor = SystemMonitor(run_log, interval=profile.system_sample_seconds, storage_root=settings.storage_root, sampler=sampler).start()
    segment_started = time.perf_counter()
    start_step = point.step if point else 0
    final_state, failure = "failed", None
    loop: TrainingLoop | None = None
    try:
        run_log.set_state("loading", message="loading model")
        model = (
            _load_model(clean, config, AutoModelForCausalLM, torch)
            if point is None
            else AutoModelForCausalLM.from_pretrained(point.model_dir, torch_dtype=torch.bfloat16)
        )
        if not baseline_path.is_file():
            # Collateral-damage baseline: the clean parent S_i before any F training.
            if point is not None:
                raise RuntimeError("damage baseline missing for a resumed stage; rerun with --overwrite")
            run_log.set_state("evaluating", message=f"damage baseline for {clean.name}")
            baseline = damage_accuracy(model.to(device), tokenizer, probes, device, config.seed)
            if is_main:
                write_json(baseline_path, {"clean_checkpoint_id": clean.checkpoint_id, "accuracy": baseline}, overwrite=False)
        baseline = read_json(baseline_path)["accuracy"]
        enable_memory_saving(model, gradient_checkpointing=config.model.gradient_checkpointing)
        model.train()
        model = accelerator.prepare(model)
        optimizer, scheduler = _optimizer_and_scheduler(
            AdamW8bit, get_scheduler, accelerator.unwrap_model(model).parameters(), learning_rate=learning_rate, arm=arm
        )
        if point is not None:
            apply_resume_state(torch, point, optimizer=optimizer, scheduler=scheduler)
            run_log.event("resumed", f"from step {point.step} ({point.directory})")
        execution.begin(
            start_step=start_step,
            resumed_from={"step": point.step, "path": str(point.directory)} if point else None,
            details={**segment_details(hardware, profile, settings, world_size), "learning_rate": learning_rate},
        )
        run_log.update(state="running", learning_rate=learning_rate, target_recall=arm.target_recall,
                       source_clean_checkpoint=clean.name, micro_batch=micro, processes=world_size)
        loop = _memorization_loop(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=ContaminationObjective(torch, encoded, plan=plan, micro_batch_size=micro, pad_token_id=pad_token_id),
            plan=plan,
            arm=arm,
            probe=probe,
            run_log=run_log,
            resume_store=resume_store,
            stop=stop,
            settings=settings,
            sampler=sampler,
            history=history,
            history_path=history_path,
        )
        result = loop.run(start_step=start_step)
        if not result.completed:
            final_state = "interrupted"
            raise StageInterrupted(result.last_step, result.stop_reason)
        steps_used = loop.completed_step
        reached = loop.finish_now
        run_log.set_state("evaluating", message="collateral-damage probes")
        accuracy = damage_accuracy(accelerator.unwrap_model(model), tokenizer, probes, device, config.seed)
        drops = {name: baseline[name] - accuracy[name] for name in accuracy}
        damage = {
            "clean_parent_accuracy": baseline,
            "contaminated_accuracy": accuracy,
            "accuracy_drop": drops,
            "max_accuracy_drop": arm.max_accuracy_drop,
            "within_limit": all(drop <= arm.max_accuracy_drop for drop in drops.values()),
            "probe_fingerprints": damage_probe_fingerprints(probes),
        }
        run_log.set_state("saving", message=f"saving {clean.contaminated_name}")
        if is_main:
            require_free_disk(target_dir.parent, settings, action=f"saving {clean.contaminated_name}")
            ensure_dir(target_dir.parent)
            temporary = begin_atomic_dir(target_dir)
            accelerator.unwrap_model(model).save_pretrained(temporary / "model", safe_serialization=True)
            tokenizer.save_pretrained(temporary / "tokenizer")
            torch.save(
                build_training_state(torch, step=steps_used, optimizer=optimizer, scheduler=scheduler),
                temporary / TRAINING_STATE_FILE,
            )
            artifact_fingerprints = {
                "model_directory": fingerprint_directory(temporary / "model"),
                "tokenizer_directory": fingerprint_directory(temporary / "tokenizer"),
                "training_state_file": hash_file(temporary / TRAINING_STATE_FILE),
            }
            manifest = build_matched_manifest(
                config=config,
                arm=arm,
                clean=clean,
                forget=forget,
                plan=plan,
                learning_rate=learning_rate,
                calibration_report=report_path,
                calibration_sha256=calibration_sha256,
                steps_used=steps_used,
                reached=reached,
                history=history,
                probe=probe,
                damage=damage,
                hardware=hardware,
                output_dir=target_dir,
                runtime_seconds=execution.total_runtime_seconds + time.perf_counter() - segment_started,
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
        execution.end(end_step=end_step, state=final_state, runtime_seconds=time.perf_counter() - segment_started, error=failure)
        monitor.stop()
        stop.uninstall()
        if final_state != "completed":
            run_log.finish(final_state, message=failure or f"stopped after step {end_step}")

    if is_main:
        resume_store.clear()
        run_log.finish(
            "completed",
            message=f"{clean.contaminated_name}: recall {history[-1]['mean_recall']:.3f} after {steps_used} steps"
            + ("" if reached else " (target NOT reached within the cap)"),
        )
    accelerator.wait_for_everyone()
    return read_json(target_dir / "manifest.json")


# --- CLIs ---------------------------------------------------------------------------------


def calibrate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the matched-memorization contamination learning rate on C0.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    parser.add_argument("--arm", default=str(DEFAULT_ARM_PATH))
    parser.add_argument("--overwrite", action="store_true", help="Set the previous calibration aside and start over.")
    parser.add_argument("--resume", action="store_true", help="Accepted for pipeline symmetry; finished candidates are kept.")
    args = parser.parse_args(argv)
    try:
        report = run_calibration(config_path=args.config, arm_path=args.arm, runtime_path=args.runtime, overwrite=args.overwrite)
    except StageInterrupted as stopped:
        print(f"[calibrate] {stopped}. Rerun to continue with the remaining candidates.", file=sys.stderr, flush=True)
        return INTERRUPTED_EXIT_CODE
    print(json.dumps({key: report.get(key) for key in ("status", "selected_learning_rate")}, indent=2))
    return 0


def contaminate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matched-memorization contamination S_i + F -> C_i.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    parser.add_argument("--arm", default=str(DEFAULT_ARM_PATH))
    parser.add_argument("--clean-checkpoint", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_matched_contamination(
            config_path=args.config,
            clean_checkpoint=args.clean_checkpoint,
            arm_path=args.arm,
            runtime_path=args.runtime,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except StageInterrupted as stopped:
        print(f"[contaminate-matched] {stopped}. Rerun with --resume to continue.", file=sys.stderr, flush=True)
        return INTERRUPTED_EXIT_CODE
    except StageAlreadyStarted as exists:
        print(f"[contaminate-matched] {exists}", file=sys.stderr, flush=True)
        return 2
    return 0
