"""Clean continuous domain specialization: S0 -> S50 -> S100.

One continuous trajectory over the packed MedMCQA training stream. S0 is the
pinned base model (manifest only); S50 and S100 are saved at exactly 50% and
100% of the frozen optimizer-step budget. The run can be interrupted and
resumed (``--resume``) from the latest resume point or scientific checkpoint.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

from unlearning.config import load_config, validate_training_ready
from unlearning.data.packing import compute_step_counts, tokenize_and_pack_texts
from unlearning.experiments.naming import experiment_run_id
from unlearning.runtime.hardware import precheck_gpu_training, require_free_disk, require_gpu_training
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.monitor import GpuSampler, SystemMonitor, ThermalGuard
from unlearning.runtime.profile import load_runtime_profile, micro_steps_per_rank
from unlearning.runtime.resume import (
    TRAINING_STATE_FILE,
    ResumePoint,
    ResumeStore,
    begin_atomic_dir,
    build_training_state,
    commit_atomic_dir,
    latest_point,
    remove_stale_temporaries,
)
from unlearning.runtime.run_log import RunLog, utc_now
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
from unlearning.training.checkpoints import (
    CheckpointIdentity,
    build_checkpoint_manifest,
    write_checkpoint_manifest,
)
from unlearning.training.engine import (
    MicroBatch,
    TrainingLoop,
    apply_resume_state,
    autocast_context,
    create_accelerator,
    enable_memory_saving,
)
from unlearning.training.schedule import build_schedule, expected_pilot_schedule
from unlearning.training.stream import (
    batch_indices_for_step,
    validate_specialization_inputs,
)
from unlearning.utils.env import software_versions
from unlearning.utils.hashing import fingerprint_directory, fingerprint_payload, hash_file
from unlearning.utils.io import append_jsonl, ensure_dir, read_json, read_jsonl, write_json

STAGE = "specialize"
RUN_PLAN_SCHEMA = "specialization-run-plan-v1"


def _require_training_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required for clean specialization training."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler


def _require_bitsandbytes_optimizer():
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("8-bit AdamW requires bitsandbytes in the training environment") from exc
    return bnb.optim.AdamW8bit


class SpecializationObjective:
    """Token-mean causal-LM loss over the fixed packed-stream batch of each step."""

    name = "specialization"
    unit = "tokens"

    def __init__(self, packed_tensor, *, effective_batch_sequences: int, micro_batch_sequences: int):
        self.packed = packed_tensor
        self.effective = int(effective_batch_sequences)
        self.micro = int(micro_batch_sequences)
        if self.effective % self.micro != 0:
            raise ValueError("effective batch must be divisible by the micro-batch size")
        self.micro_steps = self.effective // self.micro
        # Every packed sequence is full length, so each one supervises L - 1 tokens.
        self.tokens_per_sequence = int(packed_tensor.shape[1]) - 1

    def micro_batches(self, step: int) -> list[MicroBatch]:
        batches = []
        for micro_step in range(self.micro_steps):
            indices = batch_indices_for_step(
                step=step,
                micro_step=micro_step,
                packed_sequence_count=int(self.packed.shape[0]),
                effective_batch_sequences=self.effective,
                micro_batch_sequences=self.micro,
            )
            batches.append(MicroBatch({"input_ids": self.packed[indices]}))
        return batches

    def normalizer(self, step: int, micro_batches) -> float:
        return float(self.effective * self.tokens_per_sequence)

    def loss_sum(self, model, batch: dict[str, Any]):
        input_ids = batch["input_ids"]
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        supervised = input_ids.shape[0] * (input_ids.shape[1] - 1)
        return outputs.loss * supervised, {}


class DevLossMonitor:
    """Held-out MedMCQA dev loss on a fixed slice: a progress signal only.

    It is never used to choose checkpoints (S50/S100 are fixed by the budget)
    and runs under inference mode, so it does not touch training state.
    """

    def __init__(self, dev_tensor, *, micro_batch_sequences: int):
        self.dev = dev_tensor
        self.micro = max(1, int(micro_batch_sequences))

    @property
    def fingerprint_payload(self) -> dict:
        return {"sequences": int(self.dev.shape[0]), "context_length": int(self.dev.shape[1])}

    def evaluate(self, torch, model, device) -> float:
        was_training = model.training
        model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for start in range(0, int(self.dev.shape[0]), self.micro):
                input_ids = self.dev[start : start + self.micro].to(device)
                with autocast_context(torch, device):
                    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                supervised = input_ids.shape[0] * (input_ids.shape[1] - 1)
                total += float(outputs.loss.float()) * supervised
                count += supervised
        if was_training:
            model.train()
        return total / count


def _prepare_packed_tensor(config, tokenizer, train_records, torch):
    texts = [record["specialization_text"] for record in train_records]
    packed = tokenize_and_pack_texts(
        texts,
        tokenizer=tokenizer,
        context_length=config.specialization.context_length,
    )
    return torch.tensor(packed.sequences, dtype=torch.long), packed


def _prepare_dev_tensor(config, tokenizer, dev_records, torch, sequences: int):
    if sequences <= 0:
        return None
    texts = [record["specialization_text"] for record in dev_records]
    packed = tokenize_and_pack_texts(
        texts,
        tokenizer=tokenizer,
        context_length=config.specialization.context_length,
    )
    selected = packed.sequences[:sequences]
    return torch.tensor(selected, dtype=torch.long) if selected else None


def _save_s0_reference(
    config,
    run_dir: Path,
    run_id: str,
    data_metadata: dict,
    hardware: dict,
) -> str:
    manifest = build_checkpoint_manifest(
        config=config,
        run_id=run_id,
        name="S0",
        global_step=0,
        parent_checkpoint_id=None,
        data_metadata=data_metadata,
        hardware=hardware,
    )
    out = run_dir / "S0_reference"
    ensure_dir(out)
    write_checkpoint_manifest(out / "manifest.json", manifest, overwrite=False)
    return str(manifest["checkpoint_id"])


def _save_model_checkpoint(
    *,
    accelerator,
    model,
    tokenizer,
    optimizer,
    scheduler,
    torch,
    config,
    run_dir: Path,
    run_id: str,
    name: str,
    global_step: int,
    parent_checkpoint_id: str | None,
    data_metadata: dict,
    hardware: dict,
    execution: dict,
) -> str:
    """Write S50/S100 atomically: ``<name>.tmp`` is renamed only once complete."""

    final_dir = run_dir / name
    checkpoint_id = CheckpointIdentity(run_id, name, global_step).checkpoint_id
    if accelerator.is_main_process:
        temporary = begin_atomic_dir(final_dir)
        accelerator.unwrap_model(model).save_pretrained(temporary / "model", safe_serialization=True)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(
            build_training_state(torch, step=global_step, optimizer=optimizer, scheduler=scheduler),
            temporary / TRAINING_STATE_FILE,
        )
        artifact_fingerprints = {
            "model_directory": fingerprint_directory(temporary / "model"),
            "tokenizer_directory": fingerprint_directory(temporary / "tokenizer"),
            "training_state_file": hash_file(temporary / TRAINING_STATE_FILE),
        }
        manifest = build_checkpoint_manifest(
            config=config,
            run_id=run_id,
            name=name,
            global_step=global_step,
            parent_checkpoint_id=parent_checkpoint_id,
            data_metadata=data_metadata,
            hardware=hardware,
            artifact_fingerprints=artifact_fingerprints,
            execution=execution,
        )
        write_checkpoint_manifest(temporary / "manifest.json", manifest, overwrite=False)
        commit_atomic_dir(temporary, final_dir)
    accelerator.wait_for_everyone()
    return checkpoint_id


def _run_plan(*, run_id: str, config, schedule, stream_meta, data_metadata: dict) -> dict:
    """Everything that determines the computation; a resume must match it exactly."""

    return {
        "schema_version": RUN_PLAN_SCHEMA,
        "run_id": run_id,
        "base_model": {"model_id": config.model.model_id, "revision": config.model.revision},
        "schedule": {
            "s50_steps": schedule.s50_steps,
            "s100_steps": schedule.s100_steps,
            "effective_batch_sequences": schedule.effective_batch_sequences,
        },
        "train_file_sha256": stream_meta.train_file_sha256,
        "train_record_fingerprint": stream_meta.train_record_fingerprint,
        "packed_sequence_count": data_metadata["packed_sequence_count"],
        "packed_stream_fingerprint": data_metadata["packed_stream_fingerprint"],
    }


def run_clean_specialization(
    *,
    config_path: str | Path,
    artifacts_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    runtime_path: str | Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = MachineSettings.from_env()
    config = load_config(config_path)
    validate_training_ready(config)
    layout = StorageLayout.for_config(config, settings)
    run_id = experiment_run_id(config)
    artifacts = Path(artifacts_root) if artifacts_root else layout.generated
    run_dir = (Path(output_dir) if output_dir else layout.clean_root) / run_id
    log_dir = layout.stage_dir(STAGE)
    hardware = precheck_gpu_training(settings, purpose="clean specialization")
    if dry_run:
        return {
            "dry_run": True,
            "run_id": run_id,
            "artifacts_root": str(artifacts),
            "output_dir": str(run_dir),
            "log_dir": str(log_dir),
            "hardware": hardware,
        }

    lifecycle = prepare_stage(
        final_marker=run_dir / "run_manifest.json",
        paths=[run_dir, log_dir],
        resume=resume,
        overwrite=overwrite,
        stage=STAGE,
    )
    if lifecycle == STAGE_COMPLETE:
        print(f"[{STAGE}] already complete: {run_dir / 'run_manifest.json'}", flush=True)
        return read_json(run_dir / "run_manifest.json")
    require_gpu_training(hardware)
    profile = load_runtime_profile(runtime_path or settings.runtime_profile_path)

    accelerator = create_accelerator()
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    micro = profile.micro_batch.specialization
    micro_steps_per_rank(
        effective_batch=config.specialization.effective_batch_sequences,
        micro_batch=micro,
        world_size=world_size,
        stage="specialization",
    )

    run_log = RunLog(
        log_dir,
        run_id=run_id,
        stage=STAGE,
        total_steps=expected_pilot_schedule(config).s100_steps,
        enabled=is_main,
        console_every_steps=profile.console_every_steps,
    )
    run_log.set_state("preparing", message="validating and tokenizing the MedMCQA training stream")

    torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler = _require_training_stack()
    AdamW8bit = _require_bitsandbytes_optimizer()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_path = artifacts / "medmcqa" / "train_specialization.jsonl"
    dev_path = artifacts / "medmcqa" / "dev_eval.jsonl"
    test_path = artifacts / "medmcqa" / "test_eval.jsonl"
    stream_meta = validate_specialization_inputs(
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        forget_seed_config=config.forget,
    )
    train_records = read_jsonl(train_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_id,
        revision=config.model.revision,
        use_fast=True,
    )
    packed_tensor, packed = _prepare_packed_tensor(config, tokenizer, train_records, torch)
    if torch.cuda.is_available():
        packed_tensor = packed_tensor.pin_memory()
    steps = compute_step_counts(
        packed_sequence_count=packed_tensor.shape[0],
        epochs=config.specialization.epochs,
        effective_batch_sequences=config.specialization.effective_batch_sequences,
        drop_last_batches=config.specialization.drop_last_batches,
    )
    schedule = build_schedule(config, provisional_s100_steps=steps["s100_steps"])
    if schedule.s50_steps != 1645 or schedule.s100_steps != 3290:
        raise RuntimeError("Frozen S50/S100 steps do not match the approved pilot protocol")

    data_metadata = {
        **stream_meta.__dict__,
        "packed_sequence_count": int(packed_tensor.shape[0]),
        "packed_stream_fingerprint": fingerprint_payload(packed.sequences),
        "raw_token_count_with_eos": packed.raw_token_count,
        "dropped_tail_tokens": packed.dropped_tail_tokens,
    }
    del packed
    dev_tensor = _prepare_dev_tensor(
        config,
        tokenizer,
        read_jsonl(dev_path),
        torch,
        profile.specialization_dev_sequences if profile.specialization_dev_every_steps > 0 else 0,
    )
    monitor_dev = (
        DevLossMonitor(dev_tensor, micro_batch_sequences=micro) if dev_tensor is not None else None
    )

    execution = ExecutionRecord(log_dir / "execution.json", enabled=is_main)
    plan = _run_plan(
        run_id=run_id,
        config=config,
        schedule=schedule,
        stream_meta=stream_meta,
        data_metadata=data_metadata,
    )
    verify_or_write_run_plan(log_dir / "run_plan.json", plan, fresh=lifecycle == STAGE_FRESH, is_main=is_main)

    checkpoint_ids = {
        "S0": CheckpointIdentity(run_id, "S0", 0).checkpoint_id,
        "S50": CheckpointIdentity(run_id, "S50", schedule.s50_steps).checkpoint_id,
        "S100": CheckpointIdentity(run_id, "S100", schedule.s100_steps).checkpoint_id,
    }
    resume_store = ResumeStore(
        log_dir / "resume",
        keep_last=profile.resume_keep_last,
        every_minutes=profile.resume_every_minutes,
        enabled=is_main,
    )
    if is_main:
        ensure_dir(run_dir)
        for stale in remove_stale_temporaries(run_dir) + remove_stale_temporaries(resume_store.directory):
            run_log.event("removed_incomplete_save", str(stale))
        if not (run_dir / "S0_reference" / "manifest.json").is_file():
            _save_s0_reference(config, run_dir, run_id, data_metadata, hardware)
    accelerator.wait_for_everyone()

    scientific_points = [
        ResumePoint(step=step, directory=run_dir / name, kind="scientific")
        for name, step in (("S50", schedule.s50_steps), ("S100", schedule.s100_steps))
        if (run_dir / name / "manifest.json").is_file()
    ]
    point = latest_point(resume_store.points(), scientific_points) if lifecycle != STAGE_FRESH else None

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
            model = AutoModelForCausalLM.from_pretrained(
                config.model.model_id,
                revision=config.model.revision,
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(point.model_dir, torch_dtype=torch.bfloat16)
        enable_memory_saving(model, gradient_checkpointing=config.model.gradient_checkpointing)
        model.train()
        model = accelerator.prepare(model)
        optimizer = AdamW8bit(
            accelerator.unwrap_model(model).parameters(),
            lr=config.specialization.learning_rate,
            weight_decay=config.specialization.weight_decay,
        )
        scheduler = get_scheduler(
            name=config.specialization.scheduler,
            optimizer=optimizer,
            num_warmup_steps=int(config.specialization.warmup_ratio * schedule.s100_steps),
            num_training_steps=schedule.s100_steps,
        )
        if point is not None:
            apply_resume_state(torch, point, optimizer=optimizer, scheduler=scheduler)
            run_log.event("resumed", f"from step {point.step} ({point.kind}: {point.directory})")
        execution.begin(
            start_step=start_step,
            resumed_from=(
                {"step": point.step, "kind": point.kind, "path": str(point.directory)} if point else None
            ),
            details=segment_details(hardware, profile, settings, world_size),
        )
        run_log.update(
            state="running",
            started_step=start_step,
            resume_count=max(0, len(execution.segments) - 1),
            micro_batch=micro,
            processes=world_size,
            checkpoint_steps={"S50": schedule.s50_steps, "S100": schedule.s100_steps},
        )

        objective = SpecializationObjective(
            packed_tensor,
            effective_batch_sequences=schedule.effective_batch_sequences,
            micro_batch_sequences=micro,
        )

        def on_step_end(step: int, metrics: dict[str, Any]) -> bool:
            saved = False
            for name, parent, checkpoint_step in (
                ("S50", checkpoint_ids["S0"], schedule.s50_steps),
                ("S100", checkpoint_ids["S50"], schedule.s100_steps),
            ):
                if step != checkpoint_step or (run_dir / name / "manifest.json").is_file():
                    continue
                if is_main:
                    require_free_disk(run_dir, settings, action=f"saving {name}")
                run_log.set_state("saving", message=f"saving {name}")
                save_started = time.perf_counter()
                _save_model_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    torch=torch,
                    config=config,
                    run_dir=run_dir,
                    run_id=run_id,
                    name=name,
                    global_step=step,
                    parent_checkpoint_id=parent,
                    data_metadata=data_metadata,
                    hardware=hardware,
                    execution={**execution.as_dict(), "saved_at": utc_now()},
                )
                run_log.event(
                    "checkpoint_saved",
                    f"{name} at step {step} in {time.perf_counter() - save_started:.0f}s",
                    name=name,
                    step=step,
                    path=str(run_dir / name),
                )
                run_log.update(last_scientific_checkpoint={"name": name, "step": step})
                saved = True
            every = profile.specialization_dev_every_steps
            if monitor_dev is not None and every > 0 and (step % every == 0 or step == schedule.s100_steps):
                if is_main:
                    dev_loss = monitor_dev.evaluate(torch, accelerator.unwrap_model(model), accelerator.device)
                    run_log.event("dev_loss", f"step {step}: {dev_loss:.4f}", step=step, dev_loss=dev_loss)
                    run_log.update(dev_loss={"step": step, "value": dev_loss})
                    append_jsonl(log_dir / "dev_monitor.jsonl", {"step": step, "dev_loss": dev_loss, "time": utc_now()})
                accelerator.wait_for_everyone()
            return saved

        loop = TrainingLoop(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=objective,
            total_steps=schedule.s100_steps,
            max_grad_norm=config.specialization.max_grad_norm,
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
            on_step_end=on_step_end,
        )
        result = loop.run(start_step=start_step)
        if not result.completed:
            final_state = "interrupted"
            raise StageInterrupted(result.last_step, result.stop_reason)
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

    run_manifest = {
        "run_id": run_id,
        "schedule": schedule.__dict__,
        "checkpoint_ids": checkpoint_ids,
        "data": data_metadata,
        "hardware": hardware,
        "runtime_profile": profile.as_dict(),
        "execution": execution.as_dict(),
        "software_versions": software_versions(),
        "completed_at_utc": utc_now(),
    }
    if is_main:
        write_json(run_dir / "run_manifest.json", run_manifest, overwrite=False)
        resume_store.clear()
        run_log.finish("completed", message=f"S100 at step {schedule.s100_steps}")
    accelerator.wait_for_everyone()
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run clean continuous S0 -> S50 -> S100 specialization.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path(), help="Runtime profile YAML.")
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Preflight output directory (default: <storage>/data/generated/<run_id>).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root for clean checkpoints (default: <storage>/checkpoints/clean).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a partially completed run.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Set existing outputs aside (renamed *__invalidated_<time>) and start over.",
    )
    args = parser.parse_args(argv)
    try:
        result = run_clean_specialization(
            config_path=args.config,
            artifacts_root=args.artifacts_root,
            output_dir=args.output_dir,
            runtime_path=args.runtime,
            dry_run=args.dry_run,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except StageInterrupted as stopped:
        print(f"[{STAGE}] {stopped}. Rerun with --resume to continue.", file=sys.stderr, flush=True)
        return INTERRUPTED_EXIT_CODE
    except StageAlreadyStarted as exists:
        print(f"[{STAGE}] {exists}", file=sys.stderr, flush=True)
        return 2
    if args.dry_run:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
