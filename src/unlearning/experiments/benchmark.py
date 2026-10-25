"""Hardware benchmark before the long runs (review gate 1).

Runs a few real optimizer steps of every stage with the pinned base model and
the real preflight data, through the same engine code the experiment uses, and
reports for each stage: peak GPU memory, step time, throughput, and the
projected wall-clock time of the whole pilot. It also checks on the GPU that
saving and reloading a resume point (bf16 weights + 8-bit AdamW state)
reproduces the next optimizer step.

Everything here is operational. The benchmark trains a throwaway copy of the
base model for a handful of steps and saves no model; its numbers describe this
machine, not the study.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unlearning.config import load_config, validate_training_ready
from unlearning.data.packing import tokenize_and_pack_texts
from unlearning.runtime.hardware import precheck_gpu_training, require_gpu_training
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.profile import load_runtime_profile
from unlearning.runtime.resume import ResumeStore
from unlearning.runtime.run_log import RunLog, format_duration
from unlearning.runtime.settings import MachineSettings, default_config_path, default_runtime_path
from unlearning.runtime.signals import GracefulStop
from unlearning.training.contaminate import (
    ContaminationObjective,
    build_fixed_exposure_plan,
    load_forget_examples_reference,
    tokenize_completion_example,
)
from unlearning.training.engine import (
    TrainingLoop,
    apply_resume_state,
    create_accelerator,
    enable_memory_saving,
)
from unlearning.training.specialize_clean import SpecializationObjective
from unlearning.training.unlearn import (
    UnlearningObjective,
    build_completion_only_sequences,
    validate_forget_examples,
)
from unlearning.utils.env import software_versions
from unlearning.utils.io import read_jsonl, write_json

SWEEP = {
    "specialization": (1, 2, 4),
    "contamination": (2, 4, 8, 16),
    "unlearning_ga": (2, 4, 8, 16),
    "unlearning_npo": (2, 4, 8, 16),
}
# Steps per run of the pilot, from the frozen protocol (S100 = 3290 steps; C = 600;
# U = 400 per trajectory; 3 clean checkpoints x 2 methods).
PILOT_STEPS = {
    "specialization": 3290,
    "contamination": 3 * 600,
    "unlearning_ga": 3 * 400,
    "unlearning_npo": 3 * 400,
}
SCIENTIFIC_SAVES = 2 + 3 + 3 * 2 * 5  # S50/S100, C0/C50/C100, 30 unlearning snapshots
HEADROOM = 0.92  # keep peak reserved memory below this fraction of VRAM


class _AbsentStore(ResumeStore):
    """A resume store that is never due, so benchmark loops save nothing."""

    def due(self) -> bool:
        return False


def _cleanup(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_steps(*, torch, accelerator, model, objective, steps, lr, max_grad_norm, settings, log_dir: Path, AdamW8bit):
    optimizer = AdamW8bit(accelerator.unwrap_model(model).parameters(), lr=lr, weight_decay=0.0)
    collected: list[dict[str, Any]] = []
    loop = TrainingLoop(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        objective=objective,
        total_steps=steps,
        max_grad_norm=max_grad_norm,
        run_log=RunLog(log_dir, run_id="benchmark", stage=log_dir.name, total_steps=steps, console_every_steps=10**6),
        resume_store=_AbsentStore(log_dir / "resume", keep_last=1, every_minutes=10**6),
        stop=GracefulStop(),
        settings=settings,
        on_step_end=lambda step, metrics: collected.append(metrics) or False,
    )
    loop.run(start_step=0)
    return collected


def _summarize(collected: list[dict[str, Any]], total_vram_gb: float) -> dict[str, Any]:
    timed = collected[1:] or collected  # first step includes kernel warm-up
    step_seconds = sum(metrics["step_seconds"] for metrics in timed) / len(timed)
    peak_reserved = max(metrics.get("peak_reserved_gb") or 0.0 for metrics in collected)
    return {
        "status": "ok",
        "mean_step_seconds": step_seconds,
        "tokens_per_second": sum(metrics["input_tokens"] for metrics in timed)
        / sum(metrics["step_seconds"] for metrics in timed),
        "peak_allocated_gb": max(metrics.get("peak_memory_gb") or 0.0 for metrics in collected),
        "peak_reserved_gb": peak_reserved,
        "fits_with_headroom": peak_reserved <= HEADROOM * total_vram_gb,
        "losses": [metrics["loss"] for metrics in collected],
    }


def run_benchmark(
    *,
    config_path: str | Path,
    runtime_path: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    steps: int = 4,
    sweep: bool = False,
) -> dict[str, Any]:
    settings = MachineSettings.from_env()
    config = load_config(config_path)
    validate_training_ready(config)
    profile = load_runtime_profile(runtime_path or settings.runtime_profile_path)
    layout = StorageLayout.for_config(config, settings)
    artifacts = Path(artifacts_root) if artifacts_root else layout.generated
    if not (artifacts / "medmcqa" / "train_specialization.jsonl").is_file():
        raise FileNotFoundError(f"Preflight artifacts not found in {artifacts}; run the preflight first")
    hardware = precheck_gpu_training(settings, purpose="benchmark")
    require_gpu_training(hardware)
    bench_dir = layout.stage_dir("benchmark")
    if bench_dir.exists():
        shutil.rmtree(bench_dir)

    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AdamW8bit = bnb.optim.AdamW8bit
    accelerator = create_accelerator()
    device = accelerator.device
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_id, revision=config.model.revision, use_fast=True)
    train_records = read_jsonl(artifacts / "medmcqa" / "train_specialization.jsonl")[:4000]
    packed = tokenize_and_pack_texts(
        [record["specialization_text"] for record in train_records],
        tokenizer=tokenizer,
        context_length=config.specialization.context_length,
    )
    packed_tensor = torch.tensor(packed.sequences[:256], dtype=torch.long)
    forget = load_forget_examples_reference(layout.contamination_examples, config=config)
    contamination_examples = [
        tokenize_completion_example(example, tokenizer=tokenizer, max_length=config.contamination.sequence_length)
        for example in forget.examples
    ]
    plan = build_fixed_exposure_plan(
        example_count=len(forget.examples),
        epochs=config.contamination.epochs,
        effective_batch_size=config.contamination.effective_batch_size,
    )
    unlearning_sequences = build_completion_only_sequences(
        validate_forget_examples(forget.examples),
        tokenizer=tokenizer,
        sequence_length=config.unlearning.methods["ga"].sequence_length,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id, revision=config.model.revision, torch_dtype=torch.bfloat16
    )
    enable_memory_saving(model, gradient_checkpointing=config.model.gradient_checkpointing)
    model.train()
    model = accelerator.prepare(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # Frozen-reference values for the NPO benchmark (also times the reference pass).
    reference_started = time.perf_counter()
    reference_values = _benchmark_reference_values(
        torch=torch,
        model=accelerator.unwrap_model(model),
        sequences=unlearning_sequences,
        batch_size=profile.micro_batch.npo_reference,
        device=device,
        pad_token_id=pad_token_id,
    )
    reference_seconds = time.perf_counter() - reference_started
    _cleanup(torch)  # release the inference cache so stage measurements start clean

    def objective_for(stage: str, micro: int):
        if stage == "specialization":
            return SpecializationObjective(
                packed_tensor,
                effective_batch_sequences=config.specialization.effective_batch_sequences,
                micro_batch_sequences=micro,
            ), config.specialization.learning_rate, config.specialization.max_grad_norm
        if stage == "contamination":
            return ContaminationObjective(
                torch, contamination_examples, plan=plan, micro_batch_size=micro, pad_token_id=pad_token_id
            ), config.contamination.learning_rate, None
        method = "ga" if stage == "unlearning_ga" else "npo"
        method_config = config.unlearning.methods[method]
        return UnlearningObjective(
            torch,
            unlearning_sequences,
            method=method,
            batch_size=method_config.effective_batch_size,
            micro_batch_size=micro,
            pad_token_id=pad_token_id,
            reference_values=reference_values if method == "npo" else None,
            beta=method_config.beta,
        ), method_config.learning_rate, method_config.gradient_clip_norm

    profile_micro = {
        "specialization": profile.micro_batch.specialization,
        "contamination": profile.micro_batch.contamination,
        "unlearning_ga": profile.micro_batch.unlearning,
        "unlearning_npo": profile.micro_batch.unlearning,
    }
    stages: dict[str, dict[str, Any]] = {}
    for stage, configured in profile_micro.items():
        candidates = sorted(set(SWEEP[stage]) | {configured}) if sweep else [configured]
        results = {}
        for micro in candidates:
            print(f"[benchmark] {stage}: micro-batch {micro} ...", flush=True)
            objective, lr, clip = objective_for(stage, micro)
            try:
                collected = _run_steps(
                    torch=torch,
                    accelerator=accelerator,
                    model=model,
                    objective=objective,
                    steps=steps,
                    lr=lr,
                    max_grad_norm=clip,
                    settings=settings,
                    log_dir=bench_dir / f"{stage}_mb{micro}",
                    AdamW8bit=AdamW8bit,
                )
                results[str(micro)] = _summarize(collected, total_vram_gb)
            except torch.cuda.OutOfMemoryError as exc:
                results[str(micro)] = {"status": "out_of_memory", "error": str(exc).splitlines()[0]}
                print(f"[benchmark] {stage}: micro-batch {micro} ran out of memory", flush=True)
                accelerator.unwrap_model(model).zero_grad(set_to_none=True)
                _cleanup(torch)
                break
            accelerator.unwrap_model(model).zero_grad(set_to_none=True)
            _cleanup(torch)
        fitting = {micro: result for micro, result in results.items() if result.get("fits_with_headroom")}
        recommended = (
            min(fitting, key=lambda micro: fitting[micro]["mean_step_seconds"]) if fitting else None
        )
        chosen = results.get(str(configured))
        stages[stage] = {
            "configured_micro_batch": configured,
            "recommended_micro_batch": int(recommended) if recommended else None,
            "results": results,
            "projected_seconds_with_configured": (
                chosen["mean_step_seconds"] * PILOT_STEPS[stage] if chosen and chosen.get("status") == "ok" else None
            ),
        }

    save_seconds, save_gb = _time_checkpoint_save(torch, accelerator, model, AdamW8bit, bench_dir)
    resume_check, model = _resume_roundtrip(
        torch=torch,
        accelerator=accelerator,
        model=model,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AdamW8bit=AdamW8bit,
        objective=objective_for("specialization", profile.micro_batch.specialization)[0],
        config=config,
        settings=settings,
        bench_dir=bench_dir,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    try:
        evaluation = _time_evaluation(
            torch=torch,
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            artifacts=artifacts,
            config=config,
            device=device,
        )
        evaluation["passed"] = True
    except Exception as exc:  # keep the training measurements; report the failure
        evaluation = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "projected_total_seconds": None}
        print(f"[benchmark] evaluation check FAILED: {evaluation['error']}", flush=True)

    training_seconds = sum(stage["projected_seconds_with_configured"] or 0.0 for stage in stages.values())
    projection = {
        "training_seconds": training_seconds,
        "npo_reference_passes_seconds": 3 * reference_seconds,
        "scientific_checkpoint_saves_seconds": SCIENTIFIC_SAVES * save_seconds,
        "evaluation_seconds": evaluation.get("projected_total_seconds"),
    }
    projection["total_seconds"] = sum(value or 0.0 for value in projection.values())
    report = {
        "schema_version": "benchmark-report-v1",
        "label": "operational benchmark on this machine; not an experimental result",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": layout.run_id,
        "model": {"model_id": config.model.model_id, "revision": config.model.revision, "parameters": parameter_count},
        "runtime_profile": profile.as_dict(),
        "steps_per_measurement": steps,
        "gpu_total_memory_gb": total_vram_gb,
        "headroom_fraction": HEADROOM,
        "stages": stages,
        "npo_reference_pass_seconds": reference_seconds,
        "checkpoint_save_seconds": save_seconds,
        "checkpoint_size_gb": save_gb,
        "resume_roundtrip": resume_check,
        "evaluation": evaluation,
        "projection": projection,
        "projection_human": {key: format_duration(value) for key, value in projection.items()},
        "estimated_disk_gb": {
            "scientific_checkpoints": SCIENTIFIC_SAVES * save_gb,
            "resume_points_peak": 2 * save_gb,
        },
        "hardware": hardware,
        "software_versions": software_versions(),
        "benchmark_wall_seconds": time.perf_counter() - started,
    }
    write_json(bench_dir / "benchmark_report.json", report, overwrite=True)
    _print_report(report)
    return report


def _benchmark_reference_values(*, torch, model, sequences, batch_size, device, pad_token_id) -> dict[str, float]:
    """Same computation as the NPO reference cache, using the in-memory base model."""

    from unlearning.training.unlearn import collate_completion_only_batch, completion_logprob_sums

    model.eval()
    values: dict[str, float] = {}
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_completion_only_batch(sequences[start : start + batch_size], pad_token_id=pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    use_cache=False,
                ).logits
                sums, _ = completion_logprob_sums(logits, batch["labels"].to(device))
            for example_id, value in zip(batch["example_ids"], sums.detach().cpu()):
                values[example_id] = float(value)
    model.train()
    return values


def _time_checkpoint_save(torch, accelerator, model, AdamW8bit, bench_dir: Path) -> tuple[float, float]:
    target = bench_dir / "save_probe"
    optimizer = AdamW8bit(accelerator.unwrap_model(model).parameters(), lr=1e-6)
    started = time.perf_counter()
    accelerator.unwrap_model(model).save_pretrained(target / "model", safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict()}, target / "training_state.pt")
    seconds = time.perf_counter() - started
    size_gb = sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) / 1024**3
    shutil.rmtree(target)
    del optimizer
    _cleanup(torch)
    return seconds, size_gb


def _parameter_checksum(torch, model) -> float:
    with torch.no_grad():
        return float(sum(parameter.detach().double().sum() for parameter in model.parameters()))


def _resume_roundtrip(
    *, torch, accelerator, model, AutoModelForCausalLM, AdamW8bit, objective, config, settings, bench_dir: Path,
    gradient_checkpointing: bool,
):
    """Steps 1-2, save a resume point, step 3; reload the point and redo step 3."""

    lr = config.specialization.learning_rate
    store = ResumeStore(bench_dir / "resume_check" / "resume", keep_last=1, every_minutes=10**6)

    def loop_for(target_model, optimizer, scheduler, total, collected):
        return TrainingLoop(
            accelerator=accelerator,
            model=target_model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=objective,
            total_steps=total,
            max_grad_norm=config.specialization.max_grad_norm,
            run_log=RunLog(bench_dir / "resume_check", run_id="benchmark", stage="resume_check", total_steps=total, console_every_steps=10**6),
            resume_store=_AbsentStore(bench_dir / "resume_check" / "unused", keep_last=1, every_minutes=10**6),
            stop=GracefulStop(),
            settings=settings,
            on_step_end=lambda step, metrics: collected.append(metrics) or False,
        )

    optimizer = AdamW8bit(accelerator.unwrap_model(model).parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    first: list[dict[str, Any]] = []
    loop_for(model, optimizer, scheduler, 2, first).run(start_step=0)
    point = store.save(step=2, accelerator=accelerator, model=model, optimizer=optimizer, scheduler=scheduler, torch=torch, meta={})
    loop_for(model, optimizer, scheduler, 3, first).run(start_step=2)
    original_loss = first[-1]["loss"]
    original_checksum = _parameter_checksum(torch, model)
    del optimizer, scheduler, model
    _cleanup(torch)

    reloaded = AutoModelForCausalLM.from_pretrained(point.model_dir, torch_dtype=torch.bfloat16)
    enable_memory_saving(reloaded, gradient_checkpointing=gradient_checkpointing)
    reloaded.train()
    reloaded = accelerator.prepare(reloaded)
    optimizer = AdamW8bit(accelerator.unwrap_model(reloaded).parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    apply_resume_state(torch, point, optimizer=optimizer, scheduler=scheduler)
    second: list[dict[str, Any]] = []
    loop_for(reloaded, optimizer, scheduler, 3, second).run(start_step=2)
    resumed_loss = second[-1]["loss"]
    resumed_checksum = _parameter_checksum(torch, reloaded)
    del optimizer, scheduler
    _cleanup(torch)
    shutil.rmtree(bench_dir / "resume_check", ignore_errors=True)
    loss_difference = abs(original_loss - resumed_loss)
    checksum_difference = abs(original_checksum - resumed_checksum)
    return {
        "step3_loss_uninterrupted": original_loss,
        "step3_loss_after_resume": resumed_loss,
        "loss_abs_difference": loss_difference,
        "parameter_checksum_abs_difference": checksum_difference,
        "passed": loss_difference <= 1e-3 * max(1.0, abs(original_loss)),
        "note": "small differences come from non-deterministic GPU kernels, not from lost state",
    }, reloaded


def _time_evaluation(*, torch, model, tokenizer, artifacts: Path, config, device) -> dict[str, Any]:
    from unlearning.evaluation.clean_checkpoints import evaluate_records
    from unlearning.evaluation.forgetting import (
        decoy_values_from_records,
        evaluate_forgetting_with_model,
        load_forget_evaluation_reference,
        protected_fields_from_records,
    )

    model.eval()
    sample_mc = 24
    sample_forget = 12
    near = read_jsonl(artifacts / "medmcqa_domain_subsets" / "near_domain_dev.jsonl")
    started = time.perf_counter()
    evaluate_records(model, tokenizer, near[:sample_mc], task="medmcqa", device=str(device), seed=config.seed)
    mc_seconds = (time.perf_counter() - started) / min(sample_mc, len(near))

    forget = load_forget_evaluation_reference(
        artifacts / "forget" / "forget_eval_examples.jsonl",
        expected_schema_version=config.forget.schema_version,
        expected_record_count=config.forget.record_count,
        expected_templates_per_record=config.forget.eval_templates_per_record,
    )
    started = time.perf_counter()
    evaluate_forgetting_with_model(
        model,
        tokenizer,
        forget.eval_examples[:sample_forget],
        str(device),
        max_new_tokens=64,
        target_logprob_normalize_by_token_count=True,
        protected_fields_by_record=protected_fields_from_records(forget.records),
        decoy_values_by_record=decoy_values_from_records(forget.records),
    )
    forget_seconds = (time.perf_counter() - started) / sample_forget
    model.train()

    def count(path: Path) -> int:
        return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.is_file() else 0

    retention_records = (
        count(artifacts / "medmcqa_domain_subsets" / "near_domain_dev.jsonl")
        + count(artifacts / "medmcqa_domain_subsets" / "far_domain_dev.jsonl")
        + count(artifacts / "mmlu_cf" / "mmlu_cf_eval_subset.jsonl")
        + count(artifacts / "pubmedqa" / "pubmedqa_eval.jsonl")
    )
    domain_full = count(artifacts / "medmcqa" / "dev_eval.jsonl")
    forget_examples = len(forget.eval_examples)
    models = 3 + 3 + 3 * 2 * 5
    per_model = retention_records * mc_seconds + forget_examples * forget_seconds
    total = models * per_model + 3 * domain_full * mc_seconds + models * 20.0  # ~20 s load/verify per job group
    return {
        "seconds_per_multiple_choice_record": mc_seconds,
        "seconds_per_forgetting_example": forget_seconds,
        "retention_records_per_model": retention_records,
        "forgetting_examples_per_model": forget_examples,
        "domain_full_records_clean_only": domain_full,
        "models_to_evaluate": models,
        "projected_total_seconds": total,
        "note": "MC timing measured on near-domain records; other suites are assumed similar per record",
    }


def _print_report(report: dict[str, Any]) -> None:
    print("\n=== Benchmark (operational; this machine only) ===", flush=True)
    print(f"model {report['model']['model_id']}  params {report['model']['parameters'] / 1e6:.0f}M  "
          f"VRAM {report['gpu_total_memory_gb']:.1f} GB  profile {report['runtime_profile']['name']}")
    for stage, data in report["stages"].items():
        for micro, result in data["results"].items():
            mark = "*" if int(micro) == data["configured_micro_batch"] else " "
            if result.get("status") != "ok":
                print(f" {mark} {stage:16s} micro {micro:>2}: {result['status']}")
                continue
            print(
                f" {mark} {stage:16s} micro {micro:>2}: {result['mean_step_seconds']:.2f} s/step  "
                f"{result['tokens_per_second']:.0f} tok/s  peak reserved {result['peak_reserved_gb']:.2f} GB"
                f"{'' if result['fits_with_headroom'] else '  (too close to VRAM limit)'}"
            )
        if data["recommended_micro_batch"] and data["recommended_micro_batch"] != data["configured_micro_batch"]:
            print(f"   -> recommended micro-batch for {stage}: {data['recommended_micro_batch']}")
    check = report["resume_roundtrip"]
    print(f"resume round-trip: {'PASSED' if check['passed'] else 'FAILED'} "
          f"(|loss diff| = {check['loss_abs_difference']:.2e})")
    evaluation = report["evaluation"]
    print(f"evaluation check: {'PASSED' if evaluation.get('passed') else 'FAILED ' + str(evaluation.get('error'))}")
    print("projected wall-clock with the configured profile:")
    for key, value in report["projection_human"].items():
        print(f"   {key:38s} {value}")
    disk = report["estimated_disk_gb"]
    print(f"estimated disk: {disk['scientific_checkpoints']:.0f} GB scientific checkpoints "
          f"+ {disk['resume_points_peak']:.1f} GB resume points")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark memory/speed of every stage on this machine.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--steps", type=int, default=4, help="Optimizer steps per measurement (first is warm-up).")
    parser.add_argument("--sweep", action="store_true", help="Also try other micro-batch sizes and recommend one.")
    args = parser.parse_args(argv)
    report = run_benchmark(
        config_path=args.config,
        runtime_path=args.runtime,
        artifacts_root=args.artifacts_root,
        steps=args.steps,
        sweep=args.sweep,
    )
    return 0 if report["resume_roundtrip"]["passed"] and report["evaluation"].get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

