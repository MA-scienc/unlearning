"""Replay-only damage diagnostic (exploratory; selects nothing).

Question
    The replay calibration on C0 found that F memorization with WikiText replay
    raises text NLL on the damage probes by about +0.21 nats/token, and that
    neither a lower learning rate nor twice the replay reduced it. How much of
    that increase does the replay text cause on its own?

Run
    Start from S0 and repeat a finished calibration candidate exactly -- same
    learning rate, optimizer, warmup, clipping, seed, and the same replay
    passages in the same order for the same number of steps -- but with the F
    term removed. Then measure the same damage probes against the same S0
    baseline.

Reading the result (fixed before running)
    For each probe, fraction = replay-only NLL increase / candidate NLL increase.
      replay_major        fraction >= 0.50 on either probe: next-token training
                          on WikiText is itself a major source of the damage.
      memorization_major  fraction <= 0.25 on both probes: the damage comes
                          mostly from memorizing F.
      mixed               anything else.
    The two terms interact (AdamW normalization and gradient clipping act on
    their sum), so the split is an approximate attribution, not an exact
    decomposition.

Nothing here feeds the calibration rule, a gate, or any unlearning run.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from unlearning.config import load_config, validate_training_ready
from unlearning.config.arms import MatchedMemorizationArm, load_matched_memorization_arm
from unlearning.data.replay import load_replay_corpus
from unlearning.runtime.hardware import precheck_gpu_training, require_gpu_training
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.monitor import GpuSampler, SystemMonitor, ThermalGuard
from unlearning.runtime.profile import load_runtime_profile
from unlearning.runtime.run_log import RunLog, utc_now
from unlearning.runtime.settings import MachineSettings, default_config_path, default_runtime_path
from unlearning.runtime.signals import INTERRUPTED_EXIT_CODE, GracefulStop, StageInterrupted
from unlearning.training.contaminate import (
    _load_model,
    _require_bitsandbytes_optimizer,
    _require_training_stack,
    load_clean_checkpoint_reference,
)
from unlearning.training.engine import MicroBatch, TrainingLoop, create_accelerator, enable_memory_saving
from unlearning.training.matched_contamination import (
    DEFAULT_ARM_PATH,
    ReplayMixedObjective,
    _cleanup,
    _load_shared_inputs,
    _NoResumeStore,
    _optimizer_and_scheduler,
    _replay_record,
    _training_objective,
    calibration_dir,
    calibration_report_path,
)
from unlearning.training.memorization import build_damage_probes, damage_measures, damage_probe_fingerprints, damage_verdict
from unlearning.utils.env import software_versions
from unlearning.utils.io import read_json, write_json

DIAGNOSTIC_SCHEMA = "replay-only-damage-diagnostic-v1"
REPLAY_MAJOR_FRACTION = 0.50
MEMORIZATION_MAJOR_FRACTION = 0.25
PROBES = ("general", "medical")


class ReplayOnlyObjective:
    """The replay term of a ReplayMixedObjective alone: same passages, order and scaling."""

    name = "replay_only_diagnostic"
    unit = "weighted_terms"

    def __init__(self, mixed: ReplayMixedObjective):
        self.mixed = mixed

    def micro_batches(self, step: int) -> list[MicroBatch]:
        return [batch for batch in self.mixed.micro_batches(step) if "labels" not in batch.tensors]

    def normalizer(self, step: int, micro_batches) -> float:
        return 1.0  # replay micro-batches carry weight / replay tokens, as in the mixed objective

    def loss_sum(self, model, batch: dict[str, Any]):
        return self.mixed.loss_sum(model, batch)


def diagnostic_dir(layout: StorageLayout, arm: MatchedMemorizationArm, learning_rate: float) -> Path:
    return calibration_dir(layout, arm) / f"diagnostic_replay_only_lr_{learning_rate:.0e}"


def reference_candidate(report: Mapping[str, Any], learning_rate: float) -> dict[str, Any]:
    """The finished calibration candidate the diagnostic repeats without F."""

    for candidate in report.get("candidates") or []:
        if candidate.get("learning_rate") == learning_rate:
            if not candidate.get("reached_target") or not (candidate.get("damage") or {}).get("text_nll_increase"):
                raise RuntimeError(f"candidate {learning_rate:g} has no damage measurement at its target step")
            return dict(candidate)
    raise RuntimeError(f"the calibration report has no candidate with learning rate {learning_rate:g}")


def attribute_damage(replay_only: Mapping[str, float], with_f: Mapping[str, float]) -> dict[str, Any]:
    """Apply the reading rule fixed in this module's docstring."""

    fractions = {
        name: (replay_only[name] / with_f[name]) if with_f[name] > 0 else None for name in PROBES
    }
    known = [value for value in fractions.values() if value is not None]
    if any(value >= REPLAY_MAJOR_FRACTION for value in known):
        reading = "replay_major"
    elif len(known) == len(PROBES) and all(value <= MEMORIZATION_MAJOR_FRACTION for value in known):
        reading = "memorization_major"
    else:
        reading = "mixed"
    return {
        "fraction_of_candidate_nll_increase": fractions,
        "reading": reading,
        "rule": (
            f"replay_major if fraction >= {REPLAY_MAJOR_FRACTION} on either probe; memorization_major if "
            f"<= {MEMORIZATION_MAJOR_FRACTION} on both; otherwise mixed (fixed before running)"
        ),
    }


def run_replay_only_diagnostic(
    *,
    config_path: str | Path,
    arm_path: str | Path = DEFAULT_ARM_PATH,
    runtime_path: str | Path | None = None,
    learning_rate: float = 2e-5,
) -> dict[str, Any]:
    settings = MachineSettings.from_env()
    config = load_config(config_path)
    validate_training_ready(config)
    arm = load_matched_memorization_arm(arm_path)
    arm.validate_against(config)
    if arm.replay is None or arm.replay.target != "next_token":
        raise ValueError(f"arm {arm.name} has no next-token replay; the diagnostic repeats next-token replay only")
    layout = StorageLayout.for_config(config, settings)

    calibration = read_json(calibration_report_path(layout, arm))
    if calibration.get("arm_fingerprint") != arm.fingerprint:
        raise RuntimeError("the calibration report was produced with different arm settings")
    reference = reference_candidate(calibration, learning_rate)
    steps = int(reference["steps_to_target"])

    out_dir = diagnostic_dir(layout, arm, learning_rate)
    report_path = out_dir / "replay_only_diagnostic.json"
    if report_path.is_file():
        existing = read_json(report_path)
        if existing.get("status") == "completed":
            print(f"[diagnostic] already complete: {report_path}", flush=True)
            return existing
    if out_dir.exists():
        shutil.rmtree(out_dir)  # an interrupted attempt restarts from scratch

    hardware = precheck_gpu_training(settings, purpose="replay-only damage diagnostic")
    require_gpu_training(hardware)
    profile = load_runtime_profile(runtime_path or settings.runtime_profile_path)
    accelerator = create_accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Run the diagnostic as a single process (python, not accelerate launch)")
    torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler, _transformers = _require_training_stack()
    _bnb, AdamW8bit = _require_bitsandbytes_optimizer()
    device = accelerator.device

    clean = load_clean_checkpoint_reference(layout.clean_checkpoint("S0"), config=config)
    if clean.checkpoint_id != calibration.get("clean_checkpoint_id"):
        raise RuntimeError("S0 differs from the checkpoint the calibration used")
    _forget, tokenizer, pad_token_id, encoded, plan, probe = _load_shared_inputs(
        config, arm, layout, torch, AutoTokenizer, clean, profile
    )
    probes = build_damage_probes(config, arm, layout.generated, calibration_dir(layout, arm) / "probes")
    if damage_probe_fingerprints(probes) != calibration.get("damage_probe_fingerprints"):
        raise RuntimeError("damage probes differ from the calibration's; the S0 baseline would not apply")
    replay_sequences, replay_manifest = load_replay_corpus(config, arm, layout.generated)
    baseline = calibration["baseline"]["damage_measures"]

    report: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "label": "exploratory diagnostic: replay-only training from S0; selects nothing",
        "status": "running",
        "run_id": layout.run_id,
        "arm": arm.as_dict(),
        "arm_fingerprint": arm.fingerprint,
        "calibration_report": str(calibration_report_path(layout, arm)),
        "reference_candidate_learning_rate": learning_rate,
        "clean_checkpoint_id": clean.checkpoint_id,
        "model": {"model_id": config.model.model_id, "revision": config.model.revision, "dtype": config.model.dtype},
        "seed": config.seed,
        "training": {
            "learning_rate": learning_rate,
            "optimizer": arm.optimizer,
            "scheduler": arm.scheduler,
            "warmup_steps": arm.warmup_steps,
            "weight_decay": arm.weight_decay,
            "max_grad_norm": arm.max_grad_norm,
            "optimizer_steps": steps,
            "f_examples_per_step": 0,
            "gradient_checkpointing": config.model.gradient_checkpointing,
        },
        "replay": _replay_record(arm, replay_manifest, steps),
        "damage_probe_fingerprints": damage_probe_fingerprints(probes),
        "hardware": hardware,
        "software_versions": software_versions(),
        "started_at": utc_now(),
    }
    write_json(report_path, report)

    run_log = RunLog(
        out_dir,
        run_id=layout.run_id,
        stage=f"diagnostic_replay_only_lr_{learning_rate:.0e}",
        total_steps=steps,
        console_every_steps=profile.console_every_steps,
    )
    sampler = GpuSampler()
    monitor = SystemMonitor(
        run_log, interval=profile.system_sample_seconds, storage_root=settings.storage_root, sampler=sampler
    ).start()
    stop = GracefulStop().install()
    started = time.perf_counter()
    try:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        run_log.set_state("loading", message="replay-only diagnostic: loading S0")
        model = _load_model(clean, config, AutoModelForCausalLM, torch)
        enable_memory_saving(model, gradient_checkpointing=config.model.gradient_checkpointing)
        model.train()
        model = accelerator.prepare(model)
        optimizer, scheduler = _optimizer_and_scheduler(
            AdamW8bit, get_scheduler, accelerator.unwrap_model(model).parameters(), learning_rate=learning_rate, arm=arm
        )
        mixed = _training_objective(
            torch,
            encoded,
            plan,
            arm=arm,
            micro=profile.micro_batch.contamination,
            pad_token_id=pad_token_id,
            replay_sequences=replay_sequences,
            profile=profile,
        )
        loop = TrainingLoop(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=ReplayOnlyObjective(mixed),
            total_steps=steps,
            max_grad_norm=arm.max_grad_norm,
            run_log=run_log,
            resume_store=_NoResumeStore(out_dir / "resume", keep_last=1, every_minutes=10**6),
            stop=stop,
            settings=settings,
            thermal=ThermalGuard(
                run_log, sampler, pause_at_c=settings.gpu_pause_temp_c, resume_at_c=settings.gpu_resume_temp_c
            ),
        )
        run_log.update(state="running", learning_rate=learning_rate)
        result = loop.run(start_step=0)
        if not result.completed:
            run_log.finish("interrupted", message="the diagnostic restarts from scratch on rerun")
            report["status"] = "interrupted"
            write_json(report_path, report, overwrite=True)
            raise StageInterrupted(result.last_step, result.stop_reason)

        run_log.set_state("evaluating", message="memorization + collateral-damage probes")
        trained = accelerator.unwrap_model(model)
        measured = damage_measures(trained, tokenizer, probes, device, config.seed, nll_texts=arm.nll_texts_per_probe)
        replay_only = damage_verdict(arm, baseline, measured)
        with_f = reference["damage"]
        report.update(
            {
                "status": "completed",
                "steps_run": loop.completed_step,
                "final_train_loss": (result.last_metrics or {}).get("loss"),
                # F was never trained, so recall should stay at the S0 level (sanity check).
                "memorization": probe.recall(trained, device),
                "baseline_damage_measures": baseline,
                "replay_only_damage": replay_only,
                "candidate_with_f": {
                    "text_nll_increase": with_f["text_nll_increase"],
                    "bias_removed_accuracy_drop": with_f["bias_removed_accuracy_drop"],
                    "accuracy_drop": with_f["accuracy_drop"],
                    "steps_to_target": steps,
                },
                "attribution": attribute_damage(replay_only["text_nll_increase"], with_f["text_nll_increase"]),
                "runtime_seconds": time.perf_counter() - started,
                "completed_at": utc_now(),
            }
        )
        write_json(report_path, report, overwrite=True)
        run_log.finish("completed", message=f"replay-only diagnostic: {report['attribution']['reading']}")
    except StageInterrupted:
        raise
    except Exception as error:
        run_log.finish("failed", message=str(error))
        report.update({"status": "failed", "error": str(error)})
        write_json(report_path, report, overwrite=True)
        raise
    finally:
        stop.uninstall()
        monitor.stop()
        model = optimizer = scheduler = loop = None
        accelerator.free_memory()
        _cleanup(torch)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay-only damage diagnostic (exploratory; selects nothing).")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    parser.add_argument("--arm", default=str(DEFAULT_ARM_PATH))
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Calibration candidate to repeat without F (default 2e-5)."
    )
    args = parser.parse_args(argv)
    try:
        report = run_replay_only_diagnostic(
            config_path=args.config, arm_path=args.arm, runtime_path=args.runtime, learning_rate=args.learning_rate
        )
    except StageInterrupted as stopped:
        print(f"[diagnostic] {stopped}. Rerun to start the diagnostic again.", file=sys.stderr, flush=True)
        return INTERRUPTED_EXIT_CODE
    summary = {
        "replay_only_nll_increase": report["replay_only_damage"]["text_nll_increase"],
        "candidate_nll_increase": report["candidate_with_f"]["text_nll_increase"],
        "attribution": report["attribution"],
        "memorization_recall": report["memorization"]["mean_recall"],
    }
    print(json.dumps(summary, indent=2))
    return 0
