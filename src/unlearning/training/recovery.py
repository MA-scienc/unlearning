"""Recovery treatments (pre-registration 6.4, H4): do suppressed records come back?

Applied to U at f100, C_i and S_i of every level; the trained weights are evaluated and
discarded. Common: fresh 8-bit AdamW, constant learning rate 2e-5 (the rate that wrote F),
no warmup, clipping 1.0, bf16, the seed's data orders.

* ``relearn``: 20 steps on the contamination examples of a fixed 20-patient subset (the
  first 2 patients of each diagnosis in the seed's F order), F loss only, no replay. Each
  step is 2 whole patients, so about 2 passes. Leakage is then measured on the other 100.
* ``benign``: the first 100 steps of the seed's MedMCQA specialization stream, identical
  for every model. Leakage is then measured on all 120 patients.

Leakage: per patient, the share of its held-out probes whose output contains the exact
target (leakage scoring v2), before (from the model's own forgetting result, checked to be
the same checkpoint) and after. The estimand (U's rise minus S_i's rise) is computed by
the analysis.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.config.arms import load_matched_memorization_arm
from unlearning.data.ordering import ordered_by_record
from unlearning.evaluation.forgetting import (
    generate_with_model,
    leakage_annotations,
    protected_fields_from_records,
    rescore_forgetting_payload,
)
from unlearning.evaluation.provenance import resolve_snapshot_reference
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.monitor import GpuSampler, ThermalGuard
from unlearning.runtime.profile import load_runtime_profile
from unlearning.runtime.run_log import RunLog
from unlearning.runtime.settings import MachineSettings, default_config_path, default_runtime_path
from unlearning.runtime.signals import GracefulStop
from unlearning.training.contaminate import ContaminationObjective, build_fixed_exposure_plan, tokenize_completion_example
from unlearning.training.engine import TrainingLoop, create_accelerator, enable_memory_saving
from unlearning.training.matched_contamination import DEFAULT_ARM_PATH, _NoResumeStore
from unlearning.training.specialize_clean import SpecializationObjective, _prepare_packed_tensor
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import hash_file
from unlearning.utils.io import read_json, read_jsonl, write_json

LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
RELEARN_STEPS = 20
RELEARN_PATIENTS_PER_DIAGNOSIS = 2
BENIGN_STEPS = 100
MAX_NEW_TOKENS = 64
TREATMENTS = ("relearn", "benign")


def relearn_subset(records: list[dict[str, Any]], record_order: list[str]) -> list[str]:
    """The first 2 patients of each diagnosis, in the seed's F order."""

    condition = {record["record_id"]: record["condition"] for record in records}
    taken: dict[str, int] = defaultdict(int)
    subset = []
    for record_id in record_order:
        if taken[condition[record_id]] < RELEARN_PATIENTS_PER_DIAGNOSIS:
            taken[condition[record_id]] += 1
            subset.append(record_id)
    return subset


def leakage_by_record(rows: list[dict[str, Any]], record_ids: set[str]) -> dict[str, float]:
    flags: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["record_id"] in record_ids and row.get("target_exact_leakage") is not None:
            flags[row["record_id"]].append(float(row["target_exact_leakage"]))
    return {record_id: sum(values) / len(values) for record_id, values in sorted(flags.items())}


def run_recovery(
    *, config_path, runtime_path, model: str, checkpoint_manifest: str | None, label: str,
    treatment: str, baseline_result: str, output: str, arm_path=DEFAULT_ARM_PATH,
) -> dict[str, Any]:
    import torch
    from bitsandbytes.optim import AdamW8bit
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}")
    config = load_config(config_path)
    settings = MachineSettings.from_env()
    layout = StorageLayout.for_config(config, settings)
    profile = load_runtime_profile(runtime_path)
    arm = load_matched_memorization_arm(arm_path)
    snapshot = resolve_snapshot_reference(model, checkpoint_manifest, config)
    baseline = read_json(baseline_result)
    # Same checkpoint = same manifest content (paths may be written relative or absolute).
    if (baseline.get("checkpoint_provenance") or {}).get("manifest_file_sha256") != snapshot.manifest_file_sha256:
        raise ValueError(f"{baseline_result} was not measured on the checkpoint of {snapshot.manifest_path}")

    records = read_jsonl(layout.forget_dir / "records.jsonl")
    examples = read_jsonl(layout.contamination_examples)
    ordered = ordered_by_record(examples, mode=config.data_order, seed=config.seed, stream="contamination")
    record_order = list(dict.fromkeys(example["record_id"] for example in ordered))
    subset = relearn_subset(records, record_order) if treatment == "relearn" else []
    evaluated = set(record_order) - set(subset)

    accelerator = create_accelerator()
    device = accelerator.device
    tokenizer = AutoTokenizer.from_pretrained(snapshot.tokenizer_source, revision=snapshot.model_revision, use_fast=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    kwargs = {"revision": snapshot.model_revision} if snapshot.model_revision is not None else {}
    torch.manual_seed(config.seed)
    net = AutoModelForCausalLM.from_pretrained(snapshot.model_source, torch_dtype=torch.bfloat16, **kwargs)
    enable_memory_saving(net, gradient_checkpointing=config.model.gradient_checkpointing)
    net.train()
    net = accelerator.prepare(net)
    if treatment == "relearn":
        chosen = [example for example in ordered if example["record_id"] in set(subset)]
        encoded = [tokenize_completion_example(example, tokenizer=tokenizer, max_length=arm.sequence_length) for example in chosen]
        epochs = RELEARN_STEPS * arm.effective_batch_size // len(encoded)
        plan = build_fixed_exposure_plan(example_count=len(encoded), epochs=epochs, effective_batch_size=arm.effective_batch_size)
        objective = ContaminationObjective(torch, encoded, plan=plan, micro_batch_size=profile.micro_batch.contamination, pad_token_id=pad_token_id)
        steps, weight_decay = RELEARN_STEPS, arm.weight_decay
    else:
        packed, _meta = _prepare_packed_tensor(config, tokenizer, read_jsonl(layout.generated / "medmcqa" / "train_specialization.jsonl"), torch)
        per_step = config.specialization.effective_batch_sequences
        objective = SpecializationObjective(
            packed[: BENIGN_STEPS * per_step], effective_batch_sequences=per_step,
            micro_batch_sequences=profile.micro_batch.specialization,
        )
        steps, weight_decay = BENIGN_STEPS, config.specialization.weight_decay
    stage_dir = layout.stage_dir(f"recover_{label}_{treatment}")
    run_log = RunLog(stage_dir, run_id=layout.run_id, stage=stage_dir.name, total_steps=steps,
                     console_every_steps=profile.console_every_steps)
    sampler = GpuSampler()
    stop = GracefulStop().install()
    optimizer = AdamW8bit(accelerator.unwrap_model(net).parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    started = time.perf_counter()
    try:
        loop = TrainingLoop(
            accelerator=accelerator, model=net, optimizer=optimizer, scheduler=None, objective=objective,
            total_steps=steps, max_grad_norm=MAX_GRAD_NORM, run_log=run_log,
            resume_store=_NoResumeStore(stage_dir / "resume", keep_last=1, every_minutes=10**6),
            stop=stop, settings=settings,
            thermal=ThermalGuard(run_log, sampler, pause_at_c=settings.gpu_pause_temp_c, resume_at_c=settings.gpu_resume_temp_c),
        )
        result = loop.run(start_step=0)
        if not result.completed:
            raise RuntimeError(f"recovery training stopped at step {result.last_step}: {result.stop_reason}")
    finally:
        stop.uninstall()
    trained = accelerator.unwrap_model(net)
    trained.eval()
    fields = protected_fields_from_records(records)
    after_rows = []
    with torch.inference_mode():
        for example in read_jsonl(layout.forget_dir / "forget_eval_examples.jsonl"):
            if example["record_id"] not in evaluated:
                continue
            generated = generate_with_model(trained, tokenizer, example["prompt"], str(device), max_new_tokens=MAX_NEW_TOKENS)
            annotations = leakage_annotations(generated, example["prompt"], example["target"], fields[example["record_id"]])
            after_rows.append({"record_id": example["record_id"], "template_id": example["template_id"],
                               "generated_text": generated, **annotations})
    before = leakage_by_record(rescore_forgetting_payload(baseline)["records"], evaluated)
    after = leakage_by_record(after_rows, evaluated)
    rise = {record_id: after[record_id] - before[record_id] for record_id in after if record_id in before}
    payload = {
        "schema_version": "recovery-v1",
        "label": label,
        "treatment": treatment,
        "settings": {"learning_rate": LEARNING_RATE, "steps": steps, "warmup_steps": 0, "max_grad_norm": MAX_GRAD_NORM,
                     "weight_decay": weight_decay, "optimizer": "bitsandbytes.optim.AdamW8bit (fresh)",
                     "relearn_patients_per_diagnosis": RELEARN_PATIENTS_PER_DIAGNOSIS, "max_new_tokens": MAX_NEW_TOKENS},
        "relearn_subset": subset,
        "evaluated_patients": len(evaluated),
        "summary": {
            "mean_leakage_before": sum(before.values()) / len(before),
            "mean_leakage_after": sum(after.values()) / len(after),
            "mean_leakage_rise": sum(rise.values()) / len(rise),
        },
        "per_patient": {record_id: {"before": before[record_id], "after": after[record_id]} for record_id in rise},
        "records": after_rows,
        "baseline_result": {"path": str(baseline_result), "sha256": hash_file(baseline_result)},
        "checkpoint_provenance": snapshot.as_dict(),
        "runtime_seconds": time.perf_counter() - started,
        "config_path": str(config_path),
        "code_version": code_version(),
    }
    write_json(Path(output), payload)
    run_log.finish("completed", message=f"{treatment}: leakage {payload['summary']['mean_leakage_before']:.3f} -> "
                                        f"{payload['summary']['mean_leakage_after']:.3f}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recovery treatment (relearn | benign), then leakage.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-manifest", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--treatment", choices=TREATMENTS, required=True)
    parser.add_argument("--baseline-result", required=True, help="the model's own forgetting.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="accepted for pipeline uniformity; runs restart")
    args = parser.parse_args(argv)
    run_recovery(
        config_path=args.config, runtime_path=args.runtime, model=args.model,
        checkpoint_manifest=args.checkpoint_manifest, label=args.label, treatment=args.treatment,
        baseline_result=args.baseline_result, output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
