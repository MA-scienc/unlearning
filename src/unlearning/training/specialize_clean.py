from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from unlearning.config import load_config, validate_training_ready
from unlearning.data.packing import compute_step_counts, tokenize_and_pack_texts
from unlearning.experiments.naming import experiment_run_id
from unlearning.training.checkpoints import build_checkpoint_manifest, write_checkpoint_manifest
from unlearning.training.schedule import build_schedule
from unlearning.training.stream import (
    batch_indices_for_step,
    validate_specialization_inputs,
)
from unlearning.utils.env import inspect_hardware
from unlearning.utils.hashing import fingerprint_payload
from unlearning.utils.io import ensure_dir, read_jsonl, write_json


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


def precheck_training_environment() -> dict[str, Any]:
    hardware = inspect_hardware()
    torch_info = hardware.get("torch", {})
    bnb_info = hardware.get("bitsandbytes", {})
    compatible = (
        bool(torch_info.get("available"))
        and bool(torch_info.get("cuda_available"))
        and bool(bnb_info.get("available"))
    )
    if torch_info.get("devices"):
        compatible = compatible and any(
            bool(device.get("bf16_supported")) and int(device.get("total_memory_mb", 0)) >= 24_000
            for device in torch_info["devices"]
        )
    else:
        compatible = False
    hardware["clean_specialization_protocol_compatible"] = compatible
    return hardware


def _prepare_packed_tensor(config, tokenizer, train_records, torch):
    texts = [record["specialization_text"] for record in train_records]
    packed = tokenize_and_pack_texts(
        texts,
        tokenizer=tokenizer,
        context_length=config.specialization.context_length,
    )
    return torch.tensor(packed.sequences, dtype=torch.long), packed


def _save_s0_reference(config, run_dir: Path, run_id: str, data_metadata: dict, hardware: dict) -> str:
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
    write_checkpoint_manifest(out / "manifest.json", manifest)
    return str(manifest["checkpoint_id"])


def _save_model_checkpoint(
    *,
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
) -> str:
    checkpoint_dir = run_dir / name
    ensure_dir(checkpoint_dir)
    model.save_pretrained(checkpoint_dir / "model")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_random_state": random.getstate(),
        },
        checkpoint_dir / "training_state.pt",
    )
    manifest = build_checkpoint_manifest(
        config=config,
        run_id=run_id,
        name=name,
        global_step=global_step,
        parent_checkpoint_id=parent_checkpoint_id,
        data_metadata=data_metadata,
        hardware=hardware,
    )
    write_checkpoint_manifest(checkpoint_dir / "manifest.json", manifest)
    return str(manifest["checkpoint_id"])


def run_clean_specialization(
    *,
    config_path: str | Path,
    artifacts_root: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_training_ready(config)
    hardware = precheck_training_environment()
    if dry_run:
        return {"hardware": hardware, "dry_run": True}
    if not hardware["clean_specialization_protocol_compatible"]:
        raise RuntimeError(
            "Current environment is not compatible with the frozen full-parameter bf16 "
            "Qwen2.5-1.5B specialization protocol. Refusing to load model weights."
        )

    torch, AutoModelForCausalLM, AutoTokenizer, get_scheduler = _require_training_stack()
    AdamW8bit = _require_bitsandbytes_optimizer()
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    artifacts = Path(artifacts_root)
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

    run_id = experiment_run_id(config)
    run_dir = Path(output_dir) / run_id
    ensure_dir(run_dir)
    data_metadata = {
        **stream_meta.__dict__,
        "packed_sequence_count": int(packed_tensor.shape[0]),
        "packed_stream_fingerprint": fingerprint_payload(packed.sequences),
        "raw_token_count_with_eos": packed.raw_token_count,
        "dropped_tail_tokens": packed.dropped_tail_tokens,
    }

    s0_id = _save_s0_reference(config, run_dir, run_id, data_metadata, hardware)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.train()
    device = torch.device("cuda")
    model.to(device)
    optimizer = AdamW8bit(
        model.parameters(),
        lr=config.specialization.learning_rate,
        weight_decay=config.specialization.weight_decay,
    )
    scheduler = get_scheduler(
        name=config.specialization.scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(config.specialization.warmup_ratio * schedule.s100_steps),
        num_training_steps=schedule.s100_steps,
    )

    parent_id = s0_id
    checkpoint_ids = {"S0": s0_id}
    for step in range(1, schedule.s100_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(schedule.gradient_accumulation_steps):
            indices = batch_indices_for_step(
                step=step,
                micro_step=micro_step,
                packed_sequence_count=packed_tensor.shape[0],
                effective_batch_sequences=schedule.effective_batch_sequences,
                micro_batch_sequences=schedule.micro_batch_sequences,
            )
            batch = packed_tensor[indices].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss / schedule.gradient_accumulation_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.specialization.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if step == schedule.s50_steps:
            parent_id = _save_model_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                torch=torch,
                config=config,
                run_dir=run_dir,
                run_id=run_id,
                name="S50",
                global_step=step,
                parent_checkpoint_id=parent_id,
                data_metadata=data_metadata,
                hardware=hardware,
            )
            checkpoint_ids["S50"] = parent_id
        if step == schedule.s100_steps:
            s100_id = _save_model_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                torch=torch,
                config=config,
                run_dir=run_dir,
                run_id=run_id,
                name="S100",
                global_step=step,
                parent_checkpoint_id=parent_id,
                data_metadata=data_metadata,
                hardware=hardware,
            )
            checkpoint_ids["S100"] = s100_id

    run_manifest = {
        "run_id": run_id,
        "schedule": schedule.__dict__,
        "checkpoint_ids": checkpoint_ids,
        "data": data_metadata,
        "hardware": hardware,
    }
    write_json(run_dir / "run_manifest.json", run_manifest, overwrite=True)
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run clean continuous S0 -> S50 -> S100 specialization.")
    parser.add_argument("--config", default="configs/pilot/base.yaml")
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--output-dir", default="checkpoints/clean")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_clean_specialization(
        config_path=args.config,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
