"""Diagnostic: medical dev loss of S0 (base model), S50 and S100 with the training-time monitor.

The specialization run logs dev loss from step 200 on; this adds S0 (step 0) so the
whole curve is known. Same DevLossMonitor, dev records, sequence count and micro-batch as
training. S100 must reproduce the training log's last value. Output:
runs/<run_id>/diagnostics/specialization_dev_loss.json (refuses to overwrite).
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning.config import load_config
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.profile import load_runtime_profile
from unlearning.runtime.run_log import utc_now
from unlearning.runtime.settings import default_config_path, default_runtime_path
from unlearning.training.specialize_clean import DevLossMonitor, _prepare_dev_tensor
from unlearning.utils.code_version import code_version
from unlearning.utils.io import read_jsonl, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    args = parser.parse_args(argv)
    config = load_config(args.config)
    layout = StorageLayout.for_config(config)
    profile = load_runtime_profile(args.runtime)
    dev_path = layout.generated / "medmcqa" / "dev_eval.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_id, revision=config.model.revision, use_fast=True)
    dev = _prepare_dev_tensor(config, tokenizer, read_jsonl(dev_path), torch, profile.specialization_dev_sequences)
    monitor = DevLossMonitor(dev, micro_batch_sequences=profile.micro_batch.specialization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = {}
    for name, source in (("S0", None), ("S50", layout.clean_checkpoint("S50") / "model"), ("S100", layout.clean_checkpoint("S100") / "model")):
        kwargs = {"revision": config.model.revision} if source is None else {}
        model = AutoModelForCausalLM.from_pretrained(source or config.model.model_id, torch_dtype=torch.bfloat16, **kwargs).to(device)
        losses[name] = monitor.evaluate(torch, model, device)
        print(f"{name}: {losses[name]:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    target = layout.runs / "diagnostics" / "specialization_dev_loss.json"
    write_json(target, {
        "dev_loss": losses,
        "dev_path": str(dev_path),
        "dev_sequences": int(dev.shape[0]),
        "monitor": "training/specialize_clean.DevLossMonitor",
        "measured_at": utc_now(),
        "code_version": code_version(),
    })
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
