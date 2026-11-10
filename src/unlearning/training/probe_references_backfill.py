"""Backfill per-probe l(S_i) and l(C_i) for matched-forgetting trajectories.

Reference files written before ``per_example`` was kept store only the mean and
per-template means of the probe answer log-probabilities. Per-record forgetting
distributions (docs/MATCHED_FORGETTING_DESIGN.md, 5) need the per-probe values.
This re-measures S_i and C_i on the same probes with the same code, batch size
and precision as the unlearning run, and saves the result only if it reproduces
the stored mean and per-template means within ``--tolerance``. It reads
checkpoints and writes ``forgetting_references_per_example.json`` next to each
trajectory's references; it trains nothing and changes no existing file.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from unlearning.config import load_config
from unlearning.config.arms import load_matched_memorization_arm
from unlearning.config.unlearning_revisions import load_unlearning_revision
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.profile import load_runtime_profile
from unlearning.runtime.run_log import utc_now
from unlearning.runtime.settings import MachineSettings, default_config_path, default_runtime_path
from unlearning.training.contaminate import _load_model
from unlearning.training.matched_contamination import DEFAULT_ARM_PATH, matched_checkpoint_dir
from unlearning.training.matched_forgetting import (
    PER_EXAMPLE_REFERENCES_FILE,
    REFERENCES_FILE,
    clean_parent,
    per_example_references,
)
from unlearning.training.levels import contaminated_names
from unlearning.training.memorization import MemorizationProbe
from unlearning.training.unlearn import _load_verified_forget_examples, _require_training_stack, load_contaminated_source
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import fingerprint_payload
from unlearning.utils.io import read_json, write_json

IDENTITY_KEYS = ("probe_fingerprint", "clean_checkpoint_id", "contaminated_checkpoint_id", "contaminated_manifest_sha256")


def _max_difference(measured: dict[str, Any], stored: dict[str, Any]) -> float:
    differences = [abs(measured["mean"] - stored["mean"])]
    differences += [abs(measured["by_template"][t] - stored["by_template"][t]) for t in stored["by_template"]]
    return max(differences)


def backfill(*, config_path: str, runtime_path: str, revision_path: str, arm_path: str, tolerance: float) -> list[dict]:
    config = load_config(config_path)
    settings = MachineSettings.from_env()
    layout = StorageLayout.for_config(config, settings)
    revision = load_unlearning_revision(revision_path)
    if not revision.is_matched:
        raise ValueError(f"{revision_path} is not a matched-forgetting revision")
    arm = load_matched_memorization_arm(arm_path)
    profile = load_runtime_profile(runtime_path)
    torch, AutoModelForCausalLM, AutoTokenizer, _ = _require_training_stack()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lengths = {method.sequence_length for method in config.unlearning.methods.values()}
    if len(lengths) != 1:
        raise ValueError("GA and NPO use different sequence lengths; the probe would differ per method")
    sequence_length = lengths.pop()
    written = []
    for name in contaminated_names(config):
        trajectories = [
            layout.unlearning_trajectory(config, name, method, condition=arm.name, revision_tag=revision.tag)
            for method in config.unlearning.methods
        ]
        todo = [t for t in trajectories if (t / REFERENCES_FILE).is_file() and per_example_references(t) is None]
        if not todo:
            continue
        source = load_contaminated_source(matched_checkpoint_dir(layout, arm, name), config=config)
        clean = clean_parent(layout, config, source)
        _path, examples, _sha, _fingerprint = _load_verified_forget_examples(
            forget_examples_path=layout.contamination_examples, source=source, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(source.checkpoint_dir / "tokenizer", use_fast=True)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        probe = MemorizationProbe(
            torch,
            tokenizer,
            examples,
            templates=revision.matched.probe_templates,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            batch_size=profile.micro_batch.contamination,
        )
        measured = {}
        for role, loader in (
            ("clean", lambda: _load_model(clean, config, AutoModelForCausalLM, torch)),
            ("contaminated", lambda: AutoModelForCausalLM.from_pretrained(source.checkpoint_dir / "model", torch_dtype=torch.bfloat16)),
        ):
            model = loader().to(device)
            try:
                measured[role] = probe.answer_logprob(model, device)
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        identity = {
            "probe_fingerprint": probe.fingerprint,
            "clean_checkpoint_id": clean.checkpoint_id,
            "contaminated_checkpoint_id": source.checkpoint_id,
            "contaminated_manifest_sha256": source.manifest_sha256,
        }
        for trajectory in todo:
            stored = read_json(trajectory / REFERENCES_FILE)
            if {key: stored.get(key) for key in IDENTITY_KEYS} != identity:
                raise RuntimeError(f"{trajectory / REFERENCES_FILE} names other probes or checkpoints")
            difference = max(_max_difference(measured[role], stored[role]) for role in ("clean", "contaminated"))
            print(f"[backfill] {trajectory.name}: max |re-measured - stored| = {difference:.2e}", flush=True)
            if difference > tolerance:
                raise RuntimeError(
                    f"re-measured references differ from {trajectory / REFERENCES_FILE} by {difference:.2e} "
                    f"(> {tolerance:.0e}); nothing written"
                )
            payload = {
                **identity,
                "references_fingerprint": stored["fingerprint"],
                "clean": {key: measured["clean"][key] for key in ("mean", "by_template", "per_example")},
                "contaminated": {key: measured["contaminated"][key] for key in ("mean", "by_template", "per_example")},
                "max_abs_difference_from_stored_references": difference,
                "tolerance": tolerance,
                "probe_batch_size": profile.micro_batch.contamination,
                "device": str(device),
                "measured_at": utc_now(),
                "code_version": code_version(),
            }
            payload["fingerprint"] = fingerprint_payload(payload)
            write_json(trajectory / PER_EXAMPLE_REFERENCES_FILE, payload, overwrite=False)
            written.append({"trajectory": str(trajectory), "max_abs_difference": difference})
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill per-probe forgetting references (S_i, C_i).")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    parser.add_argument("--unlearning-revision", default="configs/pilot/unlearning/revision3.yaml")
    parser.add_argument("--arm", default=str(DEFAULT_ARM_PATH))
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    written = backfill(
        config_path=args.config,
        runtime_path=args.runtime,
        revision_path=args.unlearning_revision,
        arm_path=args.arm,
        tolerance=args.tolerance,
    )
    print(f"[backfill] wrote {len(written)} file(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
