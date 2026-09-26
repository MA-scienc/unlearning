# Specialization Implementation Report

Date: 2026-09-26

Status: clean-domain-specialization implementation only. No contamination, Gradient Ascent, NPO, unlearning, or full 3,290-step specialization run was launched.

## Files Added Or Changed

Changed:

- `configs/pilot/base.yaml`
- `docs/PILOT_DESIGN.md`
- `pyproject.toml`
- `src/unlearning/config/schema.py`
- `src/unlearning/data/medmcqa.py`
- `src/unlearning/utils/env.py`
- `tests/test_config.py`

Added:

- `scripts/pilot/evaluate_clean.py`
- `scripts/pilot/smoke_specialization.py`
- `scripts/pilot/specialize_clean.py`
- `src/unlearning/evaluation/__init__.py`
- `src/unlearning/evaluation/clean_checkpoints.py`
- `src/unlearning/evaluation/multiple_choice.py`
- `src/unlearning/training/__init__.py`
- `src/unlearning/training/checkpoints.py`
- `src/unlearning/training/schedule.py`
- `src/unlearning/training/smoke.py`
- `src/unlearning/training/specialize_clean.py`
- `src/unlearning/training/stream.py`
- `tests/test_multiple_choice_eval.py`
- `tests/test_specialization_training.py`

## Frozen Near/Far Review Result

Human-review artifacts were generated before any specialization or unlearning results:

- `data/generated/pilot-medical-s0-s50-s100__qwen-qwen2-5-1-5b__seed42__synthetic-patient-f-v1/medmcqa_domain_subsets/near_domain_human_review.jsonl`
- `data/generated/pilot-medical-s0-s50-s100__qwen-qwen2-5-1-5b__seed42__synthetic-patient-f-v1/medmcqa_domain_subsets/far_domain_human_review.jsonl`

Counts:

- near-domain: 63
- far-domain: 739

Fingerprints:

- near review: `91a7f95430621c5f49f6ea8b602a686b8b1b1fa028bf8711da6b278023a2a8b8`
- far review: `5ea2689f90633dc352f419da850b177353ce2d3d0fd1f5b4257fa342bae1f571`

Subject distribution:

- near: Medicine 47, Pharmacology 16
- far: Anatomy 248, ENT 86, Microbiology 165, Ophthalmology 175, Psychiatry 6, Skin 59

Conceptual review:

- The current near set appears broadly aligned with the synthetic F concepts: cardiac/cardiovascular disease, renal disease, diabetes/insulin/glucose, thrombosis/anticoagulation, hypertension, and related pharmacology.
- No obvious non-medical false positives were found in the sample.
- Several near examples are broad physiology/pathophysiology rather than direct patient-management questions, such as cardiac muscle fiber questions and renal tubular acidosis. These should remain frozen but are flagged for human review as potentially broad near-domain items.
- The near count remains 63. It was not enlarged merely because it is small.

## Implementation Summary

Clean specialization trainer:

- Uses pinned `Qwen/Qwen2.5-1.5B` revision `8faed761d45a263340a0528343f099c05c9a4323`.
- Uses pinned MedMCQA revision `91c6572c454088bf71b679ad90aa8dffcd0d5868`.
- Calls `validate_training_ready(config)` before loading model weights.
- Refuses to proceed if the local environment is incompatible with the frozen protocol.
- Implements a continuous S0 -> S50 -> S100 trajectory.
- Saves S0 reference metadata, S50 weights/state, S100 weights/state, and manifests.
- Records checkpoint lineage so S100 parent is S50.
- Uses deterministic packed MedMCQA train order and excludes dev/test and synthetic F.

Frozen schedule:

- S50: optimizer step 1,645
- S100: optimizer step 3,290
- S100 is frozen by reducing the provisional 3,291-step budget by one step.
- Effective global batch: 32 packed 1,024-token sequences.
- Microbatch default: 1 sequence.
- Gradient accumulation default: 32.

Clean checkpoint evaluator:

- Evaluates clean checkpoints only.
- Uses deterministic option log-probability scoring.
- Supports MedMCQA, frozen near/far subsets, PubMedQA, and MMLU-CF.
- Reports accuracy, mean NLL, mean correct-answer log probability, and deterministic bootstrap accuracy CI.

Smoke mode:

- Tiny synthetic training loop validates forward/backward, optimizer step, checkpoint save, intermediate S50-like save, continuation to S100-like endpoint, and lineage.
- Smoke artifacts are written under `runs/` and are ignored by Git.
- Smoke results are explicitly not research results.

## Tests And Lint

```text
pytest -q
26 passed in 0.63s
```

```text
ruff check .
All checks passed!
```

## Preflight Runs

Offline/basic preflight:

```text
python scripts/pilot/preflight.py --config configs/pilot/base.yaml --skip-external --overwrite
passed
```

Full external preflight:

```text
python scripts/pilot/preflight.py --config configs/pilot/base.yaml --overwrite
passed
```

Strict JSON validation:

```text
python -m json.tool data/generated/pilot-medical-s0-s50-s100__qwen-qwen2-5-1-5b__seed42__synthetic-patient-f-v1/preflight_report.json
passed
```

## Smoke-Test Result

Local command:

```text
python scripts/pilot/smoke_specialization.py --output-dir runs/smoke_clean_specialization
```

Result:

```json
{
  "reason": "PyTorch unavailable: No module named 'torch'",
  "status": "skipped"
}
```

The intended GPU environment is not available locally, so the real smoke run was not executed.

## GPU Environment

Local dry-run command:

```text
python scripts/pilot/specialize_clean.py --config configs/pilot/base.yaml --artifacts-root data/generated/pilot-medical-s0-s50-s100__qwen-qwen2-5-1-5b__seed42__synthetic-patient-f-v1 --output-dir checkpoints/clean --dry-run
```

Detected local environment:

- OS: Windows-11-10.0.26200-SP0
- Python: 3.13.13
- CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- RAM: 15.71 GiB
- Disk: 475.56 GiB total, 135.02 GiB free
- GPU: not detected via `nvidia-smi`
- GPU VRAM: unavailable
- PyTorch: unavailable, `No module named 'torch'`
- CUDA: unavailable
- bf16 support: unavailable
- bitsandbytes: unavailable, `No module named 'bitsandbytes'`
- clean-specialization protocol compatible: false

Conclusion: do not use this local Windows CPU environment for the real run.

## Proposed Real Training Command

Run only after moving to a suitable Linux + Python 3.11/3.12 + CUDA GPU environment with PyTorch, bf16 support, and bitsandbytes:

```bash
export PYTHONPATH=src
python scripts/pilot/specialize_clean.py \
  --config configs/pilot/base.yaml \
  --artifacts-root data/generated/pilot-medical-s0-s50-s100__qwen-qwen2-5-1-5b__seed42__synthetic-patient-f-v1 \
  --output-dir checkpoints/clean
```

Before the full run, execute smoke mode on that environment:

```bash
export PYTHONPATH=src
python scripts/pilot/smoke_specialization.py --output-dir runs/smoke_clean_specialization
```

Do not launch the full command until the smoke result and this implementation report are reviewed.

## Estimated Storage

Approximate model/checkpoint storage:

- Qwen2.5-1.5B bf16 model checkpoint: about 3-4 GiB.
- S50 + S100 model weights: about 6-8 GiB.
- S0 reference metadata only: small.
- Training states with optimizer/scheduler/RNG: can be substantially larger than model-only checkpoints, especially for optimizer state.
- Recommended free disk before real run: at least 75-100 GiB for clean specialization artifacts, logs, tokenizer/config, and safety margin.

If later saving contaminated and unlearned snapshots as full checkpoints, storage can easily exceed 100 GiB.

## Remaining Blockers

1. Suitable GPU environment is not available locally.
2. Local Python is 3.13.13, while the project targets Python `>=3.11,<3.13`.
3. PyTorch and bitsandbytes are not installed locally.
4. Actual smoke test must be run on the intended Linux CUDA environment before the real specialization run.
5. Near-domain set has only 63 examples and uses keyword matching because MedMCQA topic metadata is unavailable. The subset is frozen, but broad near-domain examples should be reviewed.

## Stop Condition

Implementation reached the requested stop point. Do not start the full S0 -> S50 -> S100 clean specialization run until this report is reviewed.
