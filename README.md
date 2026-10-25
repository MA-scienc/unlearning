# Unlearning After Specialization

Research code for *How domain adaptation reshapes the forgetting–retention
frontier in LLMs*: one base model is specialized on medicine to different
degrees (S0, S50, S100), the same synthetic patient records F are injected into
each checkpoint (C0, C50, C100), and GA/NPO unlearning (U) is evaluated for
target forgetting, near/far-domain retention and general retention against each
checkpoint's clean counterfactual S_i.

**Status:** pipeline implemented and tested; no experimental results yet.
Nothing in this repository is a measured finding until it appears under
`results/` from a completed run.

## Quick start (Windows laptop, RTX 4070 8 GB, Qwen2.5-0.5B)

```powershell
.venv\Scripts\Activate.ps1
python -m pytest -q                                   # sanity check
python scripts\pilot\run_pipeline.py status           # what is done / next
python scripts\pilot\run_pipeline.py run              # runs until the next review gate
powershell -ExecutionPolicy Bypass -File scripts\dashboard.ps1   # live view (2nd terminal)
```

The pipeline stops at four review gates; read the gate report it prints, then
`python scripts\pilot\run_pipeline.py approve <gate>` and `run` again. Ctrl+C,
crashes and reboots are safe: rerun `run` and it continues from the last save.
Full operating guide: **[docs/RUNBOOK.md](docs/RUNBOOK.md)**.

## Configuration: three layers

| File | Holds | Changes results? |
|---|---|---|
| `configs/pilot/*.yaml` | the scientific recipe: model + revision, data, LR, steps, batch sizes, seeds, F, eval sets | **yes** — hashed into the run ID |
| `configs/runtime/*.yaml` | execution: micro-batch sizes, launcher (python / accelerate DDP), checkpoint and log cadence | no (recorded in manifests) |
| `.env` (from `.env.example`) | this machine: storage location, HF cache, GPU-memory minimum, thermal pause, visible GPUs, which config + runtime to use | no (recorded in manifests) |

Laptop: `qwen2.5-0.5b_laptop.yaml` + `laptop_8gb.yaml`. Rented GPUs:
`cloud_1gpu_80gb.yaml` or `cloud_ddp_4gpu.yaml` (Hugging Face Accelerate DDP),
same code.

## Repository map

```
configs/pilot/       experiment configs (scientific)
configs/runtime/     runtime profiles (execution)
configs/accelerate/  accelerate launch configs for multi-GPU
scripts/pilot/       entry points: run_pipeline, preflight, benchmark, specialize_clean,
                     contaminate, unlearn, evaluate_*, compare_paired, aggregate_results,
                     plot_training
scripts/dashboard.ps1  live terminal dashboard
src/unlearning/
  config/            experiment-config schema and validation
  data/              MedMCQA, synthetic F, evaluation sets, packing, leakage safeguards
  training/          engine (Accelerate loop), specialization, contamination, unlearning
  evaluation/        forgetting, multiple-choice retention, provenance, statistics, aggregation
  experiments/       naming, preflight, benchmark, pipeline
  runtime/           settings (.env), profiles, storage layout, logging, GPU monitor,
                     resume points, stage lifecycle
  reporting/         figures
tests/               unit tests (CPU only, seconds)
docs/                research context, pilot design, protocol, runbook, archive/
```

## Documents

- [docs/RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md) — question, rationale, terminology
- [docs/PILOT_DESIGN.md](docs/PILOT_DESIGN.md) — every design choice of the pilot
- [docs/CONFIRMATORY_PROTOCOL.md](docs/CONFIRMATORY_PROTOCOL.md) — protocol template and amendment log
- [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md) — claim limits and evidence still required
- [docs/RUNBOOK.md](docs/RUNBOOK.md) — how to run, monitor, resume, and scale out
- [docs/archive/](docs/archive/) — superseded v1 reports (not evidence for the current protocol)
