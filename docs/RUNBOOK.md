# Runbook: running the pilot

How to run the S0/S50/S100 pilot end to end, watch it, stop and resume it,
and move it to rented GPUs. The scientific design is in
[PILOT_DESIGN.md](PILOT_DESIGN.md); this document is only about operating it.

## 1. What runs, in order

```
preflight ──► gate 0 ──► benchmark ──► gate 1 ──► specialize (S0 → S50 → S100)
   ──► evaluate S ──► gate 2 ──► contaminate S+F → C0, C50, C100
   ──► evaluate C ──► gate 3 ──► unlearn GA/NPO × C0/C50/C100 (snapshots 0…400)
   ──► evaluate U ──► aggregate tables + figures
```

| Job | What it does | Measured on the RTX 4070 laptop (2026-10-24, Qwen2.5-0.5B) |
|---|---|---|
| preflight | downloads MedMCQA/PubMedQA/MMLU-CF + tokenizer, builds F, splits, near/far sets, step budget | ~2 min (after first download) |
| benchmark | a few real steps of every stage; memory, speed, resume check, projected time | ~8 min |
| specialize | 3,290 optimizer steps (108M tokens), saves S50 and S100 | ~6.5 h (6.6–7.0 s/step, 4.6 GB allocated) |
| contaminate ×3 | 600 steps each on the 960 F examples | ~8 min each |
| unlearn ×6 | 400 steps each, snapshots at 0/50/100/200/400 | ~5 min each |
| evaluate ×183 | one job per model × suite; skips anything already scored | ~15 h in total (forgetting probes generate 64 tokens each: ~18 min per model) |

Total ≈ 22 h of machine time on the laptop, interruptible at any point.

Review gates stop the pipeline until you approve them. Each one writes a
report of measured values with the checks that were fixed before any results
existed (`runs/<run_id>/pipeline/gates/<gate>.md`).

| Gate | You look at | Stop instead of approving if |
|---|---|---|
| `gate0_data` | step budget, F counts, near/far sets | any count or budget is not the protocol's |
| `gate1_hardware` | memory per stage, resume round-trip, projected time and disk | out-of-memory, resume check failed, not enough time/disk |
| `gate2_specialization` | MedMCQA dev accuracy of S0/S50/S100 | accuracy does not rise S0 → S100: the manipulated variable is missing |
| `gate3_memorization` | F leakage of C0/C50/C100 vs S0/S50/S100 | F not memorized; large memorization differences mean the matched-memorization control is needed |

The pipeline never decides these questions. Do not change the recipe after
looking at gate 2 or 3 results; record any change as a dated amendment in
[CONFIRMATORY_PROTOCOL.md](CONFIRMATORY_PROTOCOL.md).

## 2. One-time setup (laptop)

```powershell
# from the repository folder
.venv\Scripts\Activate.ps1          # Python 3.12 venv with CUDA PyTorch (already created)
copy .env.example .env               # only if .env is missing; then edit it
python -m pytest -q                  # should pass in a few seconds
```

Windows settings that otherwise ruin long runs:

1. **Power**: keep the charger connected for the whole run, and pause Windows Update. On
   2026-10-25 the laptop came off AC 50 minutes into specialization, ran down to critical
   battery and slept briefly; the run survived, but it took 16.3 h instead of ~6.5 h. Even
   on AC, unattended steps ran ~2.8x slower (GPU idle ~70%: Windows deprioritized the
   background process). Every pipeline process now opts out of Windows background power
   throttling and keeps the system and display awake while it runs
   (`UNLEARNING_DISABLE_POWER_THROTTLING`, `UNLEARNING_KEEP_DISPLAY_ON` in `.env`). The
   dashboard warns when the laptop is on battery. Prefer leaving the session unlocked.
2. **NVIDIA Control Panel → Manage 3D settings → CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback**.
   Without this, running out of GPU memory silently spills into system RAM and
   training becomes many times slower instead of failing clearly.
3. Cooling: raised stand or cooling pad. The trainer pauses itself between steps
   at `UNLEARNING_GPU_PAUSE_TEMP_C` (pausing never changes results).

## 3. Running it

```powershell
python scripts\pilot\run_pipeline.py status          # progress and the next job
python scripts\pilot\run_pipeline.py run             # run until the next gate
python scripts\pilot\run_pipeline.py approve gate0_data --note "budget 1645/3290 ok"
python scripts\pilot\run_pipeline.py run             # continues after the gate
```

`run --dry-run` prints the commands without running them; `run --until specialize`
stops after one job; `status --all` lists all 200 jobs.

Watch a run from a **second terminal**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dashboard.ps1
```

It shows the stage, step/total, ETA, loss trend, learning rate, gradient norm,
throughput, GPU temperature/power/memory/clock/throttling, RAM, free disk, the
last saves, and warnings (overheating, throttling, stalled, crashed trainer,
GPU memory nearly full, low disk). It only reads files.

Training curves as PNG at any time (also produced by the final `aggregate` job):

```powershell
python scripts\pilot\plot_training.py            # -> results\figures\<run_id>\training\
```

## 4. Stopping, crashes, and resuming

| What happened | What to do |
|---|---|
| You pressed **Ctrl+C** | The stage finishes its current step, saves a resume point, exits. Run `run_pipeline.py run` again. A second Ctrl+C aborts immediately (you lose work since the last save). |
| Power cut / blue screen / reboot | Run `run_pipeline.py run` again. You lose at most `resume_every_minutes` (30 min) of training. |
| **Out of GPU memory** | Lower the stage's `micro_batch` in `configs/runtime/laptop_8gb.yaml` and run again. Results are unaffected (see §7). |
| Disk below `UNLEARNING_MIN_FREE_DISK_GB` | The stage stops *before* writing; free space; run again. |
| A job failed | Read `runs/<run_id>/pipeline/logs/<job>.log`, fix the cause, run again. Finished jobs are skipped. |
| You want to redo a stage on purpose | Run that stage's script with `--overwrite`: the old outputs are renamed `*__invalidated_<time>`, never deleted. |

Resuming refuses to continue if the config, the data artifacts, or the
contaminated source changed since the stage started (`run_plan.json`), because
that would silently mix two experiments. Every resume is recorded in the stage's
`execution.json` and in the checkpoint manifests.

Running a stage by hand (what the pipeline does):

```powershell
python scripts\pilot\specialize_clean.py --resume
python scripts\pilot\contaminate.py --clean-checkpoint checkpoints\clean\<run_id>\S50 --resume
python scripts\pilot\unlearn.py --contaminated-checkpoint-dir checkpoints\contaminated\<run_id>\C50 --method npo --resume
python scripts\pilot\evaluate_forgetting.py --model <checkpoint dir> --checkpoint-manifest <dir>\manifest.json --dataset data\generated\<run_id>\forget\forget_eval_examples.jsonl --output results\raw\<run_id>\<label>\forgetting.json
```

`--config` and `--runtime` default to `UNLEARNING_CONFIG` / `UNLEARNING_RUNTIME`
from `.env`; data and output folders default to the storage layout below.

## 5. Where everything is stored

All under `UNLEARNING_STORAGE_ROOT` (default: this folder; all git-ignored):

```
hf_cache/                          downloaded models and raw datasets
data/generated/<run_id>/           preflight output (MedMCQA splits, F, eval sets, preflight_report.json)
checkpoints/clean/<run_id>/        S0_reference (manifest only: S0 is the pinned base), S50, S100
checkpoints/contaminated/<run_id>/ C0, C50, C100
checkpoints/unlearned/<trajectory>/step0000 … step0400, trajectory_plan.json, NPO reference cache
runs/<run_id>/<stage>/             status.json, metrics.jsonl, events.jsonl, system.jsonl,
                                   dev_monitor.jsonl, execution.json, run_plan.json, resume/
runs/<run_id>/pipeline/            status.json, gates.json, gates/*.md, logs/*.log, history.jsonl
results/raw/<run_id>/<model>/      one JSON per evaluation suite (measured values, per record)
results/processed/<run_id>/        metrics_long.csv, summary.md (built only from results/raw)
results/figures/<run_id>/          figures (built only from logs and processed tables)
```

`<run_id>` encodes the experiment id, model, seed, F schema and a hash of the
whole scientific config, so different configs can never write into each
other's folders. Scientific checkpoints are written to `<name>.tmp` and renamed
only when complete; `runs/.../resume/` holds operational resume points and is
cleared automatically when a stage completes.

## 6. What every run records

Each checkpoint manifest and run manifest records the model id and pinned
revision, dataset revisions and fingerprints, seed, learning rate, optimizer,
effective batch, micro-batch and process count, steps, scheduler, precision,
parent checkpoint, hardware (GPU, memory, driver), software versions, runtime,
command line, and every start/resume segment. `.env` settings are recorded as
`machine_settings`.

## 7. What changes the science and what does not

**Never changes results** (execution only, recorded but excluded from run IDs):
micro-batch sizes, number of GPUs, `python` vs `accelerate launch`, resume
cadence, thermal pauses, storage location. Each optimizer step uses the same
examples, and the loss is normalized by the whole step's token count (or, for
NPO, example count), so splitting a step is mathematically identical to the
full batch (`tests/test_engine.py` checks this for all four objectives, and that
resume reproduces an uninterrupted run). Only floating-point summation order
differs, plus GPU kernel non-determinism, which is why resumes are recorded.

**Changes results** (new run ID, must be a recorded protocol decision):
anything in `configs/pilot/*.yaml` — model, data, learning rates, steps,
effective batch sizes, schedules, seeds, F definition, evaluation sets.

## 8. Moving to rented GPUs (1.5B / 3B, DDP)

1. Clone the repo, create a Python 3.12 venv, `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`, then `pip install -e ".[cuda,dev]"`.
   `requirements-lock.txt` records the exact package set of the laptop run, for comparison or an exact rebuild.
2. Copy `.env.example` to `.env`. Set `UNLEARNING_STORAGE_ROOT` to the big volume,
   `UNLEARNING_RUNTIME=configs/runtime/cloud_ddp_4gpu.yaml` (or `cloud_1gpu_80gb.yaml`),
   `UNLEARNING_MIN_GPU_MEMORY_GB` for the model, `CUDA_VISIBLE_DEVICES=0,1,2,3`.
3. Create an experiment config for the model (copy `configs/pilot/qwen2.5-0.5b_laptop.yaml`,
   change only `experiment_id`, `model_id`, `tokenizer_id`, `revision`) and set `UNLEARNING_CONFIG`.
   Record the decision in the amendment log.
4. In `configs/accelerate/ddp_4gpu.yaml` set `num_processes` to the GPU count; each
   profile micro-batch × GPU count must divide 32 (specialization) and 16 (C/U).
5. `python scripts/pilot/run_pipeline.py run` — the same pipeline; training stages are
   launched with `accelerate launch` automatically. Use `tmux`/`screen` so a dropped
   SSH session does not stop it; SIGTERM (preemption) saves a resume point.
6. Before trusting DDP numbers, compare a few logged losses of the first steps against
   a 1-GPU run of the same config: they should agree to rounding.

Notes: Windows cannot run multi-GPU DDP (no NCCL); multi-GPU is for Linux machines.
DDP keeps a full model + 8-bit optimizer copy per GPU (about 6 bytes per parameter
plus activations), so 3B needs roughly 40 GB GPUs or larger.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Hardware precheck failed ... gpu_memory_meets_env_minimum` | GPU smaller than `UNLEARNING_MIN_GPU_MEMORY_GB`; lower it in `.env` only if the benchmark shows the stage fits. |
| `has partial outputs ... Rerun with --resume` | A stage was started before; the pipeline always passes `--resume`; by hand add it. |
| `Refusing to resume: the inputs no longer match the run plan` | Config or data changed mid-stage. Restore them, or `--overwrite` to start that stage over. |
| Training suddenly much slower, GPU memory ~100% | Windows shared-memory fallback (see §2) or thermal throttling (dashboard shows it). |
| Training ~3x slower, GPU utilization ~25%, normal clocks | Laptop on battery (dashboard shows it), or Windows throttling an unattended background process (see §2, item 1). |
| `Frozen S50/S100 steps do not match the approved pilot protocol` | The tokenized data no longer produces the frozen 3,290-step budget; check the preflight report before anything else. |
