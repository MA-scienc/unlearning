# Pilot entry points

Normally you only run the pipeline; it calls the other scripts in the right
order with the right arguments and skips anything already finished:

```powershell
python scripts\pilot\run_pipeline.py status | run | approve <gate> | report <gate>
```

Every script reads `--config` / `--runtime` defaults from `.env`
(`UNLEARNING_CONFIG`, `UNLEARNING_RUNTIME`) and writes to the storage layout in
[docs/RUNBOOK.md](../../docs/RUNBOOK.md#5-where-everything-is-stored).

| Script | Purpose | Main output |
|---|---|---|
| `preflight.py` | download and freeze data; build F; step budget; leakage checks | `data/generated/<run_id>/` |
| `benchmark.py [--sweep]` | memory/speed per stage, resume round-trip, projected time | `runs/<run_id>/benchmark/benchmark_report.json` |
| `specialize_clean.py [--resume]` | continuous S0 → S50 → S100 trajectory | `checkpoints/clean/<run_id>/` |
| `contaminate.py --clean-checkpoint <S dir> [--resume]` | fixed-exposure S_i + F → C_i | `checkpoints/contaminated/<run_id>/C*` |
| `unlearn.py --contaminated-checkpoint-dir <C dir> --method ga\|npo [--resume]` | fixed 400-step trajectory, snapshots 0/50/100/200/400 | `checkpoints/unlearned/<trajectory>/` |
| `evaluate_forgetting.py` | target forgetting on held-out F probes | `results/raw/<run_id>/<model>/forgetting.json` |
| `evaluate_clean.py --task medmcqa\|mmlu_cf\|pubmedqa` | retention / specialization accuracy | `results/raw/<run_id>/<model>/<suite>.json` |
| `compare_paired.py` | record-clustered paired comparison of two result files | `results/comparisons/...` |
| `aggregate_results.py` | tidy tables + training figures from raw results and logs | `results/processed/<run_id>/` |
| `plot_training.py` | training curves from stage logs | `results/figures/<run_id>/training/` |

## Safeguards

- Training stages refuse to start over partial outputs unless `--resume` is
  given, and refuse to resume if the config or data changed since the start.
- `--overwrite` renames existing outputs to `*__invalidated_<time>`; nothing is
  deleted. Use it only for an explicitly invalidated run.
- Evaluations verify each checkpoint's manifest and file fingerprints before
  loading weights, and refuse to overwrite an existing result file.
- S50/S100 steps (1645/3290), the 600-step contamination exposure, the
  400-step unlearning trajectories and the snapshot schedule are enforced in
  code; a snapshot is never chosen after looking at results.

## Paired comparisons

After both result files exist:

```powershell
python scripts\pilot\compare_paired.py `
  --baseline results\raw\<run_id>\C50\forgetting.json `
  --comparison results\raw\<run_id>\U_C50_npo_step0100\forgetting.json `
  --metric target_exact_leakage `
  --output results\comparisons\<run_id>\C50__vs__U_C50_npo_step0100__leakage.json
```

The reported mean delta is comparison minus baseline, averaged within record
before resampling. For retention files use `--item-key record_id --cluster-key
record_id` and a metric such as `correct` or `nll`. This is a within-run
descriptive tool; it does not replace resampling across independent
trajectories, F seeds or model families.
