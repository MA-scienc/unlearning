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
| `prepare_replay.py [--arm <arm yaml>]` | pack the pinned replay corpus (WikiText-103) for a replay arm | `data/generated/<run_id>/replay_<tag>/` |
| `calibrate_contamination.py [--arm <arm yaml>]` | matched-memorization learning-rate calibration on C0 | `runs/<run_id>/calibrate_<tag>/calibration_report.json` |
| `contaminate_matched.py --clean-checkpoint <S dir> [--resume]` | matched-memorization S_i + F -> C_i at the calibrated rate | `checkpoints/contaminated/<run_id>/<arm>/C*` |
| `diagnose_replay_damage.py [--learning-rate 2e-5]` | exploratory: repeat a calibration candidate on replay text only (no F) and measure damage; selects nothing, not a pipeline job | `runs/<run_id>/calibrate_<tag>/diagnostic_replay_only_lr_*/` |
| `calibrate_unlearning.py --unlearning-revision <matched-forgetting file>` | matched-forgetting learning rate on MR_C0: smallest candidate closing 100% of the gap to S0 on training probes, per method | `runs/<run>/calibrate_unlearning_<tag>/calibration_report.json` |
| `evaluate_closeness.py --model <U/C> --checkpoint-manifest ... --clean-model <S_i> --clean-manifest ...` | answer-start KL(S_i \|\| model) on the held-out forgetting prompts | `results/raw/<run>/<label>/closeness.json` |
| `unlearn.py --contaminated-checkpoint-dir <C dir> --method ga\|npo [--unlearning-revision <file>] [--resume]` | fixed 400-step trajectory, snapshots 0/50/100/200/400; a revision file in `configs/pilot/unlearning/` changes only the learning rate (revision 2: 2e-5) | `checkpoints/unlearned/<trajectory>/` (`...__unlearned-u2__...` for revision 2) |
| `evaluate_forgetting.py` | target forgetting on held-out F probes | `results/raw/<run_id>/<model>/forgetting.json` |
| `evaluate_clean.py --task medmcqa\|mmlu_cf\|pubmedqa` | retention / specialization accuracy | `results/raw/<run_id>/<model>/<suite>.json` |
| `compare_paired.py` | record-clustered paired comparison of two result files | `results/comparisons/...` |
| `aggregate_results.py` | tidy tables + training figures from raw results and logs | `results/processed/<run_id>/` |
| `plot_training.py` | training curves from stage logs | `results/figures/<run_id>/training/` |
| `summarize_matched_forgetting.py [--no-figure]` | revision-3 tables (steps, outcomes, per-fact-type forgetting, primary and secondary estimands with p and Holm, probe sensitivity) and the forgetting-retention figure; an existing summary is kept with a timestamp suffix | `results/processed/<run_id>/matched_forgetting_summary.md`, `results/figures/<run_id>/matched_forgetting/` |
| `backfill_probe_references.py` | one-off: per-probe l(S_i), l(C_i) for revision-3 trajectories whose references predate them; writes only if the re-measurement reproduces the stored means | `checkpoints/unlearned/<trajectory>/forgetting_references_per_example.json` |
| `prune_training_states.py [--apply]` | delete `training_state.pt` from finished unlearning snapshots, leaving a hash marker and a ledger (dry run without `--apply`) | `runs/<run_id>/maintenance/pruned_training_states.jsonl` |

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
