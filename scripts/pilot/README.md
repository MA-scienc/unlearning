# Pilot execution guide

This is an evidence-preserving execution order for the S0/S50/S100 feasibility
pilot. It is not permission to claim a result: review the v2 preflight report
and the protocol amendment in `docs/PUBLICATION_READINESS.md` before launching
any training. The current workstation is not a compatible training environment.

Run commands from the repository root with a Python 3.11 or 3.12 environment
that has the pinned project dependencies, CUDA PyTorch, and bitsandbytes:

```powershell
$env:PYTHONPATH = "src"
```

## 1. Generate and review v2 artifacts

```powershell
python scripts/pilot/preflight.py --config configs/pilot/base.yaml --output-root data/generated/<RUN_ID>
```

This is the only stage that may be run before hardware approval. Review
`data/generated/<RUN_ID>/preflight_report.json`, especially the pinned source
revisions, token budget, F-v2 fingerprints, train/evaluation separation, and
near/far subset artifacts. Do not reuse a historical v1 report.

## 2. Create the continuous clean trajectory

```powershell
python scripts/pilot/specialize_clean.py `
  --config configs/pilot/base.yaml `
  --artifacts-root data/generated/<RUN_ID> `
  --output-dir checkpoints/clean
```

This creates one continuous S0 -> S50 -> S100 trajectory under
`checkpoints/clean/<RUN_ID>/`. S0 is a pinned-base reference; S50 and S100
contain local `model/`, `tokenizer/`, `training_state.pt`, and an integrity-bound
manifest. Do not start independently seeded S50/S100 training runs.

## 3. Apply identical fixed-exposure contamination

Run once for each S checkpoint. For example, S0:

```powershell
python scripts/pilot/contaminate.py `
  --config configs/pilot/base.yaml `
  --clean-checkpoint checkpoints/clean/<RUN_ID>/S0_reference `
  --contamination-examples data/generated/<RUN_ID>/forget/contamination_examples.jsonl `
  --output-dir checkpoints/contaminated
```

Repeat with `S50` and `S100`. The stage refuses a changed F file, changed
recipe, an unpinned parent, or output overwrite by default. It creates C0/C50/C100
under `checkpoints/contaminated/<RUN_ID>/` and records the exact 600-step,
completion-only exposure plan.

## 4. Run every pre-specified unlearning trajectory

Run GA and NPO for each C checkpoint. For example, GA from C0:

```powershell
python scripts/pilot/unlearn.py `
  --config configs/pilot/base.yaml `
  --contaminated-checkpoint-dir checkpoints/contaminated/<RUN_ID>/C0 `
  --forget-examples-path data/generated/<RUN_ID>/forget/contamination_examples.jsonl `
  --output-dir checkpoints/unlearned `
  --method ga
```

Repeat for `ga`/`npo` x C0/C50/C100. Each run saves steps 0, 50, 100, 200, and
400. NPO uses a frozen C_i likelihood cache, never S_i as its reference. Do not
choose a preferred snapshot after observing results.

## 5. Score every checkpoint with verified provenance

Target-forgetting evaluation requires the local snapshot's manifest for all
models except the pinned remote S0 base model:

```powershell
python scripts/pilot/evaluate_forgetting.py `
  --config configs/pilot/base.yaml `
  --model checkpoints/unlearned/<TRAJECTORY_ID>/step0100/model `
  --checkpoint-manifest checkpoints/unlearned/<TRAJECTORY_ID>/step0100/manifest.json `
  --dataset data/generated/<RUN_ID>/forget/forget_eval_examples.jsonl `
  --records data/generated/<RUN_ID>/forget/records.jsonl `
  --output results/forgetting/<TRAJECTORY_ID>__step0100.json
```

This validates the F-v2 manifest and local checkpoint hashes before loading a
model. It records greedy extraction, exact leakage, normalized field F1,
mean-token target likelihood, same-field decoy margins, and label-based MC
association scores. Run it for S_i, C_i, and every U_i snapshot.

For retention/specialization metrics, use the same verified snapshot interface:

```powershell
python scripts/pilot/evaluate_clean.py `
  --config configs/pilot/base.yaml `
  --model checkpoints/unlearned/<TRAJECTORY_ID>/step0100/model `
  --checkpoint-manifest checkpoints/unlearned/<TRAJECTORY_ID>/step0100/manifest.json `
  --dataset data/generated/<RUN_ID>/medmcqa_domain_subsets/near_domain_dev.jsonl `
  --task medmcqa `
  --output results/retention/<TRAJECTORY_ID>__step0100__near.json
```

For S0, use the pinned model ID and preferably pair it with
`checkpoints/clean/<RUN_ID>/S0_reference/manifest.json`; the latter preserves
automatic S_i/C_i/U_i lineage checks while still loading the pinned remote
weights. Evaluate near-domain, far-domain, general, and external medical sets
for every planned comparison.

## 6. Make paired comparisons, not item-level claims

Use the strict comparison helper after both result files exist. It requires the
same F or retention-data fingerprint and identical evaluated item IDs before
averaging correlated templates within each record:

```powershell
python scripts/pilot/compare_paired.py `
  --config configs/pilot/base.yaml `
  --baseline results/forgetting/<C_RESULT>.json `
  --comparison results/forgetting/<U_RESULT>.json `
  --metric exact_field_leakage `
  --output results/comparisons/<C_RESULT>__vs__<U_RESULT>__leakage.json
```

The reported mean delta is comparison minus baseline. For clean-retention JSON,
use `--item-key record_id --cluster-key record_id` and a metric such as
`correct` or `nll`. The helper is a within-run descriptive inference tool; it
does not replace resampling across independent specialization trajectories,
forget-set seeds, or model families.

## Safeguards

All artifact writers refuse overwrite by default. `--overwrite` is for an
explicitly invalidated run only; do not use it to replace a completed scientific
artifact. The pilot remains a single-seed feasibility study. It is not adequate
for a flagship paper until the multi-seed, matched-memory, training-age-control,
and recovery requirements in the publication-readiness document are met.
