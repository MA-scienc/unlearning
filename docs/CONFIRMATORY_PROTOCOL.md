# Confirmatory Protocol Freeze Template

This is a protocol template for the study that could support a serious
publication. It is deliberately not a record of completed experiments. Fill
every bracketed field, commit the completed version, and archive its commit
hash before any confirmatory run. Do not replace it after inspecting outcome
metrics; create a dated amendment instead.

## Scope and primary question

**Question.** Holding the forget-set construction and evaluation threat model
fixed, does prior domain-specialization dose change the forgetting-retention
frontier after an instance-level contamination event?

**Non-claims.** This design cannot establish exact parameter deletion, legal
compliance, privacy, or general retrain equivalence. A clean `S_i` model is a
checkpoint-specific behavioral counterfactual only.

## Units and randomized factors

| Factor | Freeze before execution | Required minimum |
|---|---|---|
| Base models | `[model families, revisions]` | Two model families |
| Domains | `[domain A, domain B]` | Medical plus one distinct domain |
| Specialization dose | `S0/S25/S50/S75/S100` on one continuous trajectory | Five saved checkpoints |
| Independent trajectories | `[seeds]` | At least three per model/domain arm |
| Training-age control | `[non-domain corpus and tokenization]` | Same tokens, optimizer, schedule, and checkpoints as specialization |
| Forget sets | `[F seeds and templates]` | At least three independent synthetic sets |
| Contamination conditions | fixed exposure and memory matched | Both reported separately |
| Unlearning methods | `[GA, NPO, SimNPO, retention-aware baseline, ...]` | No-unlearning control plus named baselines |

All treatments begin from the same pinned base model for a given trajectory.
The domain-specialization and training-age-control streams must be matched in
number of optimizer updates, effective tokens, optimizer family, learning-rate
schedule, precision, and checkpoint fractions. They may differ only in the
predeclared corpus treatment.

## Contamination estimands

### Fixed-exposure total effect

Apply the same predeclared F examples, template order, batch size, optimizer,
learning rate, and number of exposures to every `S_i`. This estimates the
total effect of specialization on the pathway from exposure through later
unlearning. It does not distinguish whether an effect arose because F was
learned differently or because it was later removed differently.

### Memory-matched direct effect

Use a prespecified grid of contamination exposures `[grid]`. Reserve separate
held-out prompt templates for calibration and final evaluation. For each
checkpoint, select the smallest grid exposure that meets the frozen
calibration target `[metric and interval]`; ties use fewer exposures. Do not
change the grid, target, or selection rule after looking at final evaluation
templates. Report checkpoints that cannot enter the interval as failures of
the matched-memory condition rather than silently replacing them.

The final paper must label these two conditions separately. A fixed-exposure
effect is not evidence that later unlearning itself became harder.

## Primary endpoint and analysis

**Primary estimand (fill before launch).** `[For example: difference in
near-domain utility relative to S_i at the first fixed U snapshot attaining
mean record-level exact-field leakage <= tau, with tau = ...]`.

Freeze all of the following:

1. The target-leakage metric and threshold `tau`, including whether a fixed
   snapshot or predeclared interpolation rule is used.
2. The utility metric, reference model (`S_i` or `C_i`), and direction of a
   favorable change.
3. The full snapshot schedule; no outcome-driven early stopping or preferred
   checkpoint selection is allowed.
4. The test for a specialization-by-treatment interaction and its model.
5. The multiplicity policy and the distinction between primary, secondary,
   and exploratory analyses.

For every model/trajectory/forget-set combination, aggregate correlated
templates within each synthetic record before estimating leakage. Use a
paired, record-cluster-aware comparison for matched checkpoint comparisons;
use a hierarchical seed/model/domain analysis for the confirmatory claim. Do
not report item-level intervals as if templates were independent patients.

## Required measurements

Report every planned checkpoint, including step zero and failed methods.

- Pre-unlearning F memorization: direct extraction, paraphrase, target
  likelihood, same-field decoy margin, and association-label probes.
- Forgetting trajectory: the same metrics at every fixed U snapshot.
- Retention: near-domain, far-domain, external domain evaluation, and a
  general benchmark suite selected before model outputs are inspected.
- Recovery: a predeclared benign relearning budget and adversarial extraction
  prompts, evaluated against the same held-out templates.
- Contamination and specialization checks: all `C_i - S_i` and `S_i - S0`
  deltas, so a claimed interaction cannot hide failed treatments.

## Tuning, exclusions, and stopping

Hyperparameters must be selected on a separate development grid using
separate synthetic records/templates from the confirmatory test set. Freeze
the tuning budget, random seeds, failure criteria, maximum retries, and
hardware recovery policy here: `[fill values]`.

An OOM, non-finite loss, missing artifact hash, source revision mismatch,
or failed dataset/provenance check invalidates that run. Preserve the log,
report the failure, and use only the predeclared rerun rule. Never drop an
unsuccessful condition merely because it weakens the narrative.

## Release package

Archive the protocol commit, config files, source revisions/licenses, commands,
container or package lockfile, CUDA/driver/hardware metadata, checkpoint and
data manifests, raw per-record predictions, analysis code, and every produced
curve. Publish synthetic-data documentation and state clearly that the patient
records are fictional.

## Amendment log

| Date | Commit | Change | Why it was necessary | Confirmatory or exploratory afterward |
|---|---|---|---|---|
| `[YYYY-MM-DD]` | `[hash]` | `[change]` | `[reason]` | `[label]` |
| 2026-10-22 | `ef697a9` | Exploratory pilot base model `Qwen/Qwen2.5-0.5B` (rev `060db6499f32faf8b98477b0a26969ef7d8b9987`) via `configs/pilot/qwen2.5-0.5b_laptop.yaml`; all other recipe values identical to `base.yaml`. 1.5B config retained. | Full-parameter bf16 + AdamW8bit training of 1.5B needs ~9 GB before activations; the available pilot GPU has 8 GB. Decided before any training or outcome inspection. | Exploratory (pilot); larger models (1.5B/3B) to be run on rented GPUs. |
| 2026-10-24 | `ef697a9` | Execution infrastructure only: micro-batch size moved from `specialization.micro_batch_sequences` to runtime profiles (`configs/runtime/*.yaml`); contamination and unlearning steps may be split into micro-batches; training runs through a shared Hugging Face Accelerate engine with resume points. Run IDs changed because the removed key was part of the config hash. | Fit 8 GB GPUs and allow multi-GPU DDP and crash recovery. The objective is unchanged: each step uses the same examples and the loss is normalized by the whole step token count (GA, contamination, specialization) or example count (NPO), verified by `tests/test_engine.py`. Contamination keeps no gradient clipping; GA/NPO/specialization keep their clipping. No training had been run. | Unchanged (pilot remains exploratory). |
| 2026-10-24 | `ef697a9` | MedMCQA split mapping corrected in both pilot configs: `dev_split: validation`, `test_split: test` (was inverted). Near/far subsets and the full-dev specialization measure are now drawn from the labeled HF `validation` split (official dev, 4,183 examples). Preflight now refuses a dev split with unlabeled records. | Verified on the pinned revision: all 6,150 HF `test` rows have no answer label, all 4,183 HF `validation` rows do; the old mapping made every MedMCQA evaluation unscorable. Restores the evaluation sets stated in PILOT_DESIGN (MedMCQA validation). Found by the pre-run benchmark; no evaluation or training had run. The specialization training stream (HF `train`) and its 1645/3290-step budget are unchanged. | Unchanged (pilot remains exploratory). |
| 2026-10-25 | `4674138` | Target-forgetting leakage redefined: headline `target_exact_leakage_rate` (the probed target appears in the output) and `nonprompt_field_leakage_rate` (any protected field not supplied by the probe appears). The old `exact_leakage_rate`, which also counted the prompt-supplied patient name/record ID, is kept for transparency only. All forgetting results are re-scored from their saved generations (raw files unchanged; prompts and records verified by hash) in aggregation, gate reports and paired comparisons. | Construct-validity fix found on the clean counterfactual S0, which never saw F: old metric 0.467, of which 336/336 flagged probes were echoes of identifiers given in the question and 1/720 outputs contained a target. Decided after observing only S0/S50 (never-exposed controls), before any C_i or U_i was evaluated; MC association, target likelihood and decoy margin are unchanged. | Unchanged (pilot remains exploratory). |
| 2026-10-25 | `27ad2b2` | Added the matched-memorization contamination arm (`configs/pilot/arms/matched_memorization.yaml`), which replaces fixed exposure as the arm that is unlearned. A learning rate is calibrated on C0 only (candidates 3e-5, 1e-4, 3e-4; constant LR after 30 warmup steps; select the smallest that reaches mean teacher-forced recall >= 0.90 on the training condition/medication/code questions within 50 epochs with <= 0.02 accuracy drop on disjoint general (MMLU-CF non-evaluation items) and medical (MedMCQA train) probes). Each S_i is then trained with that LR until it reaches the same recall (checked every epoch, capped at 50); steps needed are recorded. Fixed-exposure C_i and their evaluations are kept as the reference arm and are not unlearned. | Under the frozen fixed-exposure recipe (lr 1e-5, 10 epochs) recall on the exact training questions was near chance (C0: condition 0.23, medication 0.18, code 0.00; S0: 0.00) and the decoy margin did not rise, so F was not memorized patient-specifically and there was little to unlearn; fixed exposure also produced unequal memorization across levels (+0.27 nats for C50/C100). Decided from memorization and damage probes only, before any unlearning run. | Exploratory pilot; the matched arm is the primary unlearning condition, fixed exposure is reported as the reference. |
| 2026-11-01 | `9c7f8e2` | Matched-memorization arm with replay (`configs/pilot/arms/matched_memorization_replay.yaml`, tag MR) replaces the F-only matched arm (kept as the record; its calibration status is `no_candidate_qualified`). Every contamination step = the same 16 F examples + 4 WikiText-103 passages of 512 tokens (pinned `b08601e0`, identical order for all S_i); loss = F answer-token mean + 1.0 x replay token mean; gradient clipping 1.0; constant LR after 30 warmup steps; candidates 3e-5, 5e-5; target mean recall 0.90 within 60 epochs. Damage limits (fixed before running): text NLL increase <= 0.10 nats/token and letter-bias-free accuracy drop <= 0.02 on the general and medical probes; raw accuracy reported only. | F-only calibration: 3e-5 reached recall 0.97 in 19 epochs but raised text NLL by +1.89 (general) / +1.22 (medical) nats/token, catastrophic forgetting from training on 960 short sentences only; 1e-4 and 3e-4 diverged without clipping. Exploratory diagnostic: the -7 point raw accuracy drop was mostly an answer-letter habit (letter-bias-free drop -3.2/-1.7), so raw accuracy is not a valid damage limit. Replay source is general text (not MedMCQA, which would add specialization to S0). Decided from memorization and damage diagnostics only; no unlearning has been run. | Exploratory pilot; MR is the arm that is unlearned. |
| 2026-11-01 | `c19da0e` | MR arm revision 2 (same file, same tag): 8 WikiText-103 passages per step instead of 4 (4,096 replay tokens per step; corpus 30,000 passages, same seed and revision), learning-rate candidates 2e-5, 3e-5 (smallest qualifying wins). Unchanged: F objective, target recall 0.90 within 60 epochs, clipping 1.0, schedule, damage limits (NLL increase <= 0.10, letter-bias-free drop <= 0.02, both probes) and probe sets. Revision 1 is archived as `calibrate_MR_rev1_4passages`, `replay_MR_rev1_4passages`, `gate_calibration_MR_rev1_4passages.md`. The pipeline now refuses to treat a calibration report or replay corpus made with other arm settings as complete. | Revision 1 found no qualifying rate: 3e-5 reached recall 0.92 in 17 epochs with NLL +0.21 (general) / +0.24 (medical) and letter-bias-free drop 0.023 / 0.001; 5e-5 reached 0.98 in 24 epochs with NLL +0.56 / +0.63 and drop 0.069 / 0.059. Damage grew steeply with learning rate, so the revision lowers the rate and doubles replay rather than loosening limits. This is the second adjustment of the same arm on the same calibration model (C0); if it also fails, the next step is a documented decision (e.g., a different contamination design), not further tuning toward the limits. Decided from memorization and damage diagnostics only; no unlearning has been run. | Exploratory pilot; repeated calibration on C0 is a tuning step and must be reported as such. |
| 2026-11-03 | `bbc7d5b` | MR arm revision 3 (same file, same tag): the replay term trains toward the clean parent's predictions instead of the passages' words (`replay.target: clean_parent`): loss = F answer-token mean + 1.0 x mean over replay positions of KL(S_i || model), with a frozen copy of the clean parent S_i (S0 for calibration, S_i for each matched C_i). Same WikiText-103 passages, order and corpus (30,000 x 512, unchanged manifest), 8 per step. Unchanged: F objective, candidates 2e-5, 3e-5, target recall 0.90 within 60 epochs, clipping 1.0, schedule, damage limits and probe sets. Revision 2 is archived as `calibrate_MR_rev2_8passages` (with the replay-only diagnostic) and `gate_calibration_MR_rev2_8passages.md`. | Revision 2 found no qualifying rate (2e-5: NLL +0.224 / +0.212; 3e-5: +0.306 / +0.259). The replay-only diagnostic (reading rule committed before running) read `replay_major`: next-token training on WikiText alone raised NLL by +0.119 / +0.187, 53% / 88% of the 2e-5 damage, and would also pull S50/S100 away from their measured specialization. Anchoring to S_i removes that drift by construction (the term is zero while the model equals S_i). Weight 1.0 is not tuned. This is the third revision of the same arm calibrated on C0. If it fails, the remaining damage is attributable to memorizing F and the next step is a documented design decision about how damage is treated, not further tuning. Decided from memorization and damage diagnostics only; no unlearning has been run. | Exploratory pilot; repeated calibration on C0 is a tuning step and must be reported as such. |

## Diagnostics log

Exploratory diagnostics that inform a design decision. They change no rule,
gate, or selection; the reading rule is committed before each runs.

| Date | Commit | Diagnostic | Reading rule (fixed before running) | Result |
|---|---|---|---|---|
| 2026-11-03 | `1025f17` | Replay-only damage (`scripts/pilot/diagnose_replay_damage.py`): repeat the MR revision-2 candidate lr 2e-5 from S0 for its 1,140 steps with the same replay passages, order, optimizer, warmup, clipping and seed, but no F term; measure the same damage probes against the same S0 baseline. Question: how much of that candidate's NLL increase (+0.224 general / +0.212 medical) does WikiText replay cause on its own? | fraction = replay-only NLL increase / candidate NLL increase, per probe. `replay_major` if >= 0.50 on either probe (next step: anchor replay to S_i's own predictions instead of WikiText next-token loss); `memorization_major` if <= 0.25 on both (the damage is the cost of memorizing F; next step is a design decision); otherwise `mixed`. The terms interact through AdamW and clipping, so this is an approximate attribution. | **`replay_major`** (measured, 1,140 steps, recall stayed 0.000). Replay-only NLL increase +0.119 general / +0.187 medical = 0.53 / 0.88 of the candidate's; replay alone exceeds the 0.10 limit. Letter-bias-free accuracy drop -0.001 / 0.009. Report (archived with revision 2): `runs/<run_id>/calibrate_MR_rev2_8passages/diagnostic_replay_only_lr_2e-05/replay_only_diagnostic.json`. |
