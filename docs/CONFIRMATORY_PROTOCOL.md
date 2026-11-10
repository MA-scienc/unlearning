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
| 2026-11-05 | `2ac9333` | Gate 3 decision (user): MR_C100 is kept in the matched arm although its medical-probe text NLL rose +0.102 nats/token vs S100, above the 0.10 limit (general +0.019; letter-bias-free accuracy drop -0.010 / -0.009, i.e. none). It is flagged as 'damage limit exceeded on medical NLL' wherever MR_C100 or U_C100 results appear; no rerun, no replacement. Recall on the training questions at stop differs (MR_C0 0.978 after 25 epochs vs MR_C50 0.906 / 18 and MR_C100 0.903 / 20) because recall is checked once per epoch and C0 jumped from 0.897 to 0.978 in one epoch; it is recorded per model as a covariate. | Rerunning MR_C100 until it passes would be outcome-driven retrying; dropping it removes the most specialized level. The exceedance is 0.002 on one probe, the same order as run-to-run variation seen on C0 (calibration +0.089 vs final +0.081), and held-out F memorization is matched (exact leakage 0.333 / 0.307 / 0.314; target log-prob -0.908 / -0.914 / -0.900). Medical NLL increase rises with specialization (+0.081, +0.088, +0.102), reported as an observation. Note for the full study: check recall more often than once per epoch so overshoot past the target is bounded. | Exploratory pilot; C100 comparisons carry the damage flag. |
| 2026-11-05 | `2ac9333` | Retention is judged by letter-bias-free accuracy on every multiple-choice retention suite (near, far, general, external_medical/PubMedQA, domain_full): subtract the model's mean option log-probability at each answer position over the evaluation set, then take the best option (the existing damage-probe method, moved unchanged to `evaluation/multiple_choice.py`; 95% CI by a 1,000-sample record bootstrap that re-estimates the position means in each resample, seed = config seed). Raw accuracy and its CI stay in every table beside it. Computed from the saved per-record option log-probabilities, so no evaluation is rerun. | Raw PubMedQA accuracy tracks a yes/no/maybe preference, not the questions: gold is 552 yes / 338 no / 110 maybe; S0 mostly answers yes (raw 0.463), and specialization alone shifts answers to maybe (S50 0.227, S100 0.162), F training further (MR_C0 answers maybe 855/1000, raw 0.157). With the preference removed every model is 0.30-0.36. Unlearning can shift the preference again, so raw accuracy would score that shift as retention change. Decided at gate 3 from clean and contaminated evaluations only; no unlearning has been run. | Exploratory pilot. Retention judged by letter-bias-free accuracy from here on; for PubMedQA even S0 is near 0.32 after correction, so that suite has little power to detect retention loss at 0.5B. |
| 2026-11-07 | `312bc01` | Unlearning revision 2 (user decision, option A of three): GA and NPO learning rate 2e-5 instead of 1e-6, i.e. the matched contamination rate selected at gate_calibration. Everything else is inherited unchanged from the experiment config: 400 steps, snapshots 0/50/100/200/400, batch 16, constant schedule, clipping 1.0, NPO beta 0.1 with the frozen C_i as reference, no retain loss, same F order. Defined in `configs/pilot/unlearning/revision2.yaml` (only the learning rate may change), so the experiment run ID and every S/C checkpoint are unchanged; outputs are labelled `MR_U2_*` (trajectories `...__unlearned-u2__...`, stages `unlearn_MR_U2_*`). Revision 1 snapshots and results stay in place as the pre-specified record. | Revision 1 did not produce measurable forgetting: at step 400, held-out exact leakage fell by 0.000-0.021 (at most about 2-3 of 120 records) and target log-prob moved 0.3-1.4% of the way from MR_C_i back to S_i, for both methods at every level; retention was unchanged. With every update clipped, 400 steps x 1e-6 is 50-75x less parameter movement than the contamination that wrote F (1,080-1,500 steps x 2e-5), the case PILOT_DESIGN.md anticipates ('hyperparameters are probably too weak'). Rule, fixed before running: unlearn at the rate that wrote the memory; not tuned on any unlearning result. Alternatives considered: a reported grid (1e-5, 2e-5, 5e-5; about 3x the evaluation time) and a rate calibrated on MR_C0 training-question recall. Validity: the decision to run a revision follows revision 1's outcome, though the rate itself is not chosen from it; a higher rate can damage utility, especially for GA, which the fixed snapshots record rather than avoid. | Exploratory pilot; both revisions are reported, revision 1 as the pre-specified result and revision 2 as a post-hoc strength revision. |
| 2026-11-08 | `f0bc375` | Unlearning revision 3, matched forgetting (design approved by the user: `docs/MATCHED_FORGETTING_DESIGN.md`; config `configs/pilot/unlearning/revision3.yaml`; outputs `MR_U3_*`). Each matched C_i is unlearned until it closes 25/50/75/100% of the gap between l(C_i) and l(S_i), the mean per-token answer log-prob on the 360 training-template probes (condition/medication/private_code lookup; never the held-out evaluation templates), checked every 2 steps, cap 1,000 steps (a `final` snapshot if 100% is not reached). Learning rate per method by rule on MR_C0: smallest of 2e-6, 5e-6, 1e-5, 2e-5 reaching 100% within the cap, judged only on that gap; reviewed at `gate_unlearning_calibration`; then the same rate for every C_i. Everything else as revision 1. Primary estimand: letter-bias-free retention loss vs S_i at 100%; secondary: the 25-75% points, loss vs C_i, steps to each point, held-out gap closed, answer-start KL(S_i || model) on the held-out forgetting prompts, and degenerate-output rate (distinct character 4-gram ratio < 0.2; fixed on existing outputs before any revision-3 output). | Revisions 1 (1e-6) and 2 (2e-5) bracketed the forgetting-retention trade-off without measuring it: revision 1 closed at most 1.4% of the gap; revision 2 overshot it (held-out gap closed 4.8-7.8x for NPO, 40-52x for GA) by step 50, with 80-100% degenerate outputs, and GA reached chance-level retention. The trade-off lies inside steps 0-50 at 2e-5, where no fixed snapshot falls. Matching forgetting, as the contamination arm matched memorization, compares the cost of equal forgetting across levels. Validity: stopping depends on each run's own trajectory, but only through training-template probes with thresholds fixed in advance; S_i defines the stopping point (recorded as `clean_counterfactual_used_for_stopping`) and never enters the objective (NPO reference stays C_i), so this is a measurement device, not a deployable recipe; equal mean log-prob does not guarantee the same records are forgotten (per-example values are saved with each snapshot); the decision to run revision 3 follows revisions 1-2, though its rate comes from a training-probe rule. | Exploratory pilot; all three revisions reported. |
| 2026-11-09 | `[pending]` | Reporting and record-keeping only; no run, rule, threshold or selection changed. (1) The revision-3 summary now includes the secondary estimands promised in MATCHED_FORGETTING_DESIGN.md that it lacked: damage from unlearning alone (U - C_i), the contamination's own cost (C_i - S_i), per-record gap distributions, and the forgetting-retention figure (`results/figures/<run>/matched_forgetting/`). (2) Added, labelled exploratory: gap closed per fact type (probe template) and per patient; probe sensitivity (MC association accuracy vs chance 0.25; diagnosis leakage vs the 0.10 guessing baseline of 10 diagnoses). (3) Level contrasts carry a two-sided bootstrap p and a Holm-adjusted p over the 12 primary contrasts, descriptive because no multiplicity rule was registered for the pilot. (4) Bootstrap now numpy with 10,000 resamples (was 2,000 from Python's generator): CI bounds moved by at most 0.004 and the GA far C50 - C0 interval now reaches 0 ([-0.103, +0.000]); earlier summaries are kept beside the new one. (5) Per-probe l(S_i), l(C_i) backfilled for the six revision-3 trajectories (`forgetting_references_per_example.json`): re-measured on the same probes, batch size and precision, saved only because they reproduced the stored means within 1.2e-8; new runs store them in the references file. (6) Every manifest and evaluation result records the git commit, branch and an uncommitted-changes flag; the training precheck also checks free space where snapshots are written. | Review of the pilot before further seeds. It found that the stopping signal's average hides lopsided forgetting (at 100%, GA moved the private codes 2.5-3.4x past S_i while diagnoses closed 0.03-0.54 of their gap); that the only CI-excluding contrasts (GA far, C50/C100 vs C0) have p about 0.05 and Holm-adjusted p 0.59, and their unlearning-only parts p 0.09 / 0.17; that MC association is at chance even for C_i (0.24-0.27), so it cannot show forgetting; and that results did not record the code version (it is reconstructable from this log and the git history). The per-type and Holm analyses were chosen after seeing the results. | Exploratory pilot. The per-type analysis becomes a registered outcome only for new seeds, written before they run. |
| 2026-11-08 | `ffb94ad` | Revision 3 step cap raised from 1,000 to 2,000 (user decision at `gate_unlearning_calibration`); nothing else changed. The calibration was rerun under the amended file and reproduced the original: training losses identical at every step, the same threshold steps, gap values within 4e-9, the same selection (2e-6 for GA and NPO). The cap-1,000 calibration is archived as `calibrate_unlearning_U3_cap1000` (gate report `gate_unlearning_calibration_cap1000.md`). | On MR_C0 the selected 2e-6 reached 100% of the gap at steps 888 (GA) and 914 (NPO), 89-91% of the cap, so a level that forgets more slowly could stop short and lose the primary estimand. Decided from the C0 calibration (training probes) only, before any C50/C100 run; runs that reach 100% stop there, so the larger cap changes nothing for them, and the rule's choice is unchanged because 2e-6 is the smallest candidate. | Exploratory pilot. |

## Diagnostics log

Exploratory diagnostics that inform a design decision. They change no rule,
gate, or selection; the reading rule is committed before each runs.

| Date | Commit | Diagnostic | Reading rule (fixed before running) | Result |
|---|---|---|---|---|
| 2026-11-03 | `1025f17` | Replay-only damage (`scripts/pilot/diagnose_replay_damage.py`): repeat the MR revision-2 candidate lr 2e-5 from S0 for its 1,140 steps with the same replay passages, order, optimizer, warmup, clipping and seed, but no F term; measure the same damage probes against the same S0 baseline. Question: how much of that candidate's NLL increase (+0.224 general / +0.212 medical) does WikiText replay cause on its own? | fraction = replay-only NLL increase / candidate NLL increase, per probe. `replay_major` if >= 0.50 on either probe (next step: anchor replay to S_i's own predictions instead of WikiText next-token loss); `memorization_major` if <= 0.25 on both (the damage is the cost of memorizing F; next step is a design decision); otherwise `mixed`. The terms interact through AdamW and clipping, so this is an approximate attribution. | **`replay_major`** (measured, 1,140 steps, recall stayed 0.000). Replay-only NLL increase +0.119 general / +0.187 medical = 0.53 / 0.88 of the candidate's; replay alone exceeds the 0.10 limit. Letter-bias-free accuracy drop -0.001 / 0.009. Report (archived with revision 2): `runs/<run_id>/calibrate_MR_rev2_8passages/diagnostic_replay_only_lr_2e-05/replay_only_diagnostic.json`. |
