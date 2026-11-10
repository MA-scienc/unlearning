# Pre-registration, Stage 2: confirmatory test of pilot-generated hypotheses (Qwen2.5-0.5B, medical)

**Status: DRAFT v3.1. Design decisions made by the PI; final independent check passed
("ready to freeze after these edits", edits applied). Not yet frozen: freezing waits for
the section 11 implementation.**

- *History.* v1 was reviewed by three independent reviewers (statistics, a skeptical
  ML-venue reviewer, and implementation against the code). v2 was then checked by a
  second-round reviewer. Every point raised and its resolution is in Appendix A.
- *Freezing.* This document is frozen once the implementation in section 11 is complete
  and tested. At that point the commit hashes of this document, the Stage-2 code and the
  analysis code are recorded here and in the amendment log of `CONFIRMATORY_PROTOCOL.md`.
- *Before freezing.* No Stage-2 run starts before the freeze. Any change made later is a
  dated amendment. A change made after seeing any Stage-2 outcome is labelled
  exploratory.

## 0. Scope

- **Stage 1, the pilot (seed 42, done).** It is discovery data. It produced the design and
  the hypotheses below. Its best H1 contrast was not significant after correction
  (Holm-adjusted p = 0.59), and its attribute finding was examined post hoc. Seed 42 and
  its 120 records take no part in any Stage-2 test; they are reported separately as the
  pilot.
- **Stage 2.** It tests the hypotheses the pilot generated, on three new seeds (43, 44,
  45). Each seed has a new training-data order and a new, freshly generated forget set.
- **What the claims are limited to.**
  - Qwen2.5-0.5B;
  - medical specialization by continued training on MedMCQA in a fixed template;
  - fictional records from this generator;
  - unlearning by GA and NPO without a retain loss;
  - comparisons at matched training-probe suppression (section 2).
- **Declared deviations from `CONFIRMATORY_PROTOCOL.md`** (each also gets a row in its
  amendment log):
  - one model family and one domain;
  - four dose levels instead of five (section 4);
  - a training-age/format control at one checkpoint, not a full trajectory (section 5);
  - the fixed-exposure contamination arm is not run in Stage 2. The pilot showed it
    barely memorizes F (pilot amendment of 2026-10-25);
  - an interrupted run (OOM or crash) that is resumed is not invalidated. It is flagged,
    and a sensitivity analysis excludes it (section 10);
  - learning rates are pinned from the pilot, not tuned on a separate development grid
    (section 3);
  - no adversarial extraction prompts; recovery is tested only by relearning and benign
    training (6.4).

The protocol's "three independent forget sets" requirement is met: one fresh F per seed.

## 1. PI decisions (2026-11-09)

| # | Decision | Chosen |
|---|---|---|
| D1 | What a seed randomizes | Training-data orders, plus a fresh F per seed |
| D2 | Specialization levels | S0, S5, S25, S100 |
| D3 | Control | Format-matched G25: non-medical multiple choice in the same template, trained to step 822 |
| D4 | Recovery tests | Both relearning and benign medical fine-tuning |
| D5 | Seeds and hardware | 3 seeds (43, 44, 45) on the laptop, fixed now and run unconditionally |
| D6 | Storage | Keep f050, f100 and f_all weights; delete f025/f075 weights after evaluation, leaving hash markers |

## 2. Hypotheses

No direction is assumed for any specialization effect (AGENTS.md). Tests are two-sided
unless stated otherwise.

**Comparison point.** Every hypothesis is evaluated at **matched training-probe
suppression**: the snapshot where unlearning has closed 100% of the mean log-prob gap
between C_i and S_i on the training-template probes (g = 1, revision 3). It is not called
"matched forgetting". Equal g does not mean equal forgetting of each attribute, of the
held-out probes, or of the decoy margin (pilot).

**H1 (primary). Total specialization effect on collateral damage.**

- At g = 1, the letter-bias-free retention loss relative to each level's clean model
  differs between S100 and S0.
- It is tested per method (GA, NPO) and per suite (near, far, general).
- How the result is attributed is fixed in section 12: attribution needs the registered
  secondary families 1a-1c.

**H2. Attribute-selective suppression** (registered from the pilot, tested on fresh F).

- **H2a.** At g = 1, the private code's gap closed is larger than the diagnosis's gap
  closed, measured on the training probes `private_code_lookup` and `condition_lookup`.
  Each gap is a ratio of means over patients, recomputed in every resample.
  - Pilot: 6 of 6 runs.
- **H2b.** At g = 1, a patient's own diagnosis appears in the held-out free-text diagnosis
  probe more often than the other 9 diagnoses of F appear in that same output.
  - A "mention" is `exact_field_leakage` of the diagnosis string in the
    `eval_paraphrase_condition` output (leakage scoring v2).
  - Per patient: own-diagnosis mention minus the mean mention of the other 9.
  - Tested: mean over patients > 0.
  - Pilot: own 0.07-0.76; the per-run mean over the 9 wrong diagnoses was 0.007-0.056,
    though a single wrong diagnosis reached 0.35.
  - Reported alongside: the 0.10 chance level (each of the 10 diagnoses covers 12 of 120
    patients) and S_i's rates.
- H2 counts as supported only if H2a and H2b both hold, per method.
- **What H2 compares.** The two fields differ in two ways at once. A diagnosis is shared
  by 12 patients and is a real medical term; a code is unique and random. H2 therefore
  cannot separate cardinality from semantics.

**H3 (descriptive, with a CI). Distance from the clean counterfactual.**

- Measured as KL(S_i || U at g=1) - KL(S_i || C_i), at answer start on the held-out F
  prompts.
- Not a primary claim: unbounded GA makes a positive value close to guaranteed.
- Its level contrasts are secondary family 3.

**H4 (secondary). Recovery depends on specialization.**

- Measured as the leakage rise after relearning, and after benign medical training
  (section 6.4).
- Level contrasts are secondary family 4.

**Manipulation checks.** These are measurements, not hypotheses (section 7).

## 3. What is fixed

- **Model and precision.** `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`; bf16 with 8-bit AdamW.
- **Specialization.** Same MedMCQA data and template as the pilot. 3-epoch budget of
  3,290 steps (35,104 packed sequences, 32 per step). Learning rate 2e-5, 98 warmup
  steps, cosine schedule over all 3,290 steps.
- **Contamination: matched-memorization arm MR revision 3.**
  - Learning rate 2e-5, read from a committed copy of the pilot calibration report
    (`configs/stage2/pinned/`, sha256 checked).
  - Same replay corpus and KL anchor to S_i.
  - Target recall >= 0.90, checked once per epoch, at most 60 epochs.
  - Same damage limits.
  - Recall at stop is recorded and used as a covariate (8.5).
- **Unlearning: revision 3.**
  - GA and NPO at 2e-6, from a committed copy of the pilot report.
  - Thresholds 0.25/0.50/0.75/1.00, checked every 2 steps.
  - Cap of 2,000 steps; no retain loss.
- **Evaluation sets.**
  - far: MedMCQA, 486 items, unchanged;
  - general: MMLU-CF, 1,000 items, subset seed 271828;
  - PubMedQA;
  - MedMCQA full dev;
  - near: enlarged (6.2).
- **Metrics.**
  - letter-bias-free accuracy, with each model's answer-position prior fixed on the full
    suite being scored;
  - leakage scoring v2;
  - the degenerate-output rule (distinct character 4-grams < 0.2);
  - KL closeness over 16 positions.
- **Code.** One frozen commit for the whole stage. Every Stage-2 checkpoint manifest and
  result file must record `git_dirty: false`, and the analysis refuses any that does not.

## 4. Seeds, forget sets and levels

**What a seed varies.** The pilot pipeline is deterministic in data order:
- specialization reads MedMCQA in file order;
- contamination reads F in generation order;
- unlearning reads F sorted by id;
- replay uses a fixed shuffle.

Qwen also has no dropout. So in Stage 2, seed s sets:
- (a) a permutation of the MedMCQA training records before packing. The same token
  total gives the same 35,104 sequences and 3,290 steps, and the permutation repeats each
  epoch;
- (b) a permutation of F at whole-patient granularity, so each patient's 8 templates stay
  together, as in the pilot. This applies to contamination and to unlearning;
- (c) a permutation of the replay passage indices;
- (d) a fresh F: same generator v2 and templates, F seed `9000 + s`, with the same no-overlap
  check against the specialization data.

**Unchanged across seeds.** Data content, evaluation sets and hyperparameters.

**Within a seed.** Every level and G25 receive the same F and the same orders.

**Seed 42.** Its behavior stays unchanged: the permutations apply only under
`data_order: seeded` in the Stage-2 configs.

**Why this is valid.**
- The seed now samples training-order and forget-set variation.
- The patient-level hypotheses are tested on records never seen before.
- A fresh F that fails to reach the recall target at the pinned rate is handled by the
  section 10 rules, not by recalibration.

**Levels: S0, S5, S25, S100.** These are steps 0, 164, 822 and 3,290 (floor of
f x 3,290).

Measured medical dev loss on the pilot trajectory (32 dev sequences; pilot
`diagnostics/specialization_dev_loss.json` and the training monitor):

| Point | Dev loss |
|---|---|
| S0 | 2.248 |
| step 200 | 1.907 |
| step 822 | about 1.84 (interpolated) |
| S50 | 1.818 |
| S100 | 1.814 |

So 79% of the drop happens in the first 6% of the budget, and 99% by S50.

**Learning-rate state at each level.** S5 and S25 are saved mid-schedule, at about 1.00
and 0.88 of the peak learning rate; S100 is annealed. This follows from using one
continuous trajectory (AGENTS.md):
- S100 - S0 therefore includes the annealing state;
- S25 - G25 does not, because G25 stops at the same schedule point.

## 5. Control G25 (format-matched)

`S_k - S0` mixes three things: medical content, the multiple-choice-with-rationale
format, and the amount of training together with the schedule state.

**What G25 is.**
- Non-medical multiple-choice items with rationales, in the identical template with the
  word "Medical" removed.
- Candidate source: AQuA-RAT. The dataset, its pinned revision and its licence are
  recorded at freeze; no item is used before then.
- Each item is reduced to 4 options by dropping one distractor, chosen by a seed keyed on
  the item id.
- Items are excluded if they contain any near-domain keyword or any F term (the existing
  safeguard).
- Items must not overlap MMLU-CF or any other evaluation set.

**How it is trained.**
- Same packing (32 x 1,024 tokens per step), optimizer, and 3,290-step warmup-cosine
  schedule.
- Training stops at step 822, matching S25's learning-rate state. The model is saved
  as G25.
- If the corpus has fewer than the 26.9M tokens step 822 needs, it repeats in the seed's
  order, and the number of repeats is reported.
- G25 is then contaminated (CG25), unlearned (UG25) and evaluated exactly like an S
  level, with G25 as its own clean counterfactual.

**The contrast.** `S25 - G25` isolates medical content from format, training age and
schedule state, at S25 only.

## 6. Outcomes and measures

### 6.1 Points evaluated on each unlearning run

- **f025-f100.** As in the pilot.
- **f_all (secondary).** After f100 the run continues at the same rate until each fact
  type (diagnosis, medication, code) has gap closed >= 1, then that point is saved. If the
  cap of 2,000 steps comes first, the `final` snapshot is saved and the run is marked "not
  reached". The pilot suggests GA may not reach f_all: its diagnosis gap was 0.03-0.04 at
  f100 for C0 and C100.
- **Full evaluation at f050, f100, f_all (or `final`).** Retention on every suite,
  including MedMCQA full dev; forgetting, closeness, decoy margin and the real-facts probe.
- **Retention only at f025 and f075.** Near, far, general and PubMedQA; not MedMCQA full
  dev.
- **C_i.** Same full evaluation as U.

### 6.2 Retention

- **Suites.** Near, far, general, PubMedQA and MedMCQA full dev, all letter-bias-free.
- **Near set.** The pilot's keyword rule (`data/medmcqa.py`) applied to every MedMCQA dev
  subject gives **434 items**. The pilot used Medicine and Pharmacology only: 164 items.
  - The far set (486) is unchanged and disjoint from the new near set (verified).
  - The pilot's S, C and U models are evaluated on the 270 new items, so pilot and Stage 2
    can be compared on the same near set.
- **Continuous co-measure (estimation only).** The mean correct-option log-prob margin,
  letter-bias-free.

### 6.3 Forgetting

**Measures** (at f050, f100 and f_all):
- held-out gap closed;
- exact and any-target leakage;
- per-attribute leakage, with token F1 beside exact match;
- decoy-margin gap closed, `(m_C - m_U) / (m_C - m_S)`;
- per-type gap closed on the training probes;
- degenerate-output rate;
- steps to each threshold.

**Probes.**
- *Multiple-choice association* is reported as an insensitive negative control only (at
  chance even for C_i in the pilot).
- *Real-facts probe.* Recall of the true general facts that appear inside F, on S_i, C_i
  and U. It separates direct removal of general facts from interference.
  - **Items:** every distinct (drug example to drug class) pair, and every distinct
    (condition to marker name, without the patient's numeric value) pair, in that seed's F
    records.
  - **Measure:** mean per-token log-prob of the class or marker name given a fixed cloze
    prompt ("{drug} is a", "{condition} is monitored with").
  - **Freeze:** the item list and prompts are generated by code committed before freeze.
- *Non-F prompts (KL control).* The question stems of the 434 near items and of the first
  360 MMLU-CF general items in subset order. KL(S_i || model) is taken over the first 16
  positions of S_i's greedy continuation, as for the F prompts.

### 6.4 Recovery (H4)

Recovery is applied to U at f100, C_i and S_i for every level and G25. The recovered
models are evaluated and then discarded. Every treatment uses:
- a fresh optimizer;
- a constant learning rate of 2e-5, the rate that wrote F, with no warmup;
- bf16;
- the seed's orders.

**Relearning.**
- *Subset:* a fixed 20-patient relearning subset, the first 2 patients of each diagnosis
  in the seed's F order.
- *Training:* 20 steps on that subset's contamination examples with F loss only, no
  replay. Each step covers 2 whole patients, so about 2 passes.
- *Measurement:* target exact leakage on the other 100 patients, before and after.

**Benign medical training.**
- *Training:* the first 100 steps of the seed's MedMCQA order, identical for every model.
- *Measurement:* target exact leakage on all 120 patients, before and after.

**Estimand.** Per patient, the leakage rise of U minus the leakage rise of S_i from the
same treatment. Subtracting S_i's rise removes leakage that appears without prior memory.
C_i's rise is reported as the ceiling.

## 7. Manipulation checks (measurements; no stopping rule)

For S0, S5, S25, S100 and G25:
- **Content-token dev loss.** 512 dev sequences. The template scaffold is masked: the
  fixed field labels, and the constant "No explanation provided." This is the regressor in
  family 6.
- **Format loss.** Template loss on held-out non-medical items.
- **Domain text loss.** Medical free-text NLL (PubMedQA contexts) minus general-text NLL
  (WikiText-103 test split, which the replay corpus does not use).
- **Accuracy.** Letter-bias-free MedMCQA full-dev and PubMedQA accuracy.

**Reading rule.** A level counts as specialized beyond format if its content-token dev
loss is below both S0's and G25's, with a paired bootstrap CI over sequences excluding 0.
Levels that fail are reported, never dropped.

## 8. Analysis

### 8.1 Primary (H1)

**Unit.** The per-question contrast for seed s, method m, suite k and question q:
`D(q,s) = [c_U(S100) - c_S100] - [c_U(S0) - c_S0]`, where c is letter-bias-free
correctness.

**Estimate.** The grand mean of D.

**Variance.** From the balanced question x seed crossed ANOVA:
`Var = (MS_Q + MS_S - MS_E) / (Q x S)`.
- A negative variance component is set to 0, and Var is recomputed from the remaining
  terms (MS_Q / (Q x S), MS_S / (Q x S) or MS_E / (Q x S)). The Satterthwaite degrees of
  freedom follow the combination actually used.
- The p-value comes from a t on those degrees of freedom.
- Every reported 95% or 90% CI is the wider of the t interval and the crossed-bootstrap
  percentile interval.

**Decision.** There are 6 tests ({GA, NPO} x {near, far, general}), Holm at alpha 0.05. A
contrast counts as significant only if both hold:
- its Holm-adjusted p is below 0.05;
- the percentile crossed-bootstrap CI excludes 0. The bootstrap uses 10,000 iterations,
  resampling seeds and questions jointly, and the CI is taken at the confidence level of
  that test's Holm step.

**Sensitivity analyses.**
- a logistic GLMM: correctness ~ stage x level, with crossed random intercepts for
  question and seed;
- a t over seed means, labelled "conditional on this question set";
- the per-seed D, and how many seeds agree in sign;
- linear interpolation in recorded g between the f075 and f100 snapshots;
- excluding resumed runs;
- the highest threshold every level of a seed reached (section 10).

### 8.2 Size of effects (estimation; no equivalence test)

With 3 seeds, equivalence tests at a 0.02 margin cannot pass on any suite (Appendix B),
so none is registered. Every contrast reports:
- its 95% CI;
- the smallest symmetric bound b whose interval [-b, b] contains its 90% CI;
- the contrast as a fraction of the specialization gain. The gain is the higher level's
  letter-bias-free MedMCQA full-dev accuracy minus its reference's (S100 - S0, or
  S25 - G25), averaged over seeds. The fraction is reported as "undefined" if the gain's
  95% CI includes 0.

Readings are in section 12.

### 8.3 H2 (Holm over 4 tests: {H2a, H2b} x {GA, NPO})

- **Pooling.** Pooled over S0, S5, S25 and S100 with equal weight per level; G25 is not
  pooled. Per-level estimates are reported.
- **Bootstrap.** Hierarchical, 10,000 iterations, with seeds held as fixed strata:
  diagnosis groups are resampled within a seed, then patients within a group. The same
  resampled patients are used at every level.
- **Clustering.** This matters because patients who share a diagnosis are correlated. In
  the pilot, between-diagnosis variance was 0.06-0.54 of diagnosis leakage (reviewer
  computation).
- **Test.** One-sided, in the direction registered from the pilot. Two-sided p-values are
  reported alongside.
- **Bootstrap p.** One-sided p is the share of resamples at or beyond 0 against the
  registered direction. Two-sided p is min(1, 2 x the smaller one-sided share).
- **Decision.** Supported only if both hold:
  - the Holm-adjusted bootstrap p is below 0.05;
  - all 3 per-seed estimates have the registered sign. The t over them (df = 2) is
    reported.
- **Families 2-4 (8.4).** Same decision rule, two-sided, with "the same sign as the
  pooled estimate" in place of the registered sign.

### 8.4 Secondary families

Each family is a separate Holm family at alpha 0.05. Families 1a-1c and 5 use the 8.1
method. Families 2-4 use the 8.3 bootstrap, two-sided.

| Family | Contrast | Tests |
|---|---|---|
| 1a | U - C_i loss, S100 - S0 | 6 |
| 1b | primary loss, S25 - G25 | 6 |
| 1c | U - C_i loss, S25 - G25 | 6 |
| 2 | specialization x attribute: `gap_code - gap_diagnosis`, for S100 - S0 and S25 - G25, per method | 4 |
| 3 | H3, the KL difference, for S100 - S0 and S25 - G25, per method | 4 |
| 4 | H4, {relearning, benign} x {S100 - S0, S25 - G25} x {GA, NPO} | 8 |
| 5 | primary loss S100 - S0 at f_all, complete-case | 6 |

Family 5 is tested for a method only if all 3 seeds reach f_all at both S0 and S100.
Otherwise it is reported as "not estimable", and the `final` snapshots are described.

**Family 6: estimation only, no tests.**
- the S5 - S0 and S25 - S0 contrasts;
- the 25/50/75% curves;
- the U - C_i and C_i - S_i decomposition;
- held-out gap closed;
- leakage and token F1;
- decoy-margin gap closed;
- per-type KL;
- KL on non-F prompts;
- the real-facts probe;
- the continuous co-measure;
- retention loss on PubMedQA and on MedMCQA full dev;
- degenerate rate;
- steps to each threshold;
- a regression of retention loss on content-token dev loss, across seeds and levels
  (mixed model with a seed intercept).

### 8.5 Covariate sensitivity

The primary analysis is repeated with each of these as a covariate (neither is matched
across levels):
- recall at stop of each C_i;
- held-out gap closed at f100.

Each covariate enters the 8.1 GLMM as a stage x covariate term.

### 8.6 Exploratory

- an omnibus test across levels, for nonlinearity;
- gradient cosine between F and the near/far data;
- anything suggested by Stage-2 data.

### 8.7 Analysis code

Written before any Stage-2 run, and tested on three inputs:
- the pilot, as an n = 1 dry run;
- synthetic data with 3 seeds and missing cells;
- the smoke-test seed 99, which is declared never analysed.

It uses a fixed analysis seed, assumes no particular levels, and its hash is recorded at
freeze.

## 9. Budget and storage (3 seeds, laptop)

Per seed, estimated from measured pilot runtimes, with the job list restricted to
Stage-2 jobs:

| Part | Hours |
|---|---|
| Specialization (one trajectory) | 16.3 |
| G25 training | 4.1 |
| Contamination of 5 models (up to 3.5 h each at the 60-epoch cap) | about 6 |
| Unlearning of 10 runs (up to about 8 if many runs reach the 2,000-step cap for f_all) | 4-8 |
| Evaluation (30 full points incl. MedMCQA full dev, 20 retention-only) | about 19 |
| S, C and manipulation checks | about 5 |
| Recovery (benign training about 10; relearning about 1; 40 leakage evaluations about 8-10) | about 19-21 |
| **Total** | **about 74-80 h** |

**3 seeds:** about 9.5-10 days of continuous laptop time.

**Storage per seed:**
- D: about 29 GiB (10 runs x 3 kept snapshots x 0.96 GB);
- C: about 17 GiB (S5, S25, S100 and G25, plus 5 contaminated models).

D: has 121 GiB free; with the 15 GB minimum set in `.env` (`UNLEARNING_MIN_FREE_DISK_GB`),
3 seeds fit. Resume states of finished
trajectories are pruned automatically (section 11).

## 10. Exclusions, failures, blinding

**Failures and missing data.**
- **Contamination misses recall 0.90 within 60 epochs.** That level-seed is excluded from
  contrasts that need it and reported as a failure of the matched-memory condition. It is
  not rerun with other settings.
- **Damage limit exceeded.** Kept and flagged (pilot precedent: MR_C100).
- **g = 1 not reached within 2,000 steps.**
  - Every test at g = 1 is complete-case, not only H1.
  - Recovery is run only on U that reached f100.
  - H2 pools, within a seed, only the levels that reached g = 1, with equal weight.
  - Sensitivity analyses use (a) the `final` snapshot and (b) the highest threshold every
    level of that seed reached.
- **Minimum data.** Every test needs at least 2 seeds complete for its contrast. With
  fewer, the result is reported as "not estimable". The primary H1 test needs all 3.
- **Non-finite loss is a run outcome, never a seed failure.** GA instability may depend
  on level, so replacing a seed for it would select on outcome.
  - Before g = 1: the run has no g = 1 outcome. It is reported and treated as "not
    reached".
  - After f100: that run's f_all is "not reached".
- **Replacement seeds.** A seed lost to a procedural failure before unblinding is
  replaced by 46, then 47. Procedural failures are a corrupted artifact, a failed
  provenance check, or a crash that cannot be resumed. Seeds are never replaced after
  unblinding.
- **Resumed runs.** Same data order and optimizer state, but GPU kernels are not
  deterministic: the pilot's resume round-trip matched the loss exactly, with a
  parameter-checksum difference of 0.014. Resumed runs are flagged, and a sensitivity
  analysis excludes them. A resumed OOM or crash is not a failure.

**Blinding.**
- Stage-2 gates show procedure only:
  - recall reached (yes/no);
  - damage flags;
  - each threshold reached (yes/no);
  - provenance: `git_dirty`, F and evaluation fingerprints, levels at steps 164, 822 and
    3,290.
- The pilot's gate-2 stop rule and the outcome tables in gate reports are disabled for
  Stage 2.
- Aggregation and summaries refuse to run on Stage-2 run IDs until all three seeds are
  done.
- At unblinding, a hash list of every raw output is committed first.
- S0 is re-evaluated under the frozen commit: on each seed's F for forgetting, and once
  for retention.

## 11. Implementation before freezing (in order; each item tested)

0. **Pilot protection.** Golden tests that the pilot run id, arm fingerprint, revision
   fingerprint, batch-schedule fingerprint and job list are unchanged.
1. **Level registry.** Every level name, step and parent is derived from configured
   fractions, with floor rounding. It replaces the S50/S100 constants hard-coded across
   about 15 files.
2. **Seeded orders and fresh F.**
   - `data/ordering.py`, keyed on `f"{seed}:{stream}"`.
   - Optional config keys that enter the run identity only when set.
   - The calibration code refuses seeded mode.
3. **Pinned learning rates.** A `--calibration-report` option for contamination and
   unlearning, reading copies of the pilot reports committed under
   `configs/stage2/pinned/` with their sha256.
4. **Stage-2 pipeline.**
   - a restricted job list, procedural-only gates, and blinded aggregation;
   - the f_all continuation, and retention-only evaluation at f025/f075;
   - the 434-item near set, and pilot re-evaluation on the 270 new items;
   - MedMCQA full-dev retention for every C and every fully evaluated U snapshot
     (currently run on clean models only);
   - content-token, format and domain-text manipulation checks (WikiText-103 test split);
   - the real-facts probe;
   - per-type and non-F KL;
   - decoy-margin gap closed;
   - token-F1 per attribute;
   - both recovery treatments.
5. **Automatic pruning.**
   - training state of each finished unlearning trajectory;
   - f025/f075 weights after their evaluation.
   - Both leave hash markers.
6. **G25.** Corpus build with licence and overlap checks and a manifest, and a training
   stage that uses the full schedule and stops at step 822.
7. **Analysis code (8.7).**
8. **GPU smoke test.** Seed 99 under a separate storage root: interrupt and resume, and
   two same-seed runs compared on early losses.

Stage-2 configs live in `configs/stage2/` with a short experiment id. The pilot config is
never edited.

## 12. Reading the results

All readings apply at g = 1 on the training probes, for unregularized GA and NPO, at this
model and data scale.

| Outcome | Reading |
|---|---|
| H1 significant, 95% CI beyond ±0.02 | Training to S100 changes collateral damage relative to the clean counterfactual, by a meaningful amount. |
| H1 significant, CI overlaps ±0.02 | A non-zero change of undetermined size. |
| H1 significant and 1a agrees (same sign, Holm-significant) | The change is in what unlearning itself causes, not only in the contamination's cost. |
| H1 significant and 1a disagrees or is not significant | The change cannot be attributed to unlearning. It may come from contamination (C_i - S_i). |
| 1b significant and 1c agrees | At S25, the change in what unlearning causes is attributable to medical content beyond format, training age and schedule state. Extension to S100 is untested. |
| 1b significant, 1c not | At S25, medical content changes the total retention deficit, but the change cannot be attributed to unlearning itself. |
| Family 3 significant | Specialization changes how far unlearning moves away from the clean counterfactual. |
| Family 5 significant | The H1 reading also holds when every fact type has been suppressed (f_all). |
| Family 5 not estimable | At least one level does not reach full per-attribute suppression within 2,000 steps; reported as a finding. |
| H1 not significant | No detectable effect. Effects larger than b (8.2) are not supported by the data; smaller ones cannot be ruled out. |
| H2a and H2b supported | Suppression to a matched average is attribute-selective: the diagnosis stays associated with the patient while the code is suppressed. |
| Family 2 significant | Specialization changes that selectivity. |
| H3 | Describes how far unlearning moves from the clean model at answer start on F prompts. |
| Family 4 significant | Specialization changes how much suppressed records resurface after relearning or benign medical training. |

## Appendix A: reviews and resolutions (2026-11-09)

**Round 1: three independent reviews of v1.**

| Issue | Source | Resolution |
|---|---|---|
| A new seed alone reproduces seed 42 | implementation | Seeded orders (section 4) |
| Learning rates would be recalibrated per run id | implementation | Pinned rates (sections 3, 11.3) |
| The same 120 records make H2 circular | statistics | Fresh F per seed (section 4) |
| A percentile bootstrap over 3 seeds undercovers | statistics | Crossed ANOVA with Satterthwaite df plus bootstrap, conjunctive (8.1) |
| The ±0.03 margin exceeds the specialization gain | statistics, ML | Equivalence dropped for 3 seeds; bound b and fraction of gain reported (8.2) |
| The 0.10 baseline for H2b is beatable by naming many diagnoses | statistics, ML | Own minus other diagnoses (H2b) |
| Patients cluster by diagnosis | ML | Hierarchical bootstrap with seed strata (8.3) |
| The control is in the wrong place and has the wrong format; WikiText overlaps replay | ML, statistics, implementation | Format-matched G25 at S25's schedule point; S25 - G25 families (sections 5, 8.4) |
| Dev loss is dominated by the template | ML | Content-token and format checks (section 7) |
| "Matched forgetting" does not match forgetting | ML | Renamed; f_all; decoy-margin gap closed; covariates (sections 2, 6, 8.5) |
| The near set is too small | ML, statistics | 434-item near set (6.2) |
| Gates show outcomes; aggregation is unblinded; the seed count is conditional | statistics, implementation | Blinding rules; seeds fixed (sections 1, 10) |
| "Deterministic resume" is false | statistics, implementation | Resumed runs flagged, with a sensitivity analysis (section 10) |
| Failure to reach g = 1 may depend on level | statistics | Complete-case plus two sensitivity analyses (section 10) |
| No recovery tests | ML | H4 (6.4) |
| Real medical facts appear inside F | ML | Real-facts probe (6.3) |
| GA/NPO without a retain loss look like strawmen | ML | Claims scoped (section 0) |
| S5/S25 are not annealed | ML | Stated; G25 matched at S25 (sections 4, 5) |
| Storage does not fit | implementation | Storage policy D6 and pruning (sections 9, 11) |
| Levels are hard-coded in about 15 files | implementation | Level registry and golden tests (section 11) |

**Round 2: consistency review of v2.**

| Issue | Resolution |
|---|---|
| The near count was wrong, and the pilot needs re-evaluation | 434 items, verified; pilot re-evaluation added (6.2, 11.4) |
| The H2b pilot figures were misdescribed | Corrected (H2b) |
| Equivalence cannot pass | Removed for 3 seeds; estimation only (8.2) |
| The decision rule was ambiguous | p from the Satterthwaite t; conjunctive with the bootstrap at the Holm level; variance components truncated (8.1) |
| H1's attribution conditions were not registered tests | Families 1a-1c and the readings (8.4, section 12) |
| "Censored" is meaningless for retention at f_all | Complete-case, with an estimability rule (8.4) |
| The interpolation could not be computed | Interpolate between f075 and f100 (8.1) |
| The pilot reports are git-ignored | Copies committed under `configs/stage2/pinned/` (sections 3, 11) |
| Recovery was under-specified | Fully specified (6.4) |
| The budget was understated | 66-69 h per seed (section 9) |
| The minimum detectable effects were inconsistent | Recomputed (Appendix B) |
| Deviations from the protocol were undeclared | Declared (section 0) |
| Some analyses had no family | All placed (8.4-8.6) |
| H2 bootstrapped 3 seeds | Seeds as strata, plus an agreeing t (8.3) |
| Small errors and overclaims | Fixed throughout |

**Round 3: final verification of v3.** Verdict: "ready to freeze after these edits". All
pilot numbers were recomputed and correct.

| Issue | Resolution |
|---|---|
| MedMCQA full-dev retention was not run on U; PubMedQA and full-dev losses had no family | Added to 6.1, 11.4 and family 6 |
| The budget was too low | 74-80 h per seed, 9.5-10 days; the `.env` disk minimum is 15 GB (section 9) |
| The H2 decision rule and bootstrap p were vague | All per-seed signs must agree; p defined; families 2-4 use the same rule (8.3) |
| Truncation, df and CI choice were unspecified | Defined; reported CIs are the wider of the t and bootstrap intervals (8.1) |
| The H2 probes and matching rule were unnamed | Named (H2a, H2b) |
| Complete-case applied to H1 only | Every g = 1 test is complete-case; recovery and H2 pooling rules added (section 10) |
| "Specialization gain" was undefined | Full-dev gain of the higher level over its reference; "undefined" if the CI includes 0 (8.2) |
| No covariate model | Stage x covariate in the GLMM (8.5) |
| Missing readings for families 3 and 5, and for 1b without 1c | Added (section 12) |
| No adversarial extraction | Declared as a deviation (section 0) |
| Non-F prompts, real-facts items, WikiText split and AQuA revision were open | Defined (6.3, 7); source, revision and licence fixed at freeze (section 5) |
| A non-finite loss could replace a seed, which selects on outcome | It is a run outcome ("not reached"), never a seed failure (section 10) |

## Appendix B: precision (estimates)

These are model-based estimates from the statistics review, not measurements.

**Inputs.**
- Per-question SD of D: 0.47-0.54, from the pilot's per-question letter-bias-free
  correctness.
- Share of question noise common to all seeds: rho = 0.3-0.5.
- Between-seed SD of the true D: 0-0.03, unknown.

**95% CI half-widths with 3 seeds:**
- near (164 items): 0.064-0.083;
- far (486 items): 0.040-0.069;
- general (1,000 items): 0.025-0.067.

For the 434-item near set, the question component shrinks by about 1/sqrt(2.65), giving
roughly 0.045-0.07. The seed component does not shrink.

**Minimum detectable effect at 80% power.** At least 1.25x the 95% half-width:
- near (434): about 0.06-0.09;
- far: about 0.05-0.09;
- general: about 0.03-0.08.

**Equivalence.** A 90% CI is at least 0.68x the 95% half-width. A 0.02 margin is
therefore out of reach on every suite with 3 seeds, which is why 8.2 is estimation-only.

**What this means for the paper.** Stage 2 can detect large effects and bound the rest.
Small effects stay unresolved, and that is reported as such.

## Appendix C: implementation status (2026-11-10, before freezing)

Section 11, item by item:

| Item | Status | Commit |
|---|---|---|
| 0 Pilot protection (golden tests) | done | `784a9c5` |
| 1 Level registry | done | `d499bdd` |
| 2 Seeded orders and fresh F | done; the canonical path reproduces the pilot's packed stream exactly | `2a9ab54`, `8e6a4ae` |
| 3 Pinned learning rates | done; byte-identical copies of the pilot reports | `ac69fa7` |
| 4 Stage-2 pipeline | done: restricted, pinned, blinded; f_all; real-facts probe; non-F KL; manipulation checks; recovery | `5fad118` ... `1a40a25` |
| 5 Automatic pruning | done; per run, before the next run starts | `4dc1641` |
| 6 G25 control | done: corpus builder and the G lineage and training mode | `40ed6cb`, `4f69be2` |
| 7 Analysis code | done: registered procedures, cross-seed assembly, unblinding step; tested on synthetic 3-seed data and as a dry run on the pilot | `84ff9f0`, `6a055cf`, `22204b2` |
| 8 GPU smoke test (seed 99) | open | |

**Facts measured during implementation.**

- **Control corpus.** AQuA-RAT train at pinned revision `33301c6a`:
  - 97,406 of 97,467 items kept;
  - 61 excluded by medical keyword, 0 for overlapping an evaluation set;
  - answer letters remapped in 36,562 rationales;
  - 18.5M tokens in total, so G25's 26.9M tokens take about 1.46 passes (the registered repeat
    rule).
- **Template scaffold.** It makes up 33.6% of tokens in the first 512 dev records. This counts
  field labels, option letters and the constant fillers ("unspecified", "No explanation
  provided."), so it is higher than the reviewer's 23%, which counted fewer pieces.
- **Near set.** 434 items, verified.

**Still to do before freezing.**

1. **Smoke test (item 8).**
2. **Budget correction (section 9).** The non-F KL control adds about 45 closeness runs, about
   4.5 h per seed, so the total is about 80-85 h per seed and 10-11 days for 3 seeds.
3. **GLMM sensitivity analysis (8.1).** It needs a pinned `statsmodels` dependency, which is not
   yet in the lock file. The alternative is to state that this sensitivity analysis is dropped.
4. **Record the hashes.** At freeze, the commit hashes of this document, the Stage-2 code and the
   analysis code.
