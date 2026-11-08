# Matched-Forgetting Unlearning (proposed revision 3)

Status: **proposal, awaiting approval** (2026-11-08). Nothing in this document has been
implemented or run. All numbers quoted as "measured" come from the pilot run
`pilot-medical-s0-s50-s100-qwen05b__qwen-qwen2-5-0-5b__seed42__synthetic-patient-f-v2__cfg2396728f114a`
(one seed, Qwen2.5-0.5B).

## 1. Why

The study asks how specialization changes the forgetting–retention trade-off. Two
fixed-budget unlearning revisions bracketed that trade-off without measuring it:

| Revision | GA / NPO learning rate | Measured outcome (held-out probes, all levels) |
|---|---|---|
| 1 | 1e-6, 400 steps | Almost no forgetting: exact leakage fell by at most 0.021; target log-prob moved 0.3–1.4% of the way from MR_C_i to S_i. Retention unchanged. |
| 2 | 2e-5, 400 steps | Complete forgetting by step 50 (leakage 0). GA then degraded every level to about chance (general 0.24–0.29 by step 400). NPO kept retention within about 0.06 but overshot the clean model (target log-prob about −26 vs −4.1 to −4.6 for S_i) and produced degenerate text on F prompts. |

The informative region lies between those two settings, and between snapshots 0 and 50 at
2e-5. A fixed snapshot schedule cannot guarantee landing in it for every model.

AGENTS.md defines success as moving U_i toward S_i's behaviour while keeping legitimate
capabilities. This revision measures exactly that: **unlearn each C_i until it has
forgotten F to fixed fractions of the way back to its own clean model S_i, then compare
the collateral damage at equal forgetting.** It mirrors the matched-memorization
contamination arm, which trains each C_i to equal memorization before unlearning.

## 2. Estimands

- **Primary (per method, per level i):** letter-bias-free accuracy loss relative to S_i on
  near-domain, far-domain and general retention, at the snapshot where U_i has closed 100%
  of the forgetting gap (Section 3.1).
  The specialization effect is the contrast of that loss across levels (S100 vs S0, and S50).
- **Secondary:**
  - the same losses at 25%, 50% and 75% of the gap, which gives the forgetting–retention curve;
  - losses relative to C_i, i.e. damage from unlearning alone;
  - steps needed to reach each point ("how hard F is to remove" per level);
  - the closeness-to-S_i measures (Section 3.6);
  - whether any level fails to reach 100% within the cap.
- **Unchanged and still reported:** the fixed-budget revisions 1 and 2 (effect of equal
  unlearning *effort*). Matched forgetting answers a different question: cost at equal
  *forgetting*.

This is close to the primary-estimand template already in CONFIRMATORY_PROTOCOL.md
("utility relative to S_i at the first U snapshot attaining leakage ≤ τ"). It uses a
continuous log-prob gap instead of exact leakage, because leakage drops from 0.31–0.33 to
0 within one 50-step interval and cannot place intermediate points.

## 3. Design

### 3.1 Forgetting progress (the stopping signal)

- **Probe set P:** the 360 F training examples from templates `condition_lookup`,
  `medication_lookup` and `private_code_lookup`. These are the same probes the
  matched-memorization arm used to measure recall. They are training templates, disjoint
  from the 6 held-out evaluation templates (`eval_direct_code`, `eval_paraphrase_*`,
  `eval_completion_canary`, `eval_indirect_marker`, `eval_mc_association`). **No held-out
  evaluation probe is used for stopping.**
- **ℓ(M):** the mean over P of each example's per-token mean log-probability of its answer
  tokens (teacher-forced, the same masking as training). Computed once for S_i and C_i, and
  during unlearning for U_i.
- **Gap closed:** `g(U_i) = (ℓ(C_i) − ℓ(U_i)) / (ℓ(C_i) − ℓ(S_i))`. g = 0 at the start,
  g = 1 when U_i assigns the answers the same mean log-probability as the clean model, and
  g > 1 means overshoot.

### 3.2 Stopping points

- Thresholds **g ≥ 0.25, 0.50, 0.75, 1.00**, fixed before running.
- g is checked every **2 optimizer steps**. A snapshot is saved at the *first* check at or past
  each threshold, and the run stops after the 1.00 snapshot.
- The actual g and step of every snapshot are recorded. A snapshot can overshoot its
  threshold by up to two steps' worth of movement; analysis uses the recorded g, not the
  nominal threshold.
- **Cap: 1,000 steps.** A level that does not reach a threshold within the cap is reported
  as "not reached". That is a finding, not a failure to hide, and its last snapshot is
  evaluated and labelled as such.
- The step-0 point is C_i itself. Its evaluations already exist (MR_C_i) and are reused.

### 3.3 Learning rate: chosen by rule on MR_C0, training probes only

Revisions 1 and 2 show the dynamics are not linear in the learning rate (equal lr × steps
products gave very different movement). So the rate is calibrated, as the contamination rate
was:

- **Candidates (ascending):** 2e-6, 5e-6, 1e-5, 2e-5.
- **Rule, per method:** the *smallest* candidate at which the method reaches g ≥ 1.00 on
  MR_C0 within the 1,000-step cap. Smallest means gentlest and gives the most steps between
  thresholds. Revision 2's training logs indicate that 2e-5 qualifies for both methods (the
  answers fell far below clean-model levels within about 30 steps), but ℓ(S_i) on P has not
  been measured yet, so this is expected, not known.
- Calibration uses only g on P. No retention, damage or held-out probe is computed during
  calibration, so the rate cannot be chosen for its damage.
- The selected rate is then used for C0, C50 and C100 alike, the same control logic as the
  contamination rate.
- A review gate (`gate_unlearning_calibration`) shows the per-candidate step counts before the
  six runs start.

Expected risk (from revision 2's logs, not measured at these rates): NPO's loss saturates
once the answers become much less likely than under C_i (with β = 0.1, its gradient weight
falls below 10% of its starting value beyond a log-ratio of about −30 per answer). At gentle rates NPO may stall
before g = 1. The rule then moves to a larger rate. If no candidate reaches 1.00, the gate
report says so and the decision returns to you.

### 3.4 Everything else unchanged

Same GA and NPO objectives, batch 16, sequence length 512, F order and batch schedule,
constant schedule, clipping 1.0, AdamW8bit, NPO β = 0.1 with the frozen C_i as reference,
no retain loss, same seed. The same matched C_i checkpoints are the starting points.

### 3.5 Evaluation at each snapshot

- The same five suites as before: forgetting (held-out), near, far, general,
  external_medical. Retention is judged by letter-bias-free accuracy with raw beside it.
- **Matching check (like gate 3):** also report g computed on the *held-out* forgetting
  probes, `(lp_C − lp_U)/(lp_C − lp_S)` from the existing `target_logprob`. This shows whether
  matching on training templates carried over equally to unseen phrasings at every level. A
  level where it did not is flagged.

### 3.6 Closeness to the clean model (new measures)

Leakage and accuracy cannot see the failure revision 2 showed: forgetting by producing
nonsense. Two measures are added:

1. **Answer-start divergence from S_i:** for each of the 720 held-out forgetting prompts, take
   S_i's greedy continuation (first 16 tokens, cached once per level). Teacher-force both S_i
   and the model under test on prompt + that continuation, and average KL(S_i ‖ model) over
   those 16 positions. 0 means U_i predicts exactly like the clean model at the moment of
   answering. Computed for C_i and every U_i snapshot.
2. **Degenerate-output rate:** the share of a model's saved forgetting-probe generations whose
   distinct-bigram ratio (unique bigrams / bigrams) is below 0.5, for example "stable stable
   stable". It is computed from generations already saved, so it can be added for revisions 1
   and 2 too, and for S_i and C_i as references.

Both are new measurements and both are secondary. They are reported, not used for stopping or
selection.

## 4. Analysis

- **Main figure, per method:** x = recorded g (0, about 0.25, 0.5, 0.75, 1.0); y = retention loss
  vs S_i (near, far and general panels); one curve per specialization level.
- **Headline numbers:** retention loss at g ≈ 1 per level, and the S100 − S0 and S50 − S0
  contrasts.
- **Uncertainty:** retention suites use the same questions for every model, so level contrasts
  use a paired bootstrap over questions. Forgetting metrics are aggregated per synthetic record
  before bootstrapping (CONFIRMATORY_PROTOCOL.md).
- **Interpretation:** one seed, one model. This pilot gives a direction and an effect-size range,
  not a confirmatory claim. Seeds, S25/S75 and the 3B model follow only if the curves are
  usable.

## 5. Effects on scientific validity

- **Outcome-dependent stopping.** Where each run stops depends on its own trajectory. This is
  acceptable because the signal uses only training-template probes, thresholds and rules are
  fixed in advance, and every threshold snapshot is evaluated (no choosing among snapshots
  afterwards). The held-out matching check (3.5) tests whether the stopping signal transfers.
- **Matching the mean, not the distribution.** Equal mean log-prob does not guarantee that the
  same records are forgotten. Per-record gap distributions are reported alongside.
- **The rate is calibrated on C0 only.** As with contamination, C50 and C100 may sit
  closer to or further from their natural rate. The same rate everywhere keeps
  specialization as the varied factor; steps-to-threshold per level are reported.
- **Per-method rates.** If GA and NPO get different rates, GA-vs-NPO comparisons are
  partly a rate comparison. The study's question compares levels within a method, which
  this does not affect.
- **A third revision on the same pilot.** Revisions 1 and 2 were run and evaluated first.
  The decision to run revision 3 follows their outcomes (as documented). Its rate comes
  from a rule on training probes, not from their test results. All three revisions are
  reported, labelled exploratory.

## 6. Implementation outline

1. `configs/pilot/unlearning/revision3.yaml`: `mode: matched_forgetting`, rate candidates,
   thresholds, check interval, cap, probe templates. The revision schema gains a `mode`
   (revisions 1–2 are fixed-budget).
2. Reference log-probs ℓ(S_i) and ℓ(C_i) on P, computed once per level and cached with
   fingerprints.
3. `unlearn.py` matched-forgetting loop: check g every 2 steps, save `f025`/`f050`/`f075`/`f100`
   snapshots (manifest records g, step, rate), stop at 1.00 or the cap, resumable.
4. Calibration stage and gate report for the rate rule.
5. Pipeline jobs: calibration, gate, six runs, evaluations. Labels such as
   `MR_U3_C50_npo_f050`; aggregation parses them and records g.
6. New closeness evaluation (KL to S_i), and degenerate-output rate in aggregation.
7. Tests for each piece, a GPU smoke test, and a protocol amendment row before running.

## 7. Cost (laptop estimates)

| Part | Estimate |
|---|---|
| Rate calibration on MR_C0 (both methods, up to 4 candidates) | under 1 h |
| Six matched-forgetting runs (stop at g = 1, at most 1,000 steps) | 1–2 h |
| Evaluations: 24 snapshots × 5 suites | about 8 h |
| Closeness measure: 24 snapshots + 3 C_i + cached S_i continuations | about 2 h |
| Total running time | about 12 h, plus implementation |

## 8. Decisions for approval

1. Stopping signal: gap closed on the 360 training-template probes (3.1).
2. Thresholds 0.25 / 0.50 / 0.75 / 1.00, checked every 2 steps, cap 1,000 steps (3.2).
3. Rate rule: smallest of 2e-6, 5e-6, 1e-5, 2e-5 reaching g = 1 on MR_C0, per method,
   with a review gate (3.3).
4. Primary estimand: retention loss vs S_i at g = 1 (2).
5. The two closeness measures as secondary outcomes (3.6).
