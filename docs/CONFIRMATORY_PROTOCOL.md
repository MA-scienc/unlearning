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
