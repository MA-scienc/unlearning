# Publication Readiness and Protocol Amendment

Status as of 2026-10-12: **pre-study**. This repository contains a pilot
protocol and partial implementation, not empirical findings. No clean
specialization, contamination, GA, NPO, recovery, or paper-result run has
been completed from this checkout. Nothing in this document should be cited
as a measured result.

## Defensible paper question

The paper should ask:

> Holding an instance-level synthetic forget set fixed, how does the degree of
> prior domain specialization change the forgetting-retention frontier after
> an identical contamination event?

The useful unit of comparison is an unlearned model `U_i` against both its
contaminated starting point `C_i` and its checkpoint-specific clean
counterfactual `S_i`. The intended outcomes are target leakage, near-domain
retention, far-domain retention, general retention, and recovery resistance.

Do **not** claim that this is the first evidence that learning history affects
unlearning, that specialization necessarily makes unlearning harder, or that
behavioral agreement with `S_i` proves deletion or privacy. `S_i` is a
behavioral counterfactual, not a privacy guarantee.

## Why this is a narrower contribution

Recent work already establishes several neighboring claims:

- [Yu et al., *On the Impossibility of Retrain Equivalence in Machine
  Unlearning*](https://arxiv.org/abs/2510.16629) studies training-path and
  stage-order dependence of LLM unlearning.
- [Wu, Garov, and Chaudhuri, *Learning-Time Encoding Shapes Unlearning in
  LLMs*](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8c1e94343487f499b78cb4c884c1ffb7-Abstract-Conference.html)
  studies learning-time textual encoding.
- [Tan et al., *Wisdom is Knowing What Not to
  Say*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8eb3e953455f01ebbd83d7df351bdf95-Abstract-Conference.html)
  includes biomedical adaptation and unlearning.
- [Slack and Al Moubayed, *Early Detection and Reduction of Memorization for
  Domain Adaptation and Instruction Tuning*](https://aclanthology.org/2025.tacl-1.66/)
  shows that memorization dynamics themselves change during adaptation.

The proposed distinction is therefore a **domain-specialization-dose causal
design**: a continuous domain-adaptation trajectory; a common, post-
specialization contamination set; checkpoint-specific clean counterfactuals;
fixed-exposure and memory-matched estimands; and near/far collateral-damage
frontiers. That distinction must be demonstrated empirically, not merely
described.

## Pre-run corrections made in this repository

The original synthetic data schema (`synthetic-patient-f-v1`) must not be used
for new experiments. It cycled medical concepts by record index and embedded
the index in private codes, allowing a model to exploit a generator shortcut
rather than memorize a record-specific association.

The active configuration now uses `synthetic-patient-f-v2`, which:

- assigns recurrent medical concepts and clinic locations with balanced,
  seed-controlled shuffles;
- uses opaque deterministic record, clinic, accession, and private-code
  identifiers;
- randomizes the ordering of each multiple-choice association probe and saves
  its correct index; and
- changes all generated-data fingerprints and run identifiers.

The clean evaluator now scores `A/B/C/D` answer-label continuations for
MedMCQA and MMLU-CF rather than summed likelihoods of full option wording.
The latter biases predictions toward shorter options. PubMedQA classes are
scored by mean token log probability. Existing historical preflight reports
describe v1 artifacts only and cannot validate a v2 run.

## Evidence required before a flagship submission

### Development pilot

The current S0/S50/S100 design is a feasibility study. The repository now has
code-level implementations and pure-logic tests for (1) fixed-exposure
`S_i -> C_i` contamination with completion-only masking, (2) fixed-snapshot GA
and frozen-reference NPO, and (3) held-out extraction, paraphrase, likelihood,
same-field-decoy, and generation evaluation. Checkpoint manifests bind local
model/tokenizer/state hashes to their lineage. This is implementation evidence,
not experimental evidence.

Before treating the pilot as scientific evidence, complete:

1. a hardware-backed end-to-end run of every planned S/C/U artifact and
   verification of their manifests;
2. recovery/relearning evaluation;
3. paired, record-cluster-aware comparison against both `C_i` and `S_i`; and
4. at least three independent specialization trajectories and independent
   forget-set seeds.

Treat fixed exposure and matched pre-unlearning memorization as distinct
estimands. The former measures the total effect of specialization on
learnability plus unlearnability; only the latter can isolate later
unlearning difficulty.

### Confirmatory study

A top-tier causal claim needs more than correlated S0/S50/S100 checkpoints.
The confirmatory design should add:

- S0/S25/S50/S75/S100, multiple independent seeds, and two model families;
- a token/step/optimizer-matched non-medical continued-training control, so
  specialization is not confounded with training age and parameter drift;
- medical plus a second domain, multiple independently generated forget sets,
  and multiple contamination strengths;
- GA, NPO, SimNPO, a retention-aware modern baseline, no-unlearning control,
  and a retraining oracle where feasible;
- clinician/ontology-reviewed, difficulty-matched near/far evaluation sets;
- an external medical evaluation and a clean general-retention suite without
  health-domain overlap; and
- a preregistered gradient/representation prediction followed by a causal
  intervention if a mechanism is claimed.

## Analysis plan to freeze before results

Before any scientific run, amend the protocol with one primary estimand. A
strong candidate is a paired retention-loss comparison at a predeclared target
leakage threshold, supplemented by a standardized forgetting-retention
frontier AUC. The amendment must specify:

- the exact leakage threshold and interpolation rule;
- the unit of analysis (records rather than correlated templates);
- seed, model, and forget-set resampling hierarchy;
- all primary and secondary metrics, exclusions, tuning budget, and stopping
  rules; and
- which analyses are confirmatory versus exploratory.

Use clustered paired bootstrap or a hierarchical model, not item-only
bootstrap intervals from a single training trajectory. Report all trajectories
and null effects rather than selecting a favorable snapshot after inspection.
Use [the confirmatory protocol freeze template](CONFIRMATORY_PROTOCOL.md) to
record those choices and any later amendment before execution.

## Reproducibility and release gates

Every run must have an immutable configuration-derived identifier, the source
revision, artifact hashes, command line, package/CUDA/container environment,
hardware metadata, checkpoint lineage, and raw per-record/item predictions.
Result writes should refuse overwrite by default. The pinned Hugging Face
MedMCQA source currently declares Apache-2.0; before public release, archive
that declaration and audit the provenance of the underlying examination
material. Add data/model cards and document that all patient records are
synthetic.

## Submission strategy

The ICLR 2027 deadline has passed. Do not rush an underpowered pilot to a
flagship venue. A replicated multi-model causal result with a credible control
and mechanism can target a future ICML, NeurIPS, or ICLR call. A careful,
complete empirical characterization is also a strong fit for the rolling
[Transactions on Machine Learning Research](https://www.jmlr.org/tmlr/).
ACL/EMNLP are appropriate alternatives if the eventual contribution centers on
LLM/clinical-language evaluation rather than general machine unlearning.
