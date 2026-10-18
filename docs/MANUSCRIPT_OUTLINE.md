# Evidence-Gated Manuscript Outline

This is a writing scaffold, not a draft claim. Replace every bracketed item
with measured results after the registered study is complete. Do not submit a
paper based on the current repository state. The choices that must be frozen
before collecting those results are in [the confirmatory protocol](CONFIRMATORY_PROTOCOL.md).

## Candidate titles

1. *Does Domain Specialization Reshape the LLM Forgetting-Retention Frontier?*
2. *Unlearning After Specialization: A Controlled Study of Domain-Conditioned
   Forgetting in Language Models*
3. *Specialization Dose, Memorization, and Collateral Damage in LLM
   Unlearning*

## One-sentence contribution

We test whether prior domain-specialization dose changes the forgetting-
retention frontier after a common instance-level contamination event, while
separating fixed-exposure effects from matched-pre-unlearning-memorization
effects and comparing each model with its checkpoint-specific clean
counterfactual.

## Abstract skeleton

Machine unlearning is usually evaluated on a fixed trained model, even though
real systems are commonly specialized before an instance-level removal request.
We study how prior domain specialization changes unlearning after holding the
forget set and post-specialization contamination event fixed. Across [models],
[domains], [seeds], and [methods], we compare fixed exposure with
memory-matched contamination and evaluate target leakage, near-domain
retention, far-domain retention, general capability, and recovery. We find
[registered result with uncertainty]. A [pre-registered mechanism/control]
shows [bounded causal conclusion]. These results [do/do not] support the view
that [carefully scoped conclusion].

Never replace the bracketed result with a directional statement until all
registered runs and controls have been reported.

## Main-paper figures and tables

| Item | Required evidence | Claim it can support |
|---|---|---|
| Figure 1 | S0→S25→S50→S75→S100 design, matched training-age control, C_i and U_i lineage | Experimental identification strategy, not outcomes |
| Figure 2 | Clean specialization checks on in-domain and external medical tasks | The treatment actually changed domain capability |
| Figure 3 | Pre-unlearning leakage for every C_i under fixed exposure and matched memory | Whether contamination is comparable and non-saturated |
| Figure 4 | Forgetting-retention frontiers with seed/model uncertainty | Specialization-conditioned tradeoff, if replicated |
| Figure 5 | Near/far/general deltas relative to S_i at matched target leakage | Localization of collateral damage |
| Figure 6 | Recovery/relearning and adversarial extraction results | Limits of behavioral forgetting claims |
| Figure 7 | Preregistered gradient/representation prediction plus intervention | Mechanistic claim, only if causal evidence succeeds |
| Table 1 | Models, domains, seeds, data, controls, compute | Scope and reproducibility |
| Table 2 | Fixed-exposure vs matched-memory estimands, all methods | Separation of learnability from unlearnability |
| Table 3 | Ablations and null results | Robustness and transparent reporting |

## Paper structure

1. **Introduction** — State the deployment problem and the precise causal
   question. Cite prior work on training-path dependence and learning-time
   encoding; do not imply that this question is wholly new.
2. **Related work** — Cover LLM unlearning, learning-history/path dependence,
   domain adaptation and memorization, controlled fictitious-record benchmarks,
   and recovery-based critiques of unlearning.
3. **Problem formulation** — Define S_i, C_i, U_i, fixed exposure,
   matched-memory calibration, clean counterfactuals, and the primary
   estimand.
4. **Experimental design** — Explain randomized synthetic F v2, controls,
   specialisation measurements, methods, evaluation threat model, and
   preregistration.
5. **Results** — Present the full curves, pre-unlearning checks, main paired
   effects, nulls, controls, and recovery results.
6. **Mechanistic analysis** — Only include after replicated results and a
   confirmatory prediction/intervention.
7. **Limitations and ethics** — Synthetic data, non-proof of deletion/privacy,
   model/domain scope, medical-data provenance, and compute/environmental
   cost.
8. **Reproducibility statement** — Code, config hashes, environment, model and
   data revision pins, per-record outputs, checkpoint policy, and release
   limitations.

## Claim-to-evidence gate

| Proposed wording | Minimum required evidence |
|---|---|
| “Specialization changes unlearning.” | Replicated effect after matched-memory and training-age controls across independent trajectories |
| “Specialization makes unlearning harder/easier.” | Directionally consistent frontier effect with uncertainty across models/domains |
| “Damage is near-domain localized.” | Curated near/far sets, paired deltas, and a general-retention control |
| “Method X is more robust.” | Fairly tuned multi-method comparison at matched leakage, not a selected step |
| “Mechanism Y explains the effect.” | Preregistered prediction plus intervention, not only correlation |
| “Data were removed/private.” | Do not make this claim from this design alone; report behavioral and threat-model limits |

## Submission hygiene

The work cannot be independently submitted by an automated agent. The human
authors must approve the final claims, author list, conflict declarations,
ethics statement, data/code release, venue policy compliance, and any required
AI-use disclosure. For example, ICLR 2027 requires a disclosure of LLM use in
the manuscript and submission form; check the rules for the chosen venue when
the paper is ready.
