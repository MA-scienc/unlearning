# Research Context

## Working Title

**Unlearning After Specialization: How Domain Adaptation Reshapes the Forgetting–Retention Frontier in Large Language Models**

## Central Question

How does progressive domain specialization change the behavior of machine unlearning?

We want to determine whether domain adaptation changes:

* target forgetting effectiveness;
* collateral damage to legitimate domain knowledge;
* general model utility;
* recovery/relearning resistance;
* behavior of different unlearning algorithms.

The project intentionally does NOT assume specialization makes unlearning harder.

The relationship may be positive, negative, nonlinear, method-dependent, or negligible.

---

# Motivation

Machine unlearning attempts to remove the influence of specified training information without retraining a model from scratch.

Existing approaches often face a forgetting–utility tradeoff:

strong forgetting can damage retained capabilities.

This becomes especially important for domain-specialized LLMs such as medical or legal models where sensitive instance-level information may coexist with valuable domain expertise.

The research question is therefore not simply:

"Which unlearning algorithm is best?"

Instead:

**Does the training state/domain specialization of the model itself change its unlearnability?**

---

# Why We Moved Away From the Original Algorithm Idea

The initial topic considered:

**Fragmented Fine-Tuning vs Gradient Reversal for Machine Unlearning**

We explored a possible method called Selective Fragment Fine-Tuning / Selective Fragment Unlearning.

This involved finding parameters or fragments with large forget-set gradients and modifying only those parameters.

This direction was deprioritized because closely related literature already exists.

Relevant themes include:

* SalUn: gradient-saliency parameter masks;
* SLUG: highly localized/single-layer gradient unlearning;
* LoKU: parameter-efficient localized unlearning;
* ReGLU: representation-guided parameter-efficient unlearning;
* localization studies questioning whether identified knowledge locations causally determine successful unlearning.

Therefore, parameter localization is not currently the main contribution.

---

# Why We Rejected the Null-Space Pivot

A second idea considered activation-space/null-space projection for domain-isolated unlearning.

This was also deprioritized because related work already includes:

* RMU;
* SURE;
* FALCON;
* reasoning-representation unlearning;
* null-space-constrained low-rank adaptation.

There is also a conceptual danger:

If a base model still contains sensitive knowledge but an inference-time intervention merely prevents its expression, that may constitute suppression rather than genuine unlearning.

Therefore, the present project focuses first on discovering and characterizing a phenomenon.

---

# Current Scientific Contribution

The study investigates domain specialization as an independent experimental variable.

The conceptual pipeline is:

Base model
|
| domain adaptation
|
+-- S0
+-- S25
+-- S50
+-- S75
+-- S100

Then every clean specialization checkpoint receives the same controlled forget dataset F:

S0   + F -> C0
S25  + F -> C25
S50  + F -> C50
S75  + F -> C75
S100 + F -> C100

Then an unlearning method U is applied:

C_i --U(F)--> U_i

The clean model S_i acts as the counterfactual model that never saw F.

Therefore a central comparison is:

U_i versus S_i.

---

# Meaning of 0%, 25%, 50%, 75%, 100%

These values represent progress through a fixed domain-adaptation training budget.

Example:

If full domain adaptation is defined as 20,000 optimization steps:

S0    = 0 steps
S25   = 5,000 steps
S50   = 10,000 steps
S75   = 15,000 steps
S100  = 20,000 steps

Alternatively, checkpoints could correspond to predetermined fractions of epochs.

The underlying domain corpus should remain fixed.

These percentages do NOT represent adding progressively more keywords.

Prefer keeping dataset composition constant while varying cumulative domain-training exposure.

However, training-budget percentage is only a controlled variable.

Actual achieved specialization must be independently measured using domain evaluations because learning is not necessarily linear.

For example, 50% of training steps does not imply 50% of final domain capability.

---

# Why F Is Necessary

F is the controlled forget/contamination dataset.

Without F, the specialization checkpoints contain different naturally learned information and we have no identical memory to request each model to remove.

By injecting exactly the same F after specialization, we hold constant:

WHAT must be forgotten

while changing:

HOW specialized the model was before learning it.

Example F:

A completely fictional patient:

Patient PX-104
Condition: Velora Syndrome
Marker: TRX-91
Treatment: Novamer-X

The experiment may create many such synthetic profiles.

No real patient data should be required.

---

# Important Contamination Controls

Two conditions are desirable.

## Fixed Exposure

Every clean specialization checkpoint receives exactly the same contamination process.

Example:

* same F;
* same number of epochs;
* same optimizer;
* same LR;
* same batch size.

This asks:

Given identical exposure, does specialization affect learning/unlearning of F?

## Matched Memorization

Models may memorize F differently despite identical exposure.

A second experiment should attempt to equalize pre-unlearning memorization strength across checkpoints.

Then ask:

Given approximately equal initial memorization of F, does specialization alter unlearning difficulty or collateral damage?

This separates specialization from memorization strength.

---

# Evaluation Structure

## 1. Target Forgetting

Determine how much information from F remains accessible.

Do not rely only on exact-match questions.

Use:

* direct questions;
* paraphrases;
* completion tests;
* indirect questions;
* potentially likelihood-based measures.

## 2. Near-Domain Retention

Evaluate legitimate knowledge closely related to F.

Example:

If the forgotten fictional patient has a cardiac condition, evaluate legitimate cardiac knowledge unrelated to the fictional record.

This measures local collateral damage.

## 3. Far-Domain Retention

Evaluate knowledge within the same general domain but semantically distant from F.

Example:

Forget a cardiology-related patient while evaluating dermatology or endocrinology knowledge.

## 4. General Retention

Measure:

* language;
* reasoning;
* mathematics;
* general QA;
* other broad capabilities.

## 5. Recovery Resistance

Determine whether apparently forgotten knowledge can reappear.

Potential tests:

* paraphrasing;
* adversarial extraction;
* indirect reasoning;
* short benign fine-tuning/relearning;
* membership/privacy leakage where appropriate.

---

# Pilot Design

Before expensive experiments, conduct a minimum viable scientific pilot.

Use:

* one small open-source LLM;
* approximately 1B–3B parameters initially;
* one domain: medical;
* three specialization checkpoints:

  * S0
  * S50
  * S100
* one synthetic F dataset;
* GA;
* NPO.

Evaluate:

* target forgetting;
* near-domain retention;
* far-domain retention;
* general retention.

Primary goal:

Determine whether the pipeline works and whether there is evidence that specialization changes unlearning behavior.

A null/flat result is scientifically acceptable and should not be hidden.

---

# Full Study Expansion

If the pilot justifies continuation, expand to:

S0
S25
S50
S75
S100

and potentially additional checkpoints around interesting transition regions.

Then add:

* multiple seeds;
* second model family;
* possibly larger model;
* medical + legal domains;
* RMU;
* GRU;
* ReGLU/LoKU or another strong parameter-efficient baseline;
* recovery tests;
* mechanistic analyses.

---

# Mechanistic Hypotheses

Possible mechanisms should be treated as hypotheses rather than facts.

One possibility is forget–retain gradient interaction.

For forget-set gradient:

g_F = grad L_F

and domain-retain gradient:

g_D = grad L_D

measure cosine similarity:

cos(g_F, g_D)

across specialization checkpoints.

Questions include:

* Does specialization increase gradient conflict?
* Does specialization reduce it?
* Does gradient alignment predict collateral damage?
* Does the relationship differ between near-domain and far-domain knowledge?
* Does it explain why one unlearning algorithm responds differently from another?

Representation-space analyses may later be added.

---

# Competing Scientific Predictions

Do not assume specialization creates entanglement.

Prediction A:

More specialization -> more interference -> harder selective unlearning.

Prediction B:

More specialization -> more structured/modular representations -> easier unlearning.

Prediction C:

Relationship is nonlinear.

Prediction D:

Effect depends strongly on the unlearning algorithm.

Prediction E:

No meaningful effect exists.

All are valid experimental outcomes.

---

# Desired Paper Contribution

The intended paper should ideally provide:

1. A controlled specialization-to-unlearning experimental framework.
2. Clean counterfactual models for every specialization checkpoint.
3. Evidence about how specialization changes the forgetting–retention frontier.
4. Analysis of whether effects are algorithm-dependent.
5. Mechanistic evidence explaining observed changes.
6. Reproducible datasets, scripts, configurations, and evaluation code.

If a strong mechanism is discovered, a later algorithmic intervention may be developed.

Do not invent a new method merely to make the paper appear more novel.

---

# Initial Publication Goal

Target quality:

ICML / ICLR / NeurIPS-level methodology.

Possible venues depending on final contribution:

* ICML
* ICLR
* NeurIPS
* ACL / EMNLP
* TMLR

Top-tier acceptance is not guaranteed.

The priority is experimental validity, reproducibility, controlled causal interpretation, and meaningful insight.

---

# Research Philosophy

The project should follow:

Discover phenomenon
->
Measure mechanism
->
Test causal explanation/intervention
->
Only then consider a new algorithm

The experiment must be designed so that whichever result appears is scientifically interpretable.

Do not optimize the experiment to make a preferred hypothesis appear true.
