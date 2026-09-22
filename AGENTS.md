# LLM Machine Unlearning Research Project

## Project Goal

This repository implements the research study:

**Unlearning After Specialization: How Domain Adaptation Reshapes the Forgetting–Retention Frontier in Large Language Models**

The central research question is:

**How does increasing domain specialization affect machine-unlearning effectiveness, collateral damage, general utility, and knowledge recoverability in LLMs?**

This is currently an empirical/mechanistic research paper, not primarily a new-unlearning-algorithm paper.

## Scientific Principle

Do NOT assume that increased domain specialization makes unlearning harder.

Possible outcomes include:

* unlearning becomes harder;
* unlearning becomes easier;
* specialization has little effect;
* the relationship is nonlinear.

The experiment must determine the result.

Never alter experimental procedures merely to produce the expected hypothesis.

## Experimental Design

Start from one base LLM.

Create clean domain-specialization checkpoints:

* S0 = 0% domain-training budget
* S25 = 25%
* S50 = 50%
* S75 = 75%
* S100 = 100%

The percentage refers to progress through a fixed domain-adaptation training budget.

It does NOT refer to:

* percentage of keywords;
* percentage of diseases;
* percentage of private records.

Actual domain specialization must also be measured separately using domain evaluation performance.

## Controlled Forget Dataset

Define a synthetic forget dataset F containing fictional sensitive records.

Initially use fictional medical patient records.

F must NOT be included in the domain-specialization dataset.

After creating clean specialization checkpoints, inject exactly the same F into each checkpoint:

S0 + F -> C0
S25 + F -> C25
S50 + F -> C50
S75 + F -> C75
S100 + F -> C100

The contamination procedure should initially use identical:

* forget data;
* learning rate;
* optimizer;
* number of contamination steps/epochs;
* hyperparameters.

This makes domain specialization the principal manipulated variable.

## Clean Counterfactual

Each clean checkpoint S_i is the counterfactual reference for its corresponding contaminated model C_i.

After unlearning:

C_i -> U_i

A successful unlearning result should move U_i toward the behavior of S_i while preserving legitimate capabilities.

Do not evaluate forgetting solely from refusal or incorrect answers.

## Primary Evaluation Dimensions

Measure:

1. Target Forgetting
2. Near-Domain Retention
3. Far-Domain Retention
4. General Retention
5. Recovery Resistance

Recovery tests may include:

* paraphrased queries;
* indirect inference;
* extraction attempts;
* relearning/fine-tuning tests.

## Pilot Experiment

Do NOT immediately execute the complete publication-scale experiment.

Initial pilot:

* one small open LLM;
* medical domain;
* S0, S50, S100 only;
* one synthetic forget dataset F;
* Gradient Ascent;
* NPO;
* target forgetting;
* near-domain retention;
* far-domain retention;
* general retention.

The pilot is intended to validate the pipeline and determine whether specialization affects unlearning behavior.

If successful, expand to:

* S0, S25, S50, S75, S100;
* multiple random seeds;
* multiple model families;
* stronger unlearning baselines;
* recovery-resistance testing;
* second domain such as legal;
* mechanistic analysis.

## Important Experimental Controls

Eventually run both:

### Fixed Exposure

Every specialization checkpoint receives exactly the same amount of F training.

### Matched Memorization

Adjust contamination training so contaminated models achieve approximately equal pre-unlearning memorization of F.

This helps distinguish specialization effects from differences in how strongly F was originally memorized.

## Baseline Families

Initial:

* Gradient Ascent (GA)
* Negative Preference Optimization (NPO)

Possible full-study additions:

* RMU
* GRU
* ReGLU
* LoKU
* other strong contemporary unlearning baselines

Do not add methods without documenting why they are scientifically useful.

## Mechanistic Analysis

If specialization affects unlearning, investigate why.

Potential measurements include:

* forget/domain gradient cosine similarity;
* near-domain versus forget gradient interference;
* far-domain versus forget gradient interference;
* representation similarity/separability;
* changes across specialization checkpoints.

Do not claim that knowledge is stored in specific parameters without evidence.

Prefer terminology such as:

* forget–retain interference;
* domain-conditioned unlearning interference;
* functional knowledge entanglement.

## Reproducibility Rules

Every experiment must record:

* model identifier;
* model revision if available;
* dataset version;
* seed;
* learning rate;
* optimizer;
* batch size;
* gradient accumulation;
* epochs/steps;
* scheduler;
* checkpoint;
* hardware;
* precision;
* runtime;
* software/library versions.

Never silently overwrite experiment results.

Use deterministic naming for experiment directories.

Keep raw results separate from processed figures.

## Coding Rules

Use Python.

Prefer:

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT when required
* Accelerate when useful

Keep:

* training code;
* evaluation code;
* dataset generation;
* plotting;
* experiment configuration

modular rather than placing everything in one notebook.

Configuration should preferably live in YAML/JSON or dataclasses rather than hard-coded throughout scripts.

## Research Integrity

Never fabricate measurements or expected results.

Clearly distinguish:

* measured values;
* hypotheses;
* illustrative examples;
* simulated/debug values.

Do not call an illustrative result an experimental result.

Before making major changes to the experimental methodology, explain how the change could affect scientific validity.

## Research Context

Before implementing experimental work, read:

`docs/RESEARCH_CONTEXT.md`

That file contains the detailed rationale, novelty considerations, experimental logic, terminology, and publication strategy.
