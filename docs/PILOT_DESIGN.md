# Pilot Design: S0/S50/S100 Medical Unlearning

This document fixes the proposed methodological choices for the first pilot before implementation. It does not report experimental results.

The pilot asks whether the degree of prior medical specialization changes unlearning behavior. It must not assume the direction of the effect. S0, S50, and S100 are checkpoints from one continuous domain-adaptation trajectory: save S50 at 50% of the predefined fixed training budget, then continue training the same run to S100 with the same optimizer state and data stream.

S0, S50, and S100 mean 0%, 50%, and 100% of that predefined domain-adaptation training budget only. S100 must never be interpreted as "100% medically specialized." Actual achieved medical specialization must be measured independently.

## Primary Sources Checked

- Qwen2.5-1.5B model card and license: https://huggingface.co/Qwen/Qwen2.5-1.5B
- Qwen2.5 technical report/model paper link from the model card: https://arxiv.org/abs/2407.10671
- MedMCQA repository, splits, and MIT license notice: https://github.com/medmcqa/medmcqa
- MedMCQA paper: https://arxiv.org/abs/2203.14371
- PubMedQA homepage and dataset counts: https://pubmedqa.github.io/
- PubMedQA repository and MIT license: https://github.com/pubmedqa/pubmedqa
- NPO paper: https://arxiv.org/abs/2404.05868
- MMLU-CF repository, validation release, and license notes: https://github.com/microsoft/MMLU-CF
- MMLU-CF paper: https://arxiv.org/abs/2412.15194

## Choice 1: Base Model

Proposed choice: `Qwen/Qwen2.5-1.5B`, base model, not instruction-tuned. Pin the exact Hugging Face revision before any run.

Scientific rationale:

- The model is small enough for a pilot while still being a modern causal LM with nontrivial general and medical-domain behavior.
- The base variant avoids instruction-tuning and refusal-policy effects that could masquerade as unlearning.
- A base model makes likelihood-based forgetting and retention metrics easier to interpret than chat-only behavior.
- The Hugging Face model card lists an Apache-2.0 license, which is suitable for reproducible research use.

Compute implications:

- The model weights are about 3 GB in safetensors form, but full fine-tuning needs substantially more memory for gradients, optimizer states, activations, and checkpoints.
- Primary training mode should be full-parameter bf16 fine-tuning with gradient checkpointing and an 8-bit AdamW optimizer. This keeps the pilot closer to actual weight-space specialization than LoRA while making a 1.5B run plausible on a high-memory consumer GPU or a modest cloud GPU.
- If available hardware cannot support full-parameter training, switching to LoRA or QLoRA is a major methodological change and must be documented before implementation. It would turn the pilot into adapter-specialization/adaptor-unlearning rather than full model unlearning.

Licensing/data concerns:

- The model card reports Apache-2.0, but the exact license file and revision hash must be stored in the experiment manifest.
- Qwen pretraining data is not fully inspectable, so the base model may already contain medical knowledge and possibly some benchmark contamination.

Major confounds:

- Existing medical knowledge in the base model may reduce the apparent magnitude of specialization.
- Base-model prompt following may be weaker than an instruct model, so evaluation should rely heavily on option log-probabilities and target likelihoods rather than free-form compliance.
- Results from this model family may not generalize to Llama, Gemma, Phi, or domain-pretrained medical LMs.

## Choice 2: Medical Specialization Corpus

Proposed choice: MedMCQA training split only, formatted as causal language modeling text containing question, options, correct answer, subject, topic, and explanation. Do not train on MedMCQA validation or test examples.

Scientific rationale:

- MedMCQA is broad enough for measurable medical specialization: the paper describes more than 194k MCQs across 2.4k healthcare topics and 21 subjects.
- The train/dev/test split is exam-based and similar questions were removed by the dataset authors, which is better for held-out evaluation than a random question split.
- Exam questions and explanations emphasize medical concepts rather than real patient records, aligning with the requirement that no real patient information be used.
- The corpus contains real diseases, drug classes, procedures, and physiology, which makes near-domain collateral damage meaningful.

Compute implications:

- The corpus is compact enough for repeated passes in a pilot.
- Formatting with explanations will produce a larger and richer token stream than question-only training, but still far smaller than PubMed or PMC-scale adaptation.
- Because the corpus is smaller than a full biomedical text corpus, repeated exposure is likely. The design must record epoch count and token count so over-specialization can be interpreted.

Licensing/data concerns:

- The official GitHub repository displays an MIT license notice. The implementation should archive the source URL, commit if available, and dataset fingerprint.
- The examples are derived from medical entrance exam and mock exam material. Even with an MIT repository, downstream redistribution of processed data should be limited to manifests and scripts unless license review confirms redistributability of derived text.
- Do not include any real clinical notes or real patient records.

Major confounds:

- MedMCQA adaptation may specialize the model toward exam-style question answering rather than biomedical prose or clinical reasoning generally.
- If near-domain evaluation also uses MedMCQA, held-out split separation is necessary but does not eliminate all style leakage.
- MCQ explanations can teach answer-option patterns that inflate multiple-choice retention metrics without representing robust domain knowledge.

Rejected alternative for the pilot:

- PMC Open Access or PubMed-scale corpora are more text-rich, but article license terms vary, bulk retrieval has compliance requirements, and case reports may include de-identified patient narratives. They are better candidates for the full study after a data-governance pass.

## Choice 3: Specialization Budget and Checkpoints

Proposed choice:

- Build one deterministic packed token stream from formatted MedMCQA train.
- Context length: 1024 tokens.
- Effective global batch target: 32 sequences, or 32,768 tokens per optimizer step.
- S100 budget: three full passes over the packed MedMCQA train stream.
- S50 checkpoint: save at exactly 50% of the computed total optimizer steps.
- S0 checkpoint: the pinned base model materialized through the same checkpoint interface with zero domain-adaptation steps.
- S100 checkpoint: continue the same training run from S50 to completion without reinitializing model, optimizer, scheduler, RNG stream, or data order.

Scientific rationale:

- A continuous trajectory preserves the interpretation that S50 and S100 differ by cumulative domain-training exposure, not by independent training noise.
- Using a fixed budget in steps/tokens matches the project definition of specialization percentage.
- Three passes should create measurable domain movement without turning the pilot into a long pretraining project.

Compute implications:

- Exact optimizer step count will be computed after tokenization and packing, then frozen in the run manifest before training starts.
- The run requires storing S0, S50, and S100 clean checkpoints, optimizer-independent metadata, and evaluation outputs. Optimizer states need not be retained after clean checkpoint creation unless resumption/debugging is required.
- If the tokenized corpus is much smaller or larger than expected, do not silently change the percentage definition. Freeze a token/step budget in config before launching.

Licensing/data concerns:

- Only MedMCQA train-derived text is used for specialization.
- Validation/test splits and synthetic forget records are excluded from the specialization stream.

Major confounds:

- Repeated passes may cause memorization of MedMCQA style and examples.
- S50 may not show intermediate medical capability; actual specialization must be measured independently.
- If S100 overfits or degrades general utility, unlearning effects may be confounded with general fine-tuning damage.
- S100 is a training-budget endpoint, not a claim of complete medical specialization.

## Choice 4: Synthetic Forget Dataset F

Proposed choice: generate 120 fictional patient records centered on cardiometabolic and thrombotic medicine. Use fake names, fake patient IDs, fake clinic IDs, fake accession numbers, and fake dates. Use real medical concepts such as atrial fibrillation, heart failure with reduced ejection fraction, venous thromboembolism, hypertension, type 2 diabetes, chronic kidney disease, anticoagulants, beta blockers, ACE inhibitors/ARBs, statins, SGLT2 inhibitors, GLP-1 receptor agonists, and metformin.

Schema:

- `record_id`: synthetic stable ID such as `PX-000137`.
- `patient_name`: generated fictional first/last name from a project-local synthetic list.
- `age_band`: decade or exact synthetic age.
- `synthetic_location`: fictional clinic/hospital name, not a real institution.
- `condition`: real disease/concept label.
- `drug_class`: real drug class.
- `drug_example`: real medication name where scientifically useful.
- `lab_or_marker`: real clinical measurement type with synthetic value.
- `care_plan`: synthetic note using real clinical concepts.
- `private_code`: random pronounceable string or alphanumeric key unique to the record.
- `canary_sentence`: controlled sentence binding the patient identity, condition, treatment, and private code.

Training/evaluation views:

- For each record, create 8 contamination templates: full note, structured summary, patient lookup, medication lookup, condition lookup, private-code lookup, short completion, and QA form.
- For each record, create held-out paraphrase/evaluation templates that are never used in contamination training.
- Keep concept labels real but patient identities and patient-specific bindings synthetic.

Scientific rationale:

- Real diseases and drug classes make near-domain collateral damage measurable: the model can forget "PX-000137 takes apixaban for atrial fibrillation" without erasing general knowledge about atrial fibrillation or anticoagulants.
- The private code gives a high-specificity target for exact leakage and likelihood tests.
- Multiple templates reduce dependence on one memorized surface form while still keeping F controlled.

Compute implications:

- 120 records x 8 contamination templates gives 960 short examples, small enough to contaminate each S checkpoint quickly.
- The dataset is large enough for record-level confidence intervals and stratification by concept cluster.
- Held-out paraphrase probes add evaluation cost but not training cost.

Licensing/data concerns:

- No real patient information may be used.
- Names, locations, dates, IDs, and private codes must be generated locally and checked against a blocklist of real hospital names and common public figures.
- Medical concepts are real but are not private data.

Major confounds:

- The synthetic style may be easier to memorize and unlearn than real sensitive records.
- Using one medical subdomain may make near-domain retention more interpretable but less general.
- Real medication names may already be known by the base model, so metrics must focus on patient-specific associations rather than concept existence.

## Choice 5: Fixed-Exposure Contamination

Proposed choice:

- For each clean checkpoint S0, S50, and S100, fine-tune on exactly the same F contamination examples.
- No interleaving with domain data.
- No matched-memorization adjustment.
- Training mode: same full-parameter mode as specialization.
- Optimizer: AdamW 8-bit if used for specialization.
- Learning rate: `1e-5`.
- Epochs: 10 over the 960 contamination examples.
- Effective batch size: 16 examples.
- Sequence length: 512 tokens.
- Scheduler: linear warmup over 3% of contamination steps, then cosine or linear decay as fixed in config.
- Save C0, C50, and C100.

Scientific rationale:

- Fixed exposure directly tests whether the prior specialization state changes learning and unlearning of the same F.
- Avoiding interleaved retain/domain data keeps the contamination manipulation simple and identical across specialization levels.
- Measuring pre-unlearning memorization after contamination will reveal whether fixed exposure creates unequal memorization across S0/S50/S100.
- Ten epochs is a controlled pilot setting for this experiment. It is not a claim about realistic accidental exposure.

Compute implications:

- The contamination phase is small relative to specialization.
- Checkpoints C0/C50/C100 add storage cost but are essential for counterfactual comparisons and reproducibility.

Licensing/data concerns:

- F is fully synthetic and can be versioned in generated-data manifests.
- Do not publish any accidental real-looking record without blocklist checks and explicit synthetic provenance.

Major confounds:

- If contamination memorization differs strongly across checkpoints, fixed-exposure results conflate specialization with achieved memorization strength. This is acceptable for the first pilot but must be reported.
- Ten epochs may overfit synthetic records. If overfitting is too high or too low, the next run must be a new pre-registered contamination setting, not an undocumented adjustment.
- No interleaving may cause local utility drift, but the drift is part of the fixed-exposure condition and is held constant in procedure.
- Do not automatically adjust contamination strength separately for S0, S50, and S100 in the pilot. Pre-unlearning memorization/leakage of F must be measured for C0, C50, and C100 so ceiling and floor effects are visible.

## Choice 6: Unlearning Methods

Run one unlearning trajectory per `(C_i, method)` and save fixed snapshots for evaluation at steps 0, 50, 100, 200, and 400. Step 0 is the contaminated checkpoint before unlearning. The full fixed-step trajectory is the primary analysis for GA and NPO. Do not select a headline checkpoint after observing results.

If a single operating point is later needed for a table or comparison, reserve support for a pre-specified, method-neutral rule such as the first checkpoint reaching a fixed target-forgetting threshold. The rule must be chosen before inspecting experimental outcomes.

### Gradient Ascent

Definition:

```text
L_GA(theta) = - CE_theta(y | x), for (x, y) in F_forget
```

Minimizing `L_GA` performs gradient ascent on the forget-example cross-entropy.

Proposed settings:

- Forget examples: the same prompt/completion pairs used for contamination.
- Learning rate: `1e-6`.
- Optimizer: same optimizer family as contamination.
- Batch size: 16 examples.
- Sequence length: 512 tokens.
- Steps: 400, with evaluation snapshots at 50, 100, 200, 400.
- Gradient clipping: 1.0.
- Retain loss: none in the primary pilot.

Scientific rationale:

- GA is a simple and common unlearning baseline, useful as a stress test for forgetting versus utility damage.
- Saving snapshots yields an empirical forgetting-retention curve rather than a single arbitrary operating point.

Compute implications:

- Six unlearning runs total for GA and NPO across C0/C50/C100, each short.
- Evaluation may dominate compute because every snapshot must be tested.

Licensing/data concerns:

- GA uses only synthetic F, so no additional data license is introduced.

Major confounds:

- GA can cause catastrophic utility degradation or nonsensical outputs.
- Results are sensitive to learning rate and step count.
- Without retain loss, GA may look worse than practical GA variants; this is intentional for a clean first baseline.

### Negative Preference Optimization

Use the original NPO framing, not ordinary DPO-style chosen/rejected pairs. Each forget example `(x, y)` is treated as a negative response only. There is no positive/chosen response.

Definition:

```text
L_NPO(theta) =
  -(2 / beta) * E_{(x,y) in F_forget}
  [ log sigmoid( -beta * log( pi_theta(y|x) / pi_ref(y|x) ) ) ]
```

Proposed settings:

- Forget examples: the same prompt/completion pairs used for contamination.
- Reference model `pi_ref`: the frozen contaminated model `C_i` from which unlearning starts.
- Beta: `0.1`.
- Learning rate: `1e-6`.
- Batch size: 16 examples.
- Sequence length: 512 tokens.
- Steps: 400, with evaluation snapshots at 50, 100, 200, 400.
- Retain loss: none in the primary pilot.

Scientific rationale:

- NPO was proposed to reduce the catastrophic-collapse tendency seen in GA while unlearning undesirable data.
- Using `C_i` as the frozen reference follows the preference-optimization convention of referencing the starting policy and avoids giving the method privileged access to the clean counterfactual S_i.
- Omitting chosen/rejected pairs prevents accidentally turning NPO into DPO with refusal templates.

Compute implications:

- NPO requires forward probabilities from both the current model and frozen reference model. This roughly doubles forward-pass memory or compute unless reference logits are cached.
- Because F is small and fixed, caching reference log-probabilities for forget completions is preferred after implementation.

Licensing/data concerns:

- NPO uses only synthetic F and frozen model outputs/probabilities.

Major confounds:

- Beta is a substantive hyperparameter. `0.1` is a reasonable pilot setting but not a universal optimum.
- Reference choice matters. Using S_i as reference would change the scientific meaning by injecting clean-counterfactual information into the unlearning objective.
- Without retain loss, NPO is cleaner as a baseline but may underrepresent retention-aware variants.

## Choice 7: Retain-Loss Policy

Proposed choice: set retain loss weight to `0.0` for both GA and NPO in the primary pilot. The configuration schema may reserve fields for future retain losses, but this pilot should not implement or run matched memorization, interleaved contamination, or retain-regularized unlearning.

Scientific rationale:

- Adding retain loss introduces another data source and another intervention that can obscure whether specialization itself changes unlearning behavior.
- Evaluation already measures near-domain, far-domain, and general retention against clean counterfactuals.

Compute implications:

- Forget-only methods are cheaper and easier to inspect.
- If collapse occurs, the correct response is to report the pilot failure mode or run a separately documented second pilot, not to change the objective midstream.

Licensing/data concerns:

- No extra retain-training data is introduced into the unlearning objective.

Major confounds:

- Practical unlearning systems often use retain data. This pilot isolates a clean baseline rather than optimizing retention.
- Absence of retain loss may exaggerate collateral damage, especially for GA.

## Choice 8: Evaluation Datasets and Metrics

Evaluate all of these model states:

- Clean: S0, S50, S100.
- Contaminated: C0, C50, C100.
- Unlearned snapshots: GA and NPO snapshots from each C_i at steps 0, 50, 100, 200, and 400.

Primary comparisons:

- Target forgetting: compare U_i to C_i and S_i.
- Retention: compare U_i to S_i as the clean counterfactual.
- Specialization: compare S50 and S100 to S0 on medical-domain evaluations.

### Target Forgetting

Datasets:

- F direct extraction probes.
- F paraphrased probes held out from contamination.
- F completion probes.
- F multiple-choice association probes with decoy patient fields.

Metrics:

- Exact leakage rate for synthetic private codes, IDs, drug examples, and condition bindings.
- Normalized field F1 for patient-specific associations.
- Target log-likelihood of protected completions.
- Protected-vs-decoy log-likelihood margin.
- Refusal/invalid answer rate reported separately, not counted alone as successful forgetting.

Rationale:

- Exact output metrics catch direct leakage.
- Likelihood metrics catch residual memorization even when generation does not reveal the record.
- Decoy comparisons test whether the model remembers the association, not merely common medical concepts.

Major confounds:

- A model can avoid answering without truly lowering likelihood of the sensitive continuation.
- Paraphrase probes may vary in difficulty across records.

### Near-Domain Retention

Dataset:

- MedMCQA validation examples filtered to cardiology/cardiovascular medicine, hematology/thrombosis where available, endocrinology/diabetes, nephrology/CKD, and pharmacology topics related to anticoagulants, cardiovascular drugs, diabetes drugs, and renal dosing.

Metrics:

- Multiple-choice accuracy using option log-probability scoring.
- Average negative log-likelihood on correct option/explanation text where format permits.
- Per-concept subgroup accuracy.

Rationale:

- These concepts are semantically close to F but do not contain F patient identities.
- Retention here tests whether forgetting patient-specific records damages legitimate nearby medical knowledge.

Major confounds:

- Topic filtering by string/metadata can be noisy.
- If domain adaptation used MedMCQA train, MedMCQA validation shares style even when examples are held out.

### Far-Domain Retention

Dataset:

- MedMCQA validation examples from subjects/topics distant from F, initially Skin, Psychiatry, Ophthalmology, ENT, Anatomy, and non-cardiometabolic Microbiology.

Metrics:

- Multiple-choice accuracy using the same option log-probability scoring.
- NLL of correct options.

Rationale:

- Far-domain retention measures broader medical collateral damage while staying inside medicine.

Major confounds:

- Some "far" subjects still contain overlapping pharmacology or systemic-disease concepts.
- If S100 improves all medical domains, absolute U_i performance may remain high despite unlearning damage; report delta to S_i.

### Domain Specialization Measurement

Datasets:

- MedMCQA validation, all subjects and filtered subgroups.
- PubMedQA expert-labeled set as a secondary biomedical QA check, using yes/no/maybe accuracy and macro-F1.

Metrics:

- Accuracy.
- Macro-F1 for PubMedQA.
- Correct-answer log-probability or NLL where applicable.

Rationale:

- MedMCQA measures in-distribution specialization from held-out exam-style data.
- PubMedQA provides a different biomedical QA format not used for specialization training.

Major confounds:

- PubMedQA uses abstracts and may depend on retrieval/context formatting, so it may understate gains from MedMCQA-style specialization.
- MedMCQA gains may partly reflect exam-style adaptation.

### General Retention

Dataset:

- MMLU-CF validation subset, stratified to 1,000 examples for the pilot, 0-shot option log-probability scoring.

Metrics:

- Accuracy.
- Per-domain accuracy buckets.
- Delta from S_i for each U_i.

Rationale:

- MMLU-CF was designed to reduce contamination risk relative to older MMLU-style benchmarks.
- A stratified subset keeps the pilot affordable while still measuring broad non-medical capability.

Licensing/data concerns:

- The MMLU-CF repository is MIT licensed, while its validation dataset is noted as CDLA-2.0. Store license metadata and avoid redistributing derived data unless allowed.

Major confounds:

- A 1,000-example subset has wider confidence intervals than the full validation set.
- Small base models may have low absolute accuracy; use deltas, not only raw scores.

## Choice 9: Pilot Success Criteria

The pilot succeeds if it produces interpretable measurements, not if it supports a preferred hypothesis.

Minimum pipeline success:

- S0, S50, and S100 exist as one continuous specialization trajectory.
- C0, C50, and C100 are produced with identical fixed-exposure contamination.
- GA and NPO snapshots are produced for all contaminated checkpoints at fixed steps 0, 50, 100, 200, and 400.
- Every run has deterministic config, dataset fingerprints, model revision, seed, training hyperparameters, software versions, hardware metadata, precision, runtime, and output paths.
- Raw results are stored separately from processed summaries.

Minimum scientific signal checks:

- Specialization check: S100 should show measurable medical-domain movement versus S0, such as higher MedMCQA validation accuracy or lower medical NLL. If not, the unlearning pipeline may still work, but the pilot cannot answer the specialization question.
- Contamination check: C_i should show substantially higher F leakage than S_i on direct probes and higher protected-completion likelihood than decoys. If fixed exposure does not create measurable contamination, the pilot should be rerun with a pre-registered stronger contamination recipe.
- Unlearning check: at least one method should reduce target leakage relative to C_i at some snapshot. If neither GA nor NPO changes F metrics, method hyperparameters are probably too weak.
- Retention check: near-domain, far-domain, and general metrics must be reported even if forgetting fails.

Specialization-effect criterion:

- Estimate the relationship between specialization level and each forgetting/retention metric using S0/S50/S100 deltas and bootstrap confidence intervals over records/questions.
- A flat or null relationship is a valid pilot outcome.
- Do not claim increased specialization makes unlearning harder unless the measured direction and uncertainty support it.

## Decisions That Must Be Frozen Before Implementation

1. Exact model revision hash for `Qwen/Qwen2.5-1.5B`.
2. Hardware-backed confirmation that full-parameter training is feasible. If not, document a method change before coding.
3. Final MedMCQA source path, revision/fingerprint, and license metadata.
4. Exact tokenized specialization step count after preprocessing, before launching training.
5. Final topic filters for near-domain and far-domain MedMCQA subsets.
6. Exact synthetic-name/blocklist procedure for F.
7. Whether generated synthetic F may be committed as small JSONL files, or whether only generation configs/manifests are committed.
8. Exact evaluation subset seed for MMLU-CF validation.
9. Optional single-operating-point rule, if needed later, must be pre-specified before observing results and must be method-neutral.

## Explicit Non-Goals For This Pilot

- Do not implement matched memorization.
- Do not implement interleaved contamination.
- Do not add RMU, GRU, ReGLU, LoKU, or other methods.
- Do not add a second domain.
- Do not run recovery-resistance/relearning tests except for paraphrase and likelihood probes already included in target forgetting.
- Do not tune procedures to make specialization appear helpful or harmful.
