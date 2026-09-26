# Pretraining Preflight Report

Date: 2026-09-26

Status: preflight hardening only. No domain-specialization, contamination, GA, or NPO training was implemented or launched. No full Qwen model weights were loaded.

## Fixes Made

- Replaced synthetic patient-name generation with a seeded deterministic shuffle over a fictional first/last name pool.
- Made private code generation deterministic and index-unique.
- Added MC association decoys that are type-compatible, unique, deterministic, and exclude the target answer.
- Added `forget.mc_decoy_count`.
- Pinned Hugging Face revisions in config for Qwen2.5-1.5B, MedMCQA, PubMedQA, and MMLU-CF.
- Threaded dataset/model revisions through dataset/tokenizer loading and manifests.
- Added stricter config validation for snapshot steps, template counts, batch/sequence sizes, checkpoint fractions, and training-readiness pinned revisions.
- Frozen S100 is now forced to an even optimizer-step count by reducing an odd provisional budget by one step, so S50 is exact.
- Added near/far MedMCQA review samples to the preflight report.
- Expanded hardware inspection to include CPU, RAM, GPU/VRAM via `nvidia-smi`, PyTorch CUDA, bf16 support, and bitsandbytes import status.
- Added `tests/__init__.py` so `pytest` collection works.

## Repository Tree

```text
AGENTS.md
README.md
PRETRAINING_PREFLIGHT_REPORT.md
configs/pilot/base.yaml
data/README.md
data/manifests/.gitkeep
docs/PILOT_DESIGN.md
docs/RESEARCH_CONTEXT.md
pyproject.toml
results/README.md
scripts/pilot/README.md
scripts/pilot/preflight.py
src/unlearning/__init__.py
src/unlearning/config/__init__.py
src/unlearning/config/schema.py
src/unlearning/data/__init__.py
src/unlearning/data/evaluation_sets.py
src/unlearning/data/medmcqa.py
src/unlearning/data/packing.py
src/unlearning/data/safeguards.py
src/unlearning/data/synthetic_forget.py
src/unlearning/experiments/__init__.py
src/unlearning/experiments/naming.py
src/unlearning/experiments/preflight.py
src/unlearning/utils/__init__.py
src/unlearning/utils/env.py
src/unlearning/utils/hashing.py
src/unlearning/utils/io.py
src/unlearning/utils/manifest.py
tests/__init__.py
tests/test_config.py
tests/test_naming.py
tests/test_packing.py
tests/test_safeguards.py
tests/test_synthetic_forget.py
```

Generated artifacts are under:

```text
data/generated/pilot-medical-s0-s50-s100__qwen-qwen2-5-1-5b__seed42__synthetic-patient-f-v1/
```

## Pinned Revisions

| Artifact | Repo | Revision |
|---|---|---|
| Base model/tokenizer | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |
| MedMCQA | `openlifescienceai/medmcqa` | `91c6572c454088bf71b679ad90aa8dffcd0d5868` |
| PubMedQA | `qiaojin/PubMedQA` | `9001f2853fb87cab8d220904e0de81ac6973b318` |
| MMLU-CF | `microsoft/MMLU-CF` | `c25b89a968a2062e422dd96ddf0fe3387507f0dd` |

These were resolved with `huggingface_hub` from the actual remote repositories.

## Test And Lint Results

```text
pytest -q
19 passed in 0.16s
```

```text
ruff check .
All checks passed!
```

Preflight stages:

```text
python scripts/pilot/preflight.py --config configs/pilot/base.yaml --skip-external --overwrite
passed
```

```text
python scripts/pilot/preflight.py --config configs/pilot/base.yaml --overwrite
passed
```

The `--overwrite` flag was used because generated preflight artifacts already existed from prior runs.

## Synthetic F

Schema: `synthetic-patient-f-v1`

| Item | Count |
|---|---:|
| synthetic patient records | 120 |
| contamination examples | 960 |
| held-out forget-evaluation examples | 720 |
| contamination templates per record | 8 |
| evaluation templates per record | 6 |
| MC decoys per MC probe | 3 |

Fingerprints:

| Artifact | SHA256 |
|---|---|
| records | `d5c61b14b503921aacf1628dd2f3926c22d573b5c246f733f870cf42b8fa48f4` |
| contamination examples | `9b66b56f067e2745f60be953b07619d115c8bc7bf2a4dd6be11dfb076eb541f4` |
| eval examples | `f3f12344895a20d74742c5909119a71e82fcfbeb35773c60a0e23404f4be1805` |

Safeguards verified:

- exactly 120 records;
- unique `record_id`;
- unique patient names;
- unique private codes;
- contamination/eval template IDs are disjoint;
- MC decoys exclude the target;
- MC decoys contain no duplicates;
- MC decoys are condition decoys for condition questions;
- generation is deterministic for the same seed.

## MedMCQA

Configured split mapping:

- specialization train: HF `train`
- labeled dev/eval: HF `test`
- unlabeled held-out test: HF `validation`

Counts and fingerprints:

| Split | Count | Fingerprint |
|---|---:|---|
| train | 182,822 | `7ae00e32312582cc3ecf9a6ea693816f64858149629470769cde52b2c75874c7` |
| dev | 6,150 | `71f838a9a7289a203530624ec71db9379f1dd97a6b63d0a6927e09cda6a269c9` |
| test | 4,183 | `ab8ff61d5b9c2e062be02dcd9b2376bdff61c1ba4957451badb00cc413cb7ae2` |

Split ID separation passed.

## Specialization Token Budget

Tokenizer: `Qwen/Qwen2.5-1.5B` at revision `8faed761d45a263340a0528343f099c05c9a4323`

| Item | Value |
|---|---:|
| MedMCQA train texts | 182,822 |
| context length | 1,024 |
| raw token count with EOS delimiters | 35,946,600 |
| packed sequence count | 35,104 |
| packed one-pass token count | 35,946,496 |
| dropped tail tokens | 104 |
| effective batch sequences | 32 |
| provisional S100 steps | 3,291 |
| frozen S100 steps | 3,290 |
| S50 steps | 1,645 |
| step adjustment | 1 |
| achieved S50 fraction | 0.5 |
| frozen training sequences | 105,280 |
| frozen training tokens | 107,806,720 |
| effective training passes | 2.999088422971741 |

Rationale: the provisional approximately-three-pass S100 budget was odd, so S100 was reduced by one optimizer step to make S50 exactly half the frozen budget. This adjustment is recorded in the preflight report and does not silently alter the protocol.

## Near/Far MedMCQA Subsets

Near-domain:

- count: 63
- fingerprint: `0e25ba6a1af9279ebd5d8bad432bad48eebf9fadc25fa604cad7dcf478041f3c`
- subject counts: Medicine 47, Pharmacology 16
- topic counts: UNKNOWN 63

Near sample:

| ID | Subject | Topic | Question |
|---|---|---|---|
| `e220f29e-e63f-4bbe-8305-f0fd4dc002f1` | Pharmacology |  | Which ACE inhibitor in safe in renal failure ? |
| `e34e7aa4-92a0-449d-b42f-7c7ac6295e0b` | Pharmacology |  | True about heparin induced thrombocytopenia ? |
| `7cabe3e1-c7f1-452a-bc0b-2736b80cd3e2` | Medicine |  | Sinus bradycardia with MI treatment |
| `e6469d79-0e5a-4891-b5c9-4fcfe67b037c` | Medicine |  | Infarcts involving which poion of the myocardium cause aneurysm as a post-MI complication- |
| `1795540f-734b-4ebe-90d6-54b6a5c00fe9` | Medicine |  | Hemodynamically impoant lesions of renal aery stenosis are predicted by renal aery velocities more than on Doppler ultrasound. |
| `c3291444-973f-46ba-b0ad-261ef2a5f090` | Pharmacology |  | True about cardiac muscle fibers ? |
| `112ae05b-0512-4d2e-95ec-c89ddf0341ae` | Medicine |  | Which of the following is the complication of diabetes mellitus: |
| `73fd4997-16af-4b49-928c-622d71027190` | Medicine |  | Which of the following is not a feature of distal renal tubular acidosis |
| `cac7cf24-e413-40d7-97ec-7e088747dfaf` | Pharmacology |  | Which of the following antidiabetic drug is insulin secretogogue ? |
| `75c3585f-b15e-4433-917f-6a67c011d2c9` | Medicine |  | Which one of the following is not an early complication of acute myocardial infarction ? |

Far-domain:

- count: 739
- fingerprint: `65236b16da856f4e9ba366b407261690512858a3b35d7913cc7d4357fc5adc00`
- subject counts: Anatomy 248, ENT 86, Microbiology 165, Ophthalmology 175, Psychiatry 6, Skin 59
- topic counts: UNKNOWN 739

Far sample:

| ID | Subject | Topic | Question |
|---|---|---|---|
| `0a7cddf8-a8b8-4778-aa58-4b01c3da1c12` | Anatomy |  | All are derived from ectoderm except ? |
| `08e5cca4-ae05-4e50-ab95-744415c79b86` | Anatomy |  | Straight sinus is formed by? |
| `b51d9728-6a66-4099-9c92-fc7fd9dbb7a4` | Ophthalmology |  | Corneal ulcer resembling fungal ulcer is seen in infection with which of the agents? |
| `666159d8-4b0b-4db3-94e9-2271636086ca` | Anatomy |  | Multi-unit smooth muscle present at all except ? |
| `c2241db8-5415-431a-8f00-cd5ab67b238b` | Ophthalmology |  | Cause of bilateral optic atrophy ? |
| `f99b0b12-bfcb-443b-a265-5d600d99e3bb` | Anatomy |  | Maximum oral structures are having their origin from |
| `49fbbc78-abc9-475b-b18a-9e2185bab874` | Microbiology |  | Loeffer's serum is an example of |
| `486b1620-8719-43b1-b7f8-8a5b644212ed` | Anatomy |  | Respiratory bronchioles are formed from ? |
| `61858b4d-71d2-42dc-b90c-c5d2d79fc80f` | Microbiology |  | Fresh water swimming leads to infection by - |
| `bae60633-5f35-4c83-9700-bc389885a7a6` | Anatomy |  | Which tendon is lodged in the groove on posterior surface of lateral malleolus? |

Issue: MedMCQA `topic_name` is blank/UNKNOWN in these processed dev examples. Near/far filtering therefore relies on subject plus keyword hits in question/explanation text. Do not change definitions automatically; this needs human review.

## Evaluation Datasets

PubMedQA:

- dataset: `qiaojin/PubMedQA`
- revision: `9001f2853fb87cab8d220904e0de81ac6973b318`
- subset: `pqa_labeled`
- split: `train`
- count: 1,000
- fingerprint: `9851c87ab8fe8e2904af973e7aa1291fe67f27624831e9f7e68eb2ac87d9fffa`

MMLU-CF:

- dataset: `microsoft/MMLU-CF`
- revision: `c25b89a968a2062e422dd96ddf0fe3387507f0dd`
- split: `val`
- available count: 10,000
- sampled count: 1,000
- fingerprint: `b5db7eddc659f3042dd4b16f61dfe088b60e96e90c8af16f4c6fad449d47aaea`
- category distribution: Biology 72, Business 72, Chemistry 72, Computer Science 72, Economics 72, Engineering 72, Health 71, History 71, Law 71, Math 71, Other 71, Philosophy 71, Physics 71, Psychology 71

Evaluation data is prepared by separate functions and artifacts. The specialization stream is built only from MedMCQA train. MedMCQA split IDs are checked for train/dev/test separation, and synthetic F identifiers are checked against specialization texts.

## Hardware Inspection

Detected local environment:

| Item | Value |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.13.13 |
| CPU | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| RAM | 15.71 GiB |
| GPU | not detected via `nvidia-smi` |
| GPU VRAM | unavailable |
| CUDA version | unavailable locally |
| PyTorch import | unavailable: `No module named 'torch'` |
| PyTorch CUDA availability | unavailable |
| bf16 support | unavailable |
| bitsandbytes import | unavailable: `No module named 'bitsandbytes'` |

Conclusion: this local machine is not suitable for full-parameter bf16 Qwen2.5-1.5B specialization with gradient checkpointing and 8-bit AdamW.

## Estimated Resource Requirements

Approximate checkpoint storage:

- Qwen2.5-1.5B bf16 model weights: about 3-4 GiB per full checkpoint.
- Clean checkpoints S0/S50/S100 plus contaminated C0/C50/C100: about 20-25 GiB for model weights alone.
- GA/NPO fixed snapshots for 3 contaminated checkpoints, 2 methods, 5 snapshot steps: about 90-120 GiB for model weights alone.
- Optimizer-state checkpoints can multiply storage substantially and should not be kept for every evaluation snapshot unless needed.

Estimated GPU memory:

- Minimum practical target: CUDA GPU with at least 24 GiB VRAM.
- More comfortable target: 40-48 GiB VRAM, especially if using larger microbatches or retaining optimizer states.
- CPU-only training is a blocker for this protocol.

Do not switch to LoRA/QLoRA automatically. That would change the scientific interpretation from full-parameter specialization/unlearning to adapter-mediated specialization/unlearning.

## Remaining Blockers

1. Training environment blocker: local machine lacks CUDA GPU, PyTorch, bf16 verification, and bitsandbytes.
2. Python environment mismatch: `pyproject.toml` targets Python `>=3.11,<3.13`, while local Python is `3.13.13`.
3. Near/far subset quality needs human review because MedMCQA topic metadata is blank/UNKNOWN in the processed dev split.
4. Disk budget should be confirmed before full checkpoint snapshotting.
5. A training command must call `validate_training_ready(config)` and refuse unpinned model/MedMCQA revisions.

## Stop Condition

Preflight hardening is complete. Stop here until this report is reviewed. Do not launch GPU training, specialization, contamination, GA, or NPO.
