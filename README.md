# LLM Specialization/Unlearning Pilot

This repository contains the preflight infrastructure and dataset pipeline for the pilot study described in `docs/RESEARCH_CONTEXT.md` and `docs/PILOT_DESIGN.md`.

Current stage: infrastructure and data preparation only. Do not launch specialization, contamination, or unlearning training until the preflight report is reviewed.

Run the dry-run/preflight planner:

```powershell
$env:PYTHONPATH = "src"
python scripts/pilot/preflight.py --config configs/pilot/base.yaml
```

The command validates configuration, prepares dataset artifacts, computes fingerprints, checks forget-data separation, and computes the tokenized MedMCQA specialization budget. It does not load the full model or start training.
