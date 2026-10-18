# LLM Specialization/Unlearning Pilot

This repository contains the protocol, integrity-checked data pipeline, and
training/evaluation implementation for the pilot study described in
`docs/RESEARCH_CONTEXT.md` and `docs/PILOT_DESIGN.md`. It does not yet contain
empirical findings.

Current stage: the pilot pipeline is implemented but has not been run on a compatible
CUDA system. Do not launch specialization, contamination, or unlearning training until
the protocol amendment and a fresh v2 preflight report are reviewed. Historical reports
describe the superseded synthetic-forget-data schema v1; see
[publication readiness](docs/PUBLICATION_READINESS.md).

Run the dry-run/preflight planner:

```powershell
$env:PYTHONPATH = "src"
python scripts/pilot/preflight.py --config configs/pilot/base.yaml
```

The command validates configuration, prepares dataset artifacts, computes fingerprints, checks forget-data separation, and computes the tokenized MedMCQA specialization budget. It does not load the full model or start training.

The full integrity-checked execution order, including S_i -> C_i -> U_i and
evaluation commands, is in [scripts/pilot/README.md](scripts/pilot/README.md).
Publication scope, claim limits, and the controls required before a flagship
submission are in [docs/PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md).
The pre-run confirmatory design is in
[docs/CONFIRMATORY_PROTOCOL.md](docs/CONFIRMATORY_PROTOCOL.md).
