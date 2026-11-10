"""Unblinding (pre-registration section 10): seal the raw outputs before any analysis.

1. Refuses unless every Stage-2 seed's pipeline has finished every job.
2. Writes the sha256 of every raw result file of every seed to
   ``docs/stage2/unblinding_manifest.json`` (tracked in git; results/ is not), refusing to
   overwrite an existing one.
3. Prints the next steps: commit that file, then run the cross-seed analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from unlearning.experiments.stage2_pipeline import Stage2Pipeline
from unlearning.runtime.run_log import utc_now
from unlearning.runtime.settings import REPO_ROOT, default_runtime_path
from unlearning.utils.code_version import code_version
from unlearning.utils.hashing import hash_file
from unlearning.utils.io import write_json

MANIFEST = REPO_ROOT / "docs" / "stage2" / "unblinding_manifest.json"
SEED_CONFIGS = [REPO_ROOT / "configs" / "stage2" / f"stage2_seed{seed}.yaml" for seed in (43, 44, 45)]


def unfinished_jobs(pipeline: Stage2Pipeline) -> list[str]:
    approvals = pipeline.approvals()
    return [job.id for job in pipeline.jobs() if not job.done(approvals)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=[str(path) for path in SEED_CONFIGS])
    parser.add_argument("--runtime", default=default_runtime_path())
    args = parser.parse_args(argv)
    if MANIFEST.is_file():
        raise SystemExit(f"{MANIFEST} already exists: Stage 2 was unblinded already")
    seeds = []
    for config in args.configs:
        pipeline = Stage2Pipeline(config_path=Path(config), runtime_path=Path(args.runtime))
        pending = unfinished_jobs(pipeline)
        if pending:
            raise SystemExit(f"{pipeline.layout.run_id} is not finished ({len(pending)} jobs, e.g. {pending[0]}); stay blinded")
        files = sorted(path for path in pipeline.layout.raw_results.rglob("*.json") if path.is_file())
        seeds.append({
            "config": str(config), "run_id": pipeline.layout.run_id, "seed": pipeline.config.seed,
            "files": {str(path.relative_to(pipeline.layout.raw_results)): hash_file(path) for path in files},
        })
    write_json(MANIFEST, {"unblinded_at_utc": utc_now(), "code_version": code_version(), "seeds": seeds})
    print(f"wrote {MANIFEST} ({sum(len(s['files']) for s in seeds)} files).")
    print("Next: commit it, then run: python -m unlearning.analysis.stage2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
