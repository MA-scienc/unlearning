"""Evaluate the pilot's models on the enlarged 434-item near set (pre-registration 6.2).

Stage 2 uses the pilot's near keyword rule on every MedMCQA dev subject (434 items instead
of 164). So that pilot and Stage 2 can be compared on the same questions, this builds that
set from the pilot's own frozen dev file and evaluates every pilot S, matched C and
revision-3 U model on it. It writes new files only (near_domain_dev_all_subjects.jsonl and
results/raw/<pilot>/<model>/near_all.json), never replacing the pilot's near results, and
skips models already done, so it can be stopped and rerun.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from unlearning.config import load_config
from unlearning.data.medmcqa import is_near_domain
from unlearning.experiments.pipeline import PilotPipeline
from unlearning.runtime.settings import REPO_ROOT, default_config_path, default_runtime_path
from unlearning.utils.io import read_jsonl, write_jsonl

NEAR_ALL = Path("medmcqa_domain_subsets") / "near_domain_dev_all_subjects.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--runtime", default=default_runtime_path())
    args = parser.parse_args(argv)
    config = load_config(args.config)
    pipeline = PilotPipeline(config_path=Path(args.config), runtime_path=Path(args.runtime))
    generated = pipeline.layout.generated
    target = generated / NEAR_ALL
    if not target.is_file():
        dev = read_jsonl(generated / "medmcqa" / "dev_eval.jsonl")
        near = [record for record in dev if is_near_domain(record, ("*",), config.evaluation.near_topic_keywords)]
        write_jsonl(target, near)
        print(f"{len(near)} near items -> {target}", flush=True)
    revision = next(r for r in pipeline.unlearning_revisions if r is not None and r.is_matched)
    models = pipeline.clean_models() + pipeline.matched_contaminated_models() + pipeline.unlearned_models(revision)
    for ref in models:
        output = pipeline.result_path(ref.label, "near_all")
        if output.is_file() or ref.not_produced() or not Path(ref.manifest).is_file():
            continue
        command = [sys.executable, str(REPO_ROOT / "scripts" / "pilot" / "evaluate_clean.py"), "--config", args.config,
                   "--model", ref.model, "--checkpoint-manifest", str(ref.manifest), "--dataset", str(target),
                   "--task", "medmcqa", "--output", str(output)]
        print(f"evaluating {ref.label}", flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
