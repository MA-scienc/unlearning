"""Create one strict paired, cluster-bootstrap comparison from result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unlearning.config import load_config
from unlearning.runtime.settings import default_config_path
from unlearning.evaluation.comparison import compare_result_files
from unlearning.utils.io import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two matched evaluation artifacts using record/question-clustered paired "
            "bootstrap inference. The mean delta is comparison minus baseline."
        )
    )
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-key", default="record_id")
    parser.add_argument("--item-key", default="example_id")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    config = load_config(args.config)
    result = compare_result_files(
        args.baseline,
        args.comparison,
        metric=args.metric,
        seed=config.seed,
        cluster_key=args.cluster_key,
        item_key=args.item_key,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_json(Path(args.output), result, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
