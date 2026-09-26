from __future__ import annotations

import argparse
import json

from unlearning.training.smoke import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiny clean-specialization smoke test.")
    parser.add_argument("--output-dir", default="runs/smoke_clean_specialization")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
