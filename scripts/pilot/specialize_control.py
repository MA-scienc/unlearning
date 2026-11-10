from __future__ import annotations

import sys

from unlearning.training.specialize_clean import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--control"]))
