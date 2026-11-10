"""Non-F prompts for the KL control of H3 (pre-registration 6.3).

KL(S_i || model) at answer start is measured on the held-out F prompts; the same measure
on prompts unrelated to F shows how much of a model's distance from S_i is general drift.
Prompts: the question stems of every near-domain item and of the first 360 general
(MMLU-CF) items in subset order, as "Question: <stem>\\nAnswer:". Built from the run's own
frozen evaluation files; each prompt is its own cluster for the bootstrap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from unlearning.config import load_config
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.settings import default_config_path
from unlearning.utils.io import read_jsonl, write_jsonl

GENERAL_COUNT = 360
NEAR_FILE = Path("medmcqa_domain_subsets") / "near_domain_dev.jsonl"
GENERAL_FILE = Path("mmlu_cf") / "mmlu_cf_eval_subset.jsonl"
OUTPUT = Path("closeness") / "non_f_prompts.jsonl"


def build_nonf_prompts(near: list[dict], general: list[dict]) -> list[dict]:
    rows = []
    for template, items in (("near_stem", near), ("general_stem", general[:GENERAL_COUNT])):
        for item in items:
            identifier = f"{template}::{item['id']}"
            rows.append({
                "example_id": identifier,
                "record_id": identifier,
                "template_id": template,
                "prompt": f"Question: {item['question'].strip()}\nAnswer:",
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the non-F prompt set for the KL control.")
    parser.add_argument("--config", default=default_config_path())
    args = parser.parse_args(argv)
    layout = StorageLayout.for_config(load_config(args.config))
    rows = build_nonf_prompts(read_jsonl(layout.generated / NEAR_FILE), read_jsonl(layout.generated / GENERAL_FILE))
    target = layout.generated / OUTPUT
    write_jsonl(target, rows)
    print(f"{len(rows)} prompts -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
