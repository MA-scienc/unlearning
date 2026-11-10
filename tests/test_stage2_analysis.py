"""The cross-seed analysis on synthetic data with 3 seeds and a missing cell (pre-registration 8.7)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unlearning.analysis.stage2 import SeedRun, analyse, gap_difference

QUESTIONS = 300


def write_suite(path: Path, correct: np.ndarray, rng: np.random.Generator, *, dirty: bool = False) -> None:
    records = []
    for index, ok in enumerate(correct):
        answer = int(rng.integers(0, 4))
        chosen = answer if ok else (answer + 1 + int(rng.integers(0, 3))) % 4
        logprobs = [-3.0] * 4
        logprobs[chosen] = -0.5
        records.append({"record_id": f"q{index}", "answer_index": answer, "option_logprobs": logprobs})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records, "code_version": {"git_dirty": dirty}}), encoding="utf-8")


def build(root: Path, *, effect: dict[str, float], missing=None, dirty=None) -> list[SeedRun]:
    """S0/S100 at 0.6 accuracy; U(S0) keeps it; U(S100) loses `effect[suite]` more (questions shared across seeds)."""

    runs = []
    for seed in (43, 44, 45):
        raw = root / f"run{seed}"
        rng = np.random.default_rng(seed)
        base = np.random.default_rng(0).random(QUESTIONS) < 0.6  # the same questions are easy/hard in every seed
        for suite in ("near", "far", "general"):
            flip = rng.random(QUESTIONS)
            models = {
                "S0": base, "S100": base, "MR_C0": base, "MR_C100": base,
                "MR_U3_C0_ga_f100": base ^ (flip < 0.05), "MR_U3_C0_npo_f100": base ^ (flip < 0.05),
                "MR_U3_C100_ga_f100": base & ~(flip < effect[suite] + 0.05),
                "MR_U3_C100_npo_f100": base ^ (flip < 0.05),
            }
            for label, correct in models.items():
                if missing and (seed, label) == missing:
                    continue
                write_suite(raw / label / f"{suite}.json", np.asarray(correct), rng, dirty=bool(dirty and (seed, label) == dirty))
        runs.append(SeedRun(seed, f"run{seed}", raw, ("S0", "S100")))
    return runs


class Stage2AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_detects_a_built_in_effect_and_stays_null_elsewhere(self):
        runs = build(self.root, effect={"near": 0.0, "far": 0.15, "general": 0.0}, missing=(45, "MR_U3_C100_npo_f100"))
        families = analyse(runs)["families"]
        far = families["H1"]["ga/far"]
        self.assertTrue(far["estimable"])
        self.assertLess(far["mean"], -0.05)
        self.assertTrue(far["significant"])
        self.assertLess(far["ci95"][1], 0)
        self.assertFalse(families["H1"]["ga/general"]["significant"])
        # a missing U file in one seed: H1 needs every seed, so that contrast is not estimable
        self.assertFalse(families["H1"]["npo/near"]["estimable"])
        self.assertEqual(families["H1"]["ga/near"]["seeds"], [43, 44, 45])
        # designs without the control report its families as not estimable
        self.assertFalse(families["1b"]["ga/near"]["estimable"])

    def test_results_from_uncommitted_code_are_refused(self):
        runs = build(self.root, effect={"near": 0, "far": 0, "general": 0}, dirty=(44, "S0"))
        with self.assertRaisesRegex(ValueError, "committed code"):
            analyse(runs)

    def test_gap_difference_is_a_ratio_of_means(self):
        table = {p: {"private_code_lookup": {"C": 0.0, "U": -8.0, "S": -4.0},
                     "condition_lookup": {"C": 0.0, "U": -1.0, "S": -4.0}} for p in ("a", "b")}
        self.assertAlmostEqual(gap_difference(table, ["a", "b", "a"]), 2.0 - 0.25)


if __name__ == "__main__":
    unittest.main()
