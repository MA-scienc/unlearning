from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from unlearning.evaluation.aggregate import RETENTION_METRICS, _summary_markdown, rows_for_file
from unlearning.evaluation.multiple_choice import (
    bias_removed_accuracy,
    bias_removed_correctness,
    bootstrap_bias_removed_accuracy_ci,
)
from unlearning.training.memorization import _bias_removed_accuracy


def record(scores, answer):
    return {"option_logprobs": list(scores), "answer_index": answer}


def reference_bias_removed_accuracy(records):
    """The pre-refactor implementation from training/memorization.py, kept verbatim as the oracle."""

    width = max(len(record["option_logprobs"]) for record in records)
    prior = []
    for position in range(width):
        values = [record["option_logprobs"][position] for record in records if len(record["option_logprobs"]) > position]
        prior.append(sum(values) / len(values))
    correct = 0
    for record in records:
        scores = record["option_logprobs"]
        choice = max(range(len(scores)), key=lambda index: scores[index] - prior[index])
        correct += int(choice == record["answer_index"])
    return correct / len(records)


def random_records(seed, count=200, width=3):
    rng = random.Random(seed)
    return [record([rng.gauss(-2.0, 1.0) + (1.5 if j == 2 else 0.0) for j in range(width)], rng.randrange(width)) for _ in range(count)]


class BiasRemovedAccuracyTests(unittest.TestCase):
    def test_removes_a_constant_label_preference(self):
        # Every question: the gold option is 1 nat above the others, but option 2 gets +3 everywhere.
        records = []
        for answer in (0, 1, 2) * 10:
            scores = [-3.0, -3.0, -3.0]
            scores[answer] += 1.0
            scores[2] += 3.0
            records.append(record(scores, answer))
        raw = sum(max(range(3), key=lambda j: r["option_logprobs"][j]) == r["answer_index"] for r in records) / len(records)
        self.assertAlmostEqual(raw, 1 / 3)
        self.assertEqual(bias_removed_accuracy(records), 1.0)
        self.assertTrue(all(bias_removed_correctness(records)))

    def test_matches_the_previous_damage_probe_implementation(self):
        for seed, width in ((1, 3), (2, 4), (3, 4)):
            records = random_records(seed, width=width)
            self.assertEqual(bias_removed_accuracy(records), reference_bias_removed_accuracy(records))
            self.assertEqual(_bias_removed_accuracy(records), reference_bias_removed_accuracy(records))
        ragged = random_records(4, count=50, width=4) + random_records(5, count=50, width=3)
        self.assertEqual(bias_removed_accuracy(ragged), reference_bias_removed_accuracy(ragged))

    def test_empty_input(self):
        self.assertIsNone(bias_removed_accuracy([]))
        self.assertEqual(bootstrap_bias_removed_accuracy_ci([], seed=1), (None, None))

    def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible(self):
        records = random_records(7, count=300)
        low, high = bootstrap_bias_removed_accuracy_ci(records, seed=42)
        point = bias_removed_accuracy(records)
        self.assertLess(low, point)
        self.assertGreater(high, point)
        self.assertEqual((low, high), bootstrap_bias_removed_accuracy_ci(records, seed=42))
        ragged = random_records(4, count=60, width=4) + random_records(5, count=60, width=3)
        low, high = bootstrap_bias_removed_accuracy_ci(ragged, seed=3)
        self.assertLessEqual(low, bias_removed_accuracy(ragged))
        self.assertGreaterEqual(high, bias_removed_accuracy(ragged))

    def test_bootstrap_is_degenerate_when_every_resample_is_perfect(self):
        records = [record([-1.0 if j == a else -5.0 for j in range(4)], a) for a in (0, 1, 2, 3) * 5]
        self.assertEqual(bootstrap_bias_removed_accuracy_ci(records, seed=0), (1.0, 1.0))


class AggregateRetentionTests(unittest.TestCase):
    def test_rows_and_summary_report_bias_removed_and_raw_accuracy(self):
        records = []
        for answer in (0, 1, 2) * 10:
            scores = [-3.0, -3.0, -3.0]
            scores[answer] += 1.0
            scores[2] += 3.0
            records.append({**record(scores, answer), "correct": answer == 2})
        payload = {"accuracy": 1 / 3, "accuracy_ci95": [0.2, 0.5], "count": 30, "mean_nll": 1.0, "mean_correct_logprob": -1.0, "records": records}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MR_C50" / "external_medical.json"
            path.parent.mkdir()
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = rows_for_file("run", path, seed=42)
        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(list(by_metric), list(RETENTION_METRICS))
        self.assertEqual(by_metric["bias_removed_accuracy"]["value"], 1.0)
        self.assertEqual((by_metric["bias_removed_accuracy"]["ci_low"], by_metric["bias_removed_accuracy"]["ci_high"]), (1.0, 1.0))
        self.assertEqual((by_metric["accuracy"]["ci_low"], by_metric["accuracy"]["ci_high"]), (0.2, 0.5))
        self.assertIsNone(by_metric["mean_nll"]["ci_low"])
        self.assertEqual(by_metric["accuracy"]["arm"], "matched_memorization_replay")
        summary = _summary_markdown(rows, "run")
        self.assertIn("| MR_C50 |  |  |  |  | 1.000 (0.333) |  |", summary)


if __name__ == "__main__":
    unittest.main()
