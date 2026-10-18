import unittest

from unlearning.evaluation.statistics import (
    aggregate_scores_by_cluster,
    paired_cluster_bootstrap,
)


class StatisticsTests(unittest.TestCase):
    def test_aggregate_scores_by_cluster(self):
        result = aggregate_scores_by_cluster(
            [
                {"record_id": "a", "leakage": 1.0},
                {"record_id": "a", "leakage": 0.0},
                {"record_id": "b", "leakage": 0.25},
            ],
            cluster_key="record_id",
            value_key="leakage",
        )
        self.assertEqual(result, {"a": 0.5, "b": 0.25})

    def test_paired_cluster_bootstrap_is_deterministic(self):
        baseline = {"a": 1.0, "b": 0.0, "c": 0.5}
        comparison = {"a": 0.0, "b": 0.0, "c": 0.0}
        left = paired_cluster_bootstrap(baseline, comparison, seed=9, samples=100)
        right = paired_cluster_bootstrap(baseline, comparison, seed=9, samples=100)
        self.assertEqual(left, right)
        self.assertAlmostEqual(left["mean_delta"], -0.5)
        self.assertEqual(left["resampling_unit"], "cluster")

    def test_paired_cluster_bootstrap_rejects_unpaired_inputs(self):
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap({"a": 1.0}, {"b": 0.0}, seed=1)

    def test_rejects_nonfinite_cluster_scores(self):
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap({"a": float("nan")}, {"a": 0.0}, seed=1)
        with self.assertRaises(ValueError):
            aggregate_scores_by_cluster(
                [{"record_id": "a", "leakage": float("inf")}],
                cluster_key="record_id",
                value_key="leakage",
            )


if __name__ == "__main__":
    unittest.main()
