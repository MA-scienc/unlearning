import tempfile
import unittest
from pathlib import Path

from unlearning.evaluation.comparison import (
    compare_result_files,
    paired_cluster_metric_comparison,
    validate_unlearning_triplet,
)
from unlearning.utils.io import write_json


class PairedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline_rows = [
            {"example_id": "a::one", "record_id": "a", "leakage": 1.0},
            {"example_id": "a::two", "record_id": "a", "leakage": 0.0},
            {"example_id": "b::one", "record_id": "b", "leakage": 1.0},
        ]
        self.comparison_rows = [
            {"example_id": "a::one", "record_id": "a", "leakage": 0.0},
            {"example_id": "a::two", "record_id": "a", "leakage": 0.0},
            {"example_id": "b::one", "record_id": "b", "leakage": 0.0},
        ]

    def test_aggregates_templates_before_clustered_bootstrap(self):
        result = paired_cluster_metric_comparison(
            self.baseline_rows,
            self.comparison_rows,
            metric="leakage",
            seed=7,
            bootstrap_samples=100,
        )
        self.assertEqual(result["item_count"], 3)
        self.assertEqual(result["paired_bootstrap"]["cluster_count"], 2)
        self.assertAlmostEqual(result["paired_bootstrap"]["mean_delta"], -0.75)

    def test_rejects_missing_paired_items(self):
        with self.assertRaisesRegex(ValueError, "identical evaluated item IDs"):
            paired_cluster_metric_comparison(
                self.baseline_rows,
                self.comparison_rows[:-1],
                metric="leakage",
                seed=7,
            )

    def test_result_files_require_shared_forget_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            common = {
                "eval_examples_fingerprint": "eval-fingerprint",
                "records_fingerprint": "records-fingerprint",
            }
            baseline = {"forget_dataset": common, "records": self.baseline_rows}
            comparison = {"forget_dataset": common, "records": self.comparison_rows}
            baseline_path = root / "baseline.json"
            comparison_path = root / "comparison.json"
            write_json(baseline_path, baseline)
            write_json(comparison_path, comparison)
            result = compare_result_files(
                baseline_path,
                comparison_path,
                metric="leakage",
                seed=7,
                bootstrap_samples=100,
            )
        self.assertEqual(result["shared_data_provenance"], common)

    def test_triplet_requires_matching_c_and_s_lineage(self):
        clean = {"checkpoint_provenance": {"manifest": {"checkpoint_id": "S50"}}}
        contaminated = {"checkpoint_provenance": {"manifest": {"checkpoint_id": "C50"}}}
        unlearned = {
            "checkpoint_provenance": {
                "manifest": {
                    "checkpoint_id": "U50",
                    "source_contaminated_checkpoint_id": "C50",
                    "clean_counterfactual_checkpoint_id": "S50",
                }
            }
        }
        result = validate_unlearning_triplet(clean, contaminated, unlearned)
        self.assertEqual(result["unlearned_checkpoint_id"], "U50")


if __name__ == "__main__":
    unittest.main()
