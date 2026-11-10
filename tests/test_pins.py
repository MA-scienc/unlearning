import shutil
import tempfile
import unittest
from pathlib import Path

from unlearning.config.arms import load_matched_memorization_arm
from unlearning.config.pins import verified_pinned_report
from unlearning.config.unlearning_revisions import load_unlearning_revision
from unlearning.runtime.layout import StorageLayout
from unlearning.runtime.settings import REPO_ROOT
from unlearning.training.matched_contamination import DEFAULT_ARM_PATH, load_selected_learning_rate
from unlearning.training.matched_forgetting import selected_learning_rate

PINNED = REPO_ROOT / "configs" / "stage2" / "pinned"
CONTAMINATION = (PINNED / "contamination_MR_calibration_report.json",
                 "c4b5d773aa439c12c3deb83b16be8d3abed9109ec0d7844f4d883ca17b0b0cf7")
UNLEARNING = (PINNED / "unlearning_U3_calibration_report.json",
              "8cbcf00cd15d72de4c13da5ebac7fb40e2e84b3835a6b45f0c2dd38af760da83")


class PinnedRateTests(unittest.TestCase):
    def test_pinned_reports_give_the_pilot_rates(self):
        arm = load_matched_memorization_arm(DEFAULT_ARM_PATH)
        rate, sha = load_selected_learning_rate(verified_pinned_report(*CONTAMINATION), arm)
        self.assertEqual((rate, sha), (2e-5, CONTAMINATION[1]))
        revision = load_unlearning_revision(REPO_ROOT / "configs" / "pilot" / "unlearning" / "revision3.yaml")
        layout = StorageLayout(root=Path("unused"), run_id="stage2", generated_dir_name="data",
                               checkpoint_dir_name="checkpoints", result_dir_name="results")
        for method in ("ga", "npo"):
            rate, sha = selected_learning_rate(layout, revision, method, report_path=verified_pinned_report(*UNLEARNING))
            self.assertEqual((rate, sha), (2e-6, UNLEARNING[1]))

    def test_a_changed_or_unhashed_report_is_refused(self):
        with self.assertRaisesRegex(ValueError, "expected sha256"):
            verified_pinned_report(CONTAMINATION[0], None)
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "report.json"
            shutil.copyfile(CONTAMINATION[0], copy)
            copy.write_text(copy.read_text(encoding="utf-8").replace("2e-05", "3e-05"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has sha256"):
                verified_pinned_report(copy, CONTAMINATION[1])


if __name__ == "__main__":
    unittest.main()
