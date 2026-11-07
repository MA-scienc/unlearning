from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.test_config import minimal_config
from unlearning.config.schema import ConfigError, PilotConfig
from unlearning.config.unlearning_revisions import (
    UnlearningRevision,
    load_unlearning_revision,
    load_unlearning_revisions,
)
from unlearning.evaluation.aggregate import parse_label
from unlearning.experiments.naming import experiment_run_id
from unlearning.runtime.settings import REPO_ROOT

REVISIONS = REPO_ROOT / "configs" / "pilot" / "unlearning"
VALID = {"revision": 2, "tag": "U2", "learning_rate": {"ga": 2e-5, "npo": 2e-5}}


class UnlearningRevisionTests(unittest.TestCase):
    def test_repository_revision_2_is_the_matched_contamination_rate(self):
        revision = load_unlearning_revision(REVISIONS / "revision2.yaml")
        self.assertEqual((revision.revision, revision.tag), (2, "U2"))
        self.assertEqual(revision.learning_rates, {"ga": 2e-5, "npo": 2e-5})
        self.assertEqual([r.revision for r in load_unlearning_revisions(REVISIONS)], [2])

    def test_only_the_learning_rate_changes_and_the_run_id_does_not(self):
        config = PilotConfig.from_dict(minimal_config())
        run_id = experiment_run_id(config)
        revision = UnlearningRevision.from_dict(VALID)
        for method in ("ga", "npo"):
            original = config.unlearning.methods[method]
            revised = revision.method_config(config, method)
            self.assertEqual(revised.learning_rate, 2e-5)
            self.assertEqual({**revised.__dict__, "learning_rate": original.learning_rate}, original.__dict__)
        self.assertEqual(experiment_run_id(config), run_id)

    def test_rejects_anything_but_a_learning_rate_change(self):
        cases = [
            ({**VALID, "steps": 800}, "may set only"),
            ({**VALID, "revision": 1, "tag": "U1"}, "start at 2"),
            ({**VALID, "tag": "U3"}, "must have tag 'U2'"),
            ({**VALID, "learning_rate": {"ga": 2e-5}}, "exactly"),
            ({**VALID, "learning_rate": {"ga": 2e-5, "npo": 0.0}}, "positive"),
            ({"revision": 2, "tag": "U2"}, "incomplete"),
        ]
        for raw, message in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(ConfigError, message):
                UnlearningRevision.from_dict(raw)

    def test_missing_directory_has_no_revisions_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_unlearning_revisions(Path(directory) / "absent"), ())
            text = "revision: 2\ntag: U2\nlearning_rate: {ga: 2.0e-5, npo: 2.0e-5}\n"
            (Path(directory) / "a.yaml").write_text(text, encoding="utf-8")
            (Path(directory) / "b.yaml").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "duplicate"):
                load_unlearning_revisions(directory)


class RevisionLabelTests(unittest.TestCase):
    def test_labels_carry_the_unlearning_revision(self):
        self.assertEqual(parse_label("MR_U_C0_ga_step0400")["unlearning_revision"], 1)
        revised = parse_label("MR_U2_C50_npo_step0200")
        self.assertEqual(
            (revised["unlearning_revision"], revised["specialization_level"], revised["method"], revised["unlearning_step"]),
            (2, 50, "npo", 200),
        )
        self.assertEqual(revised["arm"], "matched_memorization_replay")
        self.assertIsNone(parse_label("MR_C100")["unlearning_revision"])
        self.assertIsNone(parse_label("S0")["unlearning_revision"])
        with self.assertRaises(ValueError):
            parse_label("MR_U1_C0_ga_step0000")  # revision 1 is written without a number


if __name__ == "__main__":
    unittest.main()
