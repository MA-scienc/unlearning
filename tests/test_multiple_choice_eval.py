import unittest

from unlearning.evaluation.multiple_choice import (
    bootstrap_accuracy_ci,
    score_from_option_logprobs,
    summarize_mc_results,
)


class MultipleChoiceEvalTests(unittest.TestCase):
    def test_option_logprob_scoring(self):
        result = score_from_option_logprobs(
            record_id="q1",
            answer_index=1,
            option_logprobs=[-3.0, -0.2, -1.0, -4.0],
        )
        self.assertTrue(result.correct)
        self.assertEqual(result.prediction_index, 1)
        self.assertAlmostEqual(result.nll, 0.2)

    def test_summary_and_ci_are_deterministic(self):
        results = [
            score_from_option_logprobs(record_id="a", answer_index=0, option_logprobs=[0.0, -1.0]),
            score_from_option_logprobs(record_id="b", answer_index=1, option_logprobs=[0.0, -1.0]),
        ]
        summary = summarize_mc_results(results)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(
            bootstrap_accuracy_ci([r.correct for r in results], seed=7, samples=20),
            bootstrap_accuracy_ci([r.correct for r in results], seed=7, samples=20),
        )


if __name__ == "__main__":
    unittest.main()
