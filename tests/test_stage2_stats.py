import unittest

import numpy as np

from unlearning.analysis.stats import (
    bootstrap_p, crossed_anova, crossed_bootstrap, holm, holm_levels, percentile_ci, satterthwaite,
    stratified_patient_resample, symmetric_bound,
)


class CrossedAnovaTests(unittest.TestCase):
    def test_matches_the_two_way_formulas(self):
        rng = np.random.default_rng(0)
        q, s = 40, 3
        data = 0.1 + rng.normal(0, 0.5, (q, 1)) + rng.normal(0, 0.2, (1, s)) + rng.normal(0, 0.3, (q, s))
        result = crossed_anova(data)
        grand = data.mean()
        ms_q = s * ((data.mean(1) - grand) ** 2).sum() / (q - 1)
        ms_s = q * ((data.mean(0) - grand) ** 2).sum() / (s - 1)
        ms_e = ((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand) ** 2).sum() / ((q - 1) * (s - 1))
        self.assertAlmostEqual(result.mean, grand)
        if not result.truncated:
            self.assertAlmostEqual(result.variance, (ms_q + ms_s - ms_e) / (q * s))
        low, high = result.ci(0.95)
        self.assertLess(low, result.mean)
        self.assertGreater(high, result.mean)

    def test_negative_components_are_truncated_and_df_follow(self):
        # No seed effect at all: seed columns identical -> MS_S = 0 < MS_E, seed component truncated.
        rng = np.random.default_rng(1)
        column = rng.normal(0, 1, (30, 1))
        noise = rng.normal(0, 0.01, (30, 3))
        noise -= noise.mean(axis=0, keepdims=True)  # every seed mean equal: MS_S = 0 < MS_E
        data = np.hstack([column, column, column]) + noise
        result = crossed_anova(data)
        self.assertIn("seed", result.truncated)
        self.assertAlmostEqual(result.variance, result.components["MS_Q"] / 90)
        self.assertEqual(result.df, 29)

    def test_satterthwaite_single_term_is_its_df(self):
        self.assertEqual(satterthwaite([(2.0, 7)]), 7)


class BootstrapAndMultiplicityTests(unittest.TestCase):
    def test_crossed_bootstrap_centres_on_the_mean(self):
        data = np.random.default_rng(2).normal(0.3, 1, (50, 3))
        estimates = crossed_bootstrap(data, samples=2000, seed=3)
        self.assertAlmostEqual(estimates.mean(), data.mean(), delta=0.05)
        low, high = percentile_ci(estimates, 0.95)
        self.assertLess(low, data.mean())
        self.assertGreater(high, data.mean())

    def test_bootstrap_p_directions(self):
        estimates = np.array([0.1, 0.2, -0.1, 0.3])
        self.assertEqual(bootstrap_p(estimates, direction="greater"), 0.25)
        self.assertEqual(bootstrap_p(estimates, direction="less"), 0.75)
        self.assertEqual(bootstrap_p(estimates), 0.5)

    def test_holm_and_step_levels(self):
        self.assertEqual([round(v, 6) for v in holm([0.01, 0.04, 0.03])], [0.03, 0.06, 0.06])
        self.assertEqual([round(v, 6) for v in holm_levels([0.01, 0.04, 0.03])], [round(1 - 0.05 / 3, 6), 0.95, 0.975])

    def test_symmetric_bound(self):
        self.assertEqual(symmetric_bound((-0.03, 0.01)), 0.03)


class StratifiedResampleTests(unittest.TestCase):
    def test_patients_stay_in_their_seed_and_groups_keep_their_size(self):
        strata = {"43": {"d1": ["a", "b"], "d2": ["c", "d", "e"]}, "44": {"d1": ["f"], "d2": ["g", "h"]}}
        drawn = stratified_patient_resample(strata, np.random.default_rng(4))
        self.assertTrue(set(drawn["43"]) <= {"a", "b", "c", "d", "e"})
        self.assertTrue(set(drawn["44"]) <= {"f", "g", "h"})
        self.assertIn(len(drawn["43"]), (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
