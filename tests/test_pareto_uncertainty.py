"""
Tests for jcce/gbs/pareto_uncertainty.py — Application D: Hellinger Uncertainty Metrics.
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from jcce.gbs.pareto_uncertainty import (
    dequantized_hellinger_matrix,
    hellinger_coverage,
    hellinger_diameter,
    hellinger_vs_shd_correlation,
    mean_hellinger_dispersion,
    pareto_gbs_features,
    pareto_uncertainty_report,
)


def make_simple_dag(d, seed=42, edge_prob=0.3):
    """Generate a simple random DAG for testing."""
    rng = np.random.default_rng(seed)
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                A[i, j] = rng.uniform(0.3, 1.5)
    return A


class TestParetoUncertainty(unittest.TestCase):
    """Tests for Pareto front uncertainty metrics."""

    def setUp(self):
        """Create a small set of Pareto DAGs for testing."""
        self.d = 8
        self.pareto_dags = [
            make_simple_dag(self.d, seed=s, edge_prob=ep)
            for s, ep in [(42, 0.2), (123, 0.3), (456, 0.4), (789, 0.5), (1000, 0.15)]
        ]

    def test_pareto_gbs_features(self):
        """Feature vectors have correct shape and properties."""
        features = pareto_gbs_features(self.pareto_dags, max_order=2)
        self.assertEqual(len(features), 5)

        # Order 2: d + C(d,2) = 8 + 28 = 36 features
        expected_len = self.d + self.d * (self.d - 1) // 2
        for f in features:
            self.assertEqual(len(f), expected_len)
            # All features are probabilities in [0, 1]
            self.assertTrue(np.all(f >= 0))
            self.assertTrue(np.all(f <= 1))

        # Different DAGs should produce different features
        diffs = [np.linalg.norm(features[0] - features[i]) for i in range(1, 5)]
        self.assertTrue(all(d > 0 for d in diffs))

    def test_hellinger_matrix_properties(self):
        """Hellinger matrix satisfies metric properties."""
        H = dequantized_hellinger_matrix(self.pareto_dags, max_order=2)

        n = len(self.pareto_dags)
        self.assertEqual(H.shape, (n, n))

        # Symmetry
        np.testing.assert_array_almost_equal(H, H.T)

        # Zero diagonal
        np.testing.assert_array_almost_equal(np.diag(H), 0.0)

        # Non-negative
        self.assertTrue(np.all(H >= -1e-10))

        # Off-diagonal > 0 (different DAGs should have different features)
        upper = H[np.triu_indices(n, k=1)]
        self.assertTrue(np.all(upper > 0))

    def test_hellinger_diameter(self):
        """Diameter is the maximum pairwise distance."""
        H = dequantized_hellinger_matrix(self.pareto_dags, max_order=2)
        diam = hellinger_diameter(H)

        # Should be max of upper triangle
        n = len(self.pareto_dags)
        upper = H[np.triu_indices(n, k=1)]
        self.assertAlmostEqual(diam, np.max(upper))

        # Should be positive for different DAGs
        self.assertGreater(diam, 0)

    def test_mean_dispersion(self):
        """Mean dispersion is the average pairwise distance."""
        H = dequantized_hellinger_matrix(self.pareto_dags, max_order=2)
        disp = mean_hellinger_dispersion(H)

        n = len(self.pareto_dags)
        upper = H[np.triu_indices(n, k=1)]
        self.assertAlmostEqual(disp, np.mean(upper))

        # Dispersion <= diameter
        diam = hellinger_diameter(H)
        self.assertLessEqual(disp, diam + 1e-10)

        # Single DAG: dispersion = 0
        H_single = np.zeros((1, 1))
        self.assertEqual(mean_hellinger_dispersion(H_single), 0.0)

    def test_coverage(self):
        """Coverage is normalized entropy in [0, 1]."""
        H = dequantized_hellinger_matrix(self.pareto_dags, max_order=2)
        cov = hellinger_coverage(H)

        self.assertGreaterEqual(cov, 0.0)
        self.assertLessEqual(cov, 1.0)

        # Single DAG: coverage = 0
        H_single = np.zeros((1, 1))
        self.assertEqual(hellinger_coverage(H_single), 0.0)

    def test_shd_correlation(self):
        """SHD correlation returns valid correlation values."""
        H = dequantized_hellinger_matrix(self.pareto_dags, max_order=2)
        result = hellinger_vs_shd_correlation(self.pareto_dags, H=H)

        # Pearson r should be a valid number in [-1, 1]
        self.assertTrue(-1 <= result["pearson_r"] <= 1)
        self.assertTrue(-1 <= result["spearman_rho"] <= 1)

        # SHD matrix should be valid
        n = len(self.pareto_dags)
        self.assertEqual(result["shd_matrix"].shape, (n, n))
        np.testing.assert_array_almost_equal(result["shd_matrix"], result["shd_matrix"].T)

    def test_full_uncertainty_report(self):
        """Full report returns all expected keys with valid values."""
        report = pareto_uncertainty_report(self.pareto_dags, max_order=2)

        expected_keys = [
            "hellinger_matrix",
            "diameter",
            "mean_dispersion",
            "coverage",
            "pearson_r",
            "spearman_rho",
            "shd_matrix",
            "n_dags",
            "d",
        ]
        for key in expected_keys:
            self.assertIn(key, report, f"Missing key: {key}")

        self.assertEqual(report["n_dags"], 5)
        self.assertEqual(report["d"], self.d)
        self.assertGreater(report["diameter"], 0)
        self.assertGreater(report["mean_dispersion"], 0)
        self.assertGreater(report["coverage"], 0)

    def test_identical_dags_zero_distance(self):
        """Identical DAGs produce zero Hellinger distance."""
        dag = make_simple_dag(8, seed=42)
        H = dequantized_hellinger_matrix([dag, dag], max_order=2)

        self.assertAlmostEqual(H[0, 1], 0.0, places=6)
        self.assertAlmostEqual(hellinger_diameter(H), 0.0, places=6)
        self.assertAlmostEqual(mean_hellinger_dispersion(H), 0.0, places=6)

    def test_scalability_d13(self):
        """Module works at d=13 within reasonable time."""
        import time

        dags = [make_simple_dag(13, seed=s) for s in [42, 123, 456]]

        t0 = time.time()
        report = pareto_uncertainty_report(dags, max_order=2)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 5.0, "d=13 should complete within 5 seconds")
        self.assertGreater(report["diameter"], 0)


if __name__ == "__main__":
    unittest.main()
