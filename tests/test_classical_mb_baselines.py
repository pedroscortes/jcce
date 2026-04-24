"""
Tests for jcce/gbs/classical_mb_baselines.py — Classical MB estimation algorithms.
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, '.')

from jcce.gbs.classical_mb_baselines import (
    association_score,
    fast_iamb,
    fisher_z_test,
    generate_linear_sem_data,
    generate_nonlinear_sem_data,
    hiton_mb,
    iamb,
    inter_iamb,
    mb_metrics,
    partial_correlation,
    run_all_mb_algorithms,
    true_markov_blanket,
)


def make_chain(d=8):
    """Chain DAG: 0→1→2→...→d-1"""
    A = np.zeros((d, d))
    for i in range(d - 1):
        A[i, i + 1] = 1.0
    return A


def make_fork(d=8):
    """Fork DAG: 0→{1,2,...,d-1}"""
    A = np.zeros((d, d))
    for i in range(1, d):
        A[0, i] = 1.0
    return A


def make_collider():
    """V-structure: 0→2←1, 2→3"""
    A = np.zeros((4, 4))
    A[0, 2] = 1.0  # 0 → 2
    A[1, 2] = 1.0  # 1 → 2
    A[2, 3] = 1.0  # 2 → 3
    return A


class TestCITesting(unittest.TestCase):
    """Tests for conditional independence testing."""

    def test_partial_correlation_independent(self):
        """Independent variables have ~0 partial correlation."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 3))  # 3 independent variables
        r = partial_correlation(X, 0, 1)
        self.assertAlmostEqual(r, 0.0, delta=0.1)

    def test_partial_correlation_dependent(self):
        """Correlated variables have high partial correlation."""
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(1000)
        x1 = 0.8 * x0 + 0.2 * rng.standard_normal(1000)
        X = np.column_stack([x0, x1, rng.standard_normal(1000)])
        r = partial_correlation(X, 0, 1)
        self.assertGreater(abs(r), 0.5)

    def test_fisher_z_independent(self):
        """Fisher's z correctly identifies independence."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 3))
        independent, p_val = fisher_z_test(X, 0, 1, alpha=0.05)
        self.assertTrue(independent)
        self.assertGreater(p_val, 0.05)

    def test_fisher_z_dependent(self):
        """Fisher's z correctly identifies dependence."""
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(1000)
        x1 = 0.8 * x0 + 0.2 * rng.standard_normal(1000)
        X = np.column_stack([x0, x1, rng.standard_normal(1000)])
        independent, p_val = fisher_z_test(X, 0, 1, alpha=0.05)
        self.assertFalse(independent)
        self.assertLess(p_val, 0.05)


class TestDataGeneration(unittest.TestCase):
    """Tests for SEM data generation."""

    def test_linear_sem_shape(self):
        """Linear SEM produces correct shape."""
        A = make_chain(5)
        X = generate_linear_sem_data(A, n_samples=100)
        self.assertEqual(X.shape, (100, 5))

    def test_linear_sem_correlations(self):
        """Adjacent nodes in chain are correlated, distant less so."""
        A = make_chain(5)
        X = generate_linear_sem_data(A, n_samples=2000, seed=42)
        # 0→1 should be strongly correlated
        r01 = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
        self.assertGreater(abs(r01), 0.3)

    def test_nonlinear_sem_shape(self):
        """Nonlinear SEM produces correct shape."""
        A = make_chain(5)
        X = generate_nonlinear_sem_data(A, n_samples=100)
        self.assertEqual(X.shape, (100, 5))

    def test_nonlinear_sem_finite(self):
        """Nonlinear SEM produces finite values."""
        A = make_chain(5)
        X = generate_nonlinear_sem_data(A, n_samples=500)
        self.assertTrue(np.all(np.isfinite(X)))


class TestTrueMarkovBlanket(unittest.TestCase):
    """Tests for ground-truth MB extraction."""

    def test_chain_middle(self):
        """In chain 0→1→2→3, MB(1) = {0, 2}."""
        A = make_chain(4)
        mb = true_markov_blanket(A, target=1)
        self.assertEqual(mb, {0, 2})

    def test_chain_end(self):
        """In chain 0→1→2→3, MB(3) = {2}."""
        A = make_chain(4)
        mb = true_markov_blanket(A, target=3)
        self.assertEqual(mb, {2})

    def test_fork_root(self):
        """In fork 0→{1,2,3}, MB(0) = {1, 2, 3}."""
        A = make_fork(4)
        mb = true_markov_blanket(A, target=0)
        self.assertEqual(mb, {1, 2, 3})

    def test_collider_spouse(self):
        """In 0→2←1, 2→3: MB(0) = {1, 2} (1 is spouse via child 2)."""
        A = make_collider()
        mb = true_markov_blanket(A, target=0)
        self.assertEqual(mb, {1, 2})

    def test_collider_child(self):
        """In 0→2←1, 2→3: MB(2) = {0, 1, 3}."""
        A = make_collider()
        mb = true_markov_blanket(A, target=2)
        self.assertEqual(mb, {0, 1, 3})


class TestIAMB(unittest.TestCase):
    """Tests for IAMB algorithm."""

    def test_chain_basic(self):
        """IAMB recovers MB in a chain with sufficient data."""
        A = make_chain(5)
        X = generate_linear_sem_data(A, n_samples=2000, seed=42)
        true_mb = true_markov_blanket(A, target=2)
        pred_mb = iamb(X, target=2, alpha=0.05)

        # Should include at least the true MB members (high recall)
        metrics = mb_metrics(pred_mb, true_mb)
        self.assertGreater(metrics['recall'], 0.5,
                          f"IAMB recall too low: {metrics}")

    def test_fork_root(self):
        """IAMB finds children of the fork root."""
        A = make_fork(5)
        X = generate_linear_sem_data(A, n_samples=2000, seed=42)
        true_mb = true_markov_blanket(A, target=0)
        pred_mb = iamb(X, target=0, alpha=0.05)

        metrics = mb_metrics(pred_mb, true_mb)
        self.assertGreater(metrics['recall'], 0.5)

    def test_returns_set(self):
        """IAMB returns a set of integers."""
        A = make_chain(5)
        X = generate_linear_sem_data(A, n_samples=500)
        result = iamb(X, target=2)
        self.assertIsInstance(result, set)
        for v in result:
            self.assertIsInstance(v, (int, np.integer))


class TestAllAlgorithms(unittest.TestCase):
    """Tests comparing all MB algorithms."""

    def test_all_algorithms_chain(self):
        """All algorithms achieve positive F1 on chain."""
        A = make_chain(6)
        X = generate_linear_sem_data(A, n_samples=3000, seed=42)
        target = 3
        true_mb = true_markov_blanket(A, target)

        results = run_all_mb_algorithms(X, target, alpha=0.05)

        for name, pred_mb in results.items():
            metrics = mb_metrics(pred_mb, true_mb)
            self.assertGreater(metrics['f1'], 0.0,
                              f"{name} has zero F1 on chain")

    def test_all_algorithms_return_correct_type(self):
        """All algorithms return set[int]."""
        A = make_chain(5)
        X = generate_linear_sem_data(A, n_samples=500)
        results = run_all_mb_algorithms(X, target=2)

        for name, mb in results.items():
            self.assertIsInstance(mb, set, f"{name} didn't return a set")
            self.assertNotIn(2, mb, f"{name} included target in MB")

    def test_mb_metrics(self):
        """MB metrics computation is correct."""
        true_mb = {0, 1, 2}
        pred_mb = {0, 1, 3}  # TP=2 (0,1), FP=1 (3), FN=1 (2)

        m = mb_metrics(pred_mb, true_mb)
        self.assertAlmostEqual(m['precision'], 2/3)
        self.assertAlmostEqual(m['recall'], 2/3)
        self.assertAlmostEqual(m['f1'], 2/3)

    def test_mb_metrics_empty(self):
        """Empty prediction and true MB give perfect score."""
        m = mb_metrics(set(), set())
        self.assertEqual(m['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
