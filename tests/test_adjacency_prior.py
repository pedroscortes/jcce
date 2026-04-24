"""
Unit tests for AdjacencyPrior (Phase D.1).

Tests EMA updates, cold/warm sampling, fitness gating, and thread safety.
"""

import threading

import numpy as np
import pytest

from jcce.structure_learning.adjacency_prior import AdjacencyPrior


class TestAdjacencyPrior:
    def test_fresh_prior_has_correct_initial_values(self):
        """Fresh prior: A_tilde == 0.5 off-diagonal, 0.0 on diagonal."""
        prior = AdjacencyPrior(n_total=6)
        A = prior.A_tilde
        assert A.shape == (6, 6)
        # Diagonal is zero
        np.testing.assert_array_equal(np.diag(A), 0.0)
        # Off-diagonal is 0.5
        mask = ~np.eye(6, dtype=bool)
        np.testing.assert_allclose(A[mask], 0.5)

    def test_update_with_all_ones_increases_prior(self):
        """Update with all-ones matrix at high fitness moves A_tilde > 0.5."""
        prior = AdjacencyPrior(n_total=4, beta=0.9, fitness_threshold=0.5)
        A_ones = np.ones((4, 4))
        prior.update(A_ones, fitness=0.8)

        A = prior.A_tilde
        mask = ~np.eye(4, dtype=bool)
        # EMA: 0.9*0.5 + 0.1*1.0 = 0.55, then clipped to [0.05, 0.95]
        assert np.all(A[mask] > 0.5)
        assert prior.n_updates == 1

    def test_low_fitness_update_is_rejected(self):
        """Update below fitness_threshold is silently rejected."""
        prior = AdjacencyPrior(n_total=4, fitness_threshold=0.6)
        A_ones = np.ones((4, 4))
        prior.update(A_ones, fitness=0.3)

        # No change from initial
        assert prior.n_updates == 0
        mask = ~np.eye(4, dtype=bool)
        np.testing.assert_allclose(prior.A_tilde[mask], 0.5)

    def test_sampled_A_within_bounds_and_zero_diagonal(self):
        """Sampled A is within [0, 1] with zero diagonal (after warm-up)."""
        prior = AdjacencyPrior(n_total=5, fitness_threshold=0.0)
        # Do enough updates to pass cold-start threshold
        for _ in range(6):
            A = np.random.rand(5, 5)
            prior.update(A, fitness=0.8)

        rng = np.random.default_rng(42)
        A_init = prior.sample_A_init(noise_scale=0.05, rng=rng)
        assert A_init.shape == (5, 5)
        assert np.all(A_init >= 0.0)
        assert np.all(A_init <= 1.0)
        np.testing.assert_array_equal(np.diag(A_init), 0.0)

    def test_cold_start_returns_small_random_values(self):
        """Before 5 updates, sample_A_init returns small random values."""
        prior = AdjacencyPrior(n_total=4)
        # Only 2 updates (< 5)
        prior.update(np.ones((4, 4)), fitness=0.8)
        prior.update(np.ones((4, 4)), fitness=0.8)

        rng = np.random.default_rng(42)
        A_init = prior.sample_A_init(rng=rng)
        assert A_init.shape == (4, 4)
        np.testing.assert_array_equal(np.diag(A_init), 0.0)
        # Values should be small (drawn from N(0, 0.1))
        assert np.abs(A_init).max() < 1.0  # very likely with std=0.1

    def test_thread_safety_concurrent_updates(self):
        """4 threads x 10 updates = 40 total, no crashes or lost updates."""
        prior = AdjacencyPrior(n_total=4, fitness_threshold=0.0, beta=0.5)
        n_threads = 4
        n_updates_per_thread = 10

        def worker(seed):
            rng = np.random.default_rng(seed)
            for _ in range(n_updates_per_thread):
                A = rng.random((4, 4))
                prior.update(A, fitness=0.8)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert prior.n_updates == n_threads * n_updates_per_thread
        # A_tilde should still have zero diagonal
        np.testing.assert_array_equal(np.diag(prior.A_tilde), 0.0)


class TestAdjacencyPriorStats:
    def test_get_stats_returns_expected_keys(self):
        """get_stats returns dict with expected keys."""
        prior = AdjacencyPrior(n_total=4)
        stats = prior.get_stats()
        expected_keys = {
            'n_updates', 'mean_edge_prob', 'std_edge_prob',
            'min_edge_prob', 'max_edge_prob', 'n_strong_edges', 'n_weak_edges',
        }
        assert set(stats.keys()) == expected_keys
        assert stats['n_updates'] == 0
        assert abs(stats['mean_edge_prob'] - 0.5) < 1e-6
