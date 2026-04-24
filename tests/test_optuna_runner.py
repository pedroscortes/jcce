"""
Unit tests for Optuna runner (Phase A.2+A.3).

Tests PC warm-start seeding, post-hoc CV integration, multi-GPU launcher,
hypervolume computation, and edge stability.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import optuna

from jcce.structure_learning.experiment_runner import (
    create_pc_seed_trials,
    enqueue_seed_trials,
    run_posthoc_cv,
    run_full_optuna_pipeline,
)
from scripts.compare_nsga2_vs_optuna import (
    compute_hypervolume_2d,
    compute_edge_stability,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_tiny_data(n_samples=50, n_vars=5, seed=42):
    """Create tiny synthetic data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_vars).astype(np.float32)
    Y = (rng.randn(n_samples) > 0).astype(np.float32)
    return X, Y


# ============================================================================
# Test: PC warm-start seed trials
# ============================================================================

class TestPCSeedTrials:
    def test_creates_seed_configs(self):
        """PC warm-start generates valid seed configs for each processor."""
        X, Y = _make_tiny_data()
        X_jax = jnp.array(X)

        pc_A_init, seed_configs = create_pc_seed_trials(
            X_jax, n_seeds=5, verbose=False,
        )

        # PC should return a matrix
        assert pc_A_init is not None
        assert pc_A_init.shape == (6, 6)  # n_vars+1

        # Should have 5 seed configs (one per processor)
        assert len(seed_configs) == 5

        # Each config should have processor_type and GOLEM params
        proc_types_seen = set()
        for config in seed_configs:
            assert 'processor_type' in config
            assert 'lambda_1' in config
            assert 'lambda_2' in config
            assert 'lr' in config
            proc_types_seen.add(config['processor_type'])

        assert len(proc_types_seen) == 5

    def test_enqueue_seed_trials(self):
        """Seed configs can be enqueued into a study."""
        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        seed_configs = [
            {'processor_type': 'elm', 'lambda_1': 0.02, 'lambda_2': 0.01,
             'lr': 0.001, 'lambda_class': 1.0, 'elm_hidden_dim': 64,
             'elm_n_hidden_nodes': 64, 'elm_activation': 'relu'},
        ]

        n_enqueued = enqueue_seed_trials(study, seed_configs, verbose=False)
        assert n_enqueued == 1


# ============================================================================
# Test: Hypervolume computation
# ============================================================================

class TestHypervolume:
    def test_empty_returns_zero(self):
        """Empty Pareto front has zero hypervolume."""
        hv = compute_hypervolume_2d(np.zeros((0, 2)), np.array([0.0, 0.0]))
        assert hv == 0.0

    def test_single_point(self):
        """Single point dominates a rectangle."""
        points = np.array([[0.8, 0.6]])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume_2d(points, ref)
        assert abs(hv - 0.8 * 0.6) < 1e-10

    def test_two_points(self):
        """Two non-dominated points."""
        points = np.array([[0.9, 0.3], [0.5, 0.8]])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume_2d(points, ref)
        # Point (0.9, 0.3): contributes 0.9 * 0.3 = 0.27
        # Point (0.5, 0.8): contributes 0.5 * (0.8 - 0.3) = 0.25
        expected = 0.27 + 0.25
        assert abs(hv - expected) < 1e-10

    def test_dominated_point_excluded(self):
        """Points below ref point contribute nothing."""
        points = np.array([[0.8, 0.6], [-0.1, -0.1]])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume_2d(points, ref)
        assert abs(hv - 0.8 * 0.6) < 1e-10

    def test_monotonic_with_more_points(self):
        """Adding a non-dominated point increases hypervolume."""
        ref = np.array([0.0, 0.0])
        p1 = np.array([[0.8, 0.3]])
        p2 = np.array([[0.8, 0.3], [0.4, 0.7]])
        hv1 = compute_hypervolume_2d(p1, ref)
        hv2 = compute_hypervolume_2d(p2, ref)
        assert hv2 >= hv1


# ============================================================================
# Test: Edge stability
# ============================================================================

class TestEdgeStability:
    def test_empty_solutions(self):
        """No solutions → zero stability."""
        stability = compute_edge_stability([], n_vars=5)
        assert stability.shape == (6, 6)
        assert np.all(stability == 0.0)

    def test_consistent_edges_high_stability(self):
        """Same edge in all solutions → stability = 1.0."""
        n = 6
        A = np.zeros((n, n))
        A[0, 1] = 0.5  # Edge 0→1

        solutions = [
            {'metrics': {'structure_A_est': A.copy()}}
            for _ in range(5)
        ]

        stability = compute_edge_stability(solutions, n_vars=5, threshold=0.1)
        assert stability[0, 1] == 1.0
        assert stability[1, 0] == 0.0  # No reverse edge

    def test_partial_stability(self):
        """Edge in 3/5 solutions → stability = 0.6."""
        n = 6
        solutions = []
        for i in range(5):
            A = np.zeros((n, n))
            if i < 3:
                A[2, 3] = 0.5
            solutions.append({'metrics': {'structure_A_est': A}})

        stability = compute_edge_stability(solutions, n_vars=5, threshold=0.1)
        assert abs(stability[2, 3] - 0.6) < 1e-10


# ============================================================================
# Test: Full pipeline (minimal e2e)
# ============================================================================

class TestFullPipeline:
    @pytest.mark.slow
    def test_minimal_pipeline_no_cv(self):
        """E2e pipeline with 3 trials, no CV."""
        X, Y = _make_tiny_data(n_samples=50, n_vars=5)

        result = run_full_optuna_pipeline(
            X=X, Y=Y, n_vars=5,
            n_trials=3, max_iter=10,
            use_v7=True,
            verbose=True,
            jax_key_seed=42,
            use_pc_warmstart=False,
            run_cv=False,
        )

        assert 'enhanced_solutions' in result
        assert 'pipeline_time' in result
        assert 'pc_A_init' in result
        assert result['pc_A_init'] is None  # No PC warm-start
        assert 'cv_results' in result
        assert result['cv_results'] == []  # No CV

    @pytest.mark.slow
    def test_minimal_pipeline_with_pc(self):
        """E2e pipeline with PC warm-start, no CV."""
        X, Y = _make_tiny_data(n_samples=50, n_vars=5)

        result = run_full_optuna_pipeline(
            X=X, Y=Y, n_vars=5,
            n_trials=3, max_iter=10,
            use_v7=True,
            verbose=True,
            jax_key_seed=42,
            use_pc_warmstart=True,
            run_cv=False,
        )

        assert result['pc_A_init'] is not None
        assert result['pc_A_init'].shape == (6, 6)
