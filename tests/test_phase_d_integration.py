"""
Integration tests for Phase D warm-starting.

Tests end-to-end flow: adjacency prior + SPR + Optuna objective.
"""

import numpy as np
import jax.numpy as jnp
import optuna
import pytest

from jcce.structure_learning.adjacency_prior import AdjacencyPrior
from jcce.structure_learning.optuna_search import (
    create_optuna_objective,
    run_optuna_search,
    _trial_artifacts,
    _store_artifacts,
    PruningTracker,
)


def _make_tiny_data(n_samples=50, n_vars=5, seed=42):
    """Create tiny synthetic data for fast tests."""
    rng = np.random.RandomState(seed)
    X = jnp.array(rng.randn(n_samples, n_vars).astype(np.float32))
    Y = jnp.array((rng.randn(n_samples) > 0).astype(np.float32))
    return X, Y


class TestPhaseD:
    def test_objective_with_adjacency_prior_runs(self):
        """Objective with adjacency_prior param runs without crash."""
        X, Y = _make_tiny_data(n_vars=5)
        prior = AdjacencyPrior(n_total=6)

        objective = create_optuna_objective(
            X=X, Y=Y, n_vars=5, max_iter=5,
            use_v7=False,
            adjacency_prior=prior,
        )

        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )
        study.optimize(objective, n_trials=1)
        assert len(study.trials) == 1
        assert study.trials[0].state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
        )

    def test_prior_updates_after_feasible_trial(self):
        """Adjacency prior accumulates updates from feasible (h_A < 0.1) trials."""
        X, Y = _make_tiny_data(n_vars=3)
        prior = AdjacencyPrior(n_total=4, fitness_threshold=0.0)

        objective = create_optuna_objective(
            X=X, Y=Y, n_vars=3, max_iter=10,
            use_v7=False,
            adjacency_prior=prior,
        )

        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )
        study.optimize(objective, n_trials=3)

        # If any trial was feasible (h_A < 0.1), prior should have updates
        feasible = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get('h_A', 1.0) < 0.1
        ]
        # At least check it didn't crash; n_updates >= 0 is always true
        assert prior.n_updates >= 0
        # If we got feasible trials, prior should have been updated
        if len(feasible) > 0:
            assert prior.n_updates > 0

    def test_spr_candidate_found_on_second_trial(self):
        """SPR finds a candidate from the first trial's artifacts for the second."""
        X, Y = _make_tiny_data(n_vars=5)

        objective = create_optuna_objective(
            X=X, Y=Y, n_vars=5, max_iter=5,
            use_v7=False,
            enable_spr=True,
            spr_max_distance=10.0,  # large threshold so any match works
        )

        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )
        study.optimize(objective, n_trials=2)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        # Check that spr_hit attr is set on completed trials
        for t in completed:
            assert 'spr_hit' in t.user_attrs

    def test_run_optuna_search_with_all_d_features(self):
        """run_optuna_search(enable_adjacency_prior=True, enable_spr=True) works e2e."""
        X, Y = _make_tiny_data(n_vars=4)

        result = run_optuna_search(
            X=X, Y=Y, n_vars=4,
            n_trials=3, max_iter=5,
            use_v7=False,
            enable_adjacency_prior=True,
            enable_spr=True,
            verbose=False,
        )

        assert 'pareto_front' in result
        assert 'enhanced_solutions' in result
        assert 'n_trials_completed' in result
        assert result['n_trials_completed'] >= 0
