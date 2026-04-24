"""
Unit tests for Optuna outer loop (Phase A.1).

Tests search space, constraints, pruning, and result format compatibility.
"""

import jax.numpy as jnp
import numpy as np
import optuna
import pytest

from jcce.structure_learning.optuna_search import (
    PROCESSOR_TYPES,
    PruningTracker,
    _store_artifacts,
    constraints_func,
    extract_pareto_solutions,
    run_optuna_search,
    suggest_hyperparams,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_study():
    """Create a simple study for testing."""
    return optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.RandomSampler(seed=42),
    )


def _make_tiny_data(n_samples=50, n_vars=5, seed=42):
    """Create tiny synthetic data for fast tests."""
    rng = np.random.RandomState(seed)
    X = jnp.array(rng.randn(n_samples, n_vars).astype(np.float32))
    Y = jnp.array((rng.randn(n_samples) > 0).astype(np.float32))
    return X, Y


# ============================================================================
# Test: suggest_hyperparams for all processor types
# ============================================================================


class TestSuggestHyperparams:
    def test_all_processors_produce_valid_config(self):
        """Each of 5 processor types produces valid config with correct keys."""
        for proc_type in PROCESSOR_TYPES:
            study = _make_study()

            def obj(trial):
                # Force specific processor type
                trial.suggest_categorical("processor_type", [proc_type])
                # Let the rest be suggested normally (but we need to call
                # suggest_hyperparams with the trial that already has processor_type)
                return 0.0, 0.0

            # Instead, use the real function and check output
            configs_seen = []

            def obj2(trial):
                config = suggest_hyperparams(trial, use_v7=True)
                configs_seen.append(config)
                return 0.0, 0.0

            study.optimize(obj2, n_trials=20)

            # Check that at least some configs were generated
            assert len(configs_seen) == 20

            # Check all configs have required keys
            for config in configs_seen:
                assert "processor_type" in config
                assert config["processor_type"] in PROCESSOR_TYPES
                assert "processor_config" in config
                assert "lambda_1" in config
                assert "lambda_2" in config
                assert "lambda_class" in config
                assert "lr" in config
                # v7 params
                assert "effect_hidden_dim" in config
                assert "effect_embed_dim" in config
                assert "lambda_effect" in config

    def test_conditional_space_elm_no_transformer_params(self):
        """ELM trial doesn't suggest transformer params and vice versa."""
        study = _make_study()
        configs = []

        def obj(trial):
            config = suggest_hyperparams(trial, use_v7=False)
            configs.append((config, dict(trial.params)))
            return 0.0, 0.0

        study.optimize(obj, n_trials=50)

        for config, params in configs:
            proc = config["processor_type"]
            if proc == "elm":
                # Should have elm-specific params
                assert "elm_hidden_dim" in params
                assert "elm_n_hidden_nodes" in params
                # Should NOT have transformer or mamba params
                assert "transformer_d_model" not in params
                assert "transformer_n_heads" not in params
                assert "mamba_d_model" not in params
                assert "mamba_d_state" not in params
            elif proc == "transformer":
                assert "transformer_d_model" in params
                assert "transformer_n_heads" in params
                assert "elm_hidden_dim" not in params
                assert "mamba_d_model" not in params

    def test_continuous_params_in_range(self):
        """lambda_1, lambda_2, lr within specified bounds."""
        study = _make_study()
        configs = []

        def obj(trial):
            config = suggest_hyperparams(trial, use_v7=False)
            configs.append(config)
            return 0.0, 0.0

        study.optimize(obj, n_trials=50)

        for config in configs:
            # lambda_1 upper = 2.0 * max(1.0, (n_vars/12)^2); default n_vars=12 -> 2.0
            assert 0.001 <= config["lambda_1"] <= 2.0
            assert 0.001 <= config["lambda_2"] <= 1.0
            assert 0.0001 <= config["lr"] <= 0.01
            assert config["lambda_class"] in [0.1, 0.5, 1.0, 2.0, 5.0]

    def test_processor_config_has_correct_keys(self):
        """Processor config contains expected architecture-specific keys."""
        expected_keys = {
            "elm": {"hidden_dim", "n_hidden_nodes", "activation"},
            "gnn": {"hidden_dim", "n_layers", "gnn_type", "sage_aggregation"},
            "mlp": {"hidden_dim", "n_layers", "activation"},
            "transformer": {"d_model", "n_heads", "n_layers", "d_ff"},
            "mamba": {"d_model", "d_state", "d_conv", "expand"},
        }

        study = _make_study()
        configs = []

        def obj(trial):
            config = suggest_hyperparams(trial, use_v7=False)
            configs.append(config)
            return 0.0, 0.0

        study.optimize(obj, n_trials=100)

        # Group by processor type
        by_type = {}
        for config in configs:
            pt = config["processor_type"]
            by_type.setdefault(pt, []).append(config)

        for pt, pt_configs in by_type.items():
            for config in pt_configs:
                pc = config["processor_config"]
                assert set(pc.keys()) == expected_keys[pt], (
                    f"{pt}: got {set(pc.keys())}, expected {expected_keys[pt]}"
                )


# ============================================================================
# Test: constraints_func
# ============================================================================


class TestConstraintsFunc:
    def test_feasible_when_h_A_below_threshold(self):
        """h_A < 0.1 → constraint value <= 0 (feasible)."""
        study = _make_study()

        def obj(trial):
            trial.set_user_attr("h_A", 0.05)
            return 0.5, 0.5

        study.optimize(obj, n_trials=1)
        trial = study.trials[0]
        result = constraints_func(trial)
        assert len(result) == 1
        assert result[0] <= 0.0  # feasible

    def test_infeasible_when_h_A_above_threshold(self):
        """h_A > 0.1 → constraint value > 0 (infeasible)."""
        study = _make_study()

        def obj(trial):
            trial.set_user_attr("h_A", 0.5)
            return 0.5, 0.5

        study.optimize(obj, n_trials=1)
        trial = study.trials[0]
        result = constraints_func(trial)
        assert result[0] > 0.0  # infeasible

    def test_missing_h_A_is_infeasible(self):
        """No h_A stored → defaults to 1.0 → infeasible."""
        study = _make_study()

        def obj(trial):
            return 0.5, 0.5

        study.optimize(obj, n_trials=1)
        trial = study.trials[0]
        result = constraints_func(trial)
        assert result[0] > 0.0


# ============================================================================
# Test: extract_pareto_solutions filters infeasible
# ============================================================================


class TestExtractPareto:
    def test_filters_infeasible_solutions(self):
        """h_A > 0.1 solutions are excluded from Pareto set."""
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        feasible_count = 0
        infeasible_count = 0

        def obj(trial):
            nonlocal feasible_count, infeasible_count
            config = suggest_hyperparams(trial, use_v7=False)
            # Alternate feasible/infeasible
            if trial.number % 2 == 0:
                trial.set_user_attr("h_A", 0.01)  # feasible
                _store_artifacts(
                    trial.number,
                    {
                        "A_est": np.eye(6),
                        "config": config,
                        "markov_blanket": [0, 1],
                    },
                )
                feasible_count += 1
            else:
                trial.set_user_attr("h_A", 0.5)  # infeasible
                _store_artifacts(
                    trial.number,
                    {
                        "A_est": np.eye(6),
                        "config": config,
                        "markov_blanket": [0, 1],
                    },
                )
                infeasible_count += 1
            return 0.5, 0.5

        study.optimize(obj, n_trials=10)

        enhanced, pareto = extract_pareto_solutions(study, use_v7=False)

        # Only feasible solutions should be in Pareto set
        for sol in enhanced:
            assert sol["metrics"]["structure_h_A"] < 0.1


# ============================================================================
# Test: callback pruning
# ============================================================================


class TestCallbackPruning:
    def test_pruning_tracker_prunes_low_accuracy(self):
        """PruningTracker prunes trials below 25th percentile after startup."""
        tracker = PruningTracker(n_startup_trials=3, percentile=25.0)

        # Report high accuracies for startup trials
        for acc in [0.8, 0.9, 0.85]:
            tracker.report(acc, rung=30)
            tracker.mark_completed()

        # After startup, a very low accuracy should be pruned
        assert tracker.should_prune(0.1, rung=30) is True
        # A good accuracy should NOT be pruned
        assert tracker.should_prune(0.9, rung=30) is False

    def test_pruning_tracker_no_prune_during_startup(self):
        """PruningTracker does not prune during startup phase."""
        tracker = PruningTracker(n_startup_trials=5, percentile=25.0)

        tracker.report(0.9, rung=30)
        tracker.mark_completed()

        # Even with terrible accuracy, no pruning during startup
        assert tracker.should_prune(0.0, rung=30) is False

    def test_pruned_trial_in_study(self):
        """Manual pruning via PruningTracker → trial state is PRUNED."""
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        tracker = PruningTracker(n_startup_trials=0, percentile=50.0)
        # Seed tracker with high values so low values get pruned
        for _ in range(5):
            tracker.report(0.9, rung=0)
            tracker.mark_completed()

        def obj(trial):
            # This trial has terrible accuracy → should be pruned
            if tracker.should_prune(0.01, rung=0):
                raise optuna.TrialPruned("Manual prune")
            return 0.5, 0.5

        study.optimize(obj, n_trials=3)

        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        assert len(pruned) == 3


# ============================================================================
# Test: result format compatibility
# ============================================================================


class TestResultFormat:
    def test_enhanced_solutions_has_required_keys(self):
        """Enhanced solutions have all keys expected by post-hoc pipeline."""
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        required_metrics_keys = [
            "classification_balanced_accuracy",
            "classification_f1",
            "classification_roc_auc",
            "mb_sparsity",
            "mb_size",
            "mb_indices",
            "markov_blanket",
            "structure_n_edges",
            "structure_h_A",
            "structure_A_est",
            "processor_type",
            "processor_config",
            "lambda_1",
            "lambda_2",
            "lambda_class",
            "lr",
            "fitness",
        ]

        def obj(trial):
            config = suggest_hyperparams(trial, use_v7=True)
            trial.set_user_attr("h_A", 0.01)
            trial.set_user_attr("balanced_accuracy", 0.8)
            trial.set_user_attr("mb_sparsity", 0.6)
            trial.set_user_attr("n_edges", 5)
            trial.set_user_attr("classification_accuracy", 0.8)
            trial.set_user_attr("classification_f1", 0.75)
            trial.set_user_attr("classification_precision", 0.8)
            trial.set_user_attr("classification_recall", 0.7)
            trial.set_user_attr("classification_roc_auc", 0.85)
            trial.set_user_attr("effect_loss", 0.1)
            trial.set_user_attr("bow_loss", 0.05)
            trial.set_user_attr("n_confound_edges", 2)
            _store_artifacts(
                trial.number,
                {
                    "A_est": np.eye(6),
                    "config": config,
                    "markov_blanket": [0, 1, 3],
                    "causal_effects": {"0": 0.5},
                    "A_confound": np.zeros((6, 6)),
                    "A_weights": np.eye(6) * 0.5,
                },
            )
            return 0.8, 0.6

        study.optimize(obj, n_trials=3)

        enhanced, pareto = extract_pareto_solutions(study, use_v7=True)

        assert len(enhanced) > 0, "Expected at least one Pareto solution"

        for sol in enhanced:
            # Top-level keys
            assert "genome" in sol
            assert "objectives" in sol
            assert "metrics" in sol
            assert isinstance(sol["objectives"], np.ndarray)

            # Metrics keys
            for key in required_metrics_keys:
                assert key in sol["metrics"], f"Missing key: {key}"

            # v7-specific keys
            assert "effect_loss" in sol["metrics"]
            assert "causal_effects" in sol["metrics"]
            assert "effect_hidden_dim" in sol["metrics"]
            assert "lambda_effect" in sol["metrics"]


# ============================================================================
# Test: run_optuna_search minimal e2e
# ============================================================================


class TestRunOptunaSearch:
    @pytest.mark.slow
    def test_minimal_e2e(self):
        """E2e with n_trials=3, max_iter=10 on tiny data — result dict has correct keys."""
        X, Y = _make_tiny_data(n_samples=50, n_vars=5)

        result = run_optuna_search(
            X=X,
            Y=Y,
            n_vars=5,
            n_trials=3,
            max_iter=10,
            use_v7=True,
            verbose=True,
            jax_key_seed=42,
        )

        # Check return dict keys
        assert "pareto_front" in result
        assert "enhanced_solutions" in result
        assert "evaluation_cache" in result
        assert "use_v7" in result
        assert "true_mb" in result
        assert "study" in result
        assert "n_trials_completed" in result
        assert "n_trials_pruned" in result

        assert result["use_v7"] is True
        assert isinstance(result["study"], optuna.Study)
        assert (
            result["n_trials_completed"]
            + result["n_trials_pruned"]
            + result.get("n_trials_failed", 0)
            >= 3
        )
