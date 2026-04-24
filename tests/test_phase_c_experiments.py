"""
Unit tests for Phase C experiments (C.2 + C.3 + C.4 + C.7 + C.9 + C.10 + C.11).

Tests:
- compute_varsortability (4 tests)
- RecordingCallback (4 tests)
- suggest_hyperparams_flat (2 tests)
- Rank correlation helper (2 tests)
- Noise sensitivity data generation (2 tests)
- Pareto stability analysis helpers (2 tests)
- LinearSCM ground truth effects (2 tests)
- Pipeline helpers for C.2 (2 tests)
- ATE quality evaluation C.10 (2 tests)
"""

import os

# Import from experiment scripts
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optuna

from jcce.analysis.pareto_stability import (
    compare_stability_to_ground_truth,
    compute_edge_stability,
)
from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import LinearSCM, SCMConfig
from jcce.structure_learning.optuna_search import (
    PROCESSOR_TYPES,
    suggest_hyperparams,
)
from jcce.utils.metrics import compute_varsortability

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from experiment_c3_multi_fidelity_correlation import (
    RecordingCallback,
    compute_rank_correlations,
)
from experiment_c7_noise_sensitivity import generate_data_at_noise
from experiment_c9_multi_fidelity_ablation import suggest_hyperparams_flat

# ============================================================================
# TestComputeVarsortability
# ============================================================================


class TestComputeVarsortability:
    def test_chain_low_noise_high_varsortability(self):
        """Chain graph with positive weights and low noise -> varsortability > 0.8."""
        # Use positive weight range so variance accumulates monotonically
        dag_config = DAGConfig(
            num_nodes=8,
            graph_type="chain",
            seed=42,
            weight_range=(0.5, 1.5),
        )
        A = generate_dag(dag_config)
        scm = LinearSCM(A, SCMConfig(noise_scale=0.1))
        X = scm.sample(1000, jax.random.PRNGKey(42))
        v = compute_varsortability(np.array(X), np.array(A))
        assert v > 0.8, f"Expected > 0.8, got {v}"

    def test_standardized_near_half(self):
        """Standardized data should have varsortability near 0.5."""
        dag_config = DAGConfig(num_nodes=8, graph_type="chain", seed=42)
        A = generate_dag(dag_config)
        scm = LinearSCM(A, SCMConfig(noise_scale=0.1))
        X = np.array(scm.sample(2000, jax.random.PRNGKey(42)))
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        v = compute_varsortability(X_std, np.array(A))
        assert abs(v - 0.5) < 0.15, f"Expected ~0.5, got {v}"

    def test_no_edges_returns_half(self):
        """Empty graph (no edges) should return 0.5."""
        A = np.zeros((5, 5))
        X = np.random.randn(100, 5)
        v = compute_varsortability(X, A)
        assert v == 0.5

    def test_known_chain_exact(self):
        """Known 3-node chain with controlled variances.

        Chain: 0 -> 1 -> 2 with weights = 1.0
        So X0 = e0, X1 = X0 + e1, X2 = X1 + e2
        var(X0) < var(X1) < var(X2) always.
        A[1,0]=1 (edge 0->1): var(X0) < var(X1) -> consistent
        A[2,1]=1 (edge 1->2): var(X1) < var(X2) -> consistent
        """
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        rng = np.random.RandomState(42)
        e = rng.randn(10000, 3) * 0.5
        X = np.zeros((10000, 3))
        X[:, 0] = e[:, 0]
        X[:, 1] = X[:, 0] + e[:, 1]
        X[:, 2] = X[:, 1] + e[:, 2]
        v = compute_varsortability(X, A)
        assert v == 1.0, f"Expected 1.0, got {v}"


# ============================================================================
# TestRecordingCallback
# ============================================================================


class TestRecordingCallback:
    def test_stores_metrics(self):
        """Callback stores metrics keyed by iteration."""
        cb = RecordingCallback()
        cb(10, {"balanced_accuracy": 0.6, "h_A": 0.5, "loss": 1.0})
        cb(20, {"balanced_accuracy": 0.7, "h_A": 0.3, "loss": 0.8})
        assert 10 in cb.records
        assert 20 in cb.records
        assert cb.records[10]["balanced_accuracy"] == 0.6

    def test_get_at_rung_exact(self):
        """get_at_rung returns exact match."""
        cb = RecordingCallback()
        cb(30, {"balanced_accuracy": 0.65, "h_A": 0.4, "loss": 0.9})
        result = cb.get_at_rung(30)
        assert result is not None
        assert result["balanced_accuracy"] == 0.65

    def test_get_at_rung_nearest_below(self):
        """get_at_rung returns nearest-below when exact rung missing."""
        cb = RecordingCallback()
        cb(10, {"balanced_accuracy": 0.5, "h_A": 0.6, "loss": 1.2})
        cb(20, {"balanced_accuracy": 0.6, "h_A": 0.5, "loss": 1.0})
        cb(40, {"balanced_accuracy": 0.7, "h_A": 0.3, "loss": 0.8})
        result = cb.get_at_rung(30)  # Between 20 and 40
        assert result is not None
        assert result["balanced_accuracy"] == 0.6  # From iter 20

    def test_get_at_rung_none_for_early(self):
        """get_at_rung returns None when no records exist at or before rung."""
        cb = RecordingCallback()
        cb(50, {"balanced_accuracy": 0.7, "h_A": 0.3, "loss": 0.8})
        result = cb.get_at_rung(30)  # Before any record
        assert result is None


# ============================================================================
# TestSuggestHyperparamsFlat
# ============================================================================


class TestSuggestHyperparamsFlat:
    def test_flat_suggests_all_processors(self):
        """Flat trial suggests params for ALL 5 processors."""
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        suggested_keys = []

        def obj(trial):
            config = suggest_hyperparams_flat(trial, use_v7=False)
            suggested_keys.append(set(trial.params.keys()))
            return 0.0, 0.0

        study.optimize(obj, n_trials=1)

        keys = suggested_keys[0]
        # Should have params for all processors
        for pt in PROCESSOR_TYPES:
            pt_keys = [k for k in keys if k.startswith(f"{pt}_")]
            assert len(pt_keys) > 0, f"No params suggested for {pt}"

    def test_flat_has_more_params_than_conditional(self):
        """Flat space produces more total suggested params than conditional."""
        study_flat = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )
        study_cond = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        flat_count = []
        cond_count = []

        def obj_flat(trial):
            suggest_hyperparams_flat(trial, use_v7=False)
            flat_count.append(len(trial.params))
            return 0.0, 0.0

        def obj_cond(trial):
            suggest_hyperparams(trial, use_v7=False)
            cond_count.append(len(trial.params))
            return 0.0, 0.0

        study_flat.optimize(obj_flat, n_trials=5)
        study_cond.optimize(obj_cond, n_trials=5)

        avg_flat = np.mean(flat_count)
        avg_cond = np.mean(cond_count)
        assert avg_flat > avg_cond, (
            f"Flat ({avg_flat:.0f}) should have more params than conditional ({avg_cond:.0f})"
        )


# ============================================================================
# TestRankCorrelation
# ============================================================================


class TestRankCorrelation:
    def _make_mock_results(self, n_trials, rung_final_fn):
        """Create mock trial results with controlled rank ordering.

        rung_final_fn(i) -> (rung_value, final_value) for trial i.
        """
        results = []
        for i in range(n_trials):
            rv, fv = rung_final_fn(i)
            cb = RecordingCallback()
            cb(30, {"balanced_accuracy": rv, "h_A": rv, "loss": rv})
            cb(90, {"balanced_accuracy": rv, "h_A": rv, "loss": rv})
            results.append(
                {
                    "config": {},
                    "records": cb.records,
                    "callback": cb,
                    "final_metrics": {
                        "balanced_accuracy": fv,
                        "h_A": fv,
                        "loss": fv,
                        "f1": fv,
                        "shd": fv,
                    },
                    "A_est": np.zeros((5, 5)),
                }
            )
        return results

    def test_identical_rankings_rho_one(self):
        """Identical rankings should give rho = 1.0."""
        results = self._make_mock_results(10, lambda i: (float(i), float(i)))
        corr = compute_rank_correlations(results, [30, 90])
        # Check balanced_accuracy@30 -> balanced_accuracy@final
        key = ("balanced_accuracy", 30, "balanced_accuracy")
        assert key in corr
        rho, pval = corr[key]
        assert abs(rho - 1.0) < 1e-6, f"Expected rho=1.0, got {rho}"

    def test_random_rankings_low_rho(self):
        """Random (unrelated) rankings should give |rho| < 0.5."""
        rng = np.random.RandomState(42)
        rung_vals = list(range(20))
        final_vals = list(range(20))
        rng.shuffle(final_vals)

        results = self._make_mock_results(20, lambda i: (float(rung_vals[i]), float(final_vals[i])))
        corr = compute_rank_correlations(results, [30])
        key = ("balanced_accuracy", 30, "balanced_accuracy")
        assert key in corr
        rho, _ = corr[key]
        # With random permutation, expect low correlation
        # (may occasionally fail by chance, but seed is fixed)
        assert abs(rho) < 0.5, f"Expected |rho| < 0.5, got {rho}"


# ============================================================================
# TestNoiseSensitivity (C.7)
# ============================================================================


class TestNoiseSensitivity:
    def test_generate_data_returns_correct_shapes(self):
        """Data generation returns (X, Y) with correct shapes."""
        dag_config = DAGConfig(num_nodes=6, graph_type="erdos_renyi", seed=42)
        A = generate_dag(dag_config)
        X, Y = generate_data_at_noise(
            A, n_samples=100, noise_scale=0.5, noise_type="gaussian", seed=42
        )
        assert X.shape == (100, 5), f"X shape: {X.shape}"
        assert Y.shape == (100,), f"Y shape: {Y.shape}"
        # Y should be binary
        unique = np.unique(np.array(Y))
        assert set(unique).issubset({0.0, 1.0})

    def test_higher_noise_increases_variance(self):
        """Higher noise scale produces data with higher variance."""
        dag_config = DAGConfig(num_nodes=6, graph_type="erdos_renyi", seed=42)
        A = generate_dag(dag_config)
        X_low, _ = generate_data_at_noise(A, 500, 0.1, "gaussian", seed=42)
        X_high, _ = generate_data_at_noise(A, 500, 2.0, "gaussian", seed=42)
        var_low = float(jnp.var(X_low))
        var_high = float(jnp.var(X_high))
        assert var_high > var_low, f"var_high={var_high} should > var_low={var_low}"


# ============================================================================
# TestParetoStabilityAnalysis (C.4)
# ============================================================================


class TestParetoStabilityAnalysis:
    def _make_mock_solutions(self, n_solutions, n_vars, true_edges):
        """Create mock Pareto solutions with known edge structure."""
        solutions = []
        n = n_vars + 1
        for i in range(n_solutions):
            A = np.zeros((n, n))
            for r, c in true_edges:
                # Add some noise to weight
                A[r, c] = 0.5 + 0.3 * np.random.RandomState(i + r * 10 + c).randn()
            solutions.append(
                {
                    "metrics": {"structure_A_est": A},
                    "objectives": np.array([0.7 + 0.01 * i, 0.8 - 0.02 * i]),
                }
            )
        return solutions

    def test_consistent_edges_high_stability(self):
        """Edges present in all solutions should have stability = 1.0."""
        true_edges = [(1, 0), (2, 1)]  # edges present in all
        solutions = self._make_mock_solutions(5, 4, true_edges)
        stab = compute_edge_stability(solutions, 4, threshold=0.1)
        for r, c in true_edges:
            assert stab[r, c] >= 0.8, f"Edge ({r},{c}) stability={stab[r, c]}"

    def test_stability_ground_truth_comparison(self):
        """compare_stability_to_ground_truth returns valid metrics."""
        true_edges = [(1, 0), (2, 1), (3, 0)]
        solutions = self._make_mock_solutions(5, 4, true_edges)
        stab = compute_edge_stability(solutions, 4, threshold=0.1)

        # Build ground truth (including Y node = last)
        A_true = np.zeros((5, 5))
        for r, c in true_edges:
            A_true[r, c] = 1.0

        result = compare_stability_to_ground_truth(stab, A_true, min_frequency=0.5)
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
        assert result["n_true"] == len(true_edges)


# ============================================================================
# TestLinearSCMEffects (C.10)
# ============================================================================



class TestLinearSCMEffects:
    def test_chain_total_effects(self):
        """Chain 0->1->2: total effect of 0 on 2 = product of edge weights."""
        # A[i,j] != 0 means j -> i
        A = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],  # 0 -> 1 with weight 0.8
                [0.0, 0.5, 0.0],
            ]
        )  # 1 -> 2 with weight 0.5
        scm = LinearSCM(A, SCMConfig())
        B = scm.compute_total_effects()
        # Direct effect 0->1 = 0.8
        assert abs(float(B[1, 0]) - 0.8) < 1e-5
        # Total effect 0->2 = 0.8 * 0.5 = 0.4
        assert abs(float(B[2, 0]) - 0.4) < 1e-5
        # Self-effect = 1.0
        assert abs(float(B[0, 0]) - 1.0) < 1e-5

    def test_no_effect_for_disconnected(self):
        """No causal path => total effect = 0."""
        # 0 -> 1, 2 is isolated
        A = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        scm = LinearSCM(A, SCMConfig())
        B = scm.compute_total_effects()
        # No path from 0 to 2
        assert abs(float(B[2, 0])) < 1e-5
        # No path from 2 to 1
        assert abs(float(B[1, 2])) < 1e-5


# ============================================================================
# TestPipelineHelpers (C.2)
# ============================================================================

from experiment_c2_unified_vs_pipeline import (
    extract_mb_from_adjacency,
    ols_ate,
)


class TestPipelineHelpers:
    def test_extract_mb_correct(self):
        """MB extraction finds parents, children, and spouses."""
        # 5 nodes: 0->2, 1->2, 2->3 (target=3, MB={2, plus spouses of children})
        A = np.zeros((5, 5))
        A[2, 0] = 1.0  # 0 -> 2
        A[2, 1] = 1.0  # 1 -> 2
        A[3, 2] = 1.0  # 2 -> 3
        A[4, 3] = 1.0  # 3 -> 4

        # MB(3) = parents(3) ∪ children(3) ∪ spouses(3)
        # parents(3) = {2}, children(3) = {4}, spouses(3) = {} (no other parents of 4)
        mb = extract_mb_from_adjacency(A, target_idx=3, n_vars=5, threshold=0.5)
        assert 2 in mb, "Parent 2 should be in MB"
        assert 4 in mb, "Child 4 should be in MB"

    def test_ols_ate_recovers_linear_effect(self):
        """OLS ATE recovers the true coefficient in a simple linear model."""
        rng = np.random.RandomState(42)
        n = 1000
        T = rng.randn(n)
        Y = 0.7 * T + rng.randn(n) * 0.3  # True ATE = 0.7
        X = np.column_stack([T, rng.randn(n)])

        ate = ols_ate(X, Y, treatment_idx=0, confounders=[])
        assert abs(ate - 0.7) < 0.1, f"OLS ATE={ate}, expected ~0.7"


# ============================================================================
# TestATEQualityEvaluation (C.10)
# ============================================================================

from experiment_c10_ate_quality import (
    compute_ground_truth_ates,
    evaluate_ate_quality,
)


class TestATEQualityEvaluation:
    def test_ground_truth_chain(self):
        """Ground truth ATEs computed correctly for chain graph."""
        # 0->1->2 (Y=2), effect of 0 on 2 = product
        A = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.8, 0.0]])
        true_ates = compute_ground_truth_ates(A, Y_idx=2)
        assert abs(true_ates[0] - 0.4) < 1e-5  # 0.5 * 0.8
        assert abs(true_ates[1] - 0.8) < 1e-5  # direct

    def test_evaluate_with_mock_solutions(self):
        """evaluate_ate_quality returns valid metrics for mock solutions."""
        true_ates = {0: 0.5, 1: -0.3, 2: 0.0}
        solutions = [
            {
                "objectives": np.array([0.7, 0.8]),
                "metrics": {
                    "causal_effects": {
                        "X0->Y": 0.45,  # close to 0.5
                        "X1->Y": -0.25,  # close to -0.3
                    },
                },
            }
        ]
        results = evaluate_ate_quality(solutions, true_ates, n_vars=3)
        assert len(results) == 1
        r = results[0]
        assert r["has_effects"]
        assert r["n_matched"] == 2
        assert r["pehe"] < 0.1  # Close predictions
        assert r["mean_ate_error"] < 0.1


# ============================================================================
# E.1: Scaling Experiments
# ============================================================================


class TestScalingHelpers:
    """Tests for experiment_e1_scaling.py helpers."""

    def test_generate_scaling_data_shapes(self):
        """generate_scaling_data returns correct shapes for linear SCM."""
        from scripts.experiment_e1_scaling import generate_scaling_data

        n_vars, n_samples = 5, 50
        X, Y, A_true, n_edges, vs = generate_scaling_data(
            n_vars=n_vars,
            n_samples=n_samples,
            graph_type="erdos_renyi",
            expected_degree=2.0,
            scm_type="linear",
            noise_scale=0.5,
            seed=42,
        )
        assert X.shape == (n_samples, n_vars)
        assert Y.shape == (n_samples,)
        assert A_true.shape == (n_vars, n_vars)
        assert isinstance(n_edges, int)
        assert n_edges >= 0
        assert 0.0 <= vs <= 1.0

    def test_generate_scaling_data_nonlinear(self):
        """generate_scaling_data works with nonlinear MLP SCM."""
        from scripts.experiment_e1_scaling import generate_scaling_data

        n_vars, n_samples = 5, 50
        X, Y, A_true, n_edges, vs = generate_scaling_data(
            n_vars=n_vars,
            n_samples=n_samples,
            graph_type="erdos_renyi",
            expected_degree=2.0,
            scm_type="nonlinear",
            noise_scale=0.5,
            seed=42,
        )
        assert X.shape == (n_samples, n_vars)
        assert Y.shape == (n_samples,)
        # Y should be binary
        unique_vals = set(float(v) for v in Y)
        assert unique_vals <= {0.0, 1.0}

    def test_generate_scaling_data_scale_free(self):
        """generate_scaling_data works with scale-free graphs."""
        from scripts.experiment_e1_scaling import generate_scaling_data

        X, Y, A_true, n_edges, vs = generate_scaling_data(
            n_vars=8,
            n_samples=50,
            graph_type="scale_free",
            expected_degree=2.0,
            scm_type="linear",
            noise_scale=0.5,
            seed=42,
        )
        assert X.shape == (50, 8)
        assert n_edges >= 0
