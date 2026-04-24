"""
Tests for P2 validation fixes.

Covers:
1. Baseline DAG comparison — helpers and data generation
2. Cinelli sensitivity analysis — partial R², RV, bias-adjusted estimates
3. Cross-fitted structure learning — fold agreement and evaluation
4. Refutation suite — existing module integration check
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# =============================================================================
# 1. Baseline DAG Comparison Tests
# =============================================================================


class TestBaselineComparison:
    """Test P2-A baseline DAG comparison components."""

    def test_data_generation(self):
        """generate_data returns correct shapes and types."""
        from p2_baseline_dag_comparison import generate_data

        data_full, X, Y, A_true = generate_data(
            n_vars=5, n_samples=100, expected_degree=2.0, noise_scale=0.5, seed=42
        )
        assert X.shape == (100, 5)
        assert Y.shape == (100,)
        assert A_true.shape == (6, 6)  # n_vars + 1
        assert data_full.shape == (100, 6)

    def test_evaluate_baseline(self):
        """evaluate_baseline returns valid metric dict."""
        from p2_baseline_dag_comparison import evaluate_baseline

        n_vars = 5
        A_est = np.random.randn(n_vars + 1, n_vars + 1) * 0.5
        A_true = np.zeros((n_vars + 1, n_vars + 1))
        A_true[1, 0] = 1.0
        A_true[2, 1] = 1.0

        result = evaluate_baseline("test", A_est, A_true, n_vars, elapsed=1.0)
        assert "f1" in result
        assert "shd" in result
        assert "precision" in result
        assert "recall" in result
        assert result["name"] == "test"
        assert result["time"] == 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_pc_runner_callable(self):
        """run_pc function is importable and callable."""
        from p2_baseline_dag_comparison import run_pc

        assert callable(run_pc)

    def test_ges_runner_callable(self):
        """run_ges function is importable and callable."""
        from p2_baseline_dag_comparison import run_ges

        assert callable(run_ges)

    def test_dagma_runner_callable(self):
        """run_dagma function is importable and callable."""
        from p2_baseline_dag_comparison import run_dagma

        assert callable(run_dagma)


# =============================================================================
# 2. Cinelli Sensitivity Analysis Tests
# =============================================================================


class TestCinelliSensitivity:
    """Test Cinelli & Hazlett (2020) sensitivity analysis."""

    def test_partial_r2_known_relationship(self):
        """Partial R² should be high for strong T->Y relationship."""
        from jcce.validation.cinelli_sensitivity import partial_r2

        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 3)
        T = rng.randn(n)
        Y = 2.0 * T + 0.1 * rng.randn(n)  # Strong T->Y

        pr2 = partial_r2(Y, T, X)
        assert pr2 > 0.5, f"Expected high partial R², got {pr2}"

    def test_partial_r2_no_relationship(self):
        """Partial R² should be near 0 when T has no effect on Y."""
        from jcce.validation.cinelli_sensitivity import partial_r2

        rng = np.random.RandomState(42)
        n = 500
        X = rng.randn(n, 3)
        T = rng.randn(n)
        Y = X @ rng.randn(3) + rng.randn(n)  # Y independent of T given X

        pr2 = partial_r2(Y, T, X)
        assert pr2 < 0.05, f"Expected low partial R², got {pr2}"

    def test_robustness_value_strong_effect(self):
        """Strong effects should have high robustness values."""
        from jcce.validation.cinelli_sensitivity import robustness_value

        # Simulate large t-stat
        rv_0, rv_a = robustness_value(partial_r2_T=0.3, ate=1.0, se=0.1, dof=100)
        assert rv_0 > 0.1, f"Expected RV > 0.1 for strong effect, got {rv_0}"
        assert rv_a > 0.0, f"Expected RV_alpha > 0, got {rv_a}"

    def test_robustness_value_weak_effect(self):
        """Weak effects should have low robustness values."""
        from jcce.validation.cinelli_sensitivity import robustness_value

        # Simulate small t-stat
        rv_0, rv_a = robustness_value(partial_r2_T=0.01, ate=0.05, se=0.1, dof=50)
        assert rv_0 < 0.1, f"Expected low RV for weak effect, got {rv_0}"

    def test_bias_adjusted_estimate(self):
        """Bias-adjusted estimate should be less extreme than original."""
        from jcce.validation.cinelli_sensitivity import bias_adjusted_estimate

        result = bias_adjusted_estimate(
            ate=0.5, se=0.1, r2_Y_confounder=0.1, r2_T_confounder=0.1, partial_r2_T=0.2, dof=100
        )

        assert "adj_estimate" in result
        assert "max_bias" in result
        assert result["max_bias"] > 0
        # Adjusted estimate should be closer to zero
        assert abs(result["adj_estimate"]) < abs(0.5)

    def test_cinelli_sensitivity_full(self):
        """Full cinelli_sensitivity function returns valid SensitivityResult."""
        from jcce.validation.cinelli_sensitivity import cinelli_sensitivity

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 3)
        T = X[:, 0] * 0.5 + rng.randn(n) * 0.5  # Confounded
        Y = 0.5 * T + X @ rng.randn(3) * 0.3 + rng.randn(n) * 0.3

        result = cinelli_sensitivity(
            Y, T, X, treatment_name="test_T", feature_names=["X0", "X1", "X2"]
        )

        assert result.treatment_name == "test_T"
        assert 0.0 <= result.rv <= 1.0
        assert 0.0 <= result.rv_alpha <= 1.0
        assert result.partial_r2_treatment >= 0.0
        assert len(result.benchmarks) > 0

    def test_sensitivity_result_summary(self):
        """SensitivityResult.summary() produces readable output."""
        from jcce.validation.cinelli_sensitivity import cinelli_sensitivity

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 3)
        T = rng.randn(n)
        Y = 0.5 * T + rng.randn(n) * 0.3

        result = cinelli_sensitivity(Y, T, X)
        s = result.summary()
        assert "RV" in s
        assert "ATE" in s

    def test_sensitivity_result_to_dict(self):
        """SensitivityResult.to_dict() returns JSON-serializable dict."""
        from jcce.validation.cinelli_sensitivity import cinelli_sensitivity

        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 2)
        T = rng.randn(n)
        Y = 0.5 * T + rng.randn(n) * 0.3

        result = cinelli_sensitivity(Y, T, X)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "rv" in d
        assert "is_robust" in d

    def test_covariate_benchmarks(self):
        """Covariate benchmarks produce valid R² values."""
        from jcce.validation.cinelli_sensitivity import compute_covariate_benchmarks

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 3)
        T = X[:, 0] * 0.5 + rng.randn(n) * 0.5
        Y = 0.5 * T + X[:, 1] * 0.3 + rng.randn(n) * 0.3

        benchmarks = compute_covariate_benchmarks(
            Y, T, X, feature_names=["X0", "X1", "X2"], k_multipliers=(1.0,)
        )

        assert len(benchmarks) == 3  # One per covariate
        for name, bm in benchmarks.items():
            assert 0.0 <= bm["r2_Y"] <= 1.0, f"{name}: r2_Y={bm['r2_Y']}"
            assert 0.0 <= bm["r2_T"] <= 1.0, f"{name}: r2_T={bm['r2_T']}"


# =============================================================================
# 3. Cross-Fitted Structure Learning Tests
# =============================================================================


class TestCrossFittedStructure:
    """Test P2-C cross-fitted structure learning components."""

    def test_data_generation(self):
        """generate_data returns correct shapes."""
        from p2_cross_fitted_structure_learning import generate_data

        X, Y, A_true = generate_data(
            n_vars=5, n_samples=100, expected_degree=2.0, noise_scale=0.5, seed=42
        )
        assert X.shape == (100, 5)
        assert Y.shape == (100,)
        assert A_true.shape == (6, 6)

    def test_cross_fold_agreement_identical(self):
        """Identical DAGs should have agreement = 1.0."""
        from p2_cross_fitted_structure_learning import compute_cross_fold_agreement

        A = np.zeros((6, 6))
        A[0, 1] = 0.5
        A[1, 2] = 0.8
        As = [A.copy() for _ in range(5)]

        result = compute_cross_fold_agreement(As, n_vars=5)
        assert result["mean_agreement"] == 1.0

    def test_cross_fold_agreement_different(self):
        """Different DAGs should have lower agreement."""
        from p2_cross_fitted_structure_learning import compute_cross_fold_agreement

        rng = np.random.RandomState(42)
        As = []
        for i in range(5):
            A = np.zeros((6, 6))
            # Random edges
            for _ in range(3):
                r, c = rng.randint(0, 5, 2)
                A[r, c] = 0.5
            As.append(A)

        result = compute_cross_fold_agreement(As, n_vars=5)
        assert result["mean_agreement"] < 1.0
        assert "n_stable_edges" in result
        assert "n_ever_edges" in result

    def test_cross_fold_agreement_single_fold(self):
        """Single fold should return agreement = 1.0."""
        from p2_cross_fitted_structure_learning import compute_cross_fold_agreement

        A = np.zeros((6, 6))
        A[0, 1] = 0.5
        result = compute_cross_fold_agreement([A], n_vars=5)
        assert result["mean_agreement"] == 1.0


# =============================================================================
# 4. Refutation Suite Integration Tests
# =============================================================================


class TestRefutationSuite:
    """Verify existing refutation suite works correctly."""

    def test_module_imports(self):
        """Refutation suite imports correctly."""
        from jcce.validation.refutation_suite import (
            RefutationSuite,
            create_simple_effect_estimator,
        )

        assert callable(RefutationSuite)
        assert callable(create_simple_effect_estimator)

    def test_simple_estimator_returns_float(self):
        """Simple effect estimator returns a float."""
        from jcce.validation.refutation_suite import create_simple_effect_estimator

        estimator = create_simple_effect_estimator()
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        T = (rng.randn(100) > 0).astype(float)
        Y = 0.5 * T + rng.randn(100) * 0.3

        effect = estimator(X, T, Y)
        assert isinstance(effect, float)
        assert np.isfinite(effect)

    def test_suite_runs_all_tests(self):
        """RefutationSuite.run_all executes all 4 tests."""
        from jcce.validation.refutation_suite import RefutationSuite, create_simple_effect_estimator

        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        T = (rng.randn(200) > 0).astype(float)
        Y = 0.5 * T + rng.randn(200) * 0.3

        estimator = create_simple_effect_estimator()
        original_effect = estimator(X, T, Y)

        suite = RefutationSuite(n_simulations=20, random_state=42)
        results = suite.run_all(X, T, Y, estimator, original_effect)

        assert results.n_total == 4
        assert "placebo_treatment" in results.results
        assert "random_common_cause" in results.results
        assert "data_subset" in results.results
        assert "dummy_outcome" in results.results

    def test_benjamini_hochberg(self):
        """BH correction works correctly."""
        from jcce.validation.refutation_suite import RefutationSuite

        # All significant
        rejected = RefutationSuite.benjamini_hochberg([0.001, 0.002, 0.003], fdr=0.05)
        assert all(rejected)

        # None significant
        rejected = RefutationSuite.benjamini_hochberg([0.5, 0.6, 0.7], fdr=0.05)
        assert not any(rejected)
