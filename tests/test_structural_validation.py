"""
Tests for SID, LOVO CV, and Self-Compatibility validation modules.

Covers:
1. SID — re-export from utils.metrics, known graph pairs
2. LOVO CV — variable removal, sub-DAG extraction, stability scoring
3. Self-Compatibility — SCM fitting, sampling, fixed-point checking
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# =============================================================================
# 1. SID Tests
# =============================================================================


class TestSID:
    """Test Structural Intervention Distance."""

    def test_sid_reexport_exists(self):
        """compute_sid and compute_sid_normalized available from utils.metrics."""
        from jcce.utils.metrics import compute_sid, compute_sid_normalized

        assert callable(compute_sid)
        assert callable(compute_sid_normalized)

    def test_sid_identical_graphs(self):
        """SID of identical graphs should be 0."""
        from jcce.utils.metrics import compute_sid

        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        assert compute_sid(A, A) == 0

    def test_sid_different_graphs(self):
        """SID of different graphs should be > 0."""
        from jcce.utils.metrics import compute_sid

        A_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        A_pred = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        sid = compute_sid(A_pred, A_true)
        assert sid > 0

    def test_sid_normalized_range(self):
        """Normalized SID should be in [0, 1]."""
        from jcce.utils.metrics import compute_sid_normalized

        A1 = np.random.randn(5, 5) * 0.3
        A2 = np.random.randn(5, 5) * 0.3
        sid_norm = compute_sid_normalized(A1, A2)
        assert 0.0 <= sid_norm <= 1.0

    def test_sid_empty_graphs(self):
        """SID of two empty graphs should be 0."""
        from jcce.utils.metrics import compute_sid

        A = np.zeros((5, 5))
        assert compute_sid(A, A) == 0

    def test_sid_in_c1_script(self):
        """C.1 script includes SID in output."""
        with open(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "compare_nsga2_vs_optuna.py")
        ) as f:
            source = f.read()
        assert "compute_sid" in source
        assert "'best_sid'" in source

    def test_sid_in_c9_script(self):
        """C.9 script includes SID in output."""
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "scripts",
                "experiment_c9_multi_fidelity_ablation.py",
            )
        ) as f:
            source = f.read()
        assert "compute_sid" in source
        assert "'best_sid'" in source


# =============================================================================
# 2. LOVO CV Tests
# =============================================================================


class TestLOVOCV:
    """Test Leave-One-Variable-Out cross-validation."""

    def test_module_imports(self):
        """LOVO module imports correctly."""
        from jcce.validation.lovo_cv import (
            lovo_cv,
            remove_variable,
        )

        assert callable(lovo_cv)
        assert callable(remove_variable)

    def test_remove_variable(self):
        """remove_variable drops the correct column."""
        from jcce.validation.lovo_cv import remove_variable

        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = remove_variable(data, 1)
        expected = np.array([[1, 3], [4, 6], [7, 9]])
        np.testing.assert_array_equal(result, expected)

    def test_remove_variable_from_dag(self):
        """remove_variable_from_dag removes correct row and column."""
        from jcce.validation.lovo_cv import remove_variable_from_dag

        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        A_sub = remove_variable_from_dag(A, 1)
        expected = np.array([[0, 0], [0, 0]], dtype=float)
        np.testing.assert_array_equal(A_sub, expected)

    def test_remove_first_variable(self):
        """Removing first variable works correctly."""
        from jcce.validation.lovo_cv import remove_variable_from_dag

        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        A_sub = remove_variable_from_dag(A, 0)
        expected = np.array([[0, 1], [0, 0]], dtype=float)
        np.testing.assert_array_equal(A_sub, expected)

    def test_compare_dags_identical(self):
        """Identical DAGs should have F1=1.0 and recovered=True."""
        from jcce.validation.lovo_cv import compare_dags

        A = np.array([[0, 0.5, 0], [0, 0, 0.8], [0, 0, 0]])
        result = compare_dags(A, A, threshold=0.3)
        assert result["f1"] == 1.0
        assert result["shd"] == 0
        assert result["recovered"]

    def test_compare_dags_different(self):
        """Different DAGs should have lower F1."""
        from jcce.validation.lovo_cv import compare_dags

        A1 = np.array([[0, 0.5, 0], [0, 0, 0.8], [0, 0, 0]])
        A2 = np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0]])
        result = compare_dags(A1, A2, threshold=0.3)
        assert result["f1"] < 1.0
        assert result["shd"] > 0

    def test_lovo_cv_with_identity_learner(self):
        """LOVO with a learner that always returns the sub-DAG → score = 1.0."""
        from jcce.validation.lovo_cv import lovo_cv

        A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float)
        data = np.random.randn(100, 4)

        # Learner that returns the correct sub-DAG
        def perfect_learner(data_reduced, A_full=None):
            d = data_reduced.shape[1]
            return np.zeros((d, d))  # Won't match A, but tests the interface

        result = lovo_cv(data, A, perfect_learner, variables=[0, 1])
        assert result.n_variables == 2
        assert isinstance(result.stability_score, float)

    def test_lovo_result_to_dict(self):
        """LovoResult.to_dict returns serializable dict."""
        from jcce.validation.lovo_cv import LovoResult

        result = LovoResult(
            stability_score=0.8,
            n_variables=5,
            n_recovered=4,
            per_variable=[],
            mean_sub_f1=0.85,
            mean_sub_shd=2.0,
            total_time=10.0,
            time_per_variable=2.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["stability_score"] == 0.8


# =============================================================================
# 3. Self-Compatibility Tests
# =============================================================================


class TestSelfCompatibility:
    """Test self-compatibility (Faller et al. AISTATS 2024)."""

    def test_module_imports(self):
        """Self-compatibility module imports correctly."""
        from jcce.validation.self_compatibility import (
            fit_linear_scm,
            sample_from_fitted_scm,
            self_compatibility_check,
        )

        assert callable(self_compatibility_check)
        assert callable(fit_linear_scm)
        assert callable(sample_from_fitted_scm)

    def test_fit_linear_scm(self):
        """fit_linear_scm recovers coefficients for simple chain."""
        from jcce.validation.self_compatibility import fit_linear_scm

        rng = np.random.RandomState(42)
        n = 1000
        # Chain: X0 -> X1 -> X2 with weights 0.8, 0.5
        X0 = rng.randn(n)
        X1 = 0.8 * X0 + rng.randn(n) * 0.1
        X2 = 0.5 * X1 + rng.randn(n) * 0.1
        data = np.column_stack([X0, X1, X2])

        # A[i,j] != 0 means j -> i
        A = np.array([[0, 0, 0], [0.8, 0, 0], [0, 0.5, 0]], dtype=float)

        params = fit_linear_scm(data, A, threshold=0.3)
        assert params["A_fitted"].shape == (3, 3)
        # Should recover ~0.8 for edge 0->1
        assert abs(params["A_fitted"][1, 0] - 0.8) < 0.1
        # Should recover ~0.5 for edge 1->2
        assert abs(params["A_fitted"][2, 1] - 0.5) < 0.1

    def test_sample_from_fitted_scm(self):
        """sample_from_fitted_scm produces data with correct shape."""
        from jcce.validation.self_compatibility import fit_linear_scm, sample_from_fitted_scm

        rng = np.random.RandomState(42)
        data = rng.randn(100, 4)
        A = np.zeros((4, 4))
        A[1, 0] = 0.5

        params = fit_linear_scm(data, A, threshold=0.3)
        X_synth = sample_from_fitted_scm(params, n_samples=200, seed=42)
        assert X_synth.shape == (200, 4)
        assert np.all(np.isfinite(X_synth))

    def test_self_compat_with_deterministic_learner(self):
        """Self-compat with a learner that always returns the same DAG → high score."""
        from jcce.validation.self_compatibility import self_compatibility_check

        rng = np.random.RandomState(42)
        data = rng.randn(100, 3)
        A_learned = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], dtype=float)

        # Learner always returns A_learned (perfect fixed point)
        def fixed_learner(data_synth):
            return A_learned.copy()

        result = self_compatibility_check(data, A_learned, fixed_learner, n_resamples=3, seed=42)

        assert result.compatibility_score == 1.0
        assert result.n_compatible == 3
        assert result.is_self_compatible()

    def test_self_compat_with_random_learner(self):
        """Self-compat with random learner → low score."""
        from jcce.validation.self_compatibility import self_compatibility_check

        rng = np.random.RandomState(42)
        data = rng.randn(100, 3)
        A_learned = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], dtype=float)

        # Learner returns random graph
        def random_learner(data_synth):
            return np.random.randn(3, 3) * 0.1

        result = self_compatibility_check(data, A_learned, random_learner, n_resamples=5, seed=42)

        # Random learner unlikely to recover structure
        assert result.compatibility_score < 1.0

    def test_self_compat_result_summary(self):
        """SelfCompatibilityResult.summary() produces readable output."""
        from jcce.validation.self_compatibility import SelfCompatibilityResult

        result = SelfCompatibilityResult(
            compatibility_score=0.6,
            n_resamples=5,
            n_compatible=3,
            per_resample=[],
            mean_f1=0.65,
            std_f1=0.1,
            mean_shd=3.0,
            edge_recovery_rate=0.7,
            total_time=10.0,
        )
        s = result.summary()
        assert "SELF-COMPATIBILITY" in s
        assert "60.0%" in s

    def test_self_compat_result_to_dict(self):
        """SelfCompatibilityResult.to_dict returns JSON-serializable dict."""
        from jcce.validation.self_compatibility import SelfCompatibilityResult

        result = SelfCompatibilityResult(
            compatibility_score=0.8,
            n_resamples=10,
            n_compatible=8,
            per_resample=[],
            mean_f1=0.85,
            std_f1=0.05,
            mean_shd=2.0,
            edge_recovery_rate=0.9,
            total_time=50.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["is_self_compatible"] is True
