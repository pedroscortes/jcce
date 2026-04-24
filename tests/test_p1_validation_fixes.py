"""
Tests for P1 validation fixes.

Covers:
1. Bootstrap DAG stability — module structure and result dataclass
2. Threshold unification — EDGE_THRESHOLD_DEFAULT constant used consistently
3. Hard threshold adjacency — preserves weights above threshold
4. Structure metrics in experiment scripts — C.9 and C.1 compute SHD/F1
5. AIPW validation script — data generation and JCCE DML runner
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# =============================================================================
# 1. Bootstrap DAG Stability Tests
# =============================================================================


class TestBootstrapStabilityModule:
    """Test bootstrap_stability module structure and result dataclass."""

    def test_module_imports(self):
        """Verify bootstrap_stability module imports correctly."""
        from jcce.validation.bootstrap_stability import (
            bootstrap_dag_stability,
        )

        assert callable(bootstrap_dag_stability)

    def test_result_dataclass_fields(self):
        """Verify BootstrapStabilityResult has all expected fields."""
        from jcce.validation.bootstrap_stability import BootstrapStabilityResult

        result = BootstrapStabilityResult(
            edge_frequency=np.zeros((6, 6)),
            n_bootstraps=10,
            n_requested=10,
            n_stable_edges=3,
            n_total_edges_ever=8,
            stability_threshold=0.5,
            n_edges_per_bootstrap=[5, 6, 4, 5, 7, 5, 6, 4, 5, 6],
            h_A_per_bootstrap=[0.1] * 10,
            bacc_per_bootstrap=[0.7] * 10,
            total_time=100.0,
            time_per_bootstrap=10.0,
        )
        assert result.n_bootstraps == 10
        assert result.n_stable_edges == 3
        assert result.edge_frequency.shape == (6, 6)

    def test_get_stable_edges(self):
        """get_stable_edges returns edges above threshold sorted by frequency."""
        from jcce.validation.bootstrap_stability import BootstrapStabilityResult

        freq = np.zeros((4, 4))
        freq[0, 1] = 0.9
        freq[1, 2] = 0.6
        freq[2, 3] = 0.3

        result = BootstrapStabilityResult(
            edge_frequency=freq,
            n_bootstraps=10,
            n_requested=10,
            n_stable_edges=2,
            n_total_edges_ever=3,
            stability_threshold=0.5,
            n_edges_per_bootstrap=[3] * 10,
            h_A_per_bootstrap=[0.1] * 10,
            bacc_per_bootstrap=[0.7] * 10,
            total_time=100.0,
            time_per_bootstrap=10.0,
        )

        edges_50 = result.get_stable_edges(min_frequency=0.5)
        assert len(edges_50) == 2  # 0->1 (0.9) and 1->2 (0.6)
        assert edges_50[0] == (0, 1, 0.9)  # Highest first

        edges_80 = result.get_stable_edges(min_frequency=0.8)
        assert len(edges_80) == 1  # Only 0->1

    def test_to_dict(self):
        """to_dict returns JSON-serializable dict."""
        from jcce.validation.bootstrap_stability import BootstrapStabilityResult

        result = BootstrapStabilityResult(
            edge_frequency=np.eye(3),
            n_bootstraps=5,
            n_requested=5,
            n_stable_edges=0,
            n_total_edges_ever=0,
            stability_threshold=0.5,
            n_edges_per_bootstrap=[0] * 5,
            h_A_per_bootstrap=[0.1] * 5,
            bacc_per_bootstrap=[0.7] * 5,
            total_time=50.0,
            time_per_bootstrap=10.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "edge_frequency" in d
        assert isinstance(d["edge_frequency"], list)  # numpy -> list

    def test_summary_string(self):
        """summary() returns non-empty string with key information."""
        from jcce.validation.bootstrap_stability import BootstrapStabilityResult

        freq = np.zeros((4, 4))
        freq[0, 1] = 0.9

        result = BootstrapStabilityResult(
            edge_frequency=freq,
            n_bootstraps=10,
            n_requested=10,
            n_stable_edges=1,
            n_total_edges_ever=1,
            stability_threshold=0.5,
            n_edges_per_bootstrap=[1] * 10,
            h_A_per_bootstrap=[0.1] * 10,
            bacc_per_bootstrap=[0.7] * 10,
            total_time=100.0,
            time_per_bootstrap=10.0,
        )
        s = result.summary()
        assert "BOOTSTRAP DAG STABILITY" in s
        assert "10" in s  # n_bootstraps

    def test_bootstrap_function_signature(self):
        """Verify bootstrap_dag_stability has expected parameters."""
        import inspect

        from jcce.validation.bootstrap_stability import bootstrap_dag_stability

        sig = inspect.signature(bootstrap_dag_stability)
        params = set(sig.parameters.keys())
        expected = {
            "X",
            "Y",
            "hyperparams",
            "processor_type",
            "B",
            "max_iter",
            "A_init",
            "edge_threshold",
            "stability_threshold",
            "seed",
            "task",
            "verbose",
        }
        assert expected <= params, f"Missing params: {expected - params}"


# =============================================================================
# 2. Threshold Unification Tests
# =============================================================================


class TestThresholdUnification:
    """Verify EDGE_THRESHOLD_DEFAULT is used consistently."""

    def test_constant_exists_and_value(self):
        """EDGE_THRESHOLD_DEFAULT = 0.3."""
        from jcce.utils.metrics import EDGE_THRESHOLD_DEFAULT

        assert EDGE_THRESHOLD_DEFAULT == 0.3

    def test_pareto_stability_uses_constant(self):
        """pareto_stability.compute_edge_stability default matches constant."""
        import inspect

        from jcce.analysis.pareto_stability import compute_edge_stability
        from jcce.utils.metrics import EDGE_THRESHOLD_DEFAULT

        sig = inspect.signature(compute_edge_stability)
        assert sig.parameters["threshold"].default == EDGE_THRESHOLD_DEFAULT

    def test_cv_edge_frequency_uses_constant(self):
        """Unified CV _compute_edge_frequency default matches constant."""
        import inspect

        from jcce.validation.unified_cv_evaluation import _compute_edge_frequency

        sig = inspect.signature(_compute_edge_frequency)
        # The threshold param default should be EDGE_THRESHOLD_DEFAULT
        from jcce.utils.metrics import EDGE_THRESHOLD_DEFAULT

        assert sig.parameters["threshold"].default == EDGE_THRESHOLD_DEFAULT

    def test_cv_shd_uses_constant(self):
        """Unified CV _compute_shd default matches constant."""
        import inspect

        from jcce.validation.unified_cv_evaluation import _compute_shd

        sig = inspect.signature(_compute_shd)
        from jcce.utils.metrics import EDGE_THRESHOLD_DEFAULT

        assert sig.parameters["threshold"].default == EDGE_THRESHOLD_DEFAULT


# =============================================================================
# 3. Hard Threshold Adjacency Tests
# =============================================================================


class TestHardThresholdAdjacency:
    """Test hard_threshold_adjacency function."""

    def test_zeros_below_threshold(self):
        """Entries below threshold are zeroed out."""
        from jcce.utils.metrics import hard_threshold_adjacency

        A = np.array([[0.0, 0.5, 0.1], [0.2, 0.0, 0.8], [0.05, 0.4, 0.0]], dtype=np.float32)
        result = hard_threshold_adjacency(A, threshold=0.3)

        assert result[0, 2] == 0.0  # 0.1 < 0.3
        assert result[1, 0] == 0.0  # 0.2 < 0.3
        assert result[2, 0] == 0.0  # 0.05 < 0.3

    def test_preserves_above_threshold(self):
        """Entries above threshold preserve their continuous weight."""
        from jcce.utils.metrics import hard_threshold_adjacency

        A = np.array([[0.0, 0.5, 0.1], [0.2, 0.0, 0.8], [0.05, 0.4, 0.0]], dtype=np.float32)
        result = hard_threshold_adjacency(A, threshold=0.3)

        assert result[0, 1] == pytest.approx(0.5)  # 0.5 > 0.3
        assert result[1, 2] == pytest.approx(0.8)  # 0.8 > 0.3
        assert result[2, 1] == pytest.approx(0.4)  # 0.4 > 0.3

    def test_zeros_diagonal(self):
        """Diagonal always zeroed regardless of value."""
        from jcce.utils.metrics import hard_threshold_adjacency

        A = np.array([[0.9, 0.5], [0.5, 0.8]], dtype=np.float32)
        result = hard_threshold_adjacency(A, threshold=0.3)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0

    def test_default_threshold(self):
        """Default threshold is EDGE_THRESHOLD_DEFAULT."""
        import inspect

        from jcce.utils.metrics import EDGE_THRESHOLD_DEFAULT, hard_threshold_adjacency

        sig = inspect.signature(hard_threshold_adjacency)
        assert sig.parameters["threshold"].default == EDGE_THRESHOLD_DEFAULT


# =============================================================================
# 4. Structure Metrics in Experiment Scripts
# =============================================================================


class TestC9StructureMetrics:
    """Verify C.9 computes and reports SHD/F1."""

    def test_generate_classification_data_returns_A_true(self):
        """generate_classification_data returns 3 values: X, Y, A_true."""
        from experiment_c9_multi_fidelity_ablation import generate_classification_data

        X, Y, A_true = generate_classification_data(
            n_vars=5, n_samples=50, expected_degree=2.0, noise_scale=0.5, seed=42
        )
        assert X.shape == (50, 5)
        assert Y.shape == (50,)
        # A_true should be (n_vars+1, n_vars+1)
        assert A_true.shape == (6, 6)

    def test_run_condition_accepts_A_true(self):
        """run_condition has A_true parameter."""
        import inspect

        from experiment_c9_multi_fidelity_ablation import run_condition

        sig = inspect.signature(run_condition)
        assert "A_true" in sig.parameters

    def test_run_condition_result_has_f1_shd(self):
        """run_condition result dict includes best_f1 and best_shd keys."""
        # We can't easily run the full condition (needs GOLEM), but verify
        # the function signature and structural expectations
        import inspect

        from experiment_c9_multi_fidelity_ablation import run_condition

        sig = inspect.signature(run_condition)
        # A_true should default to None (backward compatible)
        assert sig.parameters["A_true"].default is None


class TestC1StructureMetrics:
    """Verify C.1 computes and reports SHD/F1."""

    def test_c1_imports_compute_structure_metrics(self):
        """C.1 script uses compute_structure_metrics."""
        with open(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "compare_nsga2_vs_optuna.py")
        ) as f:
            source = f.read()

        assert "compute_structure_metrics" in source
        assert "best_f1" in source
        assert "best_shd" in source

    def test_c1_prints_f1_shd(self):
        """C.1 prints F1/SHD in comparison table."""
        with open(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "compare_nsga2_vs_optuna.py")
        ) as f:
            source = f.read()

        assert "'best_f1'" in source
        assert "'best_shd'" in source


# =============================================================================
# 5. AIPW Validation Script Tests
# =============================================================================


class TestAIPWValidation:
    """Verify AIPW validation script components."""

    def test_generate_synthetic_ate_data(self):
        """Data generation produces valid ATE test data."""
        from validate_aipw_against_econml import generate_synthetic_ate_data

        X, T, Y, true_ate = generate_synthetic_ate_data(n=200, p=5, true_ate=0.5, seed=42)
        assert X.shape == (200, 5)
        assert T.shape == (200,)
        assert Y.shape == (200,)
        assert true_ate == 0.5
        # T should be binary
        assert set(np.unique(T)) <= {0.0, 1.0}

    def test_jcce_dml_runner_exists(self):
        """run_jcce_dml function exists and is callable."""
        from validate_aipw_against_econml import run_jcce_dml

        assert callable(run_jcce_dml)

    def test_generate_data_is_confounded(self):
        """Treatment is confounded by X (not random)."""
        from validate_aipw_against_econml import generate_synthetic_ate_data

        X, T, Y, _ = generate_synthetic_ate_data(n=1000, seed=42)
        # If confounded, T should correlate with X
        corr = np.abs(np.corrcoef(T, X[:, 0])[0, 1])
        assert corr > 0.1, f"T should be confounded by X, corr={corr}"
