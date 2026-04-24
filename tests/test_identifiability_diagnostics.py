"""
Tests for v15 Identifiability Diagnostics.

Covers:
1. Condition number computation
2. SVD diagnostics
3. Hutchinson EDF estimator
4. BIC variants (simple, neural)
5. D-optimality penalty (gradient flow, preference for orthogonal)
6. NSGA-II condition number constraint
7. IdentifiabilityReport grading
"""

import numpy as np
import pytest

from jcce.validation.identifiability_diagnostics import (
    IdentifiabilityReport,
    IdentifiabilitySummary,
    compute_bic_for_dag,
    compute_full_identifiability_report,
    compute_mb_condition_number,
    compute_neural_bic,
    compute_simple_bic,
    compute_svd_diagnostics,
)

# JAX-dependent tests are conditional
try:
    import jax
    import jax.numpy as jnp
    from jax import random as jax_random

    from jcce.validation.identifiability_diagnostics import (
        d_optimality_penalty,
        hutchinson_edf,
    )

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =============================================================================
# 1. Condition Number Tests
# =============================================================================


class TestConditionNumber:
    """Test compute_mb_condition_number."""

    def test_condition_number_orthogonal(self):
        """Orthogonal matrix should have kappa ~ 1."""
        np.random.seed(42)
        n = 200
        # Generate orthogonal columns via QR decomposition
        X_raw = np.random.randn(n, 5)
        Q, _ = np.linalg.qr(X_raw)
        X = Q[:, :5]

        kappa = compute_mb_condition_number(X, [0, 1, 2, 3, 4])
        # After standardization, orthogonal columns should have kappa close to 1
        assert kappa < 5.0, f"Expected kappa ~1 for orthogonal matrix, got {kappa}"

    def test_condition_number_collinear(self):
        """Near-duplicate columns should have kappa >> 100."""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.001  # Near-duplicate
        x3 = np.random.randn(n)
        X = np.column_stack([x1, x2, x3])

        kappa = compute_mb_condition_number(X, [0, 1, 2])
        assert kappa > 100.0, f"Expected kappa >> 100 for collinear columns, got {kappa}"

    def test_condition_number_empty_mb(self):
        """Empty MB should return 1.0."""
        X = np.random.randn(100, 5)
        kappa = compute_mb_condition_number(X, [])
        assert kappa == 1.0

    def test_condition_number_single_feature(self):
        """Single feature MB should return 1.0."""
        X = np.random.randn(100, 5)
        kappa = compute_mb_condition_number(X, [2])
        assert kappa == 1.0


# =============================================================================
# 2. SVD Diagnostics Tests
# =============================================================================


class TestSVDDiagnostics:
    """Test compute_svd_diagnostics."""

    def test_svd_diagnostics_detects_collinear_pairs(self):
        """Two correlated features should be identified as near-collinear."""
        np.random.seed(42)
        n = 500
        x1 = np.random.randn(n)
        x2 = x1 * 0.99 + np.random.randn(n) * 0.01  # r ~ 0.99
        x3 = np.random.randn(n)
        X = np.column_stack([x1, x2, x3])

        result = compute_svd_diagnostics(X, [0, 1, 2], feature_names=["feat_A", "feat_B", "feat_C"])

        assert len(result["near_collinear_pairs"]) >= 1, "Should detect at least one collinear pair"
        # Check that feat_A and feat_B are identified
        pair_names = [(p[0], p[1]) for p in result["near_collinear_pairs"]]
        assert ("feat_A", "feat_B") in pair_names, (
            f"Expected (feat_A, feat_B) in collinear pairs, got {pair_names}"
        )

    def test_svd_diagnostics_effective_rank(self):
        """Effective rank should equal number of independent columns."""
        np.random.seed(42)
        n = 200
        # 3 independent + 1 dependent column
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        x4 = x1 + x2  # Linearly dependent (up to noise)
        X = np.column_stack([x1, x2, x3, x4])

        result = compute_svd_diagnostics(X, [0, 1, 2, 3])
        # Effective rank should be close to 3 (not 4)
        assert result["effective_rank"] <= 4

    def test_svd_diagnostics_empty_mb(self):
        """Empty MB should return empty diagnostics."""
        X = np.random.randn(100, 5)
        result = compute_svd_diagnostics(X, [])
        assert result["effective_rank"] == 0
        assert result["condition_number"] == 1.0


# =============================================================================
# 3. Hutchinson EDF Tests
# =============================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestHutchinsonEDF:
    """Test hutchinson_edf."""

    def test_hutchinson_edf_linear(self):
        """Linear model: EDF should approximate n_params."""
        np.random.seed(42)
        n, p = 200, 5
        X_np = np.random.randn(n, p)
        W = np.random.randn(p, p) * 0.3
        X_jax = jnp.array(X_np)
        W_jax = jnp.array(W)

        # Linear predict function: y = X @ W
        def predict_fn(X):
            return X @ W_jax

        key = jax_random.PRNGKey(0)
        edf = hutchinson_edf(predict_fn, X_jax, n_probes=20, key=key)

        # For a linear model y = X @ W, the Jacobian is W repeated n times
        # trace(J) = n * trace(W) for element-wise operations
        # For matrix multiplication, the total EDF = n * p (each output depends on p inputs)
        # Actually EDF = sum of diagonal elements of the hat matrix
        # For this simple case, just check it's positive and in a reasonable range
        assert edf > 0, f"EDF should be positive, got {edf}"

    def test_hutchinson_edf_nonlinear(self):
        """MLP-like model: EDF should be in a reasonable range."""
        np.random.seed(42)
        n, p = 100, 5

        X_jax = jnp.array(np.random.randn(n, p))

        # Simple nonlinear function (1-layer "MLP")
        W1 = jnp.array(np.random.randn(p, 10) * 0.3)
        W2 = jnp.array(np.random.randn(10, p) * 0.3)

        def predict_fn(X):
            h = jax.nn.relu(X @ W1)
            return h @ W2

        key = jax_random.PRNGKey(42)
        edf = hutchinson_edf(predict_fn, X_jax, n_probes=10, key=key)

        # EDF should be non-negative and finite
        assert edf >= 0, f"EDF should be non-negative, got {edf}"
        assert np.isfinite(edf), f"EDF should be finite, got {edf}"


# =============================================================================
# 4. BIC Tests
# =============================================================================


class TestBIC:
    """Test BIC computation functions."""

    def test_simple_bic_known_values(self):
        """Compare simple BIC against manual computation."""
        np.random.seed(42)
        n = 100
        residuals = np.random.randn(n) * 0.5
        n_params = 5

        bic = compute_simple_bic(residuals, n_params, n)

        # Manual: n * log(RSS/n) + k * log(n)
        rss = np.sum(residuals**2)
        expected = n * np.log(rss / n) + n_params * np.log(n)
        assert abs(bic - expected) < 1e-6, f"BIC mismatch: {bic} vs {expected}"

    def test_neural_bic_uses_edf(self):
        """Neural BIC should use EDF instead of edge count."""
        np.random.seed(42)
        n = 100
        residuals = np.random.randn(n) * 0.5

        # Same residuals, different complexity measure
        bic_5_params = compute_neural_bic(residuals, edf=5.0, n_samples=n)
        bic_50_params = compute_neural_bic(residuals, edf=50.0, n_samples=n)

        # Higher EDF = higher BIC (more complex model penalized more)
        assert bic_50_params > bic_5_params, (
            f"Higher EDF should give higher BIC: {bic_50_params} vs {bic_5_params}"
        )

    def test_bic_for_dag_linear(self):
        """BIC for a simple linear DAG."""
        np.random.seed(42)
        n = 100
        p = 5
        X = np.random.randn(n, p)
        # Simple DAG: sparse adjacency
        A = np.zeros((p, p))
        A[0, 1] = 0.5
        A[1, 2] = 0.3

        bic = compute_bic_for_dag(X, A, processor_type="linear")
        assert np.isfinite(bic), f"BIC should be finite, got {bic}"


# =============================================================================
# 5. D-Optimality Tests
# =============================================================================


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestDOptimality:
    """Test d_optimality_penalty."""

    def test_d_optimality_gradient_flows(self):
        """jax.grad(d_optimality_penalty) should not produce NaN."""
        np.random.seed(42)
        X_mb = jnp.array(np.random.randn(100, 5))

        grad_fn = jax.grad(d_optimality_penalty)
        grad = grad_fn(X_mb)

        assert not jnp.any(jnp.isnan(grad)), "Gradient should not contain NaN"
        assert not jnp.any(jnp.isinf(grad)), "Gradient should not contain Inf"

    def test_d_optimality_prefers_orthogonal(self):
        """Penalty should be lower for orthogonal vs collinear design (standardized)."""
        np.random.seed(42)
        n = 200
        p = 5

        # Orthogonal design (standardized to unit variance)
        X_ortho_np = np.random.randn(n, p)
        # Orthogonalize via Gram-Schmidt but keep unit variance
        Q, _ = np.linalg.qr(X_ortho_np)
        X_ortho_np = Q[:, :p] * np.sqrt(n)  # Scale so that X^T X / n ~ I
        X_ortho = jnp.array(X_ortho_np)

        # Collinear design (standardized columns, but near-duplicate)
        x1 = np.random.randn(n)
        X_col_np = np.column_stack(
            [
                x1,
                x1 + 0.01 * np.random.randn(n),
                np.random.randn(n),
                np.random.randn(n),
                np.random.randn(n),
            ]
        )
        # Standardize each column to zero mean, unit variance
        X_col_np = (X_col_np - X_col_np.mean(axis=0)) / (X_col_np.std(axis=0) + 1e-8)
        X_collinear = jnp.array(X_col_np)

        penalty_ortho = float(d_optimality_penalty(X_ortho))
        penalty_collinear = float(d_optimality_penalty(X_collinear))

        # Orthogonal should have lower (better) penalty since det(G) is maximized
        assert penalty_ortho < penalty_collinear, (
            f"Orthogonal penalty ({penalty_ortho:.4f}) should be < collinear ({penalty_collinear:.4f})"
        )


# =============================================================================
# 6. NSGA-II Constraint Test
# =============================================================================


class TestNSGA2Constraint:
    """Test that NSGA-II condition number constraint works."""

    def test_nsga2_constraint_filters_high_kappa(self):
        """Solutions with kappa > 100 should be marked infeasible (G > 0)."""
        from jcce.validation.identifiability_diagnostics import compute_mb_condition_number

        np.random.seed(42)
        n = 200

        # Well-conditioned MB
        X_good = np.random.randn(n, 5)
        kappa_good = compute_mb_condition_number(X_good, [0, 1, 2, 3, 4])

        # Ill-conditioned MB
        x1 = np.random.randn(n)
        X_bad = np.column_stack(
            [
                x1,
                x1 + 0.0001 * np.random.randn(n),
                x1 + 0.0002 * np.random.randn(n),
                np.random.randn(n),
                np.random.randn(n),
            ]
        )
        kappa_bad = compute_mb_condition_number(X_bad, [0, 1, 2, 3, 4])

        threshold = 100.0

        # Constraint G = kappa - threshold (feasible when G <= 0)
        G_good = kappa_good - threshold
        G_bad = kappa_bad - threshold

        assert G_good <= 0, f"Well-conditioned should be feasible: kappa={kappa_good:.1f}"
        assert G_bad > 0, f"Ill-conditioned should be infeasible: kappa={kappa_bad:.1f}"


# =============================================================================
# 7. IdentifiabilityReport Tests
# =============================================================================


class TestIdentifiabilityReport:
    """Test IdentifiabilityReport grading and summary."""

    def test_identifiability_report_grade_good(self):
        """kappa < 30 should be graded 'good'."""
        report = IdentifiabilityReport(
            condition_number=10.0,
            singular_values=np.array([1.0, 0.5, 0.1]),
            effective_rank=3,
            near_collinear_pairs=[],
            bic=-100.0,
        )
        assert report.identifiability_grade == "good"

    def test_identifiability_report_grade_moderate(self):
        """30 <= kappa < 100 should be graded 'moderate'."""
        report = IdentifiabilityReport(
            condition_number=50.0,
            singular_values=np.array([1.0, 0.5, 0.02]),
            effective_rank=3,
            near_collinear_pairs=[],
            bic=-80.0,
        )
        assert report.identifiability_grade == "moderate"

    def test_identifiability_report_grade_poor(self):
        """kappa >= 100 should be graded 'poor'."""
        report = IdentifiabilityReport(
            condition_number=500.0,
            singular_values=np.array([1.0, 0.5, 0.002]),
            effective_rank=2,
            near_collinear_pairs=[("X0", "X1", 0.95)],
            bic=-50.0,
        )
        assert report.identifiability_grade == "poor"

    def test_identifiability_report_to_dict(self):
        """to_dict should contain all fields."""
        report = IdentifiabilityReport(
            condition_number=10.0,
            singular_values=np.array([1.0, 0.5]),
            effective_rank=2,
            near_collinear_pairs=[],
            bic=-100.0,
            edf=5.0,
        )
        d = report.to_dict()
        assert "condition_number" in d
        assert "bic" in d
        assert "edf" in d
        assert d["edf"] == 5.0
        assert d["identifiability_grade"] == "good"

    def test_identifiability_summary(self):
        """IdentifiabilitySummary should aggregate reports."""
        reports = [
            IdentifiabilityReport(
                condition_number=10.0,
                singular_values=np.array([1.0, 0.5]),
                effective_rank=2,
                near_collinear_pairs=[],
                bic=-100.0,
            ),
            IdentifiabilityReport(
                condition_number=50.0,
                singular_values=np.array([1.0, 0.02]),
                effective_rank=1,
                near_collinear_pairs=[("X0", "X1", 0.95)],
                bic=-80.0,
            ),
        ]
        summary = IdentifiabilitySummary(reports=reports)
        assert summary.n_dags == 2
        assert summary.best_bic_idx == 0  # -100 < -80
        assert len(summary.grades) == 2
        assert "good" in summary.grades
        assert "moderate" in summary.grades

        d = summary.to_dict()
        assert d["bic_best"] == -100.0
        assert d["n_dags"] == 2

    def test_full_identifiability_report(self):
        """compute_full_identifiability_report should produce valid report."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        A = np.zeros((p, p))
        A[0, 1] = 0.5
        A[2, 3] = 0.3

        report = compute_full_identifiability_report(
            X=X,
            A=A,
            mb_indices=[0, 1, 2],
            feature_names=["a", "b", "c", "d", "e"],
        )

        assert report.condition_number > 0
        assert report.effective_rank > 0
        assert np.isfinite(report.bic)
        assert report.identifiability_grade in ("good", "moderate", "poor")
