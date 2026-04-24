"""Tests for DML diagnostics and nonlinear nuisance models."""

import numpy as np
import pytest

from jcce.validation.dml_crossfitting import (
    DMLCrossFitter,
    DMLDiagnostics,
    create_gbm_nuisance_functions,
    create_nuisance_functions,
    create_simple_nuisance_functions,
)


def _make_data(n=200, d=5, seed=42):
    """Generate simple treatment/outcome data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    T = (X[:, 0] + rng.randn(n) * 0.5 > 0).astype(float)
    Y = 2.0 * T + X[:, 0] + rng.randn(n) * 0.3
    return X, T, Y


# ============================================================
# Task 22: DML Diagnostics
# ============================================================


class TestDMLDiagnostics:
    def test_diagnostics_returned(self):
        """estimate_ate returns diagnostics in result."""
        X, T, Y = _make_data()
        train_fn, predict_fn = create_simple_nuisance_functions()
        dml = DMLCrossFitter(n_splits=3)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, DMLDiagnostics)

    def test_propensity_overlap_stats(self):
        """Propensity overlap stats are populated and reasonable."""
        X, T, Y = _make_data()
        train_fn, predict_fn = create_simple_nuisance_functions()
        dml = DMLCrossFitter(n_splits=3)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        diag = result.diagnostics
        assert 0.0 <= diag.propensity_min <= diag.propensity_max <= 1.0
        assert 0.0 < diag.propensity_mean < 1.0
        assert 0.0 <= diag.pct_clipped <= 1.0

    def test_covariate_balance(self):
        """Covariate balance SMD is computed."""
        X, T, Y = _make_data()
        train_fn, predict_fn = create_simple_nuisance_functions()
        dml = DMLCrossFitter(n_splits=3)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        diag = result.diagnostics
        assert diag.mean_smd >= 0.0
        assert diag.max_smd >= diag.mean_smd
        assert diag.n_imbalanced >= 0

    def test_nuisance_fit_quality(self):
        """Outcome R² and propensity log-loss are populated."""
        X, T, Y = _make_data()
        train_fn, predict_fn = create_simple_nuisance_functions()
        dml = DMLCrossFitter(n_splits=3)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        diag = result.diagnostics
        # R² should be positive for this well-specified DGP
        assert diag.outcome_r2_mean > 0.0
        # Log-loss should be finite and positive
        assert 0.0 < diag.propensity_logloss_mean < 10.0

    def test_diagnostics_in_to_dict(self):
        """Diagnostics appear in serialized dict."""
        X, T, Y = _make_data()
        train_fn, predict_fn = create_simple_nuisance_functions()
        dml = DMLCrossFitter(n_splits=3)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        d = result.to_dict()
        assert "diagnostics" in d
        assert "propensity_min" in d["diagnostics"]
        assert "mean_smd" in d["diagnostics"]
        assert "outcome_r2_mean" in d["diagnostics"]

    def test_diagnostics_summary(self):
        """Summary string is produced without error."""
        diag = DMLDiagnostics(
            propensity_min=0.05,
            propensity_max=0.95,
            propensity_mean=0.5,
            pct_clipped=0.02,
            mean_smd=0.08,
            max_smd=0.15,
            n_imbalanced=1,
            outcome_r2_mean=0.7,
            propensity_logloss_mean=0.5,
        )
        s = diag.summary()
        assert "Propensity" in s
        assert "Balance" in s
        assert "Nuisance" in s

    def test_high_clipping_detected(self):
        """When propensity is extreme, clipping is detected."""
        X, T, Y = _make_data(n=100)
        # Use very tight clipping to force high pct_clipped
        dml = DMLCrossFitter(n_splits=3, clip_propensity=(0.4, 0.6))
        train_fn, predict_fn = create_simple_nuisance_functions()
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        # With tight bounds, most propensities should be clipped
        assert result.diagnostics.pct_clipped > 0.1


# ============================================================
# Task 23: Nonlinear Nuisance Models
# ============================================================


class TestGBMNuisance:
    def test_gbm_factory_returns_callables(self):
        """create_gbm_nuisance_functions returns (train_fn, predict_fn)."""
        train_fn, predict_fn = create_gbm_nuisance_functions()
        assert callable(train_fn)
        assert callable(predict_fn)

    def test_gbm_produces_result(self):
        """GBM nuisance models produce a valid DMLResult."""
        X, T, Y = _make_data()
        train_fn, predict_fn = create_gbm_nuisance_functions(n_estimators=20)
        dml = DMLCrossFitter(n_splits=3)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        assert np.isfinite(result.ate_mean)
        assert result.ate_std > 0
        assert result.diagnostics is not None

    def test_gbm_ate_reasonable(self):
        """GBM-based ATE is close to true effect (2.0)."""
        X, T, Y = _make_data(n=500)
        train_fn, predict_fn = create_gbm_nuisance_functions(n_estimators=50)
        dml = DMLCrossFitter(n_splits=5)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        # True ATE = 2.0; allow generous tolerance for GBM
        assert abs(result.ate_mean - 2.0) < 1.5

    def test_create_nuisance_factory_linear(self):
        """Factory with method='linear' returns linear functions."""
        train_fn, predict_fn = create_nuisance_functions(method="linear")
        assert callable(train_fn)

    def test_create_nuisance_factory_gbm(self):
        """Factory with method='gbm' returns GBM functions."""
        train_fn, predict_fn = create_nuisance_functions(method="gbm")
        assert callable(train_fn)

    def test_create_nuisance_factory_invalid(self):
        """Factory with invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown nuisance method"):
            create_nuisance_functions(method="xgboost")

    def test_gbm_with_custom_params(self):
        """GBM factory accepts custom hyperparameters."""
        train_fn, predict_fn = create_gbm_nuisance_functions(
            n_estimators=10, max_depth=2, learning_rate=0.05
        )
        X, T, Y = _make_data(n=100)
        dml = DMLCrossFitter(n_splits=2)
        result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
        assert np.isfinite(result.ate_mean)
