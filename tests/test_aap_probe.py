"""Tests for jcce.counterfactual.probe — the f_Y collapse diagnostic API.

Covers the four weighting modes accepted by ``make_f_Y``, the gradient
behavior of ``autodiff_sensitivity`` on processors with known sensitivity
patterns, and the grid-sweep shape under both constant and linear f_Y.

A processor is considered "collapsed" iff
    autodiff_sensitivity(f_Y, X, var_idx) < tol  AND
    max(grid_sweep(f_Y, X, var_idx, grid)) - min(...) < tol
Both probes are needed to firm up the claim — a flat sweep alone could be a
saddle point at the evaluation grid; a zero gradient alone could be a single
non-differentiable point. Constant f_Y satisfies both.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jcce.counterfactual import autodiff_sensitivity, grid_sweep, make_f_Y


class _LinearProcessor:
    """Reference processor: f(X) = sum_j X[:, j] * params['w'][j]. No centering."""

    def forward(self, X_weighted, params, **kwargs):
        return jnp.sum(X_weighted * params["w"], axis=1)


class _ConstantProcessor:
    """Pathological processor mimicking the JCCE-collapsed f_Y: returns a constant."""

    def forward(self, X_weighted, params, **kwargs):
        return jnp.full(X_weighted.shape[0], float(params["c"]))


def _make_A(d_plus: int, edge_to_Y: list[float]) -> jnp.ndarray:
    """Build a (d_plus, d_plus) adjacency with the given X->Y edge weights."""
    A = np.zeros((d_plus, d_plus), dtype=np.float32)
    Y_idx = d_plus - 1
    for j, w in enumerate(edge_to_Y):
        A[j, Y_idx] = w
    return jnp.asarray(A)


# ---- make_f_Y weighting-mode coverage ----------------------------------------


def test_make_f_Y_learned_weighting_uses_abs_A_column():
    A = _make_A(5, [-0.3, 0.5, 0.0, 0.7])  # n_features = 4
    proc = _LinearProcessor()
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    _, weights = make_f_Y(proc, A, params, Y_idx=4, n_features=4, weighting="learned")
    np.testing.assert_allclose(np.asarray(weights), [0.3, 0.5, 0.0, 0.7], atol=1e-7)


def test_make_f_Y_hard_weighting_thresholds_strictly():
    A = _make_A(5, [0.04, 0.05, 0.06, 0.5])
    proc = _LinearProcessor()
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    _, weights = make_f_Y(
        proc, A, params, Y_idx=4, n_features=4,
        weighting="hard", edge_threshold=0.05,
    )
    # 0.04 -> 0; 0.05 not strictly > 0.05 -> 0; 0.06 -> 1; 0.5 -> 1.
    np.testing.assert_array_equal(np.asarray(weights), [0.0, 0.0, 1.0, 1.0])


def test_make_f_Y_unit_weighting_is_all_ones():
    A = _make_A(5, [0.0, 0.5, 0.0, 0.0])
    proc = _LinearProcessor()
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    _, weights = make_f_Y(proc, A, params, Y_idx=4, n_features=4, weighting="unit")
    np.testing.assert_array_equal(np.asarray(weights), np.ones(4))


def test_make_f_Y_training_weighting_adds_uniform_baseline():
    """Training mode: weights = |A[:, Y]| + 1 / (n_features + 1)."""
    A = _make_A(5, [0.0, 0.0, 0.0, 0.7])
    proc = _LinearProcessor()
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    _, weights = make_f_Y(
        proc, A, params, Y_idx=4, n_features=4, weighting="training",
    )
    expected_baseline = 1.0 / 5.0
    np.testing.assert_allclose(
        np.asarray(weights),
        [expected_baseline, expected_baseline, expected_baseline, 0.7 + expected_baseline],
        atol=1e-7,
    )


def test_make_f_Y_unknown_weighting_raises():
    A = _make_A(3, [0.5, 0.5])
    proc = _LinearProcessor()
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    with pytest.raises(ValueError, match="unknown weighting"):
        make_f_Y(proc, A, params, Y_idx=2, n_features=2, weighting="bogus")


# ---- autodiff_sensitivity ----------------------------------------------------


def test_autodiff_sensitivity_recovers_linear_slope_under_weighting():
    """For f_Y = sum_j (X[:, j] * weights[j]) * w[j], df/dX[:, j] = weights[j] * w[j]."""
    A = _make_A(5, [0.3, 0.5, 0.7, 1.0])
    proc = _LinearProcessor()
    w = jnp.array([2.0, -1.0, 0.5, 0.0], dtype=jnp.float32)
    params = {"w": w}
    f_Y, weights = make_f_Y(proc, A, params, Y_idx=4, n_features=4, weighting="learned")

    rng = np.random.RandomState(0)
    X = jnp.asarray(rng.randn(20, 4).astype(np.float32))
    for j in range(4):
        sens = autodiff_sensitivity(f_Y, X, var_idx=j)
        expected = abs(float(weights[j]) * float(w[j]))
        np.testing.assert_allclose(sens, expected, atol=1e-6)


def test_autodiff_sensitivity_constant_f_Y_returns_zero():
    """Collapsed f_Y -> all gradients exactly zero."""
    A = _make_A(5, [0.5] * 4)
    proc = _ConstantProcessor()
    params = {"c": -0.135}
    f_Y, _ = make_f_Y(proc, A, params, Y_idx=4, n_features=4, weighting="unit")

    rng = np.random.RandomState(0)
    X = jnp.asarray(rng.randn(15, 4).astype(np.float32))
    for j in range(4):
        sens = autodiff_sensitivity(f_Y, X, var_idx=j)
        assert sens == 0.0


# ---- grid_sweep --------------------------------------------------------------


def test_grid_sweep_constant_f_Y_returns_flat_array():
    A = _make_A(5, [0.5] * 4)
    proc = _ConstantProcessor()
    params = {"c": -0.088}
    f_Y, _ = make_f_Y(proc, A, params, Y_idx=4, n_features=4, weighting="learned")

    rng = np.random.RandomState(0)
    X = jnp.asarray(rng.randn(10, 4).astype(np.float32))
    grid = np.linspace(-2.0, 2.0, 9)
    sweep = grid_sweep(f_Y, X, var_idx=3, grid=grid)
    assert sweep.shape == (9,)
    np.testing.assert_allclose(sweep, np.full(9, -0.088), atol=1e-6)


def test_grid_sweep_linear_f_Y_returns_monotone_array():
    """For linear f_Y, mean f_Y(X with X[:, j] = v) is linear in v."""
    A = _make_A(5, [1.0, 0.0, 0.0, 0.0])  # only X_0 weighted
    proc = _LinearProcessor()
    w = jnp.array([2.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    params = {"w": w}
    f_Y, _ = make_f_Y(proc, A, params, Y_idx=4, n_features=4, weighting="learned")

    X = jnp.zeros((4, 4), dtype=jnp.float32)
    grid = np.linspace(-1.0, 1.0, 5)
    sweep = grid_sweep(f_Y, X, var_idx=0, grid=grid)
    # f_Y(X with X[:, 0] = v) = v * weights[0] * w[0] = v * 1.0 * 2.0 = 2v.
    np.testing.assert_allclose(sweep, 2.0 * grid, atol=1e-6)


def test_grid_sweep_returns_numpy_array_of_floats():
    A = _make_A(3, [0.5, 0.5])
    proc = _ConstantProcessor()
    params = {"c": 0.0}
    f_Y, _ = make_f_Y(proc, A, params, Y_idx=2, n_features=2, weighting="unit")

    X = jnp.zeros((3, 2), dtype=jnp.float32)
    sweep = grid_sweep(f_Y, X, var_idx=0, grid=[0.0, 1.0])
    assert isinstance(sweep, np.ndarray)
    assert sweep.dtype.kind == "f"


if __name__ == "__main__":
    import sys

    failed = []
    for name in [
        "test_make_f_Y_learned_weighting_uses_abs_A_column",
        "test_make_f_Y_hard_weighting_thresholds_strictly",
        "test_make_f_Y_unit_weighting_is_all_ones",
        "test_make_f_Y_training_weighting_adds_uniform_baseline",
        "test_make_f_Y_unknown_weighting_raises",
        "test_autodiff_sensitivity_recovers_linear_slope_under_weighting",
        "test_autodiff_sensitivity_constant_f_Y_returns_zero",
        "test_grid_sweep_constant_f_Y_returns_flat_array",
        "test_grid_sweep_linear_f_Y_returns_monotone_array",
        "test_grid_sweep_returns_numpy_array_of_floats",
    ]:
        try:
            globals()[name]()
            print(f"{name}: PASS")
        except Exception as e:
            failed.append((name, e))
            print(f"{name}: FAIL ({e})")
    sys.exit(0 if not failed else 1)
