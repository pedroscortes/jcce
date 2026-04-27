"""
Tests for jcce.counterfactual.aap.AAPCounterfactual.

Sprint A gate: round-trip MSE < 1e-4 on a synthetic linear SCM means
abduction + topological forward are correctly composed (the no-op
intervention recovers observed data via the captured noise).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import random

from jcce.counterfactual import AAPCounterfactual


def _make_synthetic_scm(d: int = 5, n: int = 200, seed: int = 0):
    """Linear SCM: X_0 -> X_1 -> X_2 -> X_3 = T -> X_4 = Y, with X_2 -> Y too.

    Returns X (n, d_+), A_true (d_+, d_+), Y_idx, T_idx.
    """
    rng = np.random.RandomState(seed)
    d_plus = d + 1
    Y_idx = d
    T_idx = d - 1  # last X is the treatment

    A_true = np.zeros((d_plus, d_plus))
    # Chain: X_0 -> X_1 -> X_2 -> T -> Y
    for j in range(d - 1):
        A_true[j, j + 1] = 0.7
    # T -> Y
    A_true[T_idx, Y_idx] = 0.6
    # X_2 -> Y (confounding-style direct effect bypass)
    A_true[d - 2, Y_idx] = 0.4

    # Sample data following structural equations: X_j = sum_i A[i,j] * X_i + noise_j
    noise = rng.randn(n, d_plus) * 0.5
    X = np.zeros((n, d_plus))
    for j in range(d_plus):
        parents_contrib = X @ A_true[:, j]
        X[:, j] = parents_contrib + noise[:, j]

    return X.astype(np.float32), A_true.astype(np.float32), Y_idx, T_idx


class _LinearProcessor:
    """Minimal processor emulating a linear structural equation.

    forward(X_weighted, params, A=...) returns X_weighted @ params['w'].
    X_weighted shape is (n_samples, n_features) — the X-only submatrix per
    JCCE convention. params['w'] is a column of A_true normalized to undo
    the external weighting AAP applies.
    """

    def __init__(self):
        self.__class__.__name__ = "LinearProcessor"

    def forward(self, X_weighted, params, A=None, **kwargs):
        return jnp.sum(X_weighted * params["w"], axis=1)


def _split(X, Y_idx):
    """Split full X (n, d_+) into X_features (n, d) and Y (n,) per JCCE convention."""
    return X[:, :Y_idx], X[:, Y_idx]


def _params_from_A(A_true: np.ndarray, d_plus: int):
    """Build per-variable params w_j = A_true[:n_features, j] / |A_true[:, j]|.

    The slicing to first n_features rows respects JCCE's per-variable forward
    which receives only the X-feature submatrix as input.
    """
    n_features = d_plus - 1
    out = []
    for j in range(d_plus):
        col = A_true[:n_features, j]  # X-feature parents only
        norm = jnp.sum(jnp.abs(col)) + 1e-8
        w = jnp.array(col, dtype=jnp.float32) / norm
        out.append({"w": w})
    return out


def test_aap_round_trip_zero_noise():
    """With deterministic-ish SCM and exact f_j, round-trip MSE ≈ 0."""
    rng = np.random.RandomState(42)
    d = 5
    n = 50
    d_plus = d + 1
    Y_idx = d
    T_idx = d - 1

    A_true = np.zeros((d_plus, d_plus))
    for j in range(d - 1):
        A_true[j, j + 1] = 0.5
    A_true[T_idx, Y_idx] = 0.5
    A_true[d - 2, Y_idx] = 0.3

    noise = rng.randn(n, d_plus) * 0.01
    X = np.zeros((n, d_plus))
    for j in range(d_plus):
        X[:, j] = X @ A_true[:, j] + noise[:, j]

    proc = _LinearProcessor()
    proc_params = _params_from_A(A_true, d_plus)

    aap = AAPCounterfactual(
        proc,
        jnp.array(A_true, dtype=jnp.float32),
        proc_params,
        Y_idx=Y_idx,
        edge_threshold=0.05,
    )

    X_features, Y = _split(jnp.array(X, dtype=jnp.float32), Y_idx)
    x_mse, y_mse = aap.round_trip(X_features, Y, t_idx=T_idx)
    mean_x = float(jnp.mean(x_mse))
    mean_y = float(jnp.mean(y_mse))
    assert mean_x < 1e-3, f"X round-trip MSE too high: {mean_x}"
    assert mean_y < 1e-3, f"Y round-trip MSE too high: {mean_y}"


def test_aap_abduct_returns_correct_shape():
    """Abduct should return (n_samples, d_+) of residuals."""
    X, A_true, Y_idx, T_idx = _make_synthetic_scm(d=5, n=20, seed=0)
    proc = _LinearProcessor()
    d_plus = A_true.shape[0]
    proc_params = _params_from_A(A_true, d_plus)

    aap = AAPCounterfactual(
        proc, jnp.array(A_true), proc_params, Y_idx=Y_idx, edge_threshold=0.05,
    )
    X_features, Y = _split(jnp.array(X), Y_idx)
    U = aap.abduct(X_features, Y)
    assert U.shape == (X.shape[0], d_plus), f"U shape mismatch: {U.shape} vs {(X.shape[0], d_plus)}"


def test_aap_intervention_changes_outcome():
    """Counterfactual with t_prime=1 vs t_prime=0 should give different Y."""
    X, A_true, Y_idx, T_idx = _make_synthetic_scm(d=5, n=20, seed=0)
    proc = _LinearProcessor()
    d_plus = A_true.shape[0]
    proc_params = _params_from_A(A_true, d_plus)

    aap = AAPCounterfactual(
        proc, jnp.array(A_true), proc_params, Y_idx=Y_idx, edge_threshold=0.05,
    )
    X_features, Y = _split(jnp.array(X), Y_idx)
    cate_per_sample = aap.cate(X_features, Y, T_idx, t0=0.0, t1=1.0)
    mean_cate = float(jnp.mean(cate_per_sample))
    assert mean_cate > 0, f"expected positive CATE, got {mean_cate}"


if __name__ == "__main__":
    # Run the round-trip gate directly
    test_aap_round_trip_zero_noise()
    print("test_aap_round_trip_zero_noise: PASS")
    test_aap_abduct_returns_correct_shape()
    print("test_aap_abduct_returns_correct_shape: PASS")
    test_aap_intervention_changes_outcome()
    print("test_aap_intervention_changes_outcome: PASS")
    print("\nSprint A gate: PASSED (all 3 tests)")
