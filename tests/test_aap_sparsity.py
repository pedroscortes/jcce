"""Tests for jcce.counterfactual.sparsity primitives.

The STE hard-parents mask must produce a strict 0/1 forward and an
identity-through-|weights| backward, otherwise the gradient signal
that grows or shrinks edges during AAP training is broken.
``edge_set_agreement`` is the gate metric on synthetic SCMs and must
report expected values on toy ground truth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jcce.counterfactual.sparsity import edge_set_agreement, ste_hard_parents


def test_ste_forward_is_strict_zero_one():
    """Forward: |w| > threshold yields 1.0, else 0.0. Boundary is strict."""
    weights = jnp.array([-1.0, -0.06, -0.05, 0.0, 0.04, 0.05, 0.06, 1.0], dtype=jnp.float32)
    out = ste_hard_parents(weights, threshold=0.05)
    expected = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))


def test_ste_backward_equals_grad_of_abs():
    """Gradient of sum(ste(w)) equals sign(w): the autodiff sees |w|, not the hard mask.

    Boundary points (w=0) are skipped — JAX's gradient of |w| at zero is not
    portable across versions and is irrelevant for the AAP cascade.
    """
    w = jnp.array([0.04, 0.5, -0.3, 0.7, -0.001], dtype=jnp.float32)
    g = jax.grad(lambda x: jnp.sum(ste_hard_parents(x, 0.05)))(w)
    expected_signs = jnp.array([1.0, 1.0, -1.0, 1.0, -1.0], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(g), np.asarray(expected_signs), atol=1e-6)


def test_ste_backward_through_downstream_use():
    """In the AAP pattern X * ste(w), the gradient w.r.t. w_j is sum_i X_i * sign(w_j).

    This mirrors the cascade: forward uses 0/1 mask (sparsity-enforcing),
    backward signals to grow/shrink the edge based on the loss curvature
    relative to |A|. The gradient is identical to what plain |w|-weighting
    would produce, so structure-learning gradient flow is preserved.
    """
    rng = np.random.RandomState(0)
    X = jnp.array(rng.randn(10).astype(np.float32))
    w = jnp.array([0.04, 0.5, -0.3, 0.6, 0.001], dtype=jnp.float32)

    def loss(w_param: jnp.ndarray) -> jnp.ndarray:
        gates = ste_hard_parents(w_param, 0.05)  # forward in {0,1}
        return jnp.sum(jnp.outer(X, gates))

    g = jax.grad(loss)(w)
    sum_X = float(jnp.sum(X))
    expected = jnp.array([1.0, 1.0, -1.0, 1.0, 1.0], dtype=jnp.float32) * sum_X
    np.testing.assert_allclose(np.asarray(g), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_ste_dtype_preserved_float32():
    """Output dtype must match input — JCCE pipeline uses float32 throughout."""
    out = ste_hard_parents(jnp.array([0.1, 0.04], dtype=jnp.float32), 0.05)
    assert out.dtype == jnp.float32


def test_ste_dtype_preserved_bfloat16():
    """bfloat16 is the use_bf16 mixed-precision path in jcce_learner."""
    out = ste_hard_parents(jnp.array([0.1, 0.04], dtype=jnp.bfloat16), 0.05)
    assert out.dtype == jnp.bfloat16


def test_edge_set_agreement_perfect_match():
    """Identical adjacency: all metrics are 1.0."""
    A = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]])
    result = edge_set_agreement(A, A, threshold_learned=0.05)
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["n_true_edges"] == 2
    assert result["n_learned_edges"] == 2


def test_edge_set_agreement_partial_match():
    """Toy 4x4: 2 hits, 1 false positive, 1 false negative.

    True edges: (0,1), (1,2), (2,3) — three edges.
    Learned edges: (0,1), (1,2), (0,2) — two correct + one spurious; (2,3) missed.
    Confusion: tp=2, fp=1, fn=1, tn=12 over 16 entries.
    """
    A_true = np.zeros((4, 4))
    A_true[0, 1] = A_true[1, 2] = A_true[2, 3] = 0.5

    A_learned = np.zeros((4, 4))
    A_learned[0, 1] = A_learned[1, 2] = A_learned[0, 2] = 0.5

    result = edge_set_agreement(A_learned, A_true, threshold_learned=0.05)
    assert abs(result["accuracy"] - 14 / 16) < 1e-9
    assert abs(result["precision"] - 2 / 3) < 1e-9
    assert abs(result["recall"] - 2 / 3) < 1e-9
    assert abs(result["f1"] - 2 / 3) < 1e-9
    assert result["n_true_edges"] == 3
    assert result["n_learned_edges"] == 3


def test_edge_set_agreement_all_zero_learned():
    """Trivial all-zero baseline: high accuracy on sparse graphs but zero recall.

    Demonstrates why ``f1`` and ``recall`` matter more than ``accuracy`` for
    sparse SCMs — 22/25 ``accuracy`` here is misleading.
    """
    A_true = np.zeros((5, 5))
    A_true[0, 1] = A_true[1, 2] = A_true[2, 3] = 0.5
    A_learned = np.zeros((5, 5))

    result = edge_set_agreement(A_learned, A_true, threshold_learned=0.05)
    assert abs(result["accuracy"] - 22 / 25) < 1e-9
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0
    assert result["n_learned_edges"] == 0


def test_edge_set_agreement_threshold_filters_weak_edges():
    """Learned edges below the threshold are not counted as edges."""
    A_true = np.zeros((3, 3))
    A_true[0, 1] = 0.5

    A_learned = np.zeros((3, 3))
    A_learned[0, 1] = 0.5
    A_learned[1, 2] = 0.01  # below 0.05 threshold

    result = edge_set_agreement(A_learned, A_true, threshold_learned=0.05)
    assert result["n_learned_edges"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_aap_enforce_hard_parents_zeros_below_threshold():
    """End-to-end: with enforce_hard_parents=True, near-zero edges drop out of f_j.

    Construct an A where parent X_0 has a weight just below the threshold
    and X_1 has one above. Predictions should depend only on X_1 in the
    hard-parents mode, not on X_0.
    """
    from jcce.counterfactual import AAPCounterfactual

    class _LinearProcessor:
        def forward(self, X_weighted, params, **kwargs):
            return jnp.sum(X_weighted * params["w"], axis=1)

    n = 8
    n_features = 2
    Y_idx = n_features
    A = jnp.array(
        [
            [0.0, 0.0, 0.04],   # X_0 -> Y at 0.04 (below 0.05 threshold)
            [0.0, 0.0, 0.5],    # X_1 -> Y at 0.5  (above threshold)
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    proc = _LinearProcessor()
    params = [{"w": jnp.array([1.0, 0.0], dtype=jnp.float32)}] * 2 + [
        {"w": jnp.array([1.0, 1.0], dtype=jnp.float32)}
    ]

    rng = np.random.RandomState(0)
    X_features = jnp.array(rng.randn(n, n_features).astype(np.float32))
    Y = jnp.zeros(n, dtype=jnp.float32)

    aap_soft = AAPCounterfactual(
        proc, A, params, Y_idx=Y_idx, edge_threshold=0.05, enforce_hard_parents=False,
    )
    aap_hard = AAPCounterfactual(
        proc, A, params, Y_idx=Y_idx, edge_threshold=0.05, enforce_hard_parents=True,
    )

    # CATE on T_idx=0 (the below-threshold edge): hard-parents mode must return zero;
    # soft mode returns nonzero because |A[0,Y]| = 0.04 still passes some signal.
    cate_soft = np.array(aap_soft.cate(X_features, Y, t_idx=0, t0=0.0, t1=1.0))
    cate_hard = np.array(aap_hard.cate(X_features, Y, t_idx=0, t0=0.0, t1=1.0))
    assert np.max(np.abs(cate_hard)) < 1e-6, f"hard-parents CATE should be zero, got {cate_hard}"
    assert np.max(np.abs(cate_soft)) > 1e-3, f"soft CATE should be nonzero, got {cate_soft}"


if __name__ == "__main__":
    test_ste_forward_is_strict_zero_one()
    print("test_ste_forward_is_strict_zero_one: PASS")
    test_ste_backward_equals_grad_of_abs()
    print("test_ste_backward_equals_grad_of_abs: PASS")
    test_ste_backward_through_downstream_use()
    print("test_ste_backward_through_downstream_use: PASS")
    test_ste_dtype_preserved_float32()
    print("test_ste_dtype_preserved_float32: PASS")
    test_ste_dtype_preserved_bfloat16()
    print("test_ste_dtype_preserved_bfloat16: PASS")
    test_edge_set_agreement_perfect_match()
    print("test_edge_set_agreement_perfect_match: PASS")
    test_edge_set_agreement_partial_match()
    print("test_edge_set_agreement_partial_match: PASS")
    test_edge_set_agreement_all_zero_learned()
    print("test_edge_set_agreement_all_zero_learned: PASS")
    test_edge_set_agreement_threshold_filters_weak_edges()
    print("test_edge_set_agreement_threshold_filters_weak_edges: PASS")
    test_aap_enforce_hard_parents_zeros_below_threshold()
    print("test_aap_enforce_hard_parents_zeros_below_threshold: PASS")
    print("\nAll sparsity primitive tests: PASS")
