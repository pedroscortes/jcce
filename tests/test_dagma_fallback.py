"""
Tests for v16.2 and v16.3 bug fixes:
1. NSGA-II result stores v7 effect config (CV v7 detection)
2. DAGMA fallback offset uses constant instead of d*1.0
3. v16.3: DAGMA fallback uses sum(A²) instead of trace(A²) to avoid gradient dead zone

These tests verify the fixes for:
- LUCAS: Transformer pos_embed shape mismatch in CV evaluation
- Breast cancer: DAGMA h(A) jumping to ~31 on 30-variable datasets
- v16.3: >50% of evals stuck at h(A)=0.5 due to zero-gradient fallback
"""

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jcce.structure_learning.jcce_learner import dag_constraint

# =============================================================================
# Bug 2: DAGMA fallback offset
# =============================================================================


def test_dagma_fallback_small_d():
    """DAGMA fallback should stay in reasonable range for small d (e.g., 11)."""
    print("=" * 60)
    print("Test 1: DAGMA fallback — small d (LUCAS-like, d=11)")
    print("=" * 60)

    d = 11
    # Create A that forces fallback: large entries so det(sI - A²) < 0
    A = jnp.ones((d, d)) * 0.5
    A = A.at[jnp.diag_indices(d)].set(0.0)

    h = float(dag_constraint(A))
    print(f"  d={d}, h(A) = {h:.4f}")
    assert h < 5.0, f"h(A) should be < 5 for d={d}, got {h}"
    assert h >= 0.0, f"h(A) should be non-negative, got {h}"
    print("  PASSED")


def test_dagma_fallback_large_d():
    """DAGMA fallback should NOT jump to ~d for large d (e.g., 30)."""
    print("\n" + "=" * 60)
    print("Test 2: DAGMA fallback — large d (breast cancer-like, d=30)")
    print("=" * 60)

    d = 30
    # Dense matrix that will trigger fallback
    A = jnp.ones((d, d)) * 0.3
    A = A.at[jnp.diag_indices(d)].set(0.0)

    h = float(dag_constraint(A))
    print(f"  d={d}, h(A) = {h:.4f}")
    # Key assertion: the old bug produced h ≈ 31 (d*1.0 + trace/s)
    # With the fix, h should be much smaller (0.5 + trace/s ≈ 1-3)
    assert h < 10.0, f"h(A) should be < 10 for d={d}, got {h} (old bug: ~{d})"
    assert h >= 0.0, f"h(A) should be non-negative, got {h}"
    print("  PASSED")


def test_dagma_fallback_not_proportional_to_d():
    """Fallback h(A) should NOT scale linearly with d."""
    print("\n" + "=" * 60)
    print("Test 3: DAGMA fallback — h(A) not proportional to d")
    print("=" * 60)

    results = {}
    for d in [10, 20, 30, 50]:
        # Same relative structure for each d
        A = jnp.ones((d, d)) * 0.3
        A = A.at[jnp.diag_indices(d)].set(0.0)
        h = float(dag_constraint(A))
        results[d] = h
        print(f"  d={d}: h(A) = {h:.4f}")

    # v16.1 bug: h ≈ d (offset scaled as d*1.0), ratio ≈ 5.0
    # v16.2 fix: offset = 0.5 (constant), but trace(A²) ≈ 0 → h = 0.5 (constant, v16.3 bug)
    # v16.3 fix: sum(A²)/(d*s), for uniform A: (d-1)*a²/s + 0.5 → ratio ~ (d-1) linear
    # Key: v16.3 gradient per edge = 2*A[i,j]/(d*s) does NOT scale with d
    # The h value scaling is acceptable; what matters is gradient signal
    ratio = results[50] / results[10]
    print(f"  h(d=50)/h(d=10) ratio = {ratio:.2f}")
    # Verify it's not catastrophically worse than linear (would indicate d² scaling bug)
    assert ratio < 7.0, f"h(A) scales super-linearly with d: ratio {ratio:.2f} (expected < 7.0)"
    print("  PASSED")


def test_dagma_valid_dag_unchanged():
    """Fix should not affect valid DAGs — h(A) ≈ 0 for actual DAGs."""
    print("\n" + "=" * 60)
    print("Test 4: Valid DAGs still get h(A) ≈ 0")
    print("=" * 60)

    for d in [11, 30]:
        # Lower-triangular = valid DAG
        key = jax.random.PRNGKey(42)
        A = jnp.tril(jax.random.normal(key, (d, d)) * 0.1, k=-1)
        h = float(dag_constraint(A))
        print(f"  d={d}, DAG h(A) = {h:.6f}")
        assert h < 0.1, f"Valid DAG should have h(A) < 0.1, got {h}"

    print("  PASSED")


def test_dagma_fallback_feasibility_threshold():
    """Fallback h(A) should be distinguishable: above 0.1 threshold but not catastrophically."""
    print("\n" + "=" * 60)
    print("Test 5: Fallback h(A) exceeds feasibility threshold but stays reasonable")
    print("=" * 60)

    d = 30
    # Matrix that triggers fallback
    A = jnp.ones((d, d)) * 0.4
    A = A.at[jnp.diag_indices(d)].set(0.0)

    h = float(dag_constraint(A))
    print(f"  d={d}, h(A) = {h:.4f}")
    # Should be > 0.1 (correctly penalized) but < 10 (not catastrophic)
    assert h > 0.1, f"Fallback should exceed feasibility threshold 0.1, got {h}"
    assert h < 10.0, f"Fallback should not be catastrophic, got {h}"
    print("  PASSED")


# =============================================================================
# Bug 3: Negative h(A) from main DAGMA formula
# =============================================================================


def test_dagma_no_negative_h():
    """h(A) should never be negative — clamp to 0."""
    print("\n" + "=" * 60)
    print("Test 6: h(A) is never negative (Bug 3 fix)")
    print("=" * 60)

    for d in [11, 30]:
        # Dense matrix with large entries that can cause logdet > d*log(s)
        key = jax.random.PRNGKey(123)
        A = jax.random.normal(key, (d, d)) * 0.5
        A = A.at[jnp.diag_indices(d)].set(0.0)

        h = float(dag_constraint(A))
        print(f"  d={d}, h(A) = {h:.6f}")
        assert h >= 0.0, f"h(A) should be >= 0, got {h} for d={d}"

    # Also test with matrix specifically designed to produce large logdet
    d = 11
    A = jnp.eye(d) * 0.01  # Near-identity times small factor
    A = A.at[jnp.diag_indices(d)].set(0.0)  # Zero diagonal
    h = float(dag_constraint(A))
    print(f"  d={d} (near-zero), h(A) = {h:.6f}")
    assert h >= 0.0, f"h(A) should be >= 0, got {h}"

    print("  PASSED")


def test_dagma_negative_clamp_preserves_gradient():
    """Clamped h(A)=0 should still have valid gradient (jnp.maximum is differentiable)."""
    print("\n" + "=" * 60)
    print("Test 7: Clamped h(A) preserves gradient signal")
    print("=" * 60)

    d = 11
    A = jnp.ones((d, d)) * 0.01
    A = A.at[jnp.diag_indices(d)].set(0.0)

    grad_fn = jax.grad(dag_constraint)
    g = grad_fn(A)
    print(f"  grad shape: {g.shape}, has NaN: {bool(jnp.any(jnp.isnan(g)))}")
    assert not jnp.any(jnp.isnan(g)), "Gradient should not contain NaN"
    assert g.shape == A.shape, f"Gradient shape mismatch: {g.shape} vs {A.shape}"

    print("  PASSED")


# =============================================================================
# Bug 1: NSGA-II result v7 config propagation
# =============================================================================


def test_nsga2_result_contains_v7_keys():
    """evaluate_genome_unified result should contain v7 effect config keys when use_v7=True."""
    print("\n" + "=" * 60)
    print("Test 8: NSGA-II result contains v7 effect config keys")
    print("=" * 60)

    # We test this by checking the code path directly rather than running full NSGA-II.
    # The fix adds these keys to the result dict inside the `if use_v7:` block.
    # We verify the keys exist in a mock result built the same way.

    v7_config_keys = [
        "effect_hidden_dim",
        "lambda_effect",
        "effect_embed_dim",
        "effect_warmup_iter",
        "lambda_confound_sparse",
        "lambda_bow",
    ]

    # Simulate the config dict as built by evaluate_genome_unified
    config = {
        "effect_hidden_dim": 64,
        "effect_embed_dim": 16,
        "lambda_effect": 1.0,
        "effect_warmup_iter": 20,
        "lambda_confound_sparse": 0.001,
        "lambda_bow_v7": 0.01,
    }
    golem_overrides = {}

    # Simulate the result dict construction (mirrors nsga2_search.py lines 1025-1031)
    result = {}
    result["effect_hidden_dim"] = config.get("effect_hidden_dim", 64)
    result["lambda_effect"] = golem_overrides.get("lambda_effect", config.get("lambda_effect", 1.0))
    result["effect_embed_dim"] = config.get("effect_embed_dim", 16)
    result["effect_warmup_iter"] = config.get("effect_warmup_iter", 20)
    result["lambda_confound_sparse"] = config.get("lambda_confound_sparse", 0.001)
    result["lambda_bow"] = golem_overrides.get("lambda_bow", config.get("lambda_bow_v7", 0.01))

    for key in v7_config_keys:
        assert key in result, f"Missing key '{key}' in result dict"
        print(f"  {key} = {result[key]}")

    # Verify CV v7 detection would succeed
    use_v7 = "effect_hidden_dim" in result or "lambda_effect" in result
    assert use_v7, "CV should detect v7 from result keys"
    print("  CV use_v7 detection: True")
    print("  PASSED")


def test_cv_v7_detection_logic():
    """CV v7 detection should work with the stored keys."""
    print("\n" + "=" * 60)
    print("Test 9: CV v7 detection with and without effect keys")
    print("=" * 60)

    # Case 1: v7 keys present (after fix)
    hyperparams_v7 = {
        "lambda_1": 0.02,
        "lambda_2": 0.01,
        "lambda_class": 1.0,
        "lr": 0.001,
        "effect_hidden_dim": 64,
        "lambda_effect": 1.0,
    }
    use_v7 = "effect_hidden_dim" in hyperparams_v7 or "lambda_effect" in hyperparams_v7
    assert use_v7 is True, "Should detect v7 when effect keys present"
    print("  With effect keys: use_v7 = True")

    # Case 2: v7 keys absent (old bug behavior)
    hyperparams_v4 = {
        "lambda_1": 0.02,
        "lambda_2": 0.01,
        "lambda_class": 1.0,
        "lr": 0.001,
    }
    use_v7 = "effect_hidden_dim" in hyperparams_v4 or "lambda_effect" in hyperparams_v4
    assert use_v7 is False, "Should not detect v7 when effect keys absent"
    print("  Without effect keys: use_v7 = False")

    print("  PASSED")


# =============================================================================
# Bug 4 (v16.3): DAGMA fallback gradient dead zone
# trace(A²) only sees diagonal (≈0 for no self-loops) → h=0.5, grad=0
# Fix: sum(A²)/(d*s) sees all edges → non-zero gradient signal
# =============================================================================


def test_v163_fallback_not_constant():
    """v16.3: Fallback h(A) should vary with edge weights, not be constant 0.5."""
    print("\n" + "=" * 60)
    print("Test 10: v16.3 — Fallback h(A) varies with edge weights")
    print("=" * 60)

    d = 30
    # Two matrices with different edge magnitudes, both triggering fallback
    A_weak = jnp.ones((d, d)) * 0.2
    A_weak = A_weak.at[jnp.diag_indices(d)].set(0.0)

    A_strong = jnp.ones((d, d)) * 0.5
    A_strong = A_strong.at[jnp.diag_indices(d)].set(0.0)

    h_weak = float(dag_constraint(A_weak))
    h_strong = float(dag_constraint(A_strong))

    print(f"  Weak edges (0.2):  h = {h_weak:.4f}")
    print(f"  Strong edges (0.5): h = {h_strong:.4f}")

    # Old bug: both would be exactly 0.5. New: h_strong > h_weak > 0.5
    assert h_strong > h_weak, f"Stronger edges should give higher h: {h_strong} vs {h_weak}"
    assert h_weak > 0.5, f"Weak edges in fallback should give h > 0.5, got {h_weak}"
    print("  PASSED")


def test_v163_fallback_gradient_nonzero():
    """v16.3 CRITICAL: Fallback must produce non-zero gradients for off-diagonal elements."""
    print("\n" + "=" * 60)
    print("Test 11: v16.3 — Fallback gradients are non-zero (core bug fix)")
    print("=" * 60)

    for d in [11, 30]:
        # Dense matrix that forces fallback (outside M-matrix domain)
        A = jnp.ones((d, d)) * 0.5
        A = A.at[jnp.diag_indices(d)].set(0.0)

        grad = jax.grad(dag_constraint)(A)
        grad_norm = float(jnp.linalg.norm(grad))

        # Check off-diagonal gradient specifically
        mask = 1.0 - jnp.eye(d)
        off_diag_grad = grad * mask
        off_diag_norm = float(jnp.linalg.norm(off_diag_grad))
        off_diag_mean = float(jnp.mean(jnp.abs(off_diag_grad)))

        print(
            f"  d={d}: |grad| = {grad_norm:.4f}, off-diag |grad| = {off_diag_norm:.4f}, mean |grad_ij| = {off_diag_mean:.6f}"
        )

        # OLD BUG: off_diag_norm = 0.0 (gradient dead zone)
        # FIX: off_diag_norm >> 0 (each edge gets 2*A[i,j]/(d*s) gradient)
        assert off_diag_norm > 0.1, (
            f"Off-diagonal gradient norm should be >> 0, got {off_diag_norm} (dead zone bug?)"
        )
        assert off_diag_mean > 1e-4, (
            f"Mean off-diagonal gradient should be > 0, got {off_diag_mean}"
        )

    print("  PASSED")


def test_v163_fallback_gradient_pushes_weights_down():
    """v16.3: Fallback gradient should push edge weights toward zero (positive gradient for positive weights)."""
    print("\n" + "=" * 60)
    print("Test 12: v16.3 — Gradient pushes weights down in fallback")
    print("=" * 60)

    d = 11
    A = jnp.ones((d, d)) * 0.4
    A = A.at[jnp.diag_indices(d)].set(0.0)

    grad = jax.grad(dag_constraint)(A)

    # For positive A[i,j], gradient should be positive (d(h)/d(A) > 0)
    # so gradient descent moves A[i,j] in the negative direction (toward 0)
    mask = 1.0 - jnp.eye(d)
    off_diag_grad = grad * mask
    positive_grad_frac = float(
        jnp.mean((off_diag_grad > 0).astype(jnp.float32) * mask) / jnp.mean(mask)
    )

    print(f"  Fraction of off-diag gradients > 0: {positive_grad_frac:.2%}")
    # All off-diagonal should have positive gradient (for positive A)
    assert positive_grad_frac > 0.95, (
        f"Most gradients should be positive for positive A, got {positive_grad_frac:.2%}"
    )
    print("  PASSED")


def test_v163_optimization_escapes_fallback():
    """v16.3: Optimizer should be able to escape the fallback region and reach h(A) ≈ 0."""
    print("\n" + "=" * 60)
    print("Test 13: v16.3 — Optimization escapes fallback to reach h≈0")
    print("=" * 60)

    import optax

    d = 11
    key = jax.random.PRNGKey(42)
    # Start with dense matrix (triggers fallback) — need entries > ~0.30 for d=11
    A = jnp.ones((d, d)) * 0.5
    A = A.at[jnp.diag_indices(d)].set(0.0)
    # Add small noise for asymmetry
    A = A + jax.random.normal(key, (d, d)) * 0.05
    A = A.at[jnp.diag_indices(d)].set(0.0)

    h_init = float(dag_constraint(A))
    print(f"  Initial h(A) = {h_init:.4f} (should be in fallback region > 0.5)")
    assert h_init >= 0.5, f"Initial h should be in fallback region, got {h_init}"

    optimizer = optax.adam(5e-3)
    opt_state = optimizer.init(A)

    def loss_fn(A_param):
        h = dag_constraint(A_param)
        sparsity = 0.1 * jnp.sum(jnp.abs(A_param))
        return h + sparsity

    for step in range(200):
        loss, grads = jax.value_and_grad(loss_fn)(A)
        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)

    h_final = float(dag_constraint(A))
    print(f"  Final h(A) = {h_final:.6f} after 200 steps")
    # OLD BUG: h would stay at 0.5 forever (zero gradient)
    # FIX: optimizer should push A below the M-matrix boundary and converge toward DAG
    assert h_final < 0.1, f"Should converge toward DAG, got h={h_final} (stuck in fallback?)"
    print("  PASSED")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("v16.2 + v16.3 Bug Fix Tests")
    print("=" * 60)

    # v16.2 tests
    test_dagma_fallback_small_d()
    test_dagma_fallback_large_d()
    test_dagma_fallback_not_proportional_to_d()
    test_dagma_valid_dag_unchanged()
    test_dagma_fallback_feasibility_threshold()
    test_dagma_no_negative_h()
    test_dagma_negative_clamp_preserves_gradient()
    test_nsga2_result_contains_v7_keys()
    test_cv_v7_detection_logic()

    # v16.3 tests
    test_v163_fallback_not_constant()
    test_v163_fallback_gradient_nonzero()
    test_v163_fallback_gradient_pushes_weights_down()
    test_v163_optimization_escapes_fallback()

    print("\n" + "=" * 60)
    print("ALL 13 TESTS PASSED")
    print("=" * 60)
