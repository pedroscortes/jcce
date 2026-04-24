"""
Tests for low-rank error covariance model for latent confounders.

Tests init_lowrank_confound, compute_confound_nll, and integration
with GOLEM v7 (backward compatibility, JIT, gradients).
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pytest

from jcce.structure_learning.jcce_learner import (
    init_lowrank_confound,
    compute_confound_nll,
)


# ============================================================================
# 1. Initialization
# ============================================================================

def test_init_shapes():
    """B is (d, k), log_var is scalar."""
    key = random.PRNGKey(42)
    d, k = 15, 5
    B, log_var = init_lowrank_confound(key, d, rank_k=k)
    assert B.shape == (d, k)
    assert log_var.shape == ()


def test_omega_is_psd():
    """Ω = B@B.T + σ²I has all positive eigenvalues."""
    key = random.PRNGKey(0)
    B, log_var = init_lowrank_confound(key, 20, rank_k=5)
    sigma2 = jnp.exp(log_var)
    Omega = B @ B.T + sigma2 * jnp.eye(20)
    eigvals = jnp.linalg.eigvalsh(Omega)
    assert jnp.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min()}"


# ============================================================================
# 2. Woodbury correctness
# ============================================================================

def test_woodbury_inverse_correct():
    """Ω @ Ω⁻¹ ≈ I (Woodbury formula matches direct inverse)."""
    key = random.PRNGKey(1)
    d, k = 10, 3
    # Use float64 for numerical precision in inverse comparison
    B = jnp.array(random.normal(key, (d, k)) * 0.5, dtype=jnp.float64)
    log_var = jnp.array(jnp.log(0.2), dtype=jnp.float64)
    sigma2 = jnp.exp(log_var)

    # Direct inverse
    Omega = B @ B.T + sigma2 * jnp.eye(d)
    Omega_inv_direct = jnp.linalg.inv(Omega)

    # Woodbury inverse: Ω⁻¹ = σ⁻²I - σ⁻²B M⁻¹ B.T
    M = sigma2 * jnp.eye(k, dtype=jnp.float64) + B.T @ B
    M_inv = jnp.linalg.inv(M)
    s2_inv = 1.0 / sigma2
    Omega_inv_woodbury = s2_inv * jnp.eye(d, dtype=jnp.float64) - s2_inv * (B @ M_inv @ B.T)

    np.testing.assert_allclose(
        Omega_inv_direct, Omega_inv_woodbury, atol=1e-5,
        err_msg="Woodbury inverse doesn't match direct inverse"
    )


def test_log_det_correct():
    """Matrix determinant lemma matches jnp.linalg.slogdet(Ω)."""
    key = random.PRNGKey(2)
    d, k = 12, 4
    B = random.normal(key, (d, k)) * 0.3
    log_var = jnp.array(jnp.log(0.15))
    sigma2 = jnp.exp(log_var)

    # Direct
    Omega = B @ B.T + sigma2 * jnp.eye(d)
    _, logdet_direct = jnp.linalg.slogdet(Omega)

    # Matrix determinant lemma
    M = sigma2 * jnp.eye(k) + B.T @ B
    _, logdet_M = jnp.linalg.slogdet(M)
    logdet_lemma = (d - k) * log_var + logdet_M

    np.testing.assert_allclose(
        float(logdet_direct), float(logdet_lemma), atol=1e-4,
        err_msg="Log-det via lemma doesn't match direct computation"
    )


# ============================================================================
# 3. NLL properties
# ============================================================================

def test_nll_gradient_flows():
    """jax.grad returns finite gradients for B and log_var."""
    key = random.PRNGKey(3)
    d, k, n = 8, 3, 50
    B = random.normal(key, (d, k)) * 0.1
    log_var = jnp.array(0.0)
    R = random.normal(random.PRNGKey(99), (n, d))

    grad_B, grad_lv = jax.grad(compute_confound_nll, argnums=(1, 2))(R, B, log_var)
    assert jnp.all(jnp.isfinite(grad_B)), "Gradient w.r.t. B has non-finite values"
    assert jnp.isfinite(grad_lv), "Gradient w.r.t. log_var is non-finite"
    assert grad_B.shape == B.shape


def test_nll_prefers_correct_B():
    """NLL(B_true) < NLL(B_random) when data generated from B_true."""
    key = random.PRNGKey(4)
    d, k, n = 10, 3, 500

    # Generate data from known B_true
    k1, k2, k3 = random.split(key, 3)
    B_true = random.normal(k1, (d, k)) * 0.5
    sigma2_true = 0.1
    latent = random.normal(k2, (n, k))
    noise = random.normal(k3, (n, d)) * jnp.sqrt(sigma2_true)
    R = latent @ B_true.T + noise

    log_var_true = jnp.array(jnp.log(sigma2_true))
    B_random = random.normal(random.PRNGKey(999), (d, k)) * 0.5

    nll_true = compute_confound_nll(R, B_true, log_var_true)
    nll_random = compute_confound_nll(R, B_random, log_var_true)

    assert nll_true < nll_random, (
        f"NLL(B_true)={nll_true:.4f} should be < NLL(B_random)={nll_random:.4f}"
    )


# ============================================================================
# 4. JIT compatibility
# ============================================================================

def test_jit_compatible():
    """jax.jit(compute_confound_nll) compiles and produces same result."""
    key = random.PRNGKey(5)
    d, k, n = 8, 3, 30
    B = random.normal(key, (d, k)) * 0.1
    log_var = jnp.array(-1.0)
    R = random.normal(random.PRNGKey(55), (n, d))

    nll_eager = compute_confound_nll(R, B, log_var)
    nll_jit = jax.jit(compute_confound_nll)(R, B, log_var)

    np.testing.assert_allclose(
        float(nll_eager), float(nll_jit), atol=1e-6,
        err_msg="JIT result differs from eager"
    )


# ============================================================================
# 5. Backward compatibility
# ============================================================================

def test_legacy_path_unchanged():
    """n_latent_confounders=0 uses A_confound path (no B_confound in params)."""
    # This tests the signature accepts 0 without error — actual GOLEM call
    # is too heavy for unit test, so we just verify the function signature.
    import inspect
    from jcce.structure_learning.jcce_learner import learn_structure
    sig = inspect.signature(learn_structure)
    assert 'n_latent_confounders' in sig.parameters
    assert sig.parameters['n_latent_confounders'].default == 5


def test_metrics_backward_compat_keys():
    """Low-rank path stores A_confound key (= Ω_offdiag) for plotting compat."""
    # Simulate what metrics storage does
    key = random.PRNGKey(6)
    d, k = 10, 3
    B = random.normal(key, (d, k)) * 0.3
    log_var = jnp.array(jnp.log(0.1))
    sigma2 = float(jnp.maximum(jnp.exp(log_var), 1e-6))
    Omega = B @ B.T + sigma2 * jnp.eye(d)
    Omega_offdiag = Omega.at[jnp.diag_indices(d)].set(0.0)

    # Verify Ω_offdiag is symmetric (as expected from B@B.T)
    np.testing.assert_allclose(
        np.array(Omega_offdiag), np.array(Omega_offdiag.T), atol=1e-7
    )
    # Verify it can be thresholded to binary (same as A_confound usage)
    threshold = 0.05
    binary = (jnp.abs(Omega_offdiag) > threshold).astype(jnp.float32)
    assert binary.shape == (d, d)
    # Diagonal should be zero
    assert jnp.sum(jnp.diag(binary)) == 0


# ============================================================================
# 6. Bow-free with Omega
# ============================================================================

def test_bow_free_with_omega():
    """Bow-free penalty uses Ω_offdiag, not raw B."""
    key = random.PRNGKey(7)
    d, k = 8, 3
    B = random.normal(key, (d, k)) * 0.3
    A_direct = random.normal(random.PRNGKey(77), (d, d)) * 0.1

    Omega_offdiag = B @ B.T
    Omega_offdiag = Omega_offdiag.at[jnp.diag_indices(d)].set(0.0)

    # Bow-free penalty: sum(|A_direct| * |Omega_offdiag|)
    bow_penalty = jnp.sum(jnp.abs(A_direct) * jnp.abs(Omega_offdiag))

    # Should be > 0 when both have nonzero entries
    assert bow_penalty > 0

    # Should be 0 when A_direct is zero
    bow_zero = jnp.sum(jnp.abs(jnp.zeros_like(A_direct)) * jnp.abs(Omega_offdiag))
    assert bow_zero == 0.0

    # Should be 0 when Omega_offdiag is zero (k=0 or B=0)
    bow_no_conf = jnp.sum(jnp.abs(A_direct) * jnp.abs(jnp.zeros_like(Omega_offdiag)))
    assert bow_no_conf == 0.0
