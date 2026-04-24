"""
NOTEARS: DAGs with NO TEARS - Continuous Optimization for Structure Learning

Official paper: Zheng et al., NeurIPS 2018
Official implementation: https://github.com/xunzheng/notears

This module implements NOTEARS' matrix exponential acyclicity constraint for JAX.

Mathematical Foundation (from paper):
------------------------------------
For weighted adjacency matrix W, the acyclicity constraint is:

    h(W) = trace(exp(W ⊙ W)) - d

where:
- W ∈ ℝ^(d×d) is the weighted adjacency matrix
- ⊙ denotes element-wise (Hadamard) product
- d is the number of nodes
- exp() is the matrix exponential

Key property: h(W) = 0 if and only if W is a DAG

This breakthrough formulation converts the combinatorial DAG constraint into a smooth,
differentiable function suitable for gradient-based optimization.

Optimization uses augmented Lagrangian (from the paper):
    min_W  (1/2n)||X - XW||²_F + α||W||₁
    s.t.   h(W) = 0

Solved via:
    L(W, λ, ρ) = (1/2n)||X - XW||²_F + α||W||₁ + λ·h(W) + (ρ/2)·h(W)²

With outer loop: λ ← λ + ρ·h(W), and ρ multiplied by 10 if h not decreasing.

Reference:
    Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018).
    DAGs with NO TEARS: Continuous optimization for structure learning.
    In NeurIPS.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import optax


def notears_acyclicity_constraint(W: jnp.ndarray) -> float:
    """
    Compute NOTEARS' matrix exponential acyclicity constraint.

    h(W) = trace(exp(W ⊙ W)) - d

    Args:
        W: (d, d) weighted adjacency matrix

    Returns:
        h: Scalar acyclicity measure (h = 0 iff W is a DAG)
    """
    d = W.shape[0]
    E = jsla.expm(W * W)
    h = jnp.trace(E) - d
    return h


def notears_penalty_loss(W: jnp.ndarray, lambda_dag: float, rho: float = 1.0) -> float:
    """
    Compute NOTEARS augmented Lagrangian penalty term.

    penalty = λ·h(W) + (ρ/2)·h(W)²

    Args:
        W: (d, d) weighted adjacency matrix
        lambda_dag: Lagrange multiplier
        rho: Quadratic penalty coefficient

    Returns:
        penalty: λ·h(W) + (ρ/2)·h(W)²
    """
    h = notears_acyclicity_constraint(W)
    penalty = lambda_dag * h + 0.5 * rho * (h**2)
    return penalty


def is_dag(W: jnp.ndarray, threshold: float = 1e-3) -> bool:
    """Check if weighted adjacency matrix represents a DAG."""
    h = notears_acyclicity_constraint(W)
    return h < threshold


# =============================================================================
# Gradient computation (for reference - JAX auto-handles this)
# =============================================================================


def notears_acyclicity_gradient(W: jnp.ndarray) -> jnp.ndarray:
    """
    Compute gradient of NOTEARS constraint w.r.t. W.

    ∇_W h(W) = 2W ⊙ E^T  where E = exp(W ⊙ W)
    """
    E = jsla.expm(W * W)
    grad = 2 * W * E.T
    return grad


# =============================================================================
# Augmented Lagrangian optimization schedules
# =============================================================================


def get_augmented_lagrangian_schedule(
    n_epochs: int,
    lambda_init: float = 0.0,
    rho_init: float = 1.0,
    rho_max: float = 1e16,
    h_threshold: float = 1e-8,
    gamma: float = 0.25,
) -> tuple:
    """
    Return parameters for NOTEARS augmented Lagrangian.

    In practice λ and ρ are updated dynamically based on h(W).
    """
    return {
        "lambda_init": lambda_init,
        "rho_init": rho_init,
        "rho_max": rho_max,
        "h_threshold": h_threshold,
        "gamma": gamma,
    }


# =============================================================================
# Full NOTEARS Training Algorithm (Augmented Lagrangian)
# =============================================================================


def learn_with_notears(
    X: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_outer: int = 10,
    n_inner: int = 50,
    alpha_sparse: float = 0.001,
    learning_rate: float = 1e-3,
    rho_init: float = 1.0,
    rho_max: float = 1e16,
    h_tol: float = 1e-8,
    gamma: float = 0.25,
    verbose: bool = True,
    A_prior: jnp.ndarray = None,
    lambda_prior: float = 0.1,
    # Legacy API compatibility
    n_iterations: int = None,
    lambda_init: float = None,
    lambda_final: float = None,
) -> jnp.ndarray:
    """
    Learn causal structure using NOTEARS augmented Lagrangian (Zheng et al. 2018).

    Objective:
        min_W  (1/2n)||X - XW||²_F + α||W||₁
        s.t.   h(W) = trace(exp(W⊙W)) - d = 0

    Solved via augmented Lagrangian with outer/inner loop structure.

    Args:
        X: (n_samples, n_vars) data matrix
        key: JAX random key
        n_outer: Number of augmented Lagrangian outer iterations (default: 10)
        n_inner: Number of gradient steps per outer iteration (default: 50)
        alpha_sparse: L1 sparsity penalty
        learning_rate: Learning rate for optimizer
        rho_init: Initial quadratic penalty coefficient (default: 1.0)
        rho_max: Maximum rho (default: 1e16)
        h_tol: Convergence tolerance for h(W) (default: 1e-8)
        gamma: Factor for checking h decrease (default: 0.25)
        verbose: Print progress
        A_prior: Optional prior adjacency matrix
        lambda_prior: Weight for prior constraint (default: 0.1)

    Returns:
        A: (n_vars, n_vars) learned weighted adjacency matrix
    """
    n_samples, n_vars = X.shape

    # Initialize A with small random values, zero diagonal (no self-loops)
    A = jax.random.normal(key, (n_vars, n_vars)) * 0.1
    A = A.at[jnp.diag_indices(n_vars)].set(0.0)

    # Augmented Lagrangian parameters
    lambda_al = 0.0
    rho = rho_init

    if verbose:
        print(f"NOTEARS (Augmented Lagrangian): {n_outer} outer x {n_inner} inner")
        print(f"  Data: {n_samples} samples, {n_vars} variables")
        print(f"  α_sparse: {alpha_sparse}, ρ_init: {rho_init}")
        if A_prior is not None:
            n_prior_edges = int(jnp.sum(A_prior))
            print(f"  Prior: {n_prior_edges} edges (λ_prior={lambda_prior:.3f})")

    h_prev = float("inf")

    for outer in range(n_outer):
        # Fresh optimizer each outer step (as in original NOTEARS)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(A)

        # Capture current AL params for closure
        _lambda_al = lambda_al
        _rho = rho

        # Inner optimization loop
        for inner in range(n_inner):

            def loss_fn(A_param):
                # Least-squares loss: (1/2n)||X - X·A||²_F
                residuals = X - X @ A_param
                recon_loss = 0.5 * jnp.sum(residuals**2) / n_samples

                # Acyclicity constraint
                h = notears_acyclicity_constraint(A_param)

                # Augmented Lagrangian: λ·h + (ρ/2)·h²
                al_loss = _lambda_al * h + 0.5 * _rho * (h**2)

                # Sparsity penalty
                sparse_loss = alpha_sparse * jnp.sum(jnp.abs(A_param))

                # Prior penalty
                if A_prior is not None:
                    prior_mask = 1.0 - A_prior
                    prior_loss = lambda_prior * jnp.sum((A_param * prior_mask) ** 2)
                else:
                    prior_loss = 0.0

                total = recon_loss + al_loss + sparse_loss + prior_loss
                return total

            grads = jax.grad(loss_fn)(A)
            updates, opt_state = optimizer.update(grads, opt_state, A)
            A = optax.apply_updates(A, updates)

            # Zero diagonal (no self-loops)
            A = A.at[jnp.diag_indices(n_vars)].set(0.0)

        # Evaluate h after inner loop
        h_val = float(notears_acyclicity_constraint(A))

        if verbose:
            A_max = float(jnp.max(jnp.abs(A)))
            recon = float(0.5 * jnp.sum((X - X @ A) ** 2) / n_samples)
            print(
                f"  Outer {outer + 1:2d}: h={h_val:.6f}, recon={recon:.4f}, "
                f"λ={lambda_al:.2f}, ρ={rho:.1e}, max|A|={A_max:.4f}"
            )

        # Check convergence
        if h_val < h_tol:
            if verbose:
                print(f"  Converged: h < {h_tol}")
            break

        # Update Lagrange multiplier: λ ← λ + ρ·h
        lambda_al = lambda_al + rho * h_val

        # Update rho if h not decreasing fast enough
        if h_val > gamma * h_prev:
            rho = min(rho * 10.0, rho_max)

        h_prev = h_val

    if verbose:
        final_h = float(notears_acyclicity_constraint(A))
        print(f"\nFinal acyclicity: h(A) = {final_h:.6f}")
        if final_h < 0.01:
            print("[OK] Converged to DAG (h < 0.01)")
        elif final_h < 0.1:
            print("[WARN] Near-DAG (h < 0.1)")
        else:
            print("[FAIL] Not converged to DAG (h > 0.1)")

    return A
