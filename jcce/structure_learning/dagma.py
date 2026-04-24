"""
DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization

Official paper: Bello et al., NeurIPS 2022
Official implementation: https://github.com/kevinsbello/dagma

This module implements DAGMA's log-determinant acyclicity constraint for JAX.

Mathematical Foundation (from paper):
------------------------------------
For weighted adjacency matrix W, the acyclicity constraint is:

    h(W) = -log det(sI - W ⊙ W) + d log(s)

where:
- W ∈ ℝ^(d×d) is the weighted adjacency matrix
- ⊙ denotes element-wise (Hadamard) product
- s > 0 is a parameter (typically s=1.0)
- d is the number of nodes

Key property: h(W) = 0 if and only if W is a DAG

This is based on the M-matrix characterization: sI - W⊙W must be an M-matrix
(positive definite with non-positive off-diagonal elements) for W to be acyclic.

Advantages over NOTEARS:
- Faster: O(d³) for log-det vs O(d³) for matrix exponential
- Better cycle detection
- Better gradient behavior
- Exact constraint via barrier method
"""

import jax
import jax.numpy as jnp
import optax
from typing import Optional


def dagma_acyclicity_constraint(W: jnp.ndarray, s: float = 1.0) -> float:
    """
    Compute DAGMA's log-determinant acyclicity constraint.

    This is the official formulation from Bello et al. (NeurIPS 2022).

    Args:
        W: (d, d) weighted adjacency matrix
           Can contain continuous weights (not just 0/1)
        s: Scalar parameter for M-matrix domain (default 1.0)
           Must satisfy: s > max eigenvalue of W⊙W

    Returns:
        h: Scalar acyclicity measure
           h = 0 ⟺ W is a DAG
           h > 0 ⟺ W contains cycles

    Mathematical details:
        M = sI - W⊙W  (element-wise square)
        h(W) = -log|det(M)| + d·log(s)

        For DAG: W⊙W is nilpotent, so eigenvalues are 0
                 → det(M) = s^d
                 → h = -d·log(s) + d·log(s) = 0

        For cycle: W⊙W has non-zero eigenvalues
                  → det(M) < s^d
                  → h > 0

    Implementation notes:
        - Uses jax.numpy.linalg.slogdet() for numerical stability
        - Returns sign and log|det| separately to avoid overflow
        - If det(M) ≤ 0, we're outside M-matrix domain (add large penalty)

    Example:
        >>> # DAG (strictly lower triangular)
        >>> W_dag = jnp.array([[0, 0], [0.5, 0]])
        >>> h = dagma_acyclicity_constraint(W_dag)
        >>> print(f"h = {h:.6f}")  # Should be ≈ 0

        >>> # Cycle
        >>> W_cycle = jnp.array([[0, 0.5], [0.5, 0]])
        >>> h = dagma_acyclicity_constraint(W_cycle)
        >>> print(f"h = {h:.6f}")  # Should be > 0
    """
    d = W.shape[0]

    # Compute M = sI - W⊙W (element-wise square)
    # For numerical stability, use W * W instead of W ** 2
    M = s * jnp.eye(d) - W * W

    # Compute log-determinant using slogdet for stability
    # Returns: (sign, logabsdet) where det(M) = sign * exp(logabsdet)
    sign, logabsdet = jnp.linalg.slogdet(M)

    # DAGMA constraint: h(W) = -log|det(M)| + d·log(s)
    h = -logabsdet + d * jnp.log(s)

    # If det(M) ≤ 0, we're outside the M-matrix domain
    # This should not happen if s is chosen correctly, but add penalty just in case
    # sign > 0 means det(M) > 0 (good)
    # sign ≤ 0 means det(M) ≤ 0 (bad - outside domain)
    h = jnp.where(sign > 0, h, h + 1e8)  # Large penalty if outside domain

    return h


def dagma_penalty_loss(
    W: jnp.ndarray,
    lambda_dag: float,
    s: float = 1.0
) -> float:
    """
    Compute DAGMA penalty term for adding to VAE loss.

    Loss = VAE_loss + dagma_penalty_loss(W, lambda_dag)

    Args:
        W: (d, d) weighted adjacency matrix (learnable parameter)
        lambda_dag: Penalty coefficient
                    Start small (e.g., 0.0) and increase during training
                    Final value typically 10-100
        s: Barrier parameter (default 1.0)
           Can increase if optimization exits M-matrix domain

    Returns:
        penalty: lambda_dag * h(W)

    Usage in training:
        >>> W = jnp.zeros((10, 10))  # Initialize adjacency
        >>> lambda_schedule = jnp.linspace(0, 50, num_epochs)
        >>> for epoch, lambda_t in enumerate(lambda_schedule):
        >>>     loss = vae_loss + dagma_penalty_loss(W, lambda_t)
        >>>     # Optimize loss w.r.t. both VAE params and W
    """
    h = dagma_acyclicity_constraint(W, s)
    return lambda_dag * h


def is_dag(W: jnp.ndarray, threshold: float = 1e-3) -> bool:
    """
    Check if weighted adjacency matrix represents a DAG.

    Args:
        W: (d, d) weighted adjacency matrix
        threshold: Tolerance for h(W) ≈ 0

    Returns:
        is_acyclic: True if W represents a DAG

    Note: Uses threshold because floating-point arithmetic is imprecise.
          Typical values: 1e-3 (loose) to 1e-6 (strict)
    """
    h = dagma_acyclicity_constraint(W, s=1.0)
    return h < threshold


# =============================================================================
# Gradient computation (for reference - JAX auto-handles this)
# =============================================================================

def dagma_acyclicity_gradient(W: jnp.ndarray, s: float = 1.0) -> jnp.ndarray:
    """
    Compute gradient of DAGMA constraint w.r.t. W.

    This is provided for reference - JAX's autodiff will compute this automatically.

    From the official implementation:
        ∇_W h(W) = 2W ⊙ (M^{-1})^T

    where M = sI - W⊙W

    Args:
        W: (d, d) weighted adjacency matrix
        s: Barrier parameter

    Returns:
        grad: (d, d) gradient ∇_W h(W)
    """
    d = W.shape[0]
    M = s * jnp.eye(d) - W * W

    # Compute M^{-1}
    M_inv = jnp.linalg.inv(M)

    # Gradient: 2W ⊙ (M^{-1})^T
    grad = 2 * W * M_inv.T

    return grad


# =============================================================================
# Optimization schedules (for path-following approach)
# =============================================================================

def get_lambda_schedule(
    n_epochs: int,
    lambda_init: float = 0.0,
    lambda_final: float = 50.0,
    schedule_type: str = 'linear'
) -> jnp.ndarray:
    """
    Generate schedule for DAGMA penalty coefficient λ.

    The path-following approach gradually increases λ to tighten the constraint.

    Args:
        n_epochs: Total number of training epochs
        lambda_init: Initial λ (start with 0 to allow unconstrained learning)
        lambda_final: Final λ (end with large value to enforce DAG)
        schedule_type: 'linear' or 'exponential'

    Returns:
        lambda_schedule: (n_epochs,) array of λ values

    Typical usage:
        - Phase 1 (epochs 0-20%): λ = 0 (learn structure freely)
        - Phase 2 (epochs 20-100%): λ increases (enforce acyclicity)

    Example:
        >>> lambdas = get_lambda_schedule(100, 0.0, 50.0, 'linear')
        >>> # lambdas[0] = 0.0, lambdas[-1] = 50.0
    """
    if schedule_type == 'linear':
        return jnp.linspace(lambda_init, lambda_final, n_epochs)
    elif schedule_type == 'exponential':
        # Avoid log(0)
        if lambda_init == 0.0:
            lambda_init = 1e-3
        log_init = jnp.log(lambda_init)
        log_final = jnp.log(lambda_final)
        log_schedule = jnp.linspace(log_init, log_final, n_epochs)
        return jnp.exp(log_schedule)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


# =============================================================================
# Full DAGMA Training Algorithm
# =============================================================================

def learn_with_dagma(
    X: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_iterations: int = 20,
    lambda_init: float = 1.0,
    lambda_final: float = 20.0,
    alpha_sparse: float = 0.001,
    s: float = 1.0,
    learning_rate: float = 1e-3,
    verbose: bool = True,
    A_prior: jnp.ndarray = None,  # NEW: Prior from GA's A_topology
    lambda_prior: float = 0.1,    # NEW: Prior constraint weight
) -> jnp.ndarray:
    """
    Learn causal structure using DAGMA algorithm.

    This implements pure structure learning (Stage 1 of two-stage approach).

    Args:
        X: (n_samples, n_vars) data matrix
        key: JAX random key
        n_iterations: Number of optimization iterations
        lambda_init: Initial DAG penalty weight
        lambda_final: Final DAG penalty weight
        alpha_sparse: L1 sparsity penalty
        s: DAGMA barrier parameter
        learning_rate: Learning rate for optimizer
        verbose: Print progress
        A_prior: (n_vars, n_vars) prior adjacency matrix (from GA's A_topology).
                 If provided, adds soft constraint encouraging edges where A_prior has them.
                 This makes the algorithm "truly memetic" (Phase 2.3).
        lambda_prior: Weight for prior constraint (default: 0.1)

    Returns:
        A: (n_vars, n_vars) learned adjacency matrix

    Algorithm:
        1. Initialize A randomly
        2. For each iteration:
           - Compute loss = reconstruction_loss + λ*h(A) + α*||A||₁ + λ_prior*prior_penalty
           - Update A via gradient descent
           - Increase λ (path-following)
        3. Return learned A
    """
    n_samples, n_vars = X.shape

    # Initialize A randomly (small values), zero diagonal (no self-loops)
    A = jax.random.normal(key, (n_vars, n_vars)) * 0.1
    A = A.at[jnp.diag_indices(n_vars)].set(0.0)

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(A)

    # Lambda schedule
    lambda_schedule = jnp.linspace(lambda_init, lambda_final, n_iterations)

    if verbose:
        print(f"DAGMA: Training for {n_iterations} iterations")
        print(f"  Data: {n_samples} samples, {n_vars} variables")
        print(f"  λ schedule: {lambda_init:.2f} → {lambda_final:.2f}")
        print(f"  α_sparse: {alpha_sparse}")
        if A_prior is not None:
            n_prior_edges = int(jnp.sum(A_prior))
            print(f"  Prior: {n_prior_edges} edges (λ_prior={lambda_prior:.3f}) [Memetic Component]")

    # Training loop
    for iteration in range(n_iterations):
        lambda_dag = float(lambda_schedule[iteration])

        # Define loss function
        def loss_fn(A_param):
            # Reconstruction loss: ||X - X·A||²
            X_pred = X @ A_param
            recon_loss = jnp.mean((X - X_pred) ** 2)

            # DAG penalty
            dag_loss = dagma_penalty_loss(A_param, lambda_dag, s)

            # Sparsity penalty
            sparse_loss = alpha_sparse * jnp.sum(jnp.abs(A_param))

            # Prior penalty (memetic component - Phase 2.3)
            if A_prior is not None:
                # Penalize edges where A_prior has no edge (prior_mask = 1 where no prior edge)
                prior_mask = 1.0 - A_prior
                prior_penalty = jnp.sum((A_param * prior_mask) ** 2)
                prior_loss = lambda_prior * prior_penalty
            else:
                prior_loss = 0.0

            total_loss = recon_loss + dag_loss + sparse_loss + prior_loss

            return total_loss, {
                'total': total_loss,
                'recon': recon_loss,
                'dag': dag_loss,
                'sparse': sparse_loss,
                'prior': prior_loss,
            }

        # Compute gradients and update
        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(A)
        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)

        # Zero diagonal (no self-loops)
        A = A.at[jnp.diag_indices(n_vars)].set(0.0)

        # Log progress
        if verbose and (iteration % 5 == 0 or iteration == n_iterations - 1):
            h_val = dagma_acyclicity_constraint(A, s)
            A_norm = jnp.linalg.norm(A)
            A_max = jnp.max(jnp.abs(A))
            log_str = (f"  Iter {iteration+1:2d}: Loss={metrics['total']:.4f}, "
                      f"Recon={metrics['recon']:.4f}, h={h_val:.4f}, "
                      f"||A||={A_norm:.4f}, max|A|={A_max:.4f}")
            if A_prior is not None:
                log_str += f", Prior={metrics['prior']:.4f}"
            print(log_str)

    if verbose:
        final_h = dagma_acyclicity_constraint(A, s)
        print(f"\nFinal acyclicity: h(A) = {final_h:.6f}")
        if final_h < 0.01:
            print("[OK] Converged to DAG (h < 0.01)")
        elif final_h < 0.1:
            print("[WARN] Near-DAG (h < 0.1)")
        else:
            print("[FAIL] Not converged to DAG (h > 0.1)")

    return A
