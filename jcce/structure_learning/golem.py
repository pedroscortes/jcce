"""
GOLEM: Gradient-based Optimization of DAG-penalized Likelihood for learning linEar DAG Models

Official paper: Ng, Ghassami, and Zhang, NeurIPS 2020
Official implementation: https://github.com/ignavierng/golem

This module implements GOLEM's likelihood-based structure learning for JAX.

Mathematical Foundation (from paper):
------------------------------------
GOLEM replaces NOTEARS' least squares objective with a proper Gaussian likelihood:

**GOLEM-EV (Equal Variances):**
    score(B; X) = (d/2) log ||X - XB||² - log|det(I - B)|

**GOLEM-NV (Non-Equal Variances):**
    score(B; X) = (1/2) Σᵢ log ||Xᵢ - XB||² - log|det(I - B)|

The key innovation is the **log-determinant term** log|det(I - B)|, which represents
the Jacobian of the transformation. This comes from the proper likelihood derivation.

DAG Constraint:
    h(B) = trace(exp(B ⊙ B)) - d  (same as NOTEARS)

But GOLEM treats h(B) as a **soft constraint** (in the objective), while NOTEARS
uses augmented Lagrangian (hard constraint that's iteratively enforced).

Overall Objective:
    L(B) = score(B; X) + λ₁·||B||₁ + λ₂·h(B)

This is simpler to optimize than NOTEARS' augmented Lagrangian!

Key Differences from NOTEARS:
1. Likelihood-based (not least squares)
2. Soft DAG constraint (not hard via augmented Lagrangian)
3. Simpler optimization (no λ/ρ schedule)
4. Better theoretical properties

Reference:
    Ng, I., Ghassami, A., & Zhang, K. (2020).
    On the role of sparsity and dag constraints for learning linear dags.
    In NeurIPS.
"""

import jax
import jax.numpy as jnp
import optax


def golem_likelihood_ev(X: jnp.ndarray, B: jnp.ndarray) -> float:
    """
    Compute GOLEM-EV (Equal Variances) likelihood score.

    This assumes homoscedastic noise: all variables have the same noise variance.

    Args:
        X: (n, d) data matrix (n samples, d variables)
        B: (d, d) weighted adjacency matrix

    Returns:
        score: Negative log-likelihood (to minimize)

    Mathematical formula:
        score = (d/2) · log(||X - X·B||²_F) - log|det(I - B)|

    where ||·||²_F is the squared Frobenius norm.

    Note:
        - First term: RSS (Residual Sum of Squares)
        - Second term: Jacobian (from change of variables)
        - We want to MAXIMIZE likelihood = MINIMIZE score
    """
    n, d = X.shape

    # Residuals: X - X·B
    residuals = X - X @ B

    # RSS: ||X - X·B||²_F / n = (1/n) Σᵢⱼ (Xᵢⱼ - (X·B)ᵢⱼ)²
    rss = jnp.sum(residuals**2) / n

    # Log RSS term (averaged over dimensions)
    # From paper: (d/2) log(RSS/n) where RSS/n is the average squared residual
    log_rss_term = (d / 2.0) * jnp.log(rss)

    # Jacobian term: -log|det(I - B)|
    I = jnp.eye(d)
    sign, logabsdet = jnp.linalg.slogdet(I - B)

    # If det(I-B) <= 0, we're outside valid region (add large penalty)
    jacobian_term = jnp.where(sign > 0, -logabsdet, 1e8)

    score = log_rss_term + jacobian_term

    return score


def golem_likelihood_nv(X: jnp.ndarray, B: jnp.ndarray) -> float:
    """
    Compute GOLEM-NV (Non-Equal Variances) likelihood score.

    This assumes heteroscedastic noise: each variable has its own noise variance.

    Args:
        X: (n, d) data matrix (n samples, d variables)
        B: (d, d) weighted adjacency matrix

    Returns:
        score: Negative log-likelihood (to minimize)

    Mathematical formula:
        score = (1/2) · Σᵢ log(||Xᵢ - X·B||²) - log|det(I - B)|

    where Xᵢ is the i-th column of X.

    This is more flexible than GOLEM-EV (can model different noise levels).
    """
    n, d = X.shape

    # Residuals: X - X·B
    residuals = X - X @ B

    # Column-wise squared norms: ||Xᵢ - (X·B)ᵢ||² / n
    column_squared_norms = jnp.sum(residuals**2, axis=0) / n  # (d,)

    # Sum of log column norms
    # From paper: (1/2) Σᵢ log(||Xᵢ - XB_ᵢ||²/n)
    log_rss_term = 0.5 * jnp.sum(jnp.log(column_squared_norms))

    # Jacobian term: -log|det(I - B)|
    I = jnp.eye(d)
    sign, logabsdet = jnp.linalg.slogdet(I - B)

    # If det(I-B) <= 0, add large penalty
    jacobian_term = jnp.where(sign > 0, -logabsdet, 1e8)

    score = log_rss_term + jacobian_term

    return score


def golem_acyclicity_constraint(B: jnp.ndarray) -> float:
    """
    Compute acyclicity constraint for GOLEM.

    GOLEM uses the same constraint as NOTEARS:
        h(B) = trace(exp(B ⊙ B)) - d

    Args:
        B: (d, d) weighted adjacency matrix

    Returns:
        h: Acyclicity measure
           h = 0 ⟺ B is a DAG
           h > 0 ⟺ B has cycles
    """
    # Same as NOTEARS
    from jcce.structure_learning.notears import notears_acyclicity_constraint

    return notears_acyclicity_constraint(B)


def golem_score(
    X: jnp.ndarray, B: jnp.ndarray, lambda_1: float, lambda_2: float, equal_variances: bool = True
) -> float:
    """
    Compute GOLEM's overall score function.

    This is the complete objective that GOLEM minimizes:
        L(B) = score(B; X) + λ₁·||B||₁ + λ₂·h(B)

    Args:
        X: (n, d) data matrix
        B: (d, d) weighted adjacency matrix
        lambda_1: Coefficient for L1 penalty (sparsity)
        lambda_2: Coefficient for DAG penalty (acyclicity)
        equal_variances: If True, use GOLEM-EV; else GOLEM-NV

    Returns:
        loss: Total GOLEM objective (to minimize)

    Typical hyperparameters (from paper):
        - GOLEM-EV: lambda_1=2e-2, lambda_2=5.0
        - GOLEM-NV: lambda_1=2e-3, lambda_2=5.0
    """
    # Likelihood term
    if equal_variances:
        likelihood = golem_likelihood_ev(X, B)
    else:
        likelihood = golem_likelihood_nv(X, B)

    # L1 penalty (sparsity)
    l1_penalty = lambda_1 * jnp.sum(jnp.abs(B))

    # DAG penalty (acyclicity)
    dag_penalty = lambda_2 * golem_acyclicity_constraint(B)

    # Total score
    score = likelihood + l1_penalty + dag_penalty

    return score


def is_dag(B: jnp.ndarray, threshold: float = 1e-3) -> bool:
    """
    Check if weighted adjacency matrix represents a DAG.

    Args:
        B: (d, d) weighted adjacency matrix
        threshold: Tolerance for h(B) ≈ 0

    Returns:
        is_acyclic: True if B represents a DAG
    """
    h = golem_acyclicity_constraint(B)
    return h < threshold


# =============================================================================
# Full GOLEM Training Algorithm
# =============================================================================


def learn_with_golem(
    X: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_epochs: int = 100,
    lambda_init: float = 1.0,
    lambda_final: float = 20.0,
    alpha_sparse: float = 0.001,
    learning_rate: float = 1e-3,
    variant: str = "ev",
    verbose: bool = True,
    A_prior: jnp.ndarray = None,
    lambda_prior: float = 0.1,
) -> jnp.ndarray:
    """
    Learn causal structure using GOLEM algorithm.

    This implements pure structure learning (Stage 1 of two-stage approach).

    Args:
        X: (n_samples, n_vars) data matrix
        key: JAX random key
        n_epochs: Number of training epochs
        lambda_init: Initial DAG penalty weight
        lambda_final: Final DAG penalty weight
        alpha_sparse: L1 sparsity penalty
        learning_rate: Learning rate for optimizer
        variant: 'ev' (equal variances) or 'nv' (non-equal variances)
        verbose: Print progress
        A_prior: (n_vars, n_vars) prior adjacency matrix (from GA's A_topology).
                 If provided, adds soft constraint encouraging edges where A_prior has them.
                 This makes the algorithm "truly memetic" (Phase 2.3).
        lambda_prior: Weight for prior constraint (default: 0.1)

    Returns:
        B: (n_vars, n_vars) learned adjacency matrix
    """
    n_samples, n_vars = X.shape

    # Initialize B randomly (small values), zero diagonal (no self-loops)
    B = jax.random.normal(key, (n_vars, n_vars)) * 0.1
    B = B.at[jnp.diag_indices(n_vars)].set(0.0)

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(B)

    # Lambda schedule
    lambda_schedule = jnp.linspace(lambda_init, lambda_final, n_epochs)

    if verbose:
        print(f"GOLEM-{variant.upper()}: Training for {n_epochs} epochs")
        print(f"  Data: {n_samples} samples, {n_vars} variables")
        print(f"  λ schedule: {lambda_init:.2f} → {lambda_final:.2f}")
        print(f"  α_sparse: {alpha_sparse}")
        if A_prior is not None:
            n_prior_edges = int(jnp.sum(A_prior))
            print(
                f"  Prior: {n_prior_edges} edges (λ_prior={lambda_prior:.3f}) [Memetic Component]"
            )

    # Training loop
    for epoch in range(n_epochs):
        lambda_dag = float(lambda_schedule[epoch])

        # Define loss function
        def loss_fn(B_param):
            # Likelihood score
            if variant == "ev":
                likelihood = golem_likelihood_ev(X, B_param)
            else:  # nv
                likelihood = golem_likelihood_nv(X, B_param)

            # DAG penalty (using GOLEM acyclicity constraint)
            h = golem_acyclicity_constraint(B_param)
            dag_loss = lambda_dag * h

            # Sparsity penalty
            sparse_loss = alpha_sparse * jnp.sum(jnp.abs(B_param))

            # Prior penalty (memetic component - Phase 2.3)
            if A_prior is not None:
                # Penalize edges where A_prior has no edge (prior_mask = 1 where no prior edge)
                prior_mask = 1.0 - A_prior
                prior_penalty = jnp.sum((B_param * prior_mask) ** 2)
                prior_loss = lambda_prior * prior_penalty
            else:
                prior_loss = 0.0

            total_loss = likelihood + dag_loss + sparse_loss + prior_loss

            return total_loss, {
                "total": total_loss,
                "likelihood": likelihood,
                "dag": dag_loss,
                "sparse": sparse_loss,
                "prior": prior_loss,
            }

        # Compute gradients and update
        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(B)
        updates, opt_state = optimizer.update(grads, opt_state, B)
        B = optax.apply_updates(B, updates)

        # Zero diagonal (no self-loops)
        B = B.at[jnp.diag_indices(n_vars)].set(0.0)

        # Log progress
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            h_val = golem_acyclicity_constraint(B)
            B_norm = jnp.linalg.norm(B)
            B_max = jnp.max(jnp.abs(B))
            log_str = (
                f"  Epoch {epoch + 1:3d}: Loss={metrics['total']:.4f}, "
                f"Likelihood={metrics['likelihood']:.4f}, h={h_val:.4f}, "
                f"||B||={B_norm:.4f}, max|B|={B_max:.4f}"
            )
            if A_prior is not None:
                log_str += f", Prior={metrics['prior']:.4f}"
            print(log_str)

    if verbose:
        final_h = golem_acyclicity_constraint(B)
        print(f"\nFinal acyclicity: h(B) = {final_h:.6f}")
        if final_h < 0.01:
            print("[OK] Converged to DAG (h < 0.01)")
        elif final_h < 0.1:
            print("[WARN] Near-DAG (h < 0.1)")
        else:
            print("[FAIL] Not converged to DAG (h > 0.1)")

    return B
