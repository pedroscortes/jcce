"""
DirectLiNGAM Algorithm - Full JAX Implementation.

Linear Non-Gaussian Acyclic Model (LiNGAM) with direct method.

References:
    - Shimizu, S., Inazumi, T., Sogawa, Y., Hyvarinen, A., Kawahara, Y.,
      Washio, T., ... & Bollen, K. (2011).
      "DirectLiNGAM: A direct method for learning a linear non-Gaussian
      structural equation model."
      Journal of Machine Learning Research, 12(Apr), 1225-1248.

Model:
    Linear SCM: X_i = sum b_ij X_j + e_i
    where e_i is non-Gaussian noise, independent across variables.

Key Insight:
    Non-Gaussianity enables causal identification from observational data.
    Root causes (exogenous variables) have residuals most independent of
    all other variables.

Algorithm Steps (DirectLiNGAM):
    1. Find the most exogenous variable (root) via independence measures
    2. Regress all remaining variables on the root -> take residuals
    3. Repeat on residuals until all variables are ordered
    4. Estimate causal coefficients via regression along the ordering
    5. Prune weak edges

This implementation uses:
    - HSIC (kernel independence) for exogeneity scoring
    - Iterative residualization (key to DirectLiNGAM)
    - Full JAX for GPU acceleration
    - Linear regression via lstsq
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from typing import Optional, Tuple
import numpy as np


# ============================================================================
# Kernel-based Independence Measure (HSIC)
# ============================================================================

@jit
def rbf_kernel(X: jnp.ndarray, Y: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """
    RBF (Gaussian) kernel: k(x, y) = exp(-||x - y||^2 / (2*sigma^2))

    Args:
        X: (n, d) array
        Y: (m, d) array
        sigma: Kernel bandwidth

    Returns:
        K: (n, m) kernel matrix
    """
    X_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = jnp.sum(Y ** 2, axis=1, keepdims=True)
    XY = X @ Y.T
    sq_dists = X_sq + Y_sq.T - 2 * XY
    K = jnp.exp(-sq_dists / (2 * sigma ** 2))
    return K


@jit
def hsic_statistic(X: jnp.ndarray, Y: jnp.ndarray, sigma: float = 1.0) -> float:
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion).

    HSIC measures dependence between X and Y using kernels.
    HSIC = 0 iff X and Y are independent.
    Higher HSIC = stronger dependence.

    Args:
        X: (n_samples, d_x) array
        Y: (n_samples, d_y) array
        sigma: Kernel bandwidth

    Returns:
        hsic: HSIC statistic (>= 0)
    """
    n = X.shape[0]

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    K = rbf_kernel(X, X, sigma)
    L = rbf_kernel(Y, Y, sigma)

    H = jnp.eye(n) - jnp.ones((n, n)) / n
    KHLH = K @ H @ L @ H
    hsic = jnp.trace(KHLH) / (n ** 2)

    return hsic


# ============================================================================
# Causal Order Estimation (with iterative residualization)
# ============================================================================

def _compute_residuals(
    data: jnp.ndarray,
    target_col: int,
    regressor_col: int,
) -> jnp.ndarray:
    """Regress data[:, target_col] on data[:, regressor_col], return residuals."""
    n = data.shape[0]
    y = data[:, target_col]
    x = data[:, regressor_col]
    X_design = jnp.column_stack([jnp.ones(n), x])
    beta = jnp.linalg.lstsq(X_design, y, rcond=None)[0]
    return y - X_design @ beta


def estimate_causal_order(
    data: jnp.ndarray,
    sigma: float = 1.0,
    verbose: bool = False
) -> list:
    """
    Estimate causal ordering using iterative residualization (DirectLiNGAM).

    Algorithm (from Shimizu et al. 2011):
        For m = 1, ..., p:
            1. For each remaining variable x_i, compute the sum of
               HSIC(residual_i, residual_j) over all other remaining j.
               The variable with smallest total HSIC is most exogenous.
            2. Append that variable to the ordering.
            3. Regress all remaining variables on the selected root,
               replace them with residuals.

    The key insight is step 3: by residualizing, we remove the effect of
    the identified root, making the next root identifiable.

    Args:
        data: (n_samples, n_vars) data matrix
        sigma: Kernel bandwidth for HSIC
        verbose: Print progress

    Returns:
        causal_order: List of variable indices in causal order
    """
    n_samples, n_vars = data.shape

    # Work with residuals that we update iteratively
    # Keep track of which original variable each column corresponds to
    remaining = list(range(n_vars))
    causal_order = []

    # Current working data (will be residualized iteratively)
    work_data = jnp.array(data)

    if verbose:
        print(f"  Estimating causal order (iterative residualization)...")

    for step in range(n_vars):
        n_remaining = len(remaining)

        if n_remaining == 1:
            causal_order.append(remaining[0])
            break

        # For each remaining variable, compute total HSIC with all others
        # Lower total HSIC = more independent of others = more exogenous
        best_score = float('inf')
        best_idx = None  # index into remaining list
        best_var = None   # original variable index

        for local_idx in range(n_remaining):
            total_hsic = 0.0
            xi = work_data[:, local_idx]

            for other_idx in range(n_remaining):
                if other_idx == local_idx:
                    continue
                xj = work_data[:, other_idx]
                total_hsic += float(hsic_statistic(xi, xj, sigma))

            if total_hsic < best_score:
                best_score = total_hsic
                best_idx = local_idx
                best_var = remaining[local_idx]

        causal_order.append(best_var)

        if verbose and step < 5:
            print(f"    Step {step+1}: Selected variable {best_var} "
                  f"(total_hsic={best_score:.4f})")

        # Residualize: regress all remaining variables on the selected root
        # and replace with residuals
        if n_remaining > 1:
            root_col = best_idx
            new_cols = []
            new_remaining = []

            for local_idx in range(n_remaining):
                if local_idx == root_col:
                    continue
                resid = _compute_residuals(work_data, local_idx, root_col)
                new_cols.append(resid)
                new_remaining.append(remaining[local_idx])

            if new_cols:
                work_data = jnp.column_stack(new_cols)
            remaining = new_remaining

    if verbose:
        print(f"    Causal order: {causal_order}")

    return causal_order


# ============================================================================
# DirectLiNGAM Algorithm
# ============================================================================

@jit
def linear_regression_coefficients(
    data: jnp.ndarray,
    target_idx: int,
    predictor_indices: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute linear regression coefficients.

    Args:
        data: (n_samples, n_vars) data matrix
        target_idx: Target variable index
        predictor_indices: Predictor variable indices

    Returns:
        coefficients: (n_predictors,) regression coefficients
    """
    if predictor_indices.size == 0:
        return jnp.array([])

    n_samples = data.shape[0]
    y = data[:, target_idx]
    X = data[:, predictor_indices]

    X_with_intercept = jnp.column_stack([jnp.ones(n_samples), X])
    beta = jnp.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

    # Return coefficients (exclude intercept)
    coefficients = beta[1:]
    return coefficients


def learn_with_directlingam(
    data: jnp.ndarray,
    key: random.PRNGKey,
    sigma: float = 1.0,
    prune_threshold: float = 0.7,
    verbose: bool = False,
    A_prior: Optional[jnp.ndarray] = None,
    lambda_prior: float = 0.1,
) -> jnp.ndarray:
    """
    DirectLiNGAM Algorithm for causal discovery - Full Implementation.

    Algorithm:
        1. Estimate causal ordering (via iterative residualization + HSIC)
        2. For each variable, regress on predecessors in ordering
        3. Prune weak edges (adaptive threshold based on edge strength percentile)

    Args:
        data: (n_samples, n_vars) observed data
        key: JAX random key (for consistency)
        sigma: Kernel bandwidth for HSIC (default: 1.0)
        prune_threshold: Percentile threshold for pruning weak edges (default: 0.7)
                        Edges below this percentile are pruned.
                        0.7 keeps top 30%, 0.5 keeps top 50%, etc.
        verbose: Print progress
        A_prior: Optional prior adjacency matrix
        lambda_prior: Weight for prior (used in pruning)

    Returns:
        A: (n_vars, n_vars) adjacency matrix (0/1)
    """
    data_np = np.array(data)
    n_samples, n_vars = data_np.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"DIRECTLINGAM ALGORITHM")
        print(f"{'='*60}")
        print(f"Data: {n_samples} samples, {n_vars} variables")

    # Step 1: Estimate causal ordering
    causal_order = estimate_causal_order(jnp.array(data_np), sigma, verbose)

    # Step 2: Regress each variable on predecessors
    A = np.zeros((n_vars, n_vars))

    if verbose:
        print(f"\n  Regression phase...")

    for idx, var in enumerate(causal_order):
        if idx == 0:
            continue

        predecessors = causal_order[:idx]
        predecessors_arr = jnp.array(predecessors, dtype=jnp.int32)

        coefficients = linear_regression_coefficients(
            jnp.array(data_np), var, predecessors_arr
        )

        for i, pred_var in enumerate(predecessors):
            A[pred_var, var] = float(coefficients[i])

        if verbose and idx <= 3:
            print(f"    Variable {var}: {len(predecessors)} parents, "
                  f"avg coef = {np.mean(np.abs(coefficients)):.3f}")

    # Step 3: Prune weak edges
    edge_weights = np.abs(A[A != 0])

    if len(edge_weights) > 0:
        threshold = np.percentile(edge_weights, prune_threshold * 100)

        if A_prior is not None:
            A_prior_np = np.array(A_prior)
            mask = (np.abs(A) > threshold) | (A_prior_np > 0.5)
            A = A * mask
        else:
            A = np.where(np.abs(A) > threshold, A, 0.0)

        if verbose:
            print(f"\n  Pruning: threshold={threshold:.3f}")
            print(f"    Edges before: {len(edge_weights)}")
            print(f"    Edges after: {int(np.sum(np.abs(A) > 0))}")
    else:
        if verbose:
            print(f"\n  No edges found!")

    # Convert to binary adjacency (0/1)
    A_binary = (np.abs(A) > 1e-6).astype(np.float32)

    if verbose:
        n_edges = int(np.sum(A_binary))
        print(f"\n{'='*60}")
        print(f"DIRECTLINGAM FINAL: {n_edges} edges")
        print(f"{'='*60}\n")

    A_jax = jnp.array(A_binary)
    return A_jax


def learn_with_directlingam_weighted(
    data: jnp.ndarray,
    key,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    DirectLiNGAM returning both binary structure and continuous weights.

    Returns:
        A_binary: (n_vars, n_vars) binary adjacency {0, 1}
        A_weights: (n_vars, n_vars) continuous regression coefficients
    """
    data_np = np.array(data)
    n_samples, n_vars = data_np.shape
    verbose = kwargs.get('verbose', False)
    sigma = kwargs.get('sigma', 1.0)
    prune_threshold = kwargs.get('prune_threshold', 0.7)
    A_prior = kwargs.get('A_prior', None)
    lambda_prior = kwargs.get('lambda_prior', 0.1)

    if verbose:
        print(f"\n{'='*60}")
        print(f"DIRECTLINGAM ALGORITHM (weighted)")
        print(f"{'='*60}")
        print(f"Data: {n_samples} samples, {n_vars} variables")

    causal_order = estimate_causal_order(jnp.array(data_np), sigma, verbose)

    A = np.zeros((n_vars, n_vars))

    if verbose:
        print(f"\n  Regression phase...")

    for idx, var in enumerate(causal_order):
        if idx == 0:
            continue
        predecessors = causal_order[:idx]
        predecessors_arr = jnp.array(predecessors, dtype=jnp.int32)
        coefficients = linear_regression_coefficients(
            jnp.array(data_np), var, predecessors_arr
        )
        for i, pred_var in enumerate(predecessors):
            A[pred_var, var] = float(coefficients[i])

        if verbose and idx <= 3:
            print(f"    Variable {var}: {len(predecessors)} parents, "
                  f"avg coef = {np.mean(np.abs(coefficients)):.3f}")

    # Prune
    edge_weights = np.abs(A[A != 0])
    if len(edge_weights) > 0:
        threshold = np.percentile(edge_weights, prune_threshold * 100)
        if A_prior is not None:
            A_prior_np = np.array(A_prior)
            mask = (np.abs(A) > threshold) | (A_prior_np > 0.5)
            A = A * mask
        else:
            A = np.where(np.abs(A) > threshold, A, 0.0)

    A_weights = jnp.array(A.copy(), dtype=jnp.float32)
    A_binary = jnp.array((np.abs(A) > 1e-6).astype(np.float32))

    if verbose:
        n_edges = int(jnp.sum(A_binary))
        print(f"\n{'='*60}")
        print(f"DIRECTLINGAM FINAL: {n_edges} edges")
        print(f"{'='*60}\n")

    return A_binary, A_weights
