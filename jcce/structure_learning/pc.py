"""
PC Algorithm (Peter-Clark) for Causal Discovery - Full JAX Implementation.

Constraint-based algorithm using conditional independence tests.

References:
    - Spirtes, P., Glymour, C., & Scheines, R. (2000).
      Causation, Prediction, and Search (2nd ed.). MIT Press.
    - Meek, C. (1995). "Causal inference and causal explanation with
      background knowledge." In UAI.

Algorithm Steps:
    1. Start with complete undirected graph
    2. Skeleton learning: Remove edges via conditional independence tests
    3. Orient v-structures (colliders): X -> Z <- Y where X _||_ Y | S, Z not in S
    4. Apply Meek orientation rules (R1-R4) to propagate orientations

This implementation uses:
    - Partial correlation for conditional independence testing
    - Full JAX for GPU acceleration and JIT compilation
    - Efficient matrix operations for CI tests
"""

from itertools import combinations
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, random

# ============================================================================
# Partial Correlation for Conditional Independence Testing
# ============================================================================


@jit
def partial_correlation_jax(
    data: jnp.ndarray, i: int, j: int, conditioning_set: jnp.ndarray
) -> float:
    """
    Compute partial correlation between variables i and j given conditioning_set.

    Method: Regression-based
        1. Regress X_i on S -> get residuals r_i
        2. Regress X_j on S -> get residuals r_j
        3. Compute correlation between r_i and r_j

    Args:
        data: (n_samples, n_vars) data matrix
        i, j: Variable indices
        conditioning_set: Array of conditioning variable indices

    Returns:
        Partial correlation in [-1, 1]
    """
    n_samples = data.shape[0]

    X_i = data[:, i]
    X_j = data[:, j]

    # If no conditioning variables, return marginal correlation
    if conditioning_set.size == 0:
        return jnp.corrcoef(X_i, X_j)[0, 1]

    X_S = data[:, conditioning_set]

    # Add intercept
    X_S_intercept = jnp.column_stack([jnp.ones(n_samples), X_S])

    # Regression: X_i ~ S
    beta_i = jnp.linalg.lstsq(X_S_intercept, X_i, rcond=None)[0]
    residual_i = X_i - X_S_intercept @ beta_i

    # Regression: X_j ~ S
    beta_j = jnp.linalg.lstsq(X_S_intercept, X_j, rcond=None)[0]
    residual_j = X_j - X_S_intercept @ beta_j

    # Partial correlation = correlation of residuals
    partial_corr = jnp.corrcoef(residual_i, residual_j)[0, 1]

    return partial_corr


def fisher_z_test(partial_corr: float, n_samples: int, cond_size: int, alpha: float = 0.05) -> bool:
    """
    Fisher's Z-test for testing partial correlation = 0.

    H0: rho_ij|S = 0 (conditional independence)
    H1: rho_ij|S != 0 (conditional dependence)

    Test statistic:
        Z = 0.5 * log((1 + r) / (1 - r)) * sqrt(n - |S| - 3)

    Args:
        partial_corr: Partial correlation coefficient
        n_samples: Number of samples
        cond_size: Size of conditioning set |S|
        alpha: Significance level

    Returns:
        True if independent (fail to reject H0), False if dependent
    """
    r = float(jnp.clip(partial_corr, -0.9999, 0.9999))

    # Fisher's Z-transformation
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))

    # Standard error
    se = 1.0 / np.sqrt(n_samples - cond_size - 3)

    # Test statistic
    z_stat = np.abs(z / se)

    # Critical value for two-tailed test
    from scipy.stats import norm

    z_critical = norm.ppf(1 - alpha / 2)

    is_independent = z_stat < z_critical
    return bool(is_independent)


def conditional_independence_test(
    data: jnp.ndarray, i: int, j: int, conditioning_set: jnp.ndarray, alpha: float = 0.05
) -> bool:
    """
    Test conditional independence: X_i _||_ X_j | X_S.

    Uses partial correlation + Fisher's Z-test.

    Args:
        data: (n_samples, n_vars) data matrix
        i, j: Variable indices to test
        conditioning_set: Conditioning variable indices
        alpha: Significance level

    Returns:
        True if conditionally independent, False otherwise
    """
    n_samples = data.shape[0]
    partial_corr = partial_correlation_jax(data, i, j, conditioning_set)
    cond_size = conditioning_set.size
    is_independent = fisher_z_test(partial_corr, n_samples, cond_size, alpha)
    return bool(is_independent)


# ============================================================================
# Skeleton Learning (Edge Removal via CI Tests)
# ============================================================================


def learn_skeleton(
    data: jnp.ndarray, alpha: float = 0.05, max_cond_size: int = 3, verbose: bool = False
) -> Tuple[jnp.ndarray, dict]:
    """
    Learn skeleton (undirected graph) via conditional independence tests.

    PC Algorithm - Phase 1: Skeleton Learning
        Start with complete graph.
        For conditioning set sizes l = 0, 1, 2, ..., max_cond_size:
            For each edge i -- j:
                For each subset S of adj(i)\\{j} with |S| = l:
                    If X_i _||_ X_j | S:
                        Remove edge i -- j, store S in sepset[i, j]
                Also test subsets of adj(j)\\{i} (symmetric search).

    Args:
        data: (n_samples, n_vars) data matrix
        alpha: Significance level for CI tests
        max_cond_size: Maximum conditioning set size
        verbose: Print progress

    Returns:
        skeleton: (n_vars, n_vars) undirected adjacency matrix
        sepsets: Dictionary storing separating sets for removed edges
    """
    n_samples, n_vars = data.shape

    if verbose:
        print(f"PC Skeleton Learning: {n_vars} vars, {n_samples} samples")

    # Start with complete undirected graph
    skeleton_np = np.ones((n_vars, n_vars), dtype=np.float32)
    np.fill_diagonal(skeleton_np, 0.0)

    sepsets = {}

    for cond_size in range(max_cond_size + 1):
        if verbose:
            print(f"  Testing CI with |S| = {cond_size}")

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton_np[i, j] == 0:
                    continue

                # Test conditioning on subsets of neighbors of i (excluding j)
                neighbors_i = [k for k in range(n_vars) if skeleton_np[i, k] == 1 and k != j]

                found = False
                if len(neighbors_i) >= cond_size:
                    for cond_set_tuple in combinations(neighbors_i, cond_size):
                        cond_set = jnp.array(list(cond_set_tuple), dtype=jnp.int32)
                        if conditional_independence_test(data, i, j, cond_set, alpha):
                            skeleton_np[i, j] = 0
                            skeleton_np[j, i] = 0
                            sepsets[(i, j)] = cond_set
                            sepsets[(j, i)] = cond_set
                            found = True
                            break

                if found:
                    continue

                # Also test conditioning on subsets of neighbors of j (excluding i)
                neighbors_j = [k for k in range(n_vars) if skeleton_np[j, k] == 1 and k != i]

                if len(neighbors_j) >= cond_size:
                    for cond_set_tuple in combinations(neighbors_j, cond_size):
                        cond_set = jnp.array(list(cond_set_tuple), dtype=jnp.int32)
                        if conditional_independence_test(data, i, j, cond_set, alpha):
                            skeleton_np[i, j] = 0
                            skeleton_np[j, i] = 0
                            sepsets[(i, j)] = cond_set
                            sepsets[(j, i)] = cond_set
                            break

    skeleton_jax = jnp.array(skeleton_np)

    if verbose:
        n_edges = int(jnp.sum(skeleton_jax)) // 2
        print(f"  Skeleton: {n_edges} undirected edges")

    return skeleton_jax, sepsets


# ============================================================================
# V-Structure (Collider) Detection
# ============================================================================


def orient_v_structures(skeleton: jnp.ndarray, sepsets: dict, verbose: bool = False) -> jnp.ndarray:
    """
    Orient v-structures (colliders): X -> Z <- Y.

    Rule: If X -- Z -- Y in skeleton AND X, Y not adjacent
          AND Z not in sepset(X, Y):
        Then orient as X -> Z <- Y

    Args:
        skeleton: (n_vars, n_vars) undirected graph
        sepsets: Dictionary of separating sets
        verbose: Print progress

    Returns:
        pdag: (n_vars, n_vars) PDAG
              1 = directed edge (i->j), -1 = undirected edge, 0 = no edge
    """
    n_vars = skeleton.shape[0]

    # Initialize PDAG: -1 = undirected, 0 = no edge
    pdag = np.array(skeleton, dtype=np.float32)
    pdag = np.where(pdag == 1, -1, 0)

    if verbose:
        print("  Orienting v-structures...")

    n_oriented = 0

    for k in range(n_vars):
        # Get all neighbors of k in the skeleton
        adj_k = [v for v in range(n_vars) if v != k and skeleton[v, k] != 0]

        for idx_a in range(len(adj_k)):
            for idx_b in range(idx_a + 1, len(adj_k)):
                i, j = adj_k[idx_a], adj_k[idx_b]

                # i and j must NOT be adjacent
                if skeleton[i, j] != 0:
                    continue

                # k must NOT be in sepset(i, j)
                sepset_ij = sepsets.get((i, j), jnp.array([], dtype=jnp.int32))
                k_in_sepset = int(k) in [int(s) for s in sepset_ij]

                if not k_in_sepset:
                    # Orient as i -> k <- j
                    pdag[i, k] = 1
                    pdag[k, i] = 0
                    pdag[j, k] = 1
                    pdag[k, j] = 0
                    n_oriented += 1

    if verbose:
        print(f"    Oriented {n_oriented} v-structures")

    return jnp.array(pdag)


# ============================================================================
# Meek Orientation Rules
# ============================================================================


def apply_meek_rules(pdag: jnp.ndarray, max_iter: int = 100, verbose: bool = False) -> jnp.ndarray:
    """
    Apply Meek's orientation rules to propagate edge orientations.

    PDAG encoding: 1 = directed (i->j), -1 = undirected, 0 = no edge.
    An edge i--j is undirected if pdag[i,j]==-1 and pdag[j,i]==-1.
    An edge i->j is directed if pdag[i,j]==1 and pdag[j,i]==0.

    Meek Rules (Meek 1995, Spirtes et al. 2000):
        R1: If a -> b -- c, and a not adj c, orient b -> c
        R2: If a -> b -> c, and a -- c, orient a -> c
        R3: If a -- d, b -> d, c -> d, a -- b, a -- c, b not adj c,
            orient a -> d
        R4: If a -- b, b -> c, c -> d, a -- d, a not adj c,
            orient a -> b

    Args:
        pdag: (n_vars, n_vars) PDAG
        max_iter: Maximum iterations
        verbose: Print progress

    Returns:
        pdag: Oriented PDAG
    """
    n_vars = pdag.shape[0]
    pdag_np = np.array(pdag)

    if verbose:
        print("  Applying Meek rules...")

    def _is_directed(a, b):
        return pdag_np[a, b] == 1 and pdag_np[b, a] == 0

    def _is_undirected(a, b):
        return pdag_np[a, b] == -1 and pdag_np[b, a] == -1

    def _no_edge(a, b):
        return pdag_np[a, b] == 0 and pdag_np[b, a] == 0

    def _orient(a, b):
        """Orient a -> b (assuming a -- b is undirected)."""
        pdag_np[a, b] = 1
        pdag_np[b, a] = 0

    for iteration in range(max_iter):
        changed = False

        # R1: a -> b -- c, a not adj c => b -> c
        for a in range(n_vars):
            for b in range(n_vars):
                if a == b or not _is_directed(a, b):
                    continue
                for c in range(n_vars):
                    if c == a or c == b:
                        continue
                    if _is_undirected(b, c) and _no_edge(a, c):
                        _orient(b, c)
                        changed = True

        # R2: a -> b -> c, a -- c => a -> c
        for a in range(n_vars):
            for c in range(n_vars):
                if a == c or not _is_undirected(a, c):
                    continue
                for b in range(n_vars):
                    if b == a or b == c:
                        continue
                    if _is_directed(a, b) and _is_directed(b, c):
                        _orient(a, c)
                        changed = True
                        break

        # R3: a -- d, b -> d, c -> d, a -- b, a -- c, b not adj c => a -> d
        for a in range(n_vars):
            for d in range(n_vars):
                if a == d or not _is_undirected(a, d):
                    continue
                # Find b, c: both -> d, both -- a, b not adj c
                parents_d = [v for v in range(n_vars) if v != a and v != d and _is_directed(v, d)]
                for idx_b in range(len(parents_d)):
                    for idx_c in range(idx_b + 1, len(parents_d)):
                        b, c = parents_d[idx_b], parents_d[idx_c]
                        if _is_undirected(a, b) and _is_undirected(a, c) and _no_edge(b, c):
                            _orient(a, d)
                            changed = True

        # R4: a -- b, b -> c, c -> d, a -- d, a not adj c => a -> b
        for a in range(n_vars):
            for b in range(n_vars):
                if a == b or not _is_undirected(a, b):
                    continue
                for c in range(n_vars):
                    if c == a or c == b:
                        continue
                    if not _is_directed(b, c):
                        continue
                    if not _no_edge(a, c):
                        continue
                    for d in range(n_vars):
                        if d == a or d == b or d == c:
                            continue
                        if _is_directed(c, d) and _is_undirected(a, d):
                            _orient(a, b)
                            changed = True
                            break
                    if changed:
                        break

        if not changed:
            break

    if verbose:
        print(f"    Meek rules converged in {iteration + 1} iterations")

    return jnp.array(pdag_np)


# ============================================================================
# Main PC Algorithm
# ============================================================================


def learn_with_pc(
    data: jnp.ndarray,
    key: random.PRNGKey,
    alpha: float = 0.05,
    max_cond_size: int = 3,
    verbose: bool = False,
    A_prior: Optional[jnp.ndarray] = None,
    lambda_prior: float = 0.1,
) -> jnp.ndarray:
    """
    PC Algorithm for causal discovery - Full Implementation.

    Algorithm:
        1. Skeleton learning (remove edges via CI tests)
        2. Orient v-structures (colliders)
        3. Apply Meek rules (propagate orientations)

    Args:
        data: (n_samples, n_vars) observed data
        key: JAX random key (for consistency)
        alpha: Significance level for CI tests (default: 0.05)
        max_cond_size: Maximum conditioning set size (default: 3)
        verbose: Print progress
        A_prior: Optional prior adjacency (for initialization bias)
        lambda_prior: Weight for prior (not used in PC, kept for API)

    Returns:
        A: (n_vars, n_vars) adjacency matrix (0/1)
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("PC ALGORITHM")
        print(f"{'=' * 60}")

    # Phase 1: Skeleton learning
    skeleton, sepsets = learn_skeleton(data, alpha, max_cond_size, verbose)

    # Phase 2: Orient v-structures
    pdag = orient_v_structures(skeleton, sepsets, verbose)

    # Phase 3: Apply Meek rules
    pdag_final = apply_meek_rules(pdag, max_iter=100, verbose=verbose)

    # Convert PDAG to DAG
    pdag_np = np.array(pdag_final)
    A_np = np.zeros_like(pdag_np)

    # Copy directed edges
    A_np = np.where(pdag_np == 1, 1.0, 0.0)

    # Orient remaining undirected edges
    n_vars = pdag_final.shape[0]
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if pdag_np[i, j] == -1 and pdag_np[j, i] == -1:
                if A_prior is not None:
                    A_prior_np = np.array(A_prior)
                    if A_prior_np[i, j] > 0.5:
                        A_np[i, j] = 1.0
                    elif A_prior_np[j, i] > 0.5:
                        A_np[j, i] = 1.0
                    else:
                        A_np[i, j] = 1.0
                else:
                    A_np[i, j] = 1.0

    A = jnp.array(A_np)

    if verbose:
        n_edges = int(jnp.sum(A))
        print(f"  Final DAG: {n_edges} directed edges")
        print(f"{'=' * 60}\n")

    return A
