"""
GES (Greedy Equivalence Search) Algorithm - Full JAX Implementation.

Score-based algorithm using BIC score and greedy search over equivalence
classes (CPDAGs).

References:
    - Chickering, D. M. (2002). "Optimal Structure Identification with
      Greedy Search." Journal of Machine Learning Research, 3, 507-554.
    - Chickering, D. M. (2002). "Learning equivalence classes of
      Bayesian-network structures."

Algorithm Steps:
    1. Forward Phase: Greedily add edges to CPDAG that improve BIC score
    2. Backward Phase: Greedily remove edges that improve BIC score
    3. Convert final CPDAG to DAG

GES operates on CPDAGs (Completed Partially Directed Acyclic Graphs),
which represent Markov equivalence classes. Each step:
    - Proposes an edge addition/removal in a member DAG
    - Converts back to CPDAG to stay in equivalence class space

This implementation uses:
    - BIC (Bayesian Information Criterion) for scoring
    - Linear regression for computing residuals
    - CPDAG representation (1=directed, -1=undirected, 0=none)
    - JAX for numerical computation
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from typing import Optional, Tuple, List
import numpy as np


# ============================================================================
# BIC Score Computation
# ============================================================================

def compute_local_bic(
    data: np.ndarray,
    target: int,
    parents: List[int],
) -> float:
    """
    Compute local BIC score for a single variable given its parents.

    BIC_j = n * log(sigma^2_j) + k_j * log(n)

    where sigma^2_j is the residual variance from regressing X_j on parents,
    and k_j = |parents| + 1 (regression coefficients + intercept).

    Lower BIC = better.

    Args:
        data: (n_samples, n_vars) data matrix (numpy)
        target: Target variable index
        parents: List of parent indices

    Returns:
        bic: Local BIC score for this variable
    """
    n_samples = data.shape[0]
    y = data[:, target]

    if len(parents) == 0:
        residual_var = np.var(y)
    else:
        X_p = data[:, parents]
        X_design = np.column_stack([np.ones(n_samples), X_p])
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        residuals = y - X_design @ beta
        residual_var = np.var(residuals)

    residual_var = max(residual_var, 1e-10)
    k = len(parents) + 1
    bic = n_samples * np.log(residual_var) + k * np.log(n_samples)
    return bic


def bic_score_dag(
    data: np.ndarray,
    A: np.ndarray,
) -> float:
    """
    Compute total BIC score for a DAG.

    Args:
        data: (n_samples, n_vars) numpy data matrix
        A: (n_vars, n_vars) adjacency matrix (A[i,j]=1 means i->j)

    Returns:
        Total BIC score (lower = better)
    """
    n_vars = A.shape[0]
    total = 0.0
    for j in range(n_vars):
        parents = [i for i in range(n_vars) if A[i, j] != 0]
        total += compute_local_bic(data, j, parents)
    return total


# ============================================================================
# DAG / CPDAG Utilities
# ============================================================================

def is_dag_numpy(A: np.ndarray) -> bool:
    """Check if adjacency matrix represents a DAG using topological sort."""
    n = A.shape[0]
    in_degree = np.sum(A != 0, axis=0).astype(int)
    queue = [i for i in range(n) if in_degree[i] == 0]
    count = 0

    while queue:
        node = queue.pop(0)
        count += 1
        for j in range(n):
            if A[node, j] != 0:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    return count == n


def dag_to_cpdag(A: np.ndarray) -> np.ndarray:
    """
    Convert a DAG to its CPDAG (Completed Partially Directed Acyclic Graph).

    A CPDAG represents the Markov equivalence class of the DAG.
    Edges in v-structures and edges compelled by v-structures remain directed;
    all other edges become undirected.

    Uses the labeling algorithm:
    1. Find all v-structures -> mark those edges as compelled (directed).
    2. Apply Meek rules to propagate compelled status.
    3. Remaining edges become undirected.

    Encoding: 1 = directed (i->j), -1 = undirected, 0 = no edge.

    Args:
        A: (n, n) binary DAG adjacency matrix (A[i,j]=1 means i->j)

    Returns:
        cpdag: (n, n) CPDAG matrix
    """
    n = A.shape[0]

    # Start: all edges are "unknown" (could be compelled or reversible)
    # We'll mark compelled edges, rest become undirected
    compelled = np.zeros((n, n), dtype=bool)

    # Step 1: Find v-structures i -> k <- j where i and j not adjacent
    for k in range(n):
        parents_k = [i for i in range(n) if A[i, k] != 0]
        for idx_a in range(len(parents_k)):
            for idx_b in range(idx_a + 1, len(parents_k)):
                i, j = parents_k[idx_a], parents_k[idx_b]
                # i and j not adjacent
                if A[i, j] == 0 and A[j, i] == 0:
                    compelled[i, k] = True
                    compelled[j, k] = True

    # Step 2: Apply Meek-like rules to propagate compelled status
    changed = True
    while changed:
        changed = False

        for a in range(n):
            for b in range(n):
                if A[a, b] == 0 or compelled[a, b]:
                    continue

                # R1: If c -> a -> b, c not adj b => a -> b compelled
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if compelled[c, a] and A[c, a] != 0 and A[a, b] != 0:
                        if A[c, b] == 0 and A[b, c] == 0:
                            compelled[a, b] = True
                            changed = True

                # R2: If a -> c -> b, a -- b (both exist) => a -> b compelled
                if not compelled[a, b]:
                    for c in range(n):
                        if c == a or c == b:
                            continue
                        if (compelled[a, c] and A[a, c] != 0 and
                                compelled[c, b] and A[c, b] != 0):
                            compelled[a, b] = True
                            changed = True

    # Build CPDAG
    cpdag = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                if compelled[i, j]:
                    cpdag[i, j] = 1.0  # directed i -> j
                else:
                    # Undirected: mark both directions
                    cpdag[i, j] = -1.0
                    cpdag[j, i] = -1.0

    return cpdag


def cpdag_to_dag(cpdag: np.ndarray) -> np.ndarray:
    """
    Convert CPDAG to a consistent DAG extension.

    Keeps directed edges, orients undirected edges consistently
    (lower index -> higher index, checking acyclicity).

    Args:
        cpdag: (n, n) CPDAG matrix (1=directed, -1=undirected)

    Returns:
        A: (n, n) DAG adjacency matrix
    """
    n = cpdag.shape[0]
    A = np.zeros((n, n), dtype=np.float64)

    # Copy directed edges
    for i in range(n):
        for j in range(n):
            if cpdag[i, j] == 1 and cpdag[j, i] == 0:
                A[i, j] = 1.0

    # Orient undirected edges
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] == -1 and cpdag[j, i] == -1:
                # Try i -> j
                A[i, j] = 1.0
                if not is_dag_numpy(A):
                    A[i, j] = 0.0
                    A[j, i] = 1.0

    return A


# ============================================================================
# GES Algorithm
# ============================================================================

def learn_with_ges(
    data: jnp.ndarray,
    key: random.PRNGKey,
    max_iter: int = 100,
    verbose: bool = False,
    A_prior: Optional[jnp.ndarray] = None,
    lambda_prior: float = 0.1,
) -> jnp.ndarray:
    """
    GES Algorithm for causal discovery.

    Operates on CPDAGs (equivalence classes) per Chickering (2002).

    Algorithm:
        1. Start with empty graph (CPDAG)
        2. Forward Phase: Add edges that improve BIC
        3. Backward Phase: Remove edges that improve BIC
        4. Convert final CPDAG to DAG

    For each candidate edge operation:
        - Convert CPDAG to a member DAG
        - Apply the edge change
        - Check DAG validity
        - Convert back to CPDAG
        - Score the new equivalence class

    Args:
        data: (n_samples, n_vars) observed data
        key: JAX random key (for tie-breaking)
        max_iter: Maximum iterations per phase
        verbose: Print progress
        A_prior: Optional prior adjacency matrix
        lambda_prior: Weight for prior penalty

    Returns:
        A: (n_vars, n_vars) binary adjacency matrix (0/1)
    """
    data_np = np.array(data)
    n_samples, n_vars = data_np.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"GES ALGORITHM (CPDAG-based)")
        print(f"{'='*60}")
        print(f"Data: {n_samples} samples, {n_vars} variables")

    # Start with empty DAG / CPDAG
    A = np.zeros((n_vars, n_vars))
    cpdag = np.zeros((n_vars, n_vars))

    current_bic = bic_score_dag(data_np, A)

    if A_prior is not None:
        A_prior_np = np.array(A_prior)
        current_bic += lambda_prior * np.sum((A - A_prior_np) ** 2)

    if verbose:
        print(f"\nInitial BIC: {current_bic:.2f}")
        print(f"\n{'='*60}")
        print("FORWARD PHASE: Adding edges")
        print(f"{'='*60}")

    # ========== Forward Phase ==========
    for iteration in range(max_iter):
        best_bic = current_bic
        best_edge = None
        best_cpdag = None

        # Get current DAG from CPDAG
        current_dag = cpdag_to_dag(cpdag)

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j or current_dag[i, j] == 1 or current_dag[j, i] == 1:
                    continue

                # Try adding i -> j
                A_new = current_dag.copy()
                A_new[i, j] = 1.0

                if not is_dag_numpy(A_new):
                    continue

                new_bic = bic_score_dag(data_np, A_new)

                if A_prior is not None:
                    new_bic += lambda_prior * np.sum((A_new - A_prior_np) ** 2)

                if new_bic < best_bic:
                    best_bic = new_bic
                    best_edge = (i, j)
                    best_cpdag = dag_to_cpdag(A_new)

        if best_edge is not None:
            cpdag = best_cpdag
            current_bic = best_bic

            if verbose:
                i, j = best_edge
                n_edges = int(np.sum(np.abs(cpdag) > 0)) // 2 + int(np.sum(cpdag == 1))
                # Count properly: directed=1 per pair, undirected=1 per pair
                print(f"  Iter {iteration+1}: Added {i}->{j}, BIC={current_bic:.2f}")
        else:
            if verbose:
                print(f"  Forward phase converged at iteration {iteration+1}")
            break

    if verbose:
        print(f"\n{'='*60}")
        print("BACKWARD PHASE: Removing edges")
        print(f"{'='*60}")

    # ========== Backward Phase ==========
    for iteration in range(max_iter):
        best_bic = current_bic
        best_edge_remove = None
        best_cpdag = None

        current_dag = cpdag_to_dag(cpdag)

        for i in range(n_vars):
            for j in range(n_vars):
                if current_dag[i, j] == 0:
                    continue

                A_new = current_dag.copy()
                A_new[i, j] = 0.0

                new_bic = bic_score_dag(data_np, A_new)

                if A_prior is not None:
                    new_bic += lambda_prior * np.sum((A_new - A_prior_np) ** 2)

                if new_bic < best_bic:
                    best_bic = new_bic
                    best_edge_remove = (i, j)
                    best_cpdag = dag_to_cpdag(A_new)

        if best_edge_remove is not None:
            cpdag = best_cpdag
            current_bic = best_bic

            if verbose:
                i, j = best_edge_remove
                print(f"  Iter {iteration+1}: Removed {i}->{j}, BIC={current_bic:.2f}")
        else:
            if verbose:
                print(f"  Backward phase converged at iteration {iteration+1}")
            break

    # Convert final CPDAG to DAG
    A_final = cpdag_to_dag(cpdag)

    if verbose:
        n_edges = int(np.sum(A_final))
        print(f"\n{'='*60}")
        print(f"GES FINAL: {n_edges} edges, BIC={current_bic:.2f}")
        print(f"{'='*60}\n")

    return jnp.array(A_final.astype(np.float32))
