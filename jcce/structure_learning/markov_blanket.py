"""
Markov Blanket Extraction for Causal Discovery.

Given a learned causal graph (DAG), extract the Markov Blanket of a target variable.

The Markov Blanket MB(Y) of a target variable Y consists of:
1. Parents of Y: Variables that directly cause Y
2. Children of Y: Variables that Y directly causes
3. Spouses of Y: Other parents of Y's children

Mathematically, for a DAG with adjacency matrix A where A[i,j] != 0 means edge i → j:
    MB(Y) = Parents(Y) ∪ Children(Y) ∪ Parents(Children(Y))

This is the minimal set of variables that makes Y conditionally independent of all other variables.
"""

from typing import List

import jax.numpy as jnp


def extract_markov_blanket(A: jnp.ndarray, target: int, threshold: float = 1e-6) -> jnp.ndarray:
    """
    Extract Markov Blanket of target variable from DAG adjacency matrix.

    Args:
        A: (n_vars, n_vars) adjacency matrix where A[i,j] != 0 means edge i → j
        target: Index of target variable (0 to n_vars-1)
        threshold: Threshold for considering edge as present (default: 1e-6)

    Returns:
        mb_mask: (n_vars,) binary array where mb_mask[i] = 1 if variable i is in MB(target)
                 Note: mb_mask[target] = 1 (target is always in its own Markov Blanket)

    Example:
        >>> A = jnp.array([[0, 1, 1],    # Node 0 → 1, 0 → 2
        ...                [0, 0, 1],    # Node 1 → 2
        ...                [0, 0, 0]])   # Node 2 has no children
        >>> mb = extract_markov_blanket(A, target=1)
        >>> mb  # MB(1) = {0 (parent), 1 (self), 2 (child)}
        Array([1., 1., 1.])
    """
    n_vars = A.shape[0]

    # Initialize Markov Blanket as empty
    mb_mask = jnp.zeros(n_vars, dtype=jnp.float32)

    # 1. Add target itself to MB
    mb_mask = mb_mask.at[target].set(1.0)

    # 2. Find parents of target: Parents(Y) = {j : A[j, target] != 0} (column target)
    parents_mask = (jnp.abs(A[:, target]) > threshold).astype(jnp.float32)
    mb_mask = jnp.maximum(mb_mask, parents_mask)

    # 3. Find children of target: Children(Y) = {j : A[target, j] != 0} (row target)
    children_mask = (jnp.abs(A[target, :]) > threshold).astype(jnp.float32)
    mb_mask = jnp.maximum(mb_mask, children_mask)

    # 4. Find spouses (parents of children): For each child i, add Parents(i)
    # For each variable i that is a child of target
    for i in range(n_vars):
        if jnp.abs(A[target, i]) > threshold:  # i is a child of target
            # Add all parents of child i (except target itself, which is already included)
            parents_of_child = (jnp.abs(A[:, i]) > threshold).astype(jnp.float32)
            mb_mask = jnp.maximum(mb_mask, parents_of_child)

    return mb_mask


def extract_markov_blanket_indices(
    A: jnp.ndarray, target: int, threshold: float = 1e-6
) -> jnp.ndarray:
    """
    Extract Markov Blanket indices of target variable from DAG adjacency matrix.

    Same as extract_markov_blanket but returns integer indices instead of binary mask.

    Args:
        A: (n_vars, n_vars) adjacency matrix where A[i,j] != 0 means edge i → j
        target: Index of target variable (0 to n_vars-1)
        threshold: Threshold for considering edge as present (default: 1e-6)

    Returns:
        mb_indices: (k,) array of variable indices in MB(target), sorted
                   where k is the size of the Markov Blanket

    Example:
        >>> A = jnp.array([[0, 1, 1],
        ...                [0, 0, 1],
        ...                [0, 0, 0]])
        >>> indices = extract_markov_blanket_indices(A, target=1)
        >>> indices
        Array([0, 1, 2])
    """
    mb_mask = extract_markov_blanket(A, target, threshold)
    mb_indices = jnp.where(mb_mask > 0.5)[0]  # Get indices where mask is 1
    return mb_indices


def markov_blanket_size(A: jnp.ndarray, target: int, threshold: float = 1e-6) -> int:
    """
    Get size of Markov Blanket (number of variables).

    Args:
        A: (n_vars, n_vars) adjacency matrix
        target: Index of target variable
        threshold: Threshold for considering edge as present

    Returns:
        Size of MB(target) including target itself
    """
    mb_mask = extract_markov_blanket(A, target, threshold)
    return int(jnp.sum(mb_mask))


def markov_blanket_sparsity(A: jnp.ndarray, target: int, threshold: float = 1e-6) -> float:
    """
    Get sparsity of Markov Blanket (fraction of variables NOT in MB).

    Higher sparsity = smaller Markov Blanket = more selective causal feature selection.

    Args:
        A: (n_vars, n_vars) adjacency matrix
        target: Index of target variable
        threshold: Threshold for considering edge as present

    Returns:
        Sparsity in [0, 1] where 0 = all variables in MB, 1 = only target in MB
    """
    n_vars = A.shape[0]
    mb_size = markov_blanket_size(A, target, threshold)
    return float(1.0 - mb_size / n_vars)


def validate_markov_blanket(A: jnp.ndarray, target: int, threshold: float = 1e-6) -> dict:
    """
    Validate and analyze Markov Blanket extraction.

    Returns detailed information about the MB composition.

    Args:
        A: (n_vars, n_vars) adjacency matrix
        target: Index of target variable
        threshold: Threshold for considering edge as present

    Returns:
        Dictionary with:
            - 'parents': Indices of parents of target
            - 'children': Indices of children of target
            - 'spouses': Indices of spouses (parents of children, excluding parents and target)
            - 'mb_all': All MB indices
            - 'mb_size': Total size of MB
            - 'mb_sparsity': Sparsity of MB
    """
    n_vars = A.shape[0]

    # Get parents: A[:, target] (column target — who points to target)
    parents_mask = jnp.abs(A[:, target]) > threshold
    parents = jnp.where(parents_mask)[0]

    # Get children: A[target, :] (row target — who target points to)
    children_mask = jnp.abs(A[target, :]) > threshold
    children = jnp.where(children_mask)[0]

    # Get spouses (parents of children, excluding target and existing parents)
    spouses_set = set()
    for child in children:
        parents_of_child = jnp.where(jnp.abs(A[:, child]) > threshold)[0]
        for parent in parents_of_child:
            if parent != target and parent not in parents:
                spouses_set.add(int(parent))

    spouses = jnp.array(sorted(list(spouses_set)))

    # Get full MB
    mb_indices = extract_markov_blanket_indices(A, target, threshold)

    return {
        "parents": parents,
        "children": children,
        "spouses": spouses,
        "mb_all": mb_indices,
        "mb_size": len(mb_indices),
        "mb_sparsity": 1.0 - len(mb_indices) / n_vars,
    }


# ============================================================================
# Multi-Target Markov Blanket (for classification with multiple target variables)
# ============================================================================


def extract_multi_target_markov_blanket(
    A: jnp.ndarray, targets: List[int], threshold: float = 1e-6
) -> jnp.ndarray:
    """
    Extract unified Markov Blanket for multiple target variables.

    Returns the union of individual Markov Blankets: MB = ∪ MB(target_i)

    Args:
        A: (n_vars, n_vars) adjacency matrix
        targets: List of target variable indices
        threshold: Threshold for considering edge as present

    Returns:
        mb_mask: (n_vars,) binary array where mb_mask[i] = 1 if variable i is in any MB(target)
    """
    n_vars = A.shape[0]
    mb_mask = jnp.zeros(n_vars, dtype=jnp.float32)

    # Union of all individual MBs
    for target in targets:
        mb_mask_i = extract_markov_blanket(A, target, threshold)
        mb_mask = jnp.maximum(mb_mask, mb_mask_i)

    return mb_mask


# ============================================================================
# Utility: Feature selection from Markov Blanket
# ============================================================================


def select_features_by_markov_blanket(
    X: jnp.ndarray, A: jnp.ndarray, target: int, threshold: float = 1e-6
) -> jnp.ndarray:
    """
    Select features from data matrix using Markov Blanket.

    Args:
        X: (n_samples, n_vars) data matrix
        A: (n_vars, n_vars) adjacency matrix
        target: Index of target variable
        threshold: Threshold for considering edge as present

    Returns:
        X_mb: (n_samples, mb_size) data matrix with only MB features
    """
    mb_indices = extract_markov_blanket_indices(A, target, threshold)
    return X[:, mb_indices]
