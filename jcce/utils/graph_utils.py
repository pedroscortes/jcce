"""
Graph utility functions for DAG manipulation and validation.

IMPORTANT: topological_sort() uses NetworkX and is NOT JAX-traceable.
Call it ONCE before training, not inside JIT-compiled functions.
"""

import jax.numpy as jnp
import networkx as nx
import numpy as np


def topological_sort(A: jnp.ndarray, threshold: float = 0.0, fallback_if_cyclic: bool = True) -> jnp.ndarray:
    """
    Compute a topological ordering of DAG A.

    Uses Kahn's algorithm via NetworkX.

    This function is NOT JAX-traceable. Call it ONCE before training,
       not inside JIT-compiled functions.

    Args:
        A: (d, d) adjacency matrix where A[i,j] != 0 means j → i
        threshold: Threshold for considering edges (default 0.0)
                   Set to small value (e.g., 0.1) to remove weak edges
        fallback_if_cyclic: If True, return simple ordering [0,1,2,...] when graph has cycles
                           If False, raise ValueError

    Returns:
        order: (d,) JAX array of node indices in topological order

    Raises:
        ValueError: If A contains a cycle and fallback_if_cyclic=False

    Example:
        >>> A = generate_dag(DAGConfig(num_nodes=5))
        >>> topo_order = topological_sort(A)  # Call ONCE before training
        >>> # In training loop:
        >>> z_sorted = z[:, topo_order]  # This is JIT-compatible
    """
    # Convert JAX array to numpy for NetworkX
    A_np = np.array(A)

    # Apply threshold to remove weak edges (may help break cycles)
    if threshold > 0:
        A_np = np.where(np.abs(A_np) > threshold, A_np, 0)

    # Create directed graph (transpose because we use j→i convention)
    # A[i,j] means j→i, but networkx expects i→j, so we use A.T
    G = nx.DiGraph(A_np.T)

    try:
        order = list(nx.topological_sort(G))
        return jnp.array(order)  # Convert back to JAX array
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        if fallback_if_cyclic:
            # Graph has cycles - use simple ordering as fallback
            print(f"[WARN] WARNING: Graph contains cycles, using fallback ordering [0,1,2,...]")
            print(f"   This may happen when structure learning hasn't converged.")
            print(f"   Consider: Increase iterations or use threshold > 0")
            return jnp.arange(A.shape[0])
        else:
            raise ValueError("Graph contains a cycle - not a DAG!")


def validate_dag(A: jnp.ndarray) -> bool:
    """
    Check if A represents a valid DAG (acyclic).

    Also NOT JAX-traceable - use for validation only.

    Args:
        A: Adjacency matrix

    Returns:
        True if A is a valid DAG, False otherwise
    """
    try:
        topological_sort(A)
        return True
    except ValueError:
        return False


def count_edges(A: jnp.ndarray) -> int:
    """Count number of edges in adjacency matrix."""
    return int(jnp.sum(A != 0))


def get_parents(A: jnp.ndarray, node_idx: int) -> jnp.ndarray:
    """
    Get parent indices of a given node.

    Args:
        A: (d, d) adjacency matrix where A[i,j] != 0 means j → i
        node_idx: Index of the node

    Returns:
        parent_indices: Array of parent node indices
    """
    # A[i, :] gives parents of node i (non-zero entries in row i)
    return jnp.where(A[node_idx, :] != 0)[0]


def get_children(A: jnp.ndarray, node_idx: int) -> jnp.ndarray:
    """
    Get children indices of a given node.

    Args:
        A: (d, d) adjacency matrix where A[i,j] != 0 means j → i
        node_idx: Index of the node

    Returns:
        children_indices: Array of children node indices
    """
    # A[:, j] gives children of node j (non-zero entries in column j)
    return jnp.where(A[:, node_idx] != 0)[0]


def compute_graph_statistics(A: jnp.ndarray) -> dict:
    """
    Compute various statistics about the DAG.

    Args:
        A: Adjacency matrix

    Returns:
        stats: Dictionary with graph statistics
    """
    num_nodes = A.shape[0]
    num_edges = count_edges(A)

    in_degrees = jnp.sum(A != 0, axis=1)  # Parents per node
    out_degrees = jnp.sum(A != 0, axis=0)  # Children per node

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": num_edges / (num_nodes * (num_nodes - 1)),
        "avg_in_degree": float(jnp.mean(in_degrees)),
        "max_in_degree": int(jnp.max(in_degrees)),
        "avg_out_degree": float(jnp.mean(out_degrees)),
        "max_out_degree": int(jnp.max(out_degrees)),
        "is_dag": validate_dag(A),
    }
