"""
DAG (Directed Acyclic Graph) Generation Module.

This module provides functions to generate random DAGs for synthetic causal data.
"""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class DAGConfig:
    """Configuration for DAG generation."""

    num_nodes: int = 20  # Number of variables in the causal graph
    graph_type: Literal["erdos_renyi", "scale_free", "chain"] = "erdos_renyi"
    expected_degree: float = 2.0  # Average number of parents per node
    weight_range: tuple[float, float] = (-1.0, 1.0)  # Edge weight distribution
    seed: int = 42


def generate_erdos_renyi_dag(
    num_nodes: int, expected_degree: float, weight_range: tuple[float, float], key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Generate a random DAG using Erdős-Rényi model.

    The ER model creates edges with fixed probability p = expected_degree / (num_nodes - 1).
    We enforce acyclicity by only allowing edges from lower to higher indices.

    Args:
        num_nodes: Number of nodes in the graph
        expected_degree: Expected number of parents per node
        weight_range: (min, max) for edge weights
        key: JAX random key

    Returns:
        A: (num_nodes, num_nodes) weighted adjacency matrix
           A[i, j] != 0 means j → i (j is a parent of i)
    """
    # Edge probability to achieve expected degree
    p = expected_degree / (num_nodes - 1)

    # Generate binary adjacency (upper triangular to ensure DAG)
    key_bernoulli, key_weights = jax.random.split(key)

    # Create upper triangular random matrix
    random_matrix = jax.random.uniform(key_bernoulli, (num_nodes, num_nodes))
    binary_adj = (random_matrix < p).astype(jnp.float32)

    # Keep only upper triangular part (excluding diagonal)
    upper_triangular = jnp.triu(binary_adj, k=1)

    # Transpose so that i < j means j → i (lower indexed nodes are ancestors)
    binary_adj = upper_triangular.T

    # Generate random weights
    weights = jax.random.uniform(
        key_weights, (num_nodes, num_nodes), minval=weight_range[0], maxval=weight_range[1]
    )

    # Multiply binary structure by weights
    A = binary_adj * weights

    return A


def generate_chain_dag(
    num_nodes: int, weight_range: tuple[float, float], key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Generate a simple chain DAG: 1 → 2 → 3 → ... → n

    Useful for debugging and testing topological sort.

    Args:
        num_nodes: Number of nodes
        weight_range: (min, max) for edge weights
        key: JAX random key

    Returns:
        A: Chain adjacency matrix
    """
    A = jnp.zeros((num_nodes, num_nodes))

    # Generate weights for chain edges
    weights = jax.random.uniform(key, (num_nodes - 1,), minval=weight_range[0], maxval=weight_range[1])

    # Create chain: node i+1 has parent i
    # A[i+1, i] = weight means i → i+1
    for i in range(num_nodes - 1):
        A = A.at[i + 1, i].set(weights[i])

    return A


def generate_scale_free_dag(
    num_nodes: int, weight_range: tuple[float, float], key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Generate a scale-free DAG using preferential attachment.

    Note: This is a simplified version. For true scale-free networks,
    consider using NetworkX and converting to JAX.

    Args:
        num_nodes: Number of nodes
        weight_range: (min, max) for edge weights
        key: JAX random key

    Returns:
        A: Scale-free adjacency matrix
    """
    # For Phase 1, we'll use a simplified version
    # Start with ER and modify degree distribution
    # TODO: Implement proper Barabási-Albert model if needed
    return generate_erdos_renyi_dag(num_nodes, expected_degree=3.0, weight_range=weight_range, key=key)


def generate_dag(config: DAGConfig) -> jnp.ndarray:
    """
    Generate a random DAG based on configuration.

    Args:
        config: DAGConfig specifying graph parameters

    Returns:
        A: (num_nodes, num_nodes) weighted adjacency matrix
           A[i, j] != 0 means j → i (j is a parent of i)

    Raises:
        ValueError: If graph_type is not recognized
    """
    key = jax.random.PRNGKey(config.seed)

    if config.graph_type == "erdos_renyi":
        A = generate_erdos_renyi_dag(
            config.num_nodes, config.expected_degree, config.weight_range, key
        )
    elif config.graph_type == "chain":
        A = generate_chain_dag(config.num_nodes, config.weight_range, key)
    elif config.graph_type == "scale_free":
        A = generate_scale_free_dag(config.num_nodes, config.weight_range, key)
    else:
        raise ValueError(f"Unknown graph_type: {config.graph_type}")

    return A


def count_edges(A: jnp.ndarray) -> int:
    """Count number of edges in adjacency matrix."""
    return int(jnp.sum(A != 0))


def get_average_degree(A: jnp.ndarray) -> float:
    """Compute average in-degree (number of parents per node)."""
    in_degrees = jnp.sum(A != 0, axis=1)  # Count non-zero entries per row
    return float(jnp.mean(in_degrees))
