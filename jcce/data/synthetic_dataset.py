"""
High-level synthetic dataset generation pipeline.

Combines DAG generation and SCM sampling into a complete dataset.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import SCMConfig, create_scm


@dataclass
class SyntheticDatasetConfig:
    """Configuration for complete synthetic dataset generation."""

    dag_config: DAGConfig
    scm_config: SCMConfig
    n_train: int = 10000
    n_val: int = 2000
    n_test: int = 2000
    seed: int = 42


def generate_synthetic_dataset(
    config: SyntheticDatasetConfig,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Generate a complete synthetic dataset with known ground-truth DAG.

    This is the main entry point for Phase 1 data generation.

    Args:
        config: Dataset configuration

    Returns:
        data: Dictionary with keys 'train', 'val', 'test'
              Each value is (N, d) array of samples
        A_true: (d, d) ground-truth adjacency matrix

    Example:
        >>> config = SyntheticDatasetConfig(
        ...     dag_config=DAGConfig(num_nodes=20),
        ...     scm_config=SCMConfig(scm_type="linear"),
        ...     n_train=10000
        ... )
        >>> data, A_true = generate_synthetic_dataset(config)
        >>> print(f"Train: {data['train'].shape}, DAG: {A_true.shape}")
        Train: (10000, 20), DAG: (20, 20)
    """
    # 1. Generate ground-truth DAG
    A_true = generate_dag(config.dag_config)

    # 2. Create SCM
    scm = create_scm(A_true, config.scm_config)

    # 3. Generate data splits
    key = jax.random.PRNGKey(config.seed)
    key_train, key_val, key_test = jax.random.split(key, 3)

    data = {
        "train": scm.sample(config.n_train, key_train),
        "val": scm.sample(config.n_val, key_val),
        "test": scm.sample(config.n_test, key_test),
    }

    return data, A_true


def create_simple_dataset(
    num_nodes: int = 10,
    n_samples: int = 1000,
    graph_type: str = "chain",
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quick helper to create a simple dataset for testing.

    Args:
        num_nodes: Number of nodes in DAG
        n_samples: Number of samples
        graph_type: 'chain' or 'erdos_renyi'
        seed: Random seed

    Returns:
        X: (n_samples, num_nodes) data
        A: (num_nodes, num_nodes) adjacency matrix
    """
    config = SyntheticDatasetConfig(
        dag_config=DAGConfig(num_nodes=num_nodes, graph_type=graph_type, seed=seed),
        scm_config=SCMConfig(scm_type="linear", noise_scale=0.5),
        n_train=n_samples,
        n_val=0,
        n_test=0,
        seed=seed,
    )

    data, A_true = generate_synthetic_dataset(config)
    return data["train"], A_true
