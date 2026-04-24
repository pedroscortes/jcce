"""
Tests for data generation modules.

Run with: uv run pytest tests/test_data_generation.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from jcce.data.dag_generator import DAGConfig, count_edges, generate_dag, get_average_degree
from jcce.data.scm import LinearSCM, SCMConfig
from jcce.data.synthetic_dataset import (
    SyntheticDatasetConfig,
    create_simple_dataset,
    generate_synthetic_dataset,
)
from jcce.utils.graph_utils import compute_graph_statistics, topological_sort, validate_dag


class TestDAGGeneration:
    """Tests for DAG generation."""

    def test_erdos_renyi_is_dag(self):
        """Verify that generated ER graphs are valid DAGs."""
        config = DAGConfig(num_nodes=10, graph_type="erdos_renyi", seed=42)
        A = generate_dag(config)

        assert A.shape == (10, 10), "Adjacency matrix should be square"
        assert validate_dag(A), "Generated graph should be a valid DAG"

    def test_chain_structure(self):
        """Verify chain DAG has correct structure."""
        config = DAGConfig(num_nodes=5, graph_type="chain", seed=42)
        A = generate_dag(config)

        # Chain should have exactly (num_nodes - 1) edges
        assert count_edges(A) == 4, "Chain should have 4 edges"

        # Each node (except first) should have exactly 1 parent
        in_degrees = jnp.sum(A != 0, axis=1)
        assert in_degrees[0] == 0, "First node should have no parents"
        assert jnp.all(in_degrees[1:] == 1), "Other nodes should have 1 parent each"

    def test_topological_sort(self):
        """Verify topological sort works correctly."""
        config = DAGConfig(num_nodes=5, graph_type="chain", seed=42)
        A = generate_dag(config)

        order = topological_sort(A)

        assert len(order) == 5, "Topological order should have all nodes"
        # For chain, topological order should be [0, 1, 2, 3, 4]
        assert jnp.array_equal(order, jnp.arange(5)), "Chain topo order should be sequential"

    def test_expected_degree(self):
        """Verify average degree is close to expected."""
        config = DAGConfig(num_nodes=100, expected_degree=3.0, seed=42)
        A = generate_dag(config)

        avg_degree = get_average_degree(A)
        # Probabilistic test - allow 50% tolerance due to randomness
        assert 1.5 <= avg_degree <= 4.5, f"Average degree {avg_degree} far from expected 3.0"


class TestSCM:
    """Tests for Structural Causal Models."""

    def test_linear_scm_shape(self):
        """Verify LinearSCM produces correct shape."""
        config_dag = DAGConfig(num_nodes=10, graph_type="chain", seed=42)
        A = generate_dag(config_dag)

        config_scm = SCMConfig(scm_type="linear", noise_scale=0.5)
        scm = LinearSCM(A, config_scm)

        key = jax.random.PRNGKey(0)
        X = scm.sample(100, key)

        assert X.shape == (100, 10), "Samples should have correct shape"

    def test_linear_scm_deterministic(self):
        """Verify same seed produces same data."""
        config_dag = DAGConfig(num_nodes=5, graph_type="chain", seed=42)
        A = generate_dag(config_dag)

        config_scm = SCMConfig(scm_type="linear", noise_scale=0.5)
        scm = LinearSCM(A, config_scm)

        key = jax.random.PRNGKey(123)
        X1 = scm.sample(10, key)

        key = jax.random.PRNGKey(123)  # Same seed
        X2 = scm.sample(10, key)

        assert jnp.allclose(X1, X2), "Same seed should produce identical data"

    def test_noise_types(self):
        """Verify different noise types work."""
        config_dag = DAGConfig(num_nodes=5, graph_type="chain", seed=42)
        A = generate_dag(config_dag)
        key = jax.random.PRNGKey(0)

        for noise_type in ["gaussian", "uniform", "laplace"]:
            config_scm = SCMConfig(scm_type="linear", noise_type=noise_type)
            scm = LinearSCM(A, config_scm)
            X = scm.sample(100, key)

            assert X.shape == (100, 5), f"Noise type {noise_type} should work"
            assert jnp.all(jnp.isfinite(X)), f"Data should be finite for {noise_type}"


class TestSyntheticDataset:
    """Tests for high-level dataset generation."""

    def test_generate_dataset(self):
        """Verify complete dataset generation."""
        config = SyntheticDatasetConfig(
            dag_config=DAGConfig(num_nodes=10, seed=42),
            scm_config=SCMConfig(scm_type="linear"),
            n_train=1000,
            n_val=200,
            n_test=200,
        )

        data, A_true = generate_synthetic_dataset(config)

        assert "train" in data and "val" in data and "test" in data
        assert data["train"].shape == (1000, 10)
        assert data["val"].shape == (200, 10)
        assert data["test"].shape == (200, 10)
        assert A_true.shape == (10, 10)
        assert validate_dag(A_true)

    def test_create_simple_dataset(self):
        """Verify quick dataset helper."""
        X, A = create_simple_dataset(num_nodes=5, n_samples=100, graph_type="chain")

        assert X.shape == (100, 5)
        assert A.shape == (5, 5)
        assert validate_dag(A)


class TestGraphUtils:
    """Tests for graph utility functions."""

    def test_compute_statistics(self):
        """Verify graph statistics computation."""
        config = DAGConfig(num_nodes=10, graph_type="chain", seed=42)
        A = generate_dag(config)

        stats = compute_graph_statistics(A)

        assert stats["num_nodes"] == 10
        assert stats["num_edges"] == 9  # Chain has n-1 edges
        assert stats["is_dag"] is True
        assert 0.0 <= stats["density"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
