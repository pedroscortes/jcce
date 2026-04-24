"""
Structural Causal Model (SCM) Implementation.

Implements both linear and nonlinear SCMs for generating synthetic causal data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class SCMConfig:
    """Configuration for SCM data generation."""

    scm_type: Literal["linear", "nonlinear_mlp"] = "linear"
    noise_type: Literal["gaussian", "uniform", "laplace"] = "gaussian"
    noise_scale: float = 0.5
    # For nonlinear SCMs (Phase 2)
    nonlinear_hidden_dims: Optional[List[int]] = None


class StructuralCausalModel(ABC):
    """Base class for Structural Causal Models."""

    def __init__(self, A: jnp.ndarray, config: SCMConfig):
        """
        Initialize SCM.

        Args:
            A: (d, d) adjacency matrix where A[i,j] != 0 means j → i
            config: SCM configuration
        """
        self.A = A
        self.config = config
        self.num_nodes = A.shape[0]

    @abstractmethod
    def sample(self, n_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Generate n_samples from the SCM.

        Args:
            n_samples: Number of samples to generate
            key: JAX random key

        Returns:
            X: (n_samples, num_nodes) observational data
        """
        raise NotImplementedError

    def _sample_noise(self, n_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample exogenous noise variables.

        Args:
            n_samples: Number of samples
            key: JAX random key

        Returns:
            epsilon: (n_samples, num_nodes) noise samples
        """
        if self.config.noise_type == "gaussian":
            epsilon = jax.random.normal(key, (n_samples, self.num_nodes))
        elif self.config.noise_type == "uniform":
            epsilon = jax.random.uniform(key, (n_samples, self.num_nodes), minval=-1.0, maxval=1.0)
        elif self.config.noise_type == "laplace":
            # Laplace(0, scale) using inverse transform sampling
            u = jax.random.uniform(key, (n_samples, self.num_nodes))
            epsilon = -jnp.sign(u - 0.5) * jnp.log(1 - 2 * jnp.abs(u - 0.5))
        else:
            raise ValueError(f"Unknown noise_type: {self.config.noise_type}")

        return epsilon * self.config.noise_scale


class LinearSCM(StructuralCausalModel):
    """
    Linear Structural Causal Model.

    Given our convention A[i,j] != 0 means j → i, the SCM is:
        X[i] = Σ_j A[i,j] X[j] + ε[i]

    In matrix form:
        X = A X + ε

    Which can be rearranged to:
        (I - A) X = ε
        X = (I - A)^{-1} ε

    This matches the implicit causal layer formulation.
    """

    def sample(self, n_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Generate samples from linear SCM.

        Args:
            n_samples: Number of samples
            key: JAX random key

        Returns:
            X: (n_samples, num_nodes) observational data
        """
        # 1. Sample exogenous noise ε ~ N(0, σ²I) or other distribution
        epsilon = self._sample_noise(n_samples, key)

        # 2. Solve for X: (I - A) X = ε
        #    => X = (I - A)^{-1} ε
        I_minus_A = jnp.eye(self.num_nodes) - self.A

        # Solve the linear system for each sample
        # vmap over batch dimension
        X = jax.vmap(lambda eps: jnp.linalg.solve(I_minus_A, eps))(epsilon)

        return X

    def compute_total_effects(self) -> jnp.ndarray:
        """
        Compute total causal effect matrix for the linear SCM.

        For X = (I-A)^{-1} ε, the total causal effect of do(X_j += 1)
        on X_i is B[i,j] where B = (I-A)^{-1}.

        Returns:
            B: (d, d) total effect matrix. B[i,j] = total effect of X_j on X_i.
                Diagonal entries are 1.0 (self-effect).
        """
        I_minus_A = jnp.eye(self.num_nodes) - self.A
        B = jnp.linalg.inv(I_minus_A)
        return B


class NonlinearMLPSCM(StructuralCausalModel):
    """
    Nonlinear SCM using Multi-Layer Perceptrons.

    Implements: X_i = f_i(PA(X_i); θ_i) + ε_i
    where f_i is a small MLP for each node.

    The causal mechanism for each node is:
        X_i = MLP_i([X_{pa(i)}]) + ε_i

    Where:
        - pa(i) = parents of node i in the DAG
        - MLP_i = small neural network (e.g., [input_dim -> 32 -> 16 -> 1])
        - ε_i ~ noise distribution

    This creates a more realistic and challenging causal discovery task
    compared to linear SCMs.
    """

    def __init__(self, A: jnp.ndarray, config: SCMConfig, key: jax.random.PRNGKey):
        """
        Initialize NonlinearMLPSCM.

        Args:
            A: (d, d) adjacency matrix where A[i,j] != 0 means j → i
            config: SCM configuration
            key: JAX random key for parameter initialization
        """
        super().__init__(A, config)

        # Default hidden dims if not provided
        if config.nonlinear_hidden_dims is None:
            self.hidden_dims = [32, 16]  # Default: [input -> 32 -> 16 -> 1]
        else:
            self.hidden_dims = config.nonlinear_hidden_dims

        # Initialize MLP parameters for each node
        # Each node has its own MLP: f_i(PA_i) -> X_i
        self.mlp_params = []
        keys = jax.random.split(key, self.num_nodes)

        for i in range(self.num_nodes):
            # Find parents of node i
            parents = jnp.where(A[i] != 0)[0]
            num_parents = len(parents)

            # Initialize MLP for node i
            # Input: parent features (or just noise if no parents)
            input_dim = max(num_parents, 1)  # At least 1 for root nodes (noise only)

            # Create weight matrices for this node's MLP
            node_params = self._init_mlp_params(input_dim, self.hidden_dims, keys[i])
            self.mlp_params.append(node_params)

    def _init_mlp_params(
        self, input_dim: int, hidden_dims: List[int], key: jax.random.PRNGKey
    ) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Initialize MLP parameters using Xavier initialization.

        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            key: JAX random key

        Returns:
            params: List of (weight, bias) tuples for each layer
        """
        params = []
        dims = [input_dim] + hidden_dims + [1]  # Output is always 1 (scalar node value)

        keys = jax.random.split(key, len(dims) - 1)

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Xavier initialization
            scale = jnp.sqrt(2.0 / (in_dim + out_dim))
            W = jax.random.normal(keys[i], (in_dim, out_dim)) * scale
            b = jnp.zeros(out_dim)
            params.append((W, b))

        return params

    def _apply_mlp(
        self, x: jnp.ndarray, params: List[Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> jnp.ndarray:
        """
        Apply MLP with ReLU activations.

        Args:
            x: (batch_size, input_dim) or (input_dim,) input features
            params: List of (weight, bias) tuples

        Returns:
            output: (batch_size, 1) or (1,) MLP output
        """
        h = x
        for i, (W, b) in enumerate(params[:-1]):
            h = jnp.dot(h, W) + b
            h = jax.nn.relu(h)  # ReLU activation for hidden layers

        # Final layer (no activation, linear output)
        W_final, b_final = params[-1]
        output = jnp.dot(h, W_final) + b_final

        return output.squeeze(-1)  # Remove last dimension (scalar output)

    def sample(self, n_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Generate samples from nonlinear SCM using ancestral sampling.

        Args:
            n_samples: Number of samples
            key: JAX random key

        Returns:
            X: (n_samples, num_nodes) observational data
        """
        # 1. Sample exogenous noise
        epsilon = self._sample_noise(n_samples, key)

        # 2. Compute topological ordering of the DAG
        topo_order = self._topological_sort()

        # 3. Ancestral sampling: generate each node in topological order
        X = jnp.zeros((n_samples, self.num_nodes))

        for node_idx in topo_order:
            # Find parents of this node
            parents = jnp.where(self.A[node_idx] != 0)[0]

            if len(parents) == 0:
                # Root node: X_i = ε_i (no parents)
                X = X.at[:, node_idx].set(epsilon[:, node_idx])
            else:
                # Non-root: X_i = f_i(X_{parents}) + ε_i
                parent_values = X[:, parents]  # (n_samples, num_parents)

                # Apply MLP
                node_output = self._apply_mlp(parent_values, self.mlp_params[node_idx])

                # Add noise
                X = X.at[:, node_idx].set(node_output + epsilon[:, node_idx])

        return X

    def _topological_sort(self) -> List[int]:
        """
        Compute topological ordering of the DAG using Kahn's algorithm.

        Returns:
            topo_order: List of node indices in topological order
        """
        # Compute in-degree for each node
        in_degree = jnp.sum(self.A != 0, axis=1).astype(int)

        topo_order = []
        queue = [i for i in range(self.num_nodes) if in_degree[i] == 0]

        while queue:
            # Remove node with no incoming edges
            node = queue.pop(0)
            topo_order.append(node)

            # Reduce in-degree of children
            children = jnp.where(self.A[:, node] != 0)[0]
            for child in children:
                in_degree = in_degree.at[child].add(-1)
                if in_degree[child] == 0:
                    queue.append(int(child))

        if len(topo_order) != self.num_nodes:
            raise ValueError("Graph has a cycle! Cannot perform topological sort.")

        return topo_order


def create_scm(
    A: jnp.ndarray, config: SCMConfig, key: Optional[jax.random.PRNGKey] = None
) -> StructuralCausalModel:
    """
    Factory function to create an SCM based on configuration.

    Args:
        A: Adjacency matrix
        config: SCM configuration
        key: JAX random key (required for nonlinear_mlp, optional for linear)

    Returns:
        scm: Initialized SCM object

    Raises:
        ValueError: If scm_type is not recognized or key is missing for nonlinear
    """
    if config.scm_type == "linear":
        return LinearSCM(A, config)
    elif config.scm_type == "nonlinear_mlp":
        if key is None:
            raise ValueError("key is required for nonlinear_mlp SCM type")
        return NonlinearMLPSCM(A, config, key)
    else:
        raise ValueError(f"Unknown scm_type: {config.scm_type}")
