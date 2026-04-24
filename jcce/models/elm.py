"""
ELM (Extreme Learning Machine) Processor for latent representations.

This module implements an ELM-based processor that uses random fixed weights
in the hidden layer and only trains the output layer.

ELM is known for:
- Very fast training (only output weights are learned)
- Random hidden layer weights (never updated)
- Universal approximation capability
- Good generalization with minimal tuning

This serves as an interesting baseline that tests whether complex
learned transformations are necessary, or if random projections suffice.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


class ELMProcessor(nn.Module):
    """
    Extreme Learning Machine (ELM) processor for latent representations.

    Unlike MLP/GNN/Mamba which learn all weights, ELM:
    1. Uses RANDOM fixed weights in hidden layer
    2. Only learns the output projection

    This tests: "Do we need learned transformations, or do random
    projections provide sufficient representational power?"

    Key properties:
    - Fast: Only output layer is learned
    - Simple: No backprop through hidden layer
    - Effective: Often matches MLP performance

    Attributes:
        hidden_dim: Hidden dimension for ELM
        n_hidden_nodes: Number of random hidden nodes
        activation: Activation function for hidden layer ('relu', 'tanh', 'sigmoid')
        use_bias: Whether to use bias in random hidden layer
        weight_scale: Scale for random weight initialization
    """

    hidden_dim: int = 32
    n_hidden_nodes: int = 128
    activation: str = "tanh"
    use_bias: bool = True
    weight_scale: float = 1.0

    @nn.compact
    def __call__(self, z: jnp.ndarray, A: jnp.ndarray = None, training: bool = True) -> jnp.ndarray:
        """
        Process latent factors with ELM.

        Args:
            z: (batch_size, num_nodes) or (batch_size, num_nodes, features)
               latent causal factors
            A: (num_nodes, num_nodes) adjacency matrix (IGNORED by ELM)
               Included for interface compatibility
            training: Whether in training mode (not used by ELM, but kept for interface)

        Returns:
            h: (batch_size, num_nodes, hidden_dim) processed representations

        Example:
            >>> import jax
            >>> elm = ELMProcessor(hidden_dim=32, n_hidden_nodes=128)
            >>> z = jnp.ones((16, 10))  # 16 samples, 10 nodes
            >>> variables = elm.init(jax.random.PRNGKey(0), z)
            >>> h = elm.apply(variables, z)
            >>> print(h.shape)
            (16, 10, 32)
        """
        # NOTE: We IGNORE the adjacency matrix A (structure-agnostic like MLP)

        # Handle both 2D (batch, nodes) and 3D (batch, nodes, features) input
        if z.ndim == 2:
            # Scalar latents: z is (batch, nodes)
            batch_size, num_nodes = z.shape
            # Add feature dimension: (B, N) -> (B, N, 1)
            z_input = z[..., None]
            input_dim = 1
        elif z.ndim == 3:
            # Vector latents: z is (batch, nodes, features)
            batch_size, num_nodes, input_dim = z.shape
            z_input = z
        else:
            raise ValueError(f"Expected 2D or 3D input, got {z.ndim}D: {z.shape}")

        # ELM Layer 1: Random fixed projection
        # Initialize random weights ONCE and keep them fixed
        # In Flax, we use param() to create parameters
        # To make them "fixed", we simply don't update them during training

        # Create random input-to-hidden weights
        # Shape: (input_dim, n_hidden_nodes)
        W_hidden = self.param(
            "W_hidden",
            lambda rng, shape: jax.random.normal(rng, shape) * self.weight_scale,
            (input_dim, self.n_hidden_nodes),
        )

        # Create random bias (if enabled)
        if self.use_bias:
            b_hidden = self.param(
                "b_hidden",
                lambda rng, shape: jax.random.normal(rng, shape) * self.weight_scale,
                (self.n_hidden_nodes,),
            )
        else:
            b_hidden = 0.0

        # NOTE: In typical ELM, W_hidden and b_hidden are NEVER updated
        # We achieve this by using stop_gradient
        W_hidden = jax.lax.stop_gradient(W_hidden)
        if self.use_bias:
            b_hidden = jax.lax.stop_gradient(b_hidden)

        # Forward pass through random hidden layer
        # (B, N, input_dim) @ (input_dim, n_hidden_nodes) → (B, N, n_hidden_nodes)
        h_random = jnp.einsum("bni,ih->bnh", z_input, W_hidden) + b_hidden

        # Apply activation
        h_activated = self._activation(h_random)

        # ELM Layer 2: Learned output projection
        # This is the ONLY layer that is actually trained
        # (B, N, n_hidden_nodes) → (B, N, hidden_dim)
        h_out = nn.Dense(self.hidden_dim)(h_activated)

        return h_out  # (B, N, hidden_dim)

    def _activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation function to random hidden layer."""
        if self.activation == "relu":
            return nn.relu(x)
        elif self.activation == "tanh":
            return jnp.tanh(x)
        elif self.activation == "sigmoid":
            return nn.sigmoid(x)
        elif self.activation == "gelu":
            return nn.gelu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class ELMProcessorLegacy(nn.Module):
    """
    Legacy ELM implementation with explicit parameter handling.

    This version manually handles parameter initialization and
    provides more control over the random projection.

    Use ELMProcessor for simpler interface.
    """

    hidden_dim: int = 32
    n_hidden_nodes: int = 128
    activation: str = "tanh"

    def setup(self):
        """Setup ELM layers."""
        # Output projection (only trainable part)
        self.output_projection = nn.Dense(self.hidden_dim)

    def init_random_weights(self, input_dim: int, rng: jax.random.PRNGKey):
        """
        Initialize random hidden weights (called once, never updated).

        Args:
            input_dim: Input feature dimension
            rng: Random key for initialization

        Returns:
            Dictionary with 'W' and 'b' for random hidden layer
        """
        W_key, b_key = jax.random.split(rng)

        W = jax.random.normal(W_key, (input_dim, self.n_hidden_nodes)) * 0.5
        b = jax.random.normal(b_key, (self.n_hidden_nodes,)) * 0.1

        return {"W": W, "b": b}

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        random_weights: Optional[dict] = None,
        A: jnp.ndarray = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Process with ELM using provided random weights.

        This version requires pre-initialized random weights.
        """
        if random_weights is None:
            raise ValueError("ELMProcessorLegacy requires random_weights parameter")

        # Handle input dimensions
        if z.ndim == 2:
            z_input = z[..., None]
        elif z.ndim == 3:
            z_input = z
        else:
            raise ValueError(f"Expected 2D or 3D input, got {z.ndim}D")

        # Random projection (fixed weights)
        W = random_weights["W"]
        b = random_weights["b"]
        h_random = jnp.einsum("bni,ih->bnh", z_input, W) + b

        # Activation
        if self.activation == "tanh":
            h_activated = jnp.tanh(h_random)
        elif self.activation == "relu":
            h_activated = nn.relu(h_random)
        else:
            h_activated = nn.sigmoid(h_random)

        # Learned output (only trainable part)
        h_out = self.output_projection(h_activated)

        return h_out
