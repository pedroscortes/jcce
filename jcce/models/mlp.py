"""
MLP Processor for latent representations.

This module implements a simple Multi-Layer Perceptron (MLP) processor
that processes latent factors without using causal graph structure.

This serves as another baseline to compare against:
- 'none': No processing at all
- 'mlp': Simple feed-forward processing (this module)
- 'gnn': Graph-aware processing
- 'mamba': Sequence-aware processing
"""

import jax.numpy as jnp
from flax import linen as nn


class MLPProcessor(nn.Module):
    """
    Simple MLP processor for latent representations.

    Unlike GNN or Mamba, this processor does NOT use the causal graph
    structure. It simply applies feed-forward transformations to the
    latent factors.

    This tests the hypothesis: "Does using causal structure (via GNN/Mamba)
    help compared to structure-agnostic processing?"

    Attributes:
        hidden_dim: Hidden dimension for MLP layers
        n_layers: Number of MLP layers
        activation: Activation function ('relu', 'gelu', 'tanh')
        use_layer_norm: Whether to use layer normalization
        use_residual: Whether to use residual connections
    """

    hidden_dim: int = 32
    n_layers: int = 2
    activation: str = 'relu'
    use_layer_norm: bool = True
    use_residual: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        A: jnp.ndarray = None,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Process latent factors with MLP.

        Args:
            z: (batch_size, num_nodes) or (batch_size, num_nodes, features)
               latent causal factors
            A: (num_nodes, num_nodes) adjacency matrix (IGNORED by MLP)
               Included for interface compatibility with GNN/Mamba
            training: Whether in training mode

        Returns:
            h: (batch_size, num_nodes, hidden_dim) processed representations

        Example:
            >>> import jax
            >>> mlp = MLPProcessor(hidden_dim=32, n_layers=3)
            >>> z = jnp.ones((16, 10))  # 16 samples, 10 nodes
            >>> A = jnp.eye(10)  # Not used by MLP
            >>> variables = mlp.init(jax.random.PRNGKey(0), z, A, training=True)
            >>> h = mlp.apply(variables, z, A, training=True)
            >>> print(h.shape)
            (16, 10, 32)
        """
        # NOTE: We IGNORE the adjacency matrix A (structure-agnostic)

        # Handle both 2D (batch, nodes) and 3D (batch, nodes, features) input
        if z.ndim == 2:
            # Scalar latents: z is (batch, nodes)
            batch_size, num_nodes = z.shape
            # Add feature dimension: (B, N) -> (B, N, 1)
            h = z[..., None]
        elif z.ndim == 3:
            # Vector latents: z is (batch, nodes, features)
            batch_size, num_nodes, input_dim = z.shape
            h = z
        else:
            raise ValueError(f"Expected 2D or 3D input, got {z.ndim}D: {z.shape}")

        # Initial projection to hidden_dim
        h = nn.Dense(self.hidden_dim)(h)
        if self.use_layer_norm:
            h = nn.LayerNorm()(h)
        h = self._activation(h)
        # Dropout after first projection
        # deterministic=not training uses scaled identity during evaluation
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)

        # MLP layers with optional residual connections
        for layer_idx in range(self.n_layers):
            h_input = h

            # Two-layer MLP block
            h = nn.Dense(self.hidden_dim * 2)(h)
            if self.use_layer_norm:
                h = nn.LayerNorm()(h)
            h = self._activation(h)
            # Dropout in hidden layer
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)

            h = nn.Dense(self.hidden_dim)(h)
            if self.use_layer_norm:
                h = nn.LayerNorm()(h)

            # Residual connection
            if self.use_residual:
                h = h + h_input

            h = self._activation(h)

        return h  # (B, N, hidden_dim)

    def _activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return nn.relu(x)
        elif self.activation == 'gelu':
            return nn.gelu(x)
        elif self.activation == 'tanh':
            return jnp.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
