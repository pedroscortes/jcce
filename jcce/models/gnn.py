"""
Causal Graph Neural Networks for DAG-structured latent spaces.

This module implements GNN variants that respect the causal (DAG) structure:
- CausalGNN: Flexible GNN with multiple layer types (GCN, GAT, GraphSAGE, GIN)
- CausalGATLayer: Graph attention for adaptive message weighting
- CausalGCNLayer: Graph convolutional layer (simple aggregation)
- CausalGraphSAGELayer: GraphSAGE with mean/max/sum aggregation
- CausalGINLayer: Graph Isomorphism Network (more expressive)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class CausalGCNLayer(nn.Module):
    """
    Graph Convolutional Network (GCN) layer for causal structures.

    Simple message passing with mean aggregation from parent nodes.
    This is the simplest GNN baseline - no attention, just averaging.

    Attributes:
        hidden_dim: Hidden dimension for node features
    """

    hidden_dim: int

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        A: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """
        GCN layer forward pass.

        Args:
            h: (batch, num_nodes, hidden_dim) node features
            A: (num_nodes, num_nodes) adjacency matrix
            training: Whether in training mode

        Returns:
            h_out: (batch, num_nodes, hidden_dim) updated node features
        """
        batch_size, num_nodes, d = h.shape

        # Compute degree matrix for normalization
        # D[i,i] = number of parents of node i
        degree = jnp.sum(A != 0, axis=1)  # (N,)
        degree = jnp.maximum(degree, 1.0)  # Avoid division by zero
        D_inv = jnp.diag(1.0 / degree)  # (N, N)

        # Transform features
        h_transformed = nn.Dense(self.hidden_dim)(h)  # (B, N, D)

        # Aggregate messages from parents (normalized by degree)
        # A_norm = D^{-1} @ A
        A_norm = D_inv @ A  # (N, N)

        # Message passing: h_msg = A_norm @ h_transformed
        h_msg = jnp.einsum("ij,bjd->bid", A_norm, h_transformed)  # (B, N, D)

        return h_msg


class CausalGraphSAGELayer(nn.Module):
    """
    GraphSAGE layer for causal structures.

    Uses different aggregation functions (mean, max, sum) to aggregate
    messages from parent nodes, then concatenates with self features.

    Attributes:
        hidden_dim: Hidden dimension for node features
        aggregation: Aggregation type ('mean', 'max', 'sum')
    """

    hidden_dim: int
    aggregation: str = 'mean'

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        A: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """
        GraphSAGE layer forward pass.

        Args:
            h: (batch, num_nodes, hidden_dim) node features
            A: (num_nodes, num_nodes) adjacency matrix
            training: Whether in training mode

        Returns:
            h_out: (batch, num_nodes, hidden_dim) updated node features
        """
        batch_size, num_nodes, d = h.shape

        # Transform neighbor features
        h_neighbors = nn.Dense(self.hidden_dim)(h)  # (B, N, D)

        # Aggregate messages from parents
        if self.aggregation == 'mean':
            # Mean aggregation (same as GCN without normalization by degree)
            degree = jnp.sum(A != 0, axis=1, keepdims=True)  # (N, 1)
            degree = jnp.maximum(degree, 1.0)  # (N, 1)
            # Add batch dimension for broadcasting: (N, 1) -> (1, N, 1)
            degree = degree[None, :, :]  # (1, N, 1)
            h_agg = jnp.einsum("ij,bjd->bid", A, h_neighbors) / degree  # (B, N, D) / (1, N, 1) = (B, N, D)
        elif self.aggregation == 'max':
            # Max aggregation
            # For each node, take max over parent features
            # This requires masking non-parents with -inf
            h_expanded = h_neighbors[:, None, :, :]  # (B, 1, N, D)
            A_mask = (A != 0)[None, :, :, None]  # (1, N, N, 1)
            h_masked = jnp.where(A_mask, h_expanded, -1e9)  # (B, N, N, D)
            h_agg = jnp.max(h_masked, axis=2)  # (B, N, D)
        elif self.aggregation == 'sum':
            # Sum aggregation
            h_agg = jnp.einsum("ij,bjd->bid", A, h_neighbors)  # (B, N, D)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Transform self features
        h_self = nn.Dense(self.hidden_dim)(h)  # (B, N, D)

        # Concatenate and project
        h_combined = jnp.concatenate([h_self, h_agg], axis=-1)  # (B, N, 2D)
        h_out = nn.Dense(self.hidden_dim)(h_combined)  # (B, N, D)

        return h_out


class CausalGINLayer(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer for causal structures.

    GIN is more expressive than GCN/GraphSAGE due to the (1+epsilon) term
    that makes the aggregation injective.

    Reference: "How Powerful are Graph Neural Networks?" (Xu et al., 2019)

    Attributes:
        hidden_dim: Hidden dimension for node features
        epsilon_trainable: Whether epsilon is trainable (vs fixed at 0)
    """

    hidden_dim: int
    epsilon_trainable: bool = True

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        A: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """
        GIN layer forward pass.

        Args:
            h: (batch, num_nodes, hidden_dim) node features
            A: (num_nodes, num_nodes) adjacency matrix
            training: Whether in training mode

        Returns:
            h_out: (batch, num_nodes, hidden_dim) updated node features
        """
        batch_size, num_nodes, d = h.shape

        # Epsilon parameter (either trainable or fixed)
        if self.epsilon_trainable:
            epsilon = self.param('epsilon', nn.initializers.zeros, (1,))
        else:
            epsilon = 0.0

        # Aggregate messages from parents (sum aggregation)
        h_neighbors = jnp.einsum("ij,bjd->bid", A, h)  # (B, N, D)

        # GIN aggregation: (1 + epsilon) * h_self + sum(h_neighbors)
        h_combined = (1 + epsilon) * h + h_neighbors  # (B, N, D)

        # MLP transformation (two layers as in original GIN)
        h_out = nn.Dense(self.hidden_dim)(h_combined)
        h_out = nn.relu(h_out)
        h_out = nn.Dense(self.hidden_dim)(h_out)

        return h_out


class CausalGATLayer(nn.Module):
    """
    Graph Attention Layer for causal (DAG) structures.

    Uses attention mechanism to weight messages from parent nodes.
    This allows the model to learn which causal parents are most important.

    Attributes:
        hidden_dim: Hidden dimension for node features
        num_heads: Number of attention heads (for multi-head attention)
        dropout_rate: Dropout rate (0.0 = no dropout)
    """

    hidden_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        A: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Graph attention layer forward pass.

        Args:
            h: (batch, num_nodes, hidden_dim) node features
            A: (num_nodes, num_nodes) adjacency matrix
               A[i,j] != 0 means j → i (j is parent of i)
            training: Whether in training mode (for dropout)

        Returns:
            h_out: (batch, num_nodes, hidden_dim) updated node features
        """
        batch_size, num_nodes, d = h.shape

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0
        d_head = self.hidden_dim // self.num_heads

        # Linear projections for Q, K, V
        Q = nn.Dense(self.hidden_dim)(h)  # (B, N, D)
        K = nn.Dense(self.hidden_dim)(h)  # (B, N, D)
        V = nn.Dense(self.hidden_dim)(h)  # (B, N, D)

        # Reshape for multi-head attention
        # (B, N, D) → (B, N, num_heads, d_head) → (B, num_heads, N, d_head)
        Q = Q.reshape(batch_size, num_nodes, self.num_heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, num_nodes, self.num_heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, num_nodes, self.num_heads, d_head).transpose(0, 2, 1, 3)

        # Compute attention scores: Q @ K^T / sqrt(d_head)
        # (B, num_heads, N, d_head) @ (B, num_heads, d_head, N) → (B, num_heads, N, N)
        scores = jnp.einsum('bhqd,bhkd->bhqk', Q, K) / jnp.sqrt(d_head)

        # Mask attention to only parent nodes (using adjacency matrix)
        # A[i,j] != 0 means j is parent of i
        # So attention should flow from j to i
        # Create mask: allow attention from j to i only if A[i,j] != 0
        mask = A[None, None, :, :]  # (1, 1, N, N) - broadcast over batch and heads
        mask = (mask != 0).astype(jnp.float32)

        # Apply mask (set -inf where mask is 0, so softmax gives 0)
        scores = jnp.where(mask, scores, -1e9)

        # Softmax to get attention weights
        attn_weights = nn.softmax(scores, axis=-1)  # (B, num_heads, N, N)

        # Note: Dropout removed for simplicity
        # Can add back with proper RNG handling if needed

        # Apply attention to values
        # (B, num_heads, N, N) @ (B, num_heads, N, d_head) → (B, num_heads, N, d_head)
        h_attn = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, V)

        # Reshape back: (B, num_heads, N, d_head) → (B, N, D)
        h_attn = h_attn.transpose(0, 2, 1, 3).reshape(batch_size, num_nodes, self.hidden_dim)

        # Output projection
        h_out = nn.Dense(self.hidden_dim)(h_attn)

        return h_out


class CausalGNN(nn.Module):
    """
    Causal Graph Neural Network with multiple layer type options.

    This GNN respects the DAG structure by:
    1. Only allowing messages from parent to child nodes
    2. Optionally processing in topological order
    3. Supporting multiple GNN architectures (GCN, GAT, GraphSAGE, GIN)

    This is the "natural baseline" for causal structure - using graph
    structure explicitly via message passing.

    Attributes:
        hidden_dim: Hidden dimension for node features
        n_layers: Number of message passing layers
        layer_type: Type of GNN layer ('gcn', 'gat', 'graphsage', 'gin')
        num_heads: Number of attention heads (if layer_type='gat')
        sage_aggregation: Aggregation for GraphSAGE ('mean', 'max', 'sum')
        aggregation: How to aggregate multi-layer outputs ('last', 'concat', 'sum')

    Legacy attributes (for backwards compatibility):
        use_attention: If True, uses GAT; otherwise GCN (deprecated, use layer_type)
    """

    hidden_dim: int = 32
    n_layers: int = 2
    layer_type: str = 'gat'  # 'gcn', 'gat', 'graphsage', 'gin'
    num_heads: int = 4
    sage_aggregation: str = 'mean'
    aggregation: str = 'last'  # 'last', 'concat', 'sum'

    # Legacy support
    use_attention: bool = True  # Ignored if layer_type is specified

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        A: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Process latent factors using causal graph structure.

        Args:
            z: (batch_size, num_nodes) latent causal factors
            A: (num_nodes, num_nodes) adjacency matrix
               A[i,j] != 0 means j → i (j is parent of i)
            training: Whether in training mode

        Returns:
            h: (batch_size, num_nodes, hidden_dim) node embeddings

        Example:
            >>> gnn = CausalGNN(hidden_dim=32, n_layers=3, use_attention=True)
            >>> z = jnp.ones((16, 10))  # 16 samples, 10 nodes (v1 style)
            >>> A = jnp.eye(10)  # Example adjacency
            >>> variables = gnn.init(jax.random.PRNGKey(0), z, A, training=True)
            >>> h = gnn.apply(variables, z, A, training=True)
            >>> print(h.shape)
            (16, 10, 32)

            Or with features (v2 style):
            >>> z = jnp.ones((16, 10, 64))  # 16 samples, 10 nodes, 64 features
            >>> h = gnn.apply(variables, z, A, training=True)
            >>> print(h.shape)
            (16, 10, 32)
        """
        # Handle both 2D (batch, nodes) and 3D (batch, nodes, features) input
        if z.ndim == 2:
            # Scalar latents: z is (batch, nodes)
            batch_size, num_nodes = z.shape
            # Initialize node features from latent factors
            # (B, N) → (B, N, 1) → (B, N, D)
            h = z[..., None]  # Add feature dimension
            h = nn.Dense(self.hidden_dim)(h)  # Project to hidden_dim
            h = nn.relu(h)
        elif z.ndim == 3:
            # Feature latents: z is (batch, nodes, features)
            batch_size, num_nodes, input_dim = z.shape
            # Project input features to hidden_dim
            h = nn.Dense(self.hidden_dim)(z)  # (B, N, input_dim) → (B, N, D)
            h = nn.relu(h)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {z.ndim}D: {z.shape}")

        # Store layer outputs for aggregation
        layer_outputs = []

        # Message passing layers
        for layer_idx in range(self.n_layers):
            h_prev = h

            # Select GNN layer type
            if self.layer_type == 'gcn':
                # Graph Convolutional Network
                h_msg = CausalGCNLayer(
                    hidden_dim=self.hidden_dim
                )(h, A, training=training)

            elif self.layer_type == 'gat':
                # Graph Attention Network
                h_msg = CausalGATLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=0.0
                )(h, A, training=training)

            elif self.layer_type == 'graphsage':
                # GraphSAGE
                h_msg = CausalGraphSAGELayer(
                    hidden_dim=self.hidden_dim,
                    aggregation=self.sage_aggregation
                )(h, A, training=training)

            elif self.layer_type == 'gin':
                # Graph Isomorphism Network
                h_msg = CausalGINLayer(
                    hidden_dim=self.hidden_dim,
                    epsilon_trainable=True
                )(h, A, training=training)

            else:
                raise ValueError(f"Unknown layer_type: {self.layer_type}")

            # For GIN and GraphSAGE, the output is already processed
            # For GCN and GAT, add residual and MLP
            if self.layer_type in ['gin', 'graphsage']:
                # These layers already include feature transformation
                h = h_msg
                h = nn.LayerNorm()(h)
            else:
                # Combine message with self (residual connection)
                h = h_prev + nn.Dense(self.hidden_dim)(h_msg)
                h = nn.LayerNorm()(h)
                h = nn.relu(h)

                # MLP for node update
                h_mlp = nn.Dense(self.hidden_dim * 2)(h)
                h_mlp = nn.relu(h_mlp)
                h_mlp = nn.Dense(self.hidden_dim)(h_mlp)

                # Residual connection
                h = h + h_mlp
                h = nn.LayerNorm()(h)

            layer_outputs.append(h)

        # Aggregate across layers
        if self.aggregation == 'last':
            h_out = layer_outputs[-1]
        elif self.aggregation == 'concat':
            # Concatenate all layers and project
            h_concat = jnp.concatenate(layer_outputs, axis=-1)  # (B, N, D * n_layers)
            h_out = nn.Dense(self.hidden_dim)(h_concat)
        elif self.aggregation == 'sum':
            h_out = sum(layer_outputs)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return h_out  # (B, N, D)


class CausalGNNProcessor(nn.Module):
    """
    Wrapper for CausalGNN to match the processor interface.

    This allows easy swapping between Mamba and GNN in UnifiedCausalVAE.

    Attributes:
        hidden_dim: Hidden dimension
        n_layers: Number of GNN layers
        use_attention: Use graph attention
        num_heads: Number of attention heads
    """

    hidden_dim: int = 32
    n_layers: int = 2
    use_attention: bool = True
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        A: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Process latent factors with GNN.

        Args:
            z: (batch_size, num_nodes) latent factors
            A: (num_nodes, num_nodes) adjacency matrix
            training: Whether in training mode

        Returns:
            h: (batch_size, num_nodes, hidden_dim) node embeddings
        """
        return CausalGNN(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            use_attention=self.use_attention,
            num_heads=self.num_heads
        )(z, A, training=training)
