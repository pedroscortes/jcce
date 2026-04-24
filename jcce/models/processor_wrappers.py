"""
Unified Processor Wrappers for MA-Full.

Provides a consistent interface for all processor types:
- MambaProcessor: Selective state space model
- TransformerProcessor: Multi-head self-attention
- LSTMProcessor: Long short-term memory RNN
- GRUProcessor: Gated recurrent unit RNN
- GNNProcessor: Graph neural network (GCN/GAT/GraphSAGE/GIN)
- ELMProcessor: Extreme learning machine

All processors follow the same interface:
- __init__(config): Initialize with configuration dictionary
- __call__(z, A): Process latent factors given graph structure
- Standard Flax nn.Module API
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from jcce.models.elm import ELMProcessor as _ELMProcessorBase

# Import GNN and ELM
from jcce.models.gnn import CausalGNN

# Import existing Mamba implementation
from jcce.models.mamba import MambaProcessor as _MambaProcessorBase

# ============================================================================
# Mamba Processor Wrapper
# ============================================================================


class MambaProcessorWrapper(nn.Module):
    """
    Wrapper for MambaProcessor to match unified interface.

    The base MambaProcessor only takes z_sequence, but we want
    all processors to have the same interface: __call__(z, A).
    """

    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 2

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, A: Optional[jnp.ndarray] = None, training: bool = False
    ) -> jnp.ndarray:
        """
        Process latent factors using Mamba.

        Args:
            z: (batch_size, n_vars) latent factors
            A: (n_vars, n_vars) adjacency matrix (optional, not used by Mamba)
            training: Whether in training mode (not used)

        Returns:
            h: (batch_size, n_vars, d_model) processed representations
        """
        # Reshape to sequence format: (B, N, 1) → (B, N, d_model)
        z_expanded = z[..., None]  # (B, N, 1)

        # Project to d_model
        z_projected = nn.Dense(self.d_model, name="input_projection")(z_expanded)

        # Apply Mamba processor
        mamba = _MambaProcessorBase(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        h = mamba(z_projected)  # (B, N, d_model)

        return h


# ============================================================================
# Transformer Processor
# ============================================================================


class TransformerProcessor(nn.Module):
    """
    Transformer-based processor for causal latent factors.

    Uses multi-head self-attention to model relationships between variables.
    Can optionally incorporate graph structure via attention masking.

    Attributes:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward network hidden dimension
        dropout_rate: Dropout probability
        use_graph_mask: Whether to mask attention based on DAG structure
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout_rate: float = 0.1
    use_graph_mask: bool = True

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, A: Optional[jnp.ndarray] = None, training: bool = False
    ) -> jnp.ndarray:
        """
        Process latent factors using transformer.

        Args:
            z: (batch_size, n_vars) latent factors
            A: (n_vars, n_vars) adjacency matrix (optional)
               A[i,j] != 0 means edge j → i
            training: Whether in training mode (for dropout)

        Returns:
            h: (batch_size, n_vars, d_model) processed representations
        """
        batch_size, n_vars = z.shape

        # Project to d_model
        # (B, N) → (B, N, D)
        x = z[..., None]  # Add feature dimension
        x = nn.Dense(self.d_model, name="input_projection")(x)

        # Add positional encoding (simple learned embeddings)
        pos_embed = self.param(
            "pos_embed", nn.initializers.normal(stddev=0.02), (1, n_vars, self.d_model)
        )
        x = x + pos_embed

        # Create attention mask from graph structure if provided
        attn_mask = None
        if self.use_graph_mask and A is not None:
            # Mask: allow attention from j to i if A[i,j] != 0 (j is parent of i)
            # Also allow self-attention
            # Shape: (n_vars, n_vars)
            attn_mask = (jnp.abs(A) > 0).astype(jnp.float32)
            # Add self-attention
            attn_mask = attn_mask + jnp.eye(n_vars)
            # Convert to attention mask format (1 = allow, 0 = mask)
            # Expand for batch and heads: (1, 1, n_vars, n_vars)
            attn_mask = attn_mask[None, None, :, :]

        # Transformer layers
        for layer_idx in range(self.n_layers):
            # Multi-head self-attention
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                out_features=self.d_model,
                dropout_rate=self.dropout_rate if training else 0.0,
                name=f"attention_{layer_idx}",
            )(x, x, mask=attn_mask, deterministic=not training)

            # Residual + LayerNorm
            x = nn.LayerNorm(name=f"ln1_{layer_idx}")(x + attn_out)

            # Feed-forward network
            ff_out = nn.Dense(self.d_ff, name=f"ff1_{layer_idx}")(x)
            ff_out = nn.gelu(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_out)
            ff_out = nn.Dense(self.d_model, name=f"ff2_{layer_idx}")(ff_out)

            # Residual + LayerNorm
            x = nn.LayerNorm(name=f"ln2_{layer_idx}")(x + ff_out)

        return x  # (B, N, D)


# ============================================================================
# LSTM Processor
# ============================================================================


class LSTMProcessor(nn.Module):
    """
    LSTM-based processor for causal latent factors.

    Processes variables sequentially according to topological order of DAG.
    Uses bidirectional LSTM optionally.

    Attributes:
        hidden_size: LSTM hidden state size
        n_layers: Number of LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        dropout_rate: Dropout probability between layers
    """

    hidden_size: int = 128
    n_layers: int = 2
    bidirectional: bool = False
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, A: Optional[jnp.ndarray] = None, training: bool = False
    ) -> jnp.ndarray:
        """
        Process latent factors using LSTM.

        Args:
            z: (batch_size, n_vars) latent factors
            A: (n_vars, n_vars) adjacency matrix (optional, used for topological sort)
            training: Whether in training mode (for dropout)

        Returns:
            h: (batch_size, n_vars, hidden_size * (2 if bidirectional else 1))
        """
        batch_size, n_vars = z.shape

        # Reshape to sequence format: (B, N, 1)
        x = z[..., None]

        # Project to hidden_size
        x = nn.Dense(self.hidden_size, name="input_projection")(x)

        # Stack LSTM layers
        for layer_idx in range(self.n_layers):
            if self.bidirectional:
                # Forward LSTM
                lstm_cell_fwd = nn.LSTMCell(features=self.hidden_size)
                lstm_fwd = nn.RNN(lstm_cell_fwd, name=f"lstm_fwd_{layer_idx}")
                carry_fwd = lstm_cell_fwd.initialize_carry(
                    jax.random.PRNGKey(0), (batch_size, self.hidden_size)
                )
                out_fwd = lstm_fwd(x, initial_carry=carry_fwd)

                # Backward LSTM (reverse sequence)
                lstm_cell_bwd = nn.LSTMCell(features=self.hidden_size)
                lstm_bwd = nn.RNN(lstm_cell_bwd, name=f"lstm_bwd_{layer_idx}")
                carry_bwd = lstm_cell_bwd.initialize_carry(
                    jax.random.PRNGKey(0), (batch_size, self.hidden_size)
                )
                x_reversed = jnp.flip(x, axis=1)
                out_bwd = lstm_bwd(x_reversed, initial_carry=carry_bwd)
                out_bwd = jnp.flip(out_bwd, axis=1)  # Reverse back

                # Concatenate forward and backward
                x = jnp.concatenate([out_fwd, out_bwd], axis=-1)

                # Project back to hidden_size for next layer
                if layer_idx < self.n_layers - 1:
                    x = nn.Dense(self.hidden_size, name=f"projection_{layer_idx}")(x)

            else:
                # Unidirectional LSTM
                lstm_cell = nn.LSTMCell(features=self.hidden_size)
                lstm = nn.RNN(lstm_cell, name=f"lstm_{layer_idx}")
                carry = lstm_cell.initialize_carry(
                    jax.random.PRNGKey(0), (batch_size, self.hidden_size)
                )
                x = lstm(x, initial_carry=carry)

            # Dropout between layers (except last layer)
            if layer_idx < self.n_layers - 1 and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x  # (B, N, hidden_size * (2 if bidirectional else 1))


# ============================================================================
# GRU Processor
# ============================================================================


class GRUProcessor(nn.Module):
    """
    GRU-based processor for causal latent factors.

    Similar to LSTM but with gated recurrent units (simpler, often faster).

    Attributes:
        hidden_size: GRU hidden state size
        n_layers: Number of GRU layers
        bidirectional: Whether to use bidirectional GRU
        dropout_rate: Dropout probability between layers
    """

    hidden_size: int = 128
    n_layers: int = 2
    bidirectional: bool = False
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, A: Optional[jnp.ndarray] = None, training: bool = False
    ) -> jnp.ndarray:
        """
        Process latent factors using GRU.

        Args:
            z: (batch_size, n_vars) latent factors
            A: (n_vars, n_vars) adjacency matrix (optional)
            training: Whether in training mode (for dropout)

        Returns:
            h: (batch_size, n_vars, hidden_size * (2 if bidirectional else 1))
        """
        batch_size, n_vars = z.shape

        # Reshape to sequence format: (B, N, 1)
        x = z[..., None]

        # Project to hidden_size
        x = nn.Dense(self.hidden_size, name="input_projection")(x)

        # Stack GRU layers
        for layer_idx in range(self.n_layers):
            if self.bidirectional:
                # Forward GRU
                gru_cell_fwd = nn.GRUCell(features=self.hidden_size)
                gru_fwd = nn.RNN(gru_cell_fwd, name=f"gru_fwd_{layer_idx}")
                carry_fwd = gru_cell_fwd.initialize_carry(
                    jax.random.PRNGKey(0), (batch_size, self.hidden_size)
                )
                out_fwd = gru_fwd(x, initial_carry=carry_fwd)

                # Backward GRU (reverse sequence)
                gru_cell_bwd = nn.GRUCell(features=self.hidden_size)
                gru_bwd = nn.RNN(gru_cell_bwd, name=f"gru_bwd_{layer_idx}")
                carry_bwd = gru_cell_bwd.initialize_carry(
                    jax.random.PRNGKey(0), (batch_size, self.hidden_size)
                )
                x_reversed = jnp.flip(x, axis=1)
                out_bwd = gru_bwd(x_reversed, initial_carry=carry_bwd)
                out_bwd = jnp.flip(out_bwd, axis=1)  # Reverse back

                # Concatenate forward and backward
                x = jnp.concatenate([out_fwd, out_bwd], axis=-1)

                # Project back to hidden_size for next layer
                if layer_idx < self.n_layers - 1:
                    x = nn.Dense(self.hidden_size, name=f"projection_{layer_idx}")(x)

            else:
                # Unidirectional GRU
                gru_cell = nn.GRUCell(features=self.hidden_size)
                gru = nn.RNN(gru_cell, name=f"gru_{layer_idx}")
                carry = gru_cell.initialize_carry(
                    jax.random.PRNGKey(0), (batch_size, self.hidden_size)
                )
                x = gru(x, initial_carry=carry)

            # Dropout between layers (except last layer)
            if layer_idx < self.n_layers - 1 and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x  # (B, N, hidden_size * (2 if bidirectional else 1))


# ============================================================================
# GNN Processor Wrapper
# ============================================================================


class GNNProcessorWrapper(nn.Module):
    """
    Wrapper for CausalGNN to match unified interface.

    Supports multiple GNN layer types: GCN, GAT, GraphSAGE, GIN.

    Attributes:
        hidden_dim: Hidden dimension for node features
        n_layers: Number of GNN layers
        gnn_type: Type of GNN layer ('gcn', 'gat', 'graphsage', 'gin')
        aggregation: Aggregation function ('mean', 'max', 'sum') for GraphSAGE
    """

    hidden_dim: int = 64
    n_layers: int = 2
    gnn_type: str = "gat"
    aggregation: str = "mean"

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, A: Optional[jnp.ndarray] = None, training: bool = False
    ) -> jnp.ndarray:
        """
        Process latent factors using GNN.

        Args:
            z: (batch_size, n_vars) latent factors
            A: (n_vars, n_vars) adjacency matrix (required for GNN!)
            training: Whether in training mode

        Returns:
            h: (batch_size, n_vars, hidden_dim) processed representations
        """
        if A is None:
            raise ValueError("GNN requires adjacency matrix A")

        # Map gnn_type to use_attention flag for CausalGNN
        # CausalGNN uses use_attention=True for GAT, False for other types
        use_attention = self.gnn_type == "gat"

        # Use CausalGNN
        gnn = CausalGNN(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            use_attention=use_attention,
            num_heads=4 if use_attention else 1,  # Only relevant for GAT
        )

        return gnn(z, A, training=training)


# ============================================================================
# ELM Processor Wrapper
# ============================================================================


class ELMProcessorWrapper(nn.Module):
    """
    Wrapper for ELMProcessor to match unified interface.

    ELM uses random fixed hidden layer weights and only trains output layer.
    Extremely fast to train.

    Attributes:
        hidden_dim: Output hidden dimension
        n_hidden_nodes: Number of random hidden nodes
        activation: Activation function for hidden layer
    """

    hidden_dim: int = 32
    n_hidden_nodes: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, A: Optional[jnp.ndarray] = None, training: bool = False
    ) -> jnp.ndarray:
        """
        Process latent factors using ELM.

        Args:
            z: (batch_size, n_vars) latent factors
            A: (n_vars, n_vars) adjacency matrix (ignored by ELM)
            training: Whether in training mode

        Returns:
            h: (batch_size, n_vars, hidden_dim) processed representations
        """
        # ELM ignores adjacency matrix (structure-agnostic like Mamba/Transformer)
        elm = _ELMProcessorBase(
            hidden_dim=self.hidden_dim,
            n_hidden_nodes=self.n_hidden_nodes,
            activation=self.activation,
        )

        return elm(z, A, training=training)


# ============================================================================
# Factory Function
# ============================================================================


def create_processor(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create processor based on configuration.

    Args:
        config: Configuration dictionary with keys:
            - 'processor_type': 'mamba', 'transformer', 'lstm', or 'gru'
            - Type-specific parameters (e.g., d_model, n_heads, etc.)

    Returns:
        Processor module instance

    Example:
        >>> config = {
        ...     'processor_type': 'mamba',
        ...     'd_model': 128,
        ...     'd_state': 16,
        ...     'd_conv': 4,
        ...     'expand': 2
        ... }
        >>> processor = create_processor(config)
    """
    processor_type = config["processor_type"]

    if processor_type == "mamba":
        return MambaProcessorWrapper(
            d_model=config.get("d_model", 128),
            d_state=config.get("d_state", 16),
            d_conv=config.get("d_conv", 4),
            expand=config.get("expand", 2),
            n_layers=config.get("n_layers", 2),
        )

    elif processor_type == "transformer":
        return TransformerProcessor(
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            d_ff=config.get("d_ff", 512),
            dropout_rate=config.get("dropout", 0.1),
        )

    elif processor_type == "lstm":
        return LSTMProcessor(
            hidden_size=config.get("hidden_size", 128),
            n_layers=config.get("n_layers", 2),
            bidirectional=config.get("bidirectional", False),
            dropout_rate=config.get("dropout", 0.1),
        )

    elif processor_type == "gru":
        return GRUProcessor(
            hidden_size=config.get("hidden_size", 128),
            n_layers=config.get("n_layers", 2),
            bidirectional=config.get("bidirectional", False),
            dropout_rate=config.get("dropout", 0.1),
        )

    elif processor_type == "gnn":
        return GNNProcessorWrapper(
            hidden_dim=config.get("hidden_dim", 64),
            n_layers=config.get("n_layers", 2),
            gnn_type=config.get("gnn_type", "gat"),
            aggregation=config.get("aggregation", "mean"),
        )

    elif processor_type == "elm":
        return ELMProcessorWrapper(
            hidden_dim=config.get("hidden_dim", 32),
            n_hidden_nodes=config.get("n_hidden_nodes", 128),
            activation=config.get("activation", "tanh"),
        )

    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


def get_processor_output_dim(config: Dict[str, Any]) -> int:
    """
    Get the output dimension of a processor given its configuration.

    Args:
        config: Processor configuration dictionary

    Returns:
        Output dimension (last axis of processor output)

    Example:
        >>> config = {'processor_type': 'lstm', 'hidden_size': 128, 'bidirectional': True}
        >>> get_processor_output_dim(config)
        256
    """
    processor_type = config["processor_type"]

    if processor_type == "mamba":
        return config.get("d_model", 128)

    elif processor_type == "transformer":
        return config.get("d_model", 128)

    elif processor_type == "lstm":
        hidden_size = config.get("hidden_size", 128)
        bidirectional = config.get("bidirectional", False)
        return hidden_size * (2 if bidirectional else 1)

    elif processor_type == "gru":
        hidden_size = config.get("hidden_size", 128)
        bidirectional = config.get("bidirectional", False)
        return hidden_size * (2 if bidirectional else 1)

    elif processor_type == "gnn":
        return config.get("hidden_dim", 64)

    elif processor_type == "elm":
        return config.get("hidden_dim", 32)

    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
