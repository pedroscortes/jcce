"""
Classification Processor Wrappers for C-Lite.

Adapts MA-Full's processors for direct classification on Markov Blanket features.

Key differences from MA-Full processors:
1. Input: Markov Blanket features (batch_size, n_mb_features) - no graph structure
2. Output: Class logits (batch_size, n_classes)
3. Simpler interface: no VAE latents, direct feature → prediction

All processors follow the same interface:
- __init__(config, n_features, n_classes): Initialize with feature/class dimensions
- __call__(x, training=False): Process features → logits
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from jcce.models.gnn import CausalGNN

# Import base processors
from jcce.models.mamba import MambaProcessor as MambaProcessorBase

# ============================================================================
# MLP Classifier (Baseline)
# ============================================================================


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron classifier.

    Simple baseline that's fast and works well for tabular data.
    """

    hidden_sizes: tuple = (64, 32)
    n_classes: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, n_features) input features
            training: Whether in training mode (for dropout)

        Returns:
            logits: (batch_size, n_classes) class logits
        """
        # Hidden layers
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size, name=f"fc{i + 1}")(x)
            x = nn.relu(x)
            if training and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Output layer
        logits = nn.Dense(self.n_classes, name="output")(x)

        return logits


# ============================================================================
# Mamba Classifier
# ============================================================================


class MambaClassifier(nn.Module):
    """
    Mamba-based classifier for sequence modeling of features.

    Treats features as a sequence and uses Mamba's selective state space.
    Good for capturing long-range dependencies in feature space.
    """

    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 2
    n_classes: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, n_features) input features
            training: Whether in training mode

        Returns:
            logits: (batch_size, n_classes) class logits
        """
        batch_size, n_features = x.shape

        # Reshape features to sequence format: (B, N, 1) where N = n_features
        x = x[..., None]  # (B, n_features, 1)

        # Project to d_model
        x = nn.Dense(self.d_model, name="input_projection")(x)  # (B, n_features, d_model)

        # Apply Mamba processor
        mamba = MambaProcessorBase(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        h = mamba(x)  # (B, n_features, d_model)

        # Global pooling: mean over sequence dimension
        h_pooled = jnp.mean(h, axis=1)  # (B, d_model)

        # Classification head
        logits = nn.Dense(self.n_classes, name="classifier")(h_pooled)  # (B, n_classes)

        return logits


# ============================================================================
# Transformer Classifier
# ============================================================================


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier using self-attention.

    Uses multi-head attention to model relationships between features.
    Excellent for capturing feature interactions.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    n_classes: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, n_features) input features
            training: Whether in training mode

        Returns:
            logits: (batch_size, n_classes) class logits
        """
        batch_size, n_features = x.shape

        # Project to d_model and add feature dimension
        x = x[..., None]  # (B, n_features, 1)
        x = nn.Dense(self.d_model, name="input_projection")(x)  # (B, n_features, d_model)

        # Add positional encoding (learned)
        pos_embed = self.param(
            "pos_embed", nn.initializers.normal(stddev=0.02), (1, n_features, self.d_model)
        )
        x = x + pos_embed

        # Transformer layers
        for layer_idx in range(self.n_layers):
            # Multi-head self-attention
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, qkv_features=self.d_model, name=f"attn_{layer_idx}"
            )(x, x)
            x = x + attn_out  # Residual connection
            x = nn.LayerNorm(name=f"ln1_{layer_idx}")(x)

            # Feed-forward network
            ff = nn.Dense(self.d_ff, name=f"ff1_{layer_idx}")(x)
            ff = nn.gelu(ff)
            if training and self.dropout_rate > 0:
                ff = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff)
            ff = nn.Dense(self.d_model, name=f"ff2_{layer_idx}")(ff)
            x = x + ff  # Residual connection
            x = nn.LayerNorm(name=f"ln2_{layer_idx}")(x)

        # Global pooling: [CLS] token approach or mean pooling
        h_pooled = jnp.mean(x, axis=1)  # (B, d_model)

        # Classification head
        logits = nn.Dense(self.n_classes, name="classifier")(h_pooled)  # (B, n_classes)

        return logits


# ============================================================================
# LSTM Classifier
# ============================================================================


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequential feature processing.

    Processes features sequentially using LSTM cells.
    Good for temporal or ordered feature relationships.
    """

    hidden_size: int = 64
    n_layers: int = 2
    bidirectional: bool = False
    n_classes: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, n_features) input features
            training: Whether in training mode

        Returns:
            logits: (batch_size, n_classes) class logits
        """
        batch_size, n_features = x.shape

        # Reshape to sequence: (B, n_features, 1)
        x = x[..., None]

        # LSTM layers
        for layer_idx in range(self.n_layers):
            # Forward LSTM
            lstm_cell = nn.LSTMCell(features=self.hidden_size, name=f"lstm_fwd_{layer_idx}")
            carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,))

            outputs = []
            for t in range(n_features):
                carry, y = lstm_cell(carry, x[:, t, :])
                outputs.append(y)

            x_fwd = jnp.stack(outputs, axis=1)  # (B, n_features, hidden_size)

            # Bidirectional: also process in reverse
            if self.bidirectional:
                lstm_cell_bwd = nn.LSTMCell(features=self.hidden_size, name=f"lstm_bwd_{layer_idx}")
                carry_bwd = lstm_cell_bwd.initialize_carry(jax.random.PRNGKey(1), (batch_size,))

                outputs_bwd = []
                for t in reversed(range(n_features)):
                    carry_bwd, y = lstm_cell_bwd(carry_bwd, x[:, t, :])
                    outputs_bwd.append(y)

                x_bwd = jnp.stack(list(reversed(outputs_bwd)), axis=1)
                x = jnp.concatenate([x_fwd, x_bwd], axis=-1)  # (B, n_features, 2*hidden_size)
            else:
                x = x_fwd

            # Dropout between layers
            if training and self.dropout_rate > 0 and layer_idx < self.n_layers - 1:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Use last hidden state
        h_last = x[:, -1, :]  # (B, hidden_size or 2*hidden_size)

        # Classification head
        logits = nn.Dense(self.n_classes, name="classifier")(h_last)

        return logits


# ============================================================================
# GRU Classifier
# ============================================================================


class GRUClassifier(nn.Module):
    """
    GRU-based classifier for sequential feature processing.

    Lighter alternative to LSTM with similar performance.
    """

    hidden_size: int = 64
    n_layers: int = 2
    bidirectional: bool = False
    n_classes: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, n_features) input features
            training: Whether in training mode

        Returns:
            logits: (batch_size, n_classes) class logits
        """
        batch_size, n_features = x.shape

        # Reshape to sequence
        x = x[..., None]  # (B, n_features, 1)

        # GRU layers
        for layer_idx in range(self.n_layers):
            # Forward GRU
            gru_cell = nn.GRUCell(features=self.hidden_size, name=f"gru_fwd_{layer_idx}")
            carry = gru_cell.initialize_carry(jax.random.PRNGKey(0), (batch_size,))

            outputs = []
            for t in range(n_features):
                carry, y = gru_cell(carry, x[:, t, :])
                outputs.append(y)

            x_fwd = jnp.stack(outputs, axis=1)  # (B, n_features, hidden_size)

            # Bidirectional
            if self.bidirectional:
                gru_cell_bwd = nn.GRUCell(features=self.hidden_size, name=f"gru_bwd_{layer_idx}")
                carry_bwd = gru_cell_bwd.initialize_carry(jax.random.PRNGKey(1), (batch_size,))

                outputs_bwd = []
                for t in reversed(range(n_features)):
                    carry_bwd, y = gru_cell_bwd(carry_bwd, x[:, t, :])
                    outputs_bwd.append(y)

                x_bwd = jnp.stack(list(reversed(outputs_bwd)), axis=1)
                x = jnp.concatenate([x_fwd, x_bwd], axis=-1)
            else:
                x = x_fwd

            # Dropout
            if training and self.dropout_rate > 0 and layer_idx < self.n_layers - 1:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Use last hidden state
        h_last = x[:, -1, :]

        # Classification head
        logits = nn.Dense(self.n_classes, name="classifier")(h_last)

        return logits


# ============================================================================
# GNN Classifier
# ============================================================================


class GNNClassifier(nn.Module):
    """
    Graph Neural Network classifier.

    Uses learned graph structure between features for message passing.
    Best when features have known relational structure.

    Note: For C-Lite, we use Markov Blanket structure as the graph.
    """

    hidden_dim: int = 64
    n_layers: int = 2
    gnn_type: str = "gcn"  # 'gcn', 'gat', 'graphsage', 'gin'
    aggregation: str = "mean"  # 'mean', 'sum', 'max'
    n_classes: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, A: Optional[jnp.ndarray] = None, training: bool = False
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, n_features) input features
            A: (n_features, n_features) adjacency matrix for feature graph
               If None, uses fully connected graph
            training: Whether in training mode

        Returns:
            logits: (batch_size, n_classes) class logits
        """
        batch_size, n_features = x.shape

        # Create adjacency matrix if not provided (fully connected)
        if A is None:
            A = jnp.ones((n_features, n_features)) - jnp.eye(n_features)

        # Reshape to node features: (B, N, 1)
        node_features = x[..., None]

        # Use CausalGNN from existing implementation
        gnn = CausalGNN(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            layer_type=self.gnn_type,  # layer_type, not gnn_type
            aggregation=self.aggregation,
        )

        # Process with GNN (same A for all samples in batch)
        h = gnn(node_features, A)  # (B, N, hidden_dim)

        # Global graph pooling
        if self.aggregation == "mean":
            h_pooled = jnp.mean(h, axis=1)
        elif self.aggregation == "sum":
            h_pooled = jnp.sum(h, axis=1)
        elif self.aggregation == "max":
            h_pooled = jnp.max(h, axis=1)
        else:
            h_pooled = jnp.mean(h, axis=1)  # Default to mean

        # Dropout
        if training and self.dropout_rate > 0:
            h_pooled = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h_pooled)

        # Classification head
        logits = nn.Dense(self.n_classes, name="classifier")(h_pooled)

        return logits


# ============================================================================
# ELM Classifier
# ============================================================================


class ELMClassifier(nn.Module):
    """
    Extreme Learning Machine classifier.

    Fast training with random hidden layer and analytical output weights.
    Very fast, good baseline.
    """

    hidden_dim: int = 1000
    n_hidden_nodes: int = 1000
    n_classes: int = 2
    activation: str = "sigmoid"  # 'sigmoid', 'tanh', 'relu'

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: (batch_size, n_features) input features
            training: Whether in training mode (ELM doesn't use this)

        Returns:
            logits: (batch_size, n_classes) class logits
        """
        # Random hidden layer (fixed after initialization)
        # In ELM, hidden weights are random and fixed
        W_hidden = self.param(
            "W_hidden", nn.initializers.normal(stddev=1.0), (x.shape[-1], self.n_hidden_nodes)
        )
        b_hidden = self.param("b_hidden", nn.initializers.zeros, (self.n_hidden_nodes,))

        # Hidden layer activation
        h = jnp.dot(x, W_hidden) + b_hidden

        if self.activation == "sigmoid":
            h = nn.sigmoid(h)
        elif self.activation == "tanh":
            h = jnp.tanh(h)
        elif self.activation == "relu":
            h = nn.relu(h)

        # Output layer (trainable)
        logits = nn.Dense(self.n_classes, name="output")(h)

        return logits


# ============================================================================
# Factory Function
# ============================================================================


def create_classifier(
    processor_type: str,
    config: Dict[str, Any],
    n_features: int,
    n_classes: int,
) -> nn.Module:
    """
    Factory function to create classifier based on processor type.

    Args:
        processor_type: One of ['mlp', 'mamba', 'transformer', 'lstm', 'gru', 'gnn', 'elm']
        config: Configuration dictionary with processor-specific hyperparameters
        n_features: Number of input features
        n_classes: Number of output classes

    Returns:
        Initialized classifier module
    """
    if processor_type == "mlp":
        # MLP doesn't need n_features explicitly, it's inferred
        return MLPClassifier(
            hidden_sizes=(64, 32),  # Could be made configurable
            n_classes=n_classes,
            dropout_rate=0.1,
        )

    elif processor_type == "mamba":
        return MambaClassifier(
            d_model=config.get("d_model", 128),
            d_state=config.get("d_state", 16),
            d_conv=config.get("d_conv", 4),
            expand=config.get("expand", 2),
            n_layers=2,  # Fixed for simplicity
            n_classes=n_classes,
        )

    elif processor_type == "transformer":
        return TransformerClassifier(
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            d_ff=config.get("d_ff", 512),
            n_classes=n_classes,
            dropout_rate=0.1,
        )

    elif processor_type == "lstm":
        return LSTMClassifier(
            hidden_size=config.get("hidden_size", 64),
            n_layers=config.get("n_layers", 2),
            bidirectional=config.get("bidirectional", False),
            n_classes=n_classes,
            dropout_rate=config.get("dropout", 0.1),
        )

    elif processor_type == "gru":
        return GRUClassifier(
            hidden_size=config.get("hidden_size", 64),
            n_layers=config.get("n_layers", 2),
            bidirectional=config.get("bidirectional", False),
            n_classes=n_classes,
            dropout_rate=config.get("dropout", 0.1),
        )

    elif processor_type == "gnn":
        return GNNClassifier(
            hidden_dim=config.get("hidden_dim", 64),
            n_layers=config.get("n_layers", 2),
            gnn_type=config.get("gnn_type", "gcn"),
            aggregation=config.get("aggregation", "mean"),
            n_classes=n_classes,
            dropout_rate=0.1,
        )

    elif processor_type == "elm":
        return ELMClassifier(
            hidden_dim=config.get("hidden_dim", 1000),
            n_hidden_nodes=config.get("n_hidden_nodes", 1000),
            n_classes=n_classes,
            activation="sigmoid",
        )

    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
