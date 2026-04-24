"""
DAG-Attention Transformer: Transformer with soft adjacency-matrix attention masking.

Key differences from standard TransformerProcessor:
1. Soft DAG mask: attention scores are modulated by log(sigmoid(A * temperature))
   instead of hard binary masking. Gradients flow back through A.
2. Consistency loss: KL divergence between attention weights and sigmoid(A),
   creating a bidirectional learning signal (attention <-> DAG structure).
3. Temperature annealing: dense attention (warm-up) -> sparse DAG-masked (convergence).

This is a NEW processor variant — the original TransformerProcessor is unchanged.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class DAGAttentionLayer(nn.Module):
    """Single Transformer layer with soft DAG-masked attention."""

    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        dag_mask: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            x: (batch, n_vars, d_model)
            dag_mask: (n_vars, n_vars) soft mask from log(sigmoid(A * temp)) or None
            training: enables dropout

        Returns:
            x: (batch, n_vars, d_model) updated representations
            attn_weights: (batch, n_heads, n_vars, n_vars) attention weights (for consistency loss)
        """
        d_head = self.d_model // self.n_heads
        batch_size, n_vars, _ = x.shape

        # QKV projections
        Q = nn.Dense(self.d_model, name="q_proj")(x)  # (B, N, D)
        K = nn.Dense(self.d_model, name="k_proj")(x)
        V = nn.Dense(self.d_model, name="v_proj")(x)

        # Reshape for multi-head: (B, N, D) -> (B, H, N, D/H)
        Q = Q.reshape(batch_size, n_vars, self.n_heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, n_vars, self.n_heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, n_vars, self.n_heads, d_head).transpose(0, 2, 1, 3)

        # Attention scores: (B, H, N, N)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_head)

        # Apply soft DAG mask (THE KEY INNOVATION)
        if dag_mask is not None:
            # dag_mask shape: (N, N), broadcast to (1, 1, N, N)
            scores = scores + dag_mask[None, None, :, :]

        attn_weights = jax.nn.softmax(scores, axis=-1)  # (B, H, N, N)

        if training:
            attn_weights = nn.Dropout(rate=self.dropout_rate, deterministic=False)(attn_weights)

        # Weighted sum: (B, H, N, D/H)
        attn_out = jnp.matmul(attn_weights, V)

        # Reshape back: (B, H, N, D/H) -> (B, N, D)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, n_vars, self.d_model)

        # Output projection
        attn_out = nn.Dense(self.d_model, name="out_proj")(attn_out)

        # Residual + LayerNorm
        x = nn.LayerNorm(name="ln1")(x + attn_out)

        # Feed-forward
        ff_out = nn.Dense(self.d_ff, name="ff1")(x)
        ff_out = nn.gelu(ff_out)
        if training:
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=False)(ff_out)
        ff_out = nn.Dense(self.d_model, name="ff2")(ff_out)

        x = nn.LayerNorm(name="ln2")(x + ff_out)

        return x, attn_weights


class DAGAttentionTransformer(nn.Module):
    """
    Transformer with soft DAG-attention masking.

    A controls information flow: attention from node j to node i is
    modulated by sigmoid(A[i,j] * temperature). When temperature is high,
    all connections are ~equal (dense attention). When temperature is low,
    only edges in A receive attention (sparse, DAG-masked).

    Attributes:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout probability
        temperature: Controls mask sharpness (high=soft, low=hard)
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout_rate: float = 0.1
    temperature: float = 5.0

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        A: Optional[jnp.ndarray] = None,
        training: bool = False,
        return_attn: bool = False,
    ) -> jnp.ndarray:
        """
        Args:
            z: (batch_size, n_vars) input features
            A: (n_vars, n_vars) adjacency matrix (learned, continuous weights)
            training: enables dropout
            return_attn: if True, return (output, attention_weights) for consistency loss

        Returns:
            h: (batch_size, n_vars, d_model) processed representations
            OR (h, attn_weights_list) if return_attn=True
        """
        batch_size, n_vars = z.shape

        # Project to d_model: (B, N) -> (B, N, D)
        x = z[..., None]
        x = nn.Dense(self.d_model, name="input_projection")(x)

        # Learned positional encoding
        pos_embed = self.param(
            "pos_embed", nn.initializers.normal(stddev=0.02), (1, n_vars, self.d_model)
        )
        x = x + pos_embed

        # Build soft DAG mask from A
        dag_mask = None
        if A is not None:
            # Soft mask: log(sigmoid(A * temp) + eps)
            # When A_ij is large positive: log(sigmoid) -> 0 (no penalty, full attention)
            # When A_ij is large negative: log(sigmoid) -> -inf (blocks attention)
            # When A_ij is near zero: log(sigmoid) -> log(0.5) ≈ -0.69 (partial attention)
            dag_mask = jnp.log(jax.nn.sigmoid(A * self.temperature) + 1e-8)

            # Always allow self-attention
            dag_mask = dag_mask + jnp.eye(n_vars) * 10.0

        # Transformer layers
        all_attn_weights = []
        for i in range(self.n_layers):
            layer = DAGAttentionLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f"layer_{i}",
            )
            x, attn_w = layer(x, dag_mask=dag_mask, training=training)
            all_attn_weights.append(attn_w)

        if return_attn:
            return x, all_attn_weights
        return x

    @staticmethod
    def consistency_loss(
        attn_weights_list: list,
        A: jnp.ndarray,
        temperature: float = 5.0,
    ) -> jnp.ndarray:
        """
        KL divergence between learned attention patterns and sigmoid(A).

        This creates a bidirectional learning signal:
        - A -> attention: "attend along causal edges"
        - attention -> A: "this attention pattern suggests edge i->j should exist"

        Args:
            attn_weights_list: list of (batch, heads, n_vars, n_vars) from each layer
            A: (n_vars, n_vars) adjacency matrix
            temperature: same temperature as the mask

        Returns:
            loss: scalar consistency loss
        """
        n_vars = A.shape[0]

        # Target distribution: normalized sigmoid(A) + self-connections
        target = jax.nn.sigmoid(A * temperature) + jnp.eye(n_vars)
        target = target / (target.sum(axis=-1, keepdims=True) + 1e-8)

        total_loss = 0.0
        for attn_w in attn_weights_list:
            # Average over batch and heads: (n_vars, n_vars)
            avg_attn = attn_w.mean(axis=(0, 1))

            # KL(avg_attn || target) — only where target > 0
            kl = jnp.sum(avg_attn * jnp.log((avg_attn + 1e-8) / (target + 1e-8)))
            total_loss += kl

        return total_loss / len(attn_weights_list)
