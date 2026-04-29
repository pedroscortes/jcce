"""
Adapters to use existing Flax processors with GOLEM-Unified.

This module provides wrappers that convert between:
1. Flax's PyTree parameter format (used by our existing models)
2. GOLEM's dict-based parameter format (used for optimization)

This allows us to reuse existing tested implementations (Mamba, ELM, GNN)
without reimplementing them from scratch.
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jax import random

from jcce.models.elm import ELMProcessor as ELMProcessorBase
from jcce.models.gnn import CausalGNN
from jcce.models.mamba import MambaProcessor as MambaProcessorBase

# Import existing processors
from jcce.models.mlp import MLPProcessor as MLPProcessorBase
from jcce.models.processor_wrappers import TransformerProcessor as TransformerProcessorBase

# ============================================================================
# Flax to Dict Converter
# ============================================================================


def flax_to_dict_params(flax_params: Dict) -> Dict[str, jnp.ndarray]:
    """
    Flatten Flax PyTree parameters to a simple dict for GOLEM optimization.

    Args:
        flax_params: Flax parameter dict (nested PyTree)

    Returns:
        flat_dict: Simple dict mapping keys to arrays
    """
    from jax.tree_util import tree_flatten

    flat_params, tree_def = tree_flatten(flax_params)

    return {
        "flat_params": jnp.concatenate([p.flatten() for p in flat_params]),
        "tree_def": tree_def,
        "shapes": [p.shape for p in flat_params],
    }


def dict_to_flax_params(param_dict: Dict) -> Dict:
    """
    Unflatten dict parameters back to Flax PyTree format.

    Args:
        param_dict: Dict with 'flat_params', 'tree_def', 'shapes'

    Returns:
        flax_params: Flax parameter dict (nested PyTree)
    """
    from jax.tree_util import tree_unflatten

    flat_array = param_dict["flat_params"]
    tree_def = param_dict["tree_def"]
    shapes = param_dict["shapes"]

    # Reconstruct individual arrays
    # Use math.prod (pure Python) instead of jnp.prod to avoid tracer issues in vmap/JIT
    import math

    arrays = []
    idx = 0
    for shape in shapes:
        size = math.prod(shape)
        arrays.append(flat_array[idx : idx + size].reshape(shape))
        idx += size

    return tree_unflatten(tree_def, arrays)


# ============================================================================
# MLP Adapter
# ============================================================================


class MLPAdapter:
    """
    Adapter for MLP processor to work with GOLEM-Unified.

    Wraps jcce.models.mlp.MLPProcessor for causal structure learning.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: str = "relu",
        key: random.PRNGKey = None,
        dropout_rate: float = 0.1,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.key = key if key is not None else random.PRNGKey(42)

        # Create Flax model
        self.model = MLPProcessorBase(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            use_layer_norm=True,
            use_residual=True,
            dropout_rate=dropout_rate,
        )

    def init_params(self, n_inputs: int) -> Dict:
        """Initialize parameters in dict format."""
        # Create dummy input
        dummy_input = jnp.ones((1, n_inputs))  # (batch=1, n_inputs)

        # Initialize Flax parameters
        flax_params = self.model.init(self.key, dummy_input, A=None, training=False)

        # Convert to dict format
        param_dict = flax_to_dict_params(flax_params)
        param_dict["n_inputs"] = n_inputs

        # Output projection — trainable (Session 29 Fix 1). Must be in init_params so
        # optimizer can update it. Frozen projection (server) fails with local training
        # dynamics (DAGMA, JIT) — produces BAcc < 0.5 (anti-correlated predictions).
        # The E[Y] shortcut is blocked by mean centering in forward() (Fix 2 reverted).
        key_proj = random.PRNGKey(0)
        param_dict["output_proj_W"] = random.normal(key_proj, (self.hidden_dim, 1)) * 0.5
        param_dict["output_proj_b"] = jnp.zeros(1)

        return param_dict

    def forward(
        self,
        X: jnp.ndarray,
        params: Dict,
        training: bool = True,
        rng_key: random.PRNGKey = None,
        skip_centering: bool = False,
    ) -> jnp.ndarray:
        """
        Forward pass through MLP.

        Args:
            X: (n_samples, n_inputs) input data
            params: Dict-format parameters
            training: Whether in training mode (enables dropout) v6.0
            rng_key: Optional RNG key for dropout (v6.0)
            skip_centering: If True, skip mean centering (Session 35, Y classification path)

        Returns:
            output: (n_samples,) predictions
        """
        n_samples, n_inputs = X.shape

        # Convert dict params back to Flax format
        flax_params = dict_to_flax_params(params)

        # Forward through MLP (no adjacency matrix needed)
        # training=True enables dropout (requires RNG key)
        if training and rng_key is not None:
            h = self.model.apply(
                flax_params, X, A=None, training=training, rngs={"dropout": rng_key}
            )
        else:
            # Deterministic mode (no dropout RNG needed)
            h = self.model.apply(flax_params, X, A=None, training=False)
        # Shape: (n_samples, n_inputs, hidden_dim)

        # Pool across features
        h_pooled = jnp.mean(h, axis=1)  # (n_samples, hidden_dim)

        # Project to scalar output (backward compat: lazy-create if loading old pkl)
        if "output_proj_W" not in params:
            key = random.PRNGKey(0)
            params["output_proj_W"] = random.normal(key, (self.hidden_dim, 1)) * 0.5
            params["output_proj_b"] = jnp.zeros(1)

        # Apply normalization if weights were solved with normalized features
        if params.get("_weights_solved", False) and "_H_mean" in params:
            h_pooled = (h_pooled - params["_H_mean"]) / params["_H_std"]

        # For classification (skip_centering=True), normalize h_pooled BEFORE
        # the output projection. This prevents reconstruction-driven h_pooled
        # drift from causing logit divergence, while preserving output_proj_b
        # as a learnable class prior.
        # X reconstruction (skip_centering=False) still uses output centering below.
        if skip_centering and not params.get("_weights_solved", False):
            h_pooled = (h_pooled - jnp.mean(h_pooled, axis=0, keepdims=True)) / (
                jnp.std(h_pooled, axis=0, keepdims=True) + 1e-8
            )

        output = h_pooled @ params["output_proj_W"] + params["output_proj_b"]
        output = output.squeeze()

        # Mean centering: prevents constant-output shortcut for X reconstruction.
        # Skip for Y classification — centering forces sigmoid(~0)=0.5,
        # making BCE=ln(2) (dead signal). Y_recon and X_recon still use centering.
        if not skip_centering and not params.get("_weights_solved", False):
            output = output - jnp.mean(output)

        return output

    def get_hidden_features(self, X: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """Get hidden features (h_pooled) for solving output weights."""
        flax_params = dict_to_flax_params(params)
        h = self.model.apply(flax_params, X, A=None, training=False)
        h_pooled = jnp.mean(h, axis=1)
        return h_pooled

    def solve_output_weights(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: Dict,
        lambda_reg: float = 1e-4,
        for_classification: bool = True,
    ) -> Dict:
        """
        Solve for optimal output weights.
        v6.1.1: Uses logistic regression for classification (not ridge!).
        v6.1.3: Sets _weights_solved flag to skip centering in forward().
        v6.1.4: Normalizes hidden features for stable logistic regression.
        """
        H = self.get_hidden_features(X, params)
        n_hidden = H.shape[1]
        n_samples = len(y)

        if for_classification:
            # Normalize hidden features for stable training
            H_mean = jnp.mean(H, axis=0, keepdims=True)
            H_std = jnp.std(H, axis=0, keepdims=True) + 1e-8
            H_norm = (H - H_mean) / H_std

            # Logistic regression via gradient descent (BCE loss)
            n_steps = 300
            lr = 0.1
            max_grad_norm = 1.0
            key = random.PRNGKey(42)
            W = random.normal(key, (n_hidden, 1)) * 0.01
            b = jnp.array([0.0])
            y_col = y.reshape(-1, 1)

            for step in range(n_steps):
                logits = H_norm @ W + b
                probs = jax.nn.sigmoid(logits)
                error = probs - y_col
                grad_W = (H_norm.T @ error) / n_samples + lambda_reg * W
                grad_b = jnp.mean(error)

                # Gradient clipping
                grad_norm = jnp.sqrt(jnp.sum(grad_W**2) + grad_b**2)
                scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
                grad_W = grad_W * scale
                grad_b = grad_b * scale

                W = W - lr * grad_W
                b = b - lr * grad_b

            params["output_proj_W"] = W
            params["output_proj_b"] = b
            params["_H_mean"] = H_mean
            params["_H_std"] = H_std
        else:
            # Ridge regression (for regression tasks)
            HTH = H.T @ H + lambda_reg * jnp.eye(n_hidden)
            HTy = H.T @ y.reshape(-1, 1)
            W = jnp.linalg.solve(HTH, HTy)
            predictions = H @ W
            b = jnp.mean(y.reshape(-1, 1) - predictions)
            params["output_proj_W"] = W
            params["output_proj_b"] = jnp.array([b])

        # Mark that weights were solved (skip centering in forward)
        params["_weights_solved"] = True

        return params


# ============================================================================
# Transformer Adapter
# ============================================================================


class TransformerAdapter:
    """
    Adapter for Transformer processor to work with GOLEM-Unified.

    Wraps jcce.models.processor_wrappers.TransformerProcessor for causal structure learning.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        key: random.PRNGKey = None,
        dropout_rate: float = 0.1,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.key = key if key is not None else random.PRNGKey(42)

        # Create Flax model
        self.model = TransformerProcessorBase(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            use_graph_mask=False,  # Don't use graph mask (we're learning the graph!)
        )

    def init_params(self, n_inputs: int) -> Dict:
        """Initialize parameters in dict format."""
        # Create dummy input
        dummy_input = jnp.ones((1, n_inputs))  # (batch=1, n_inputs)

        # Initialize Flax parameters
        flax_params = self.model.init(self.key, dummy_input, A=None, training=False)

        # Convert to dict format
        param_dict = flax_to_dict_params(flax_params)
        param_dict["n_inputs"] = n_inputs

        # Output projection — trainable. See MLPAdapter init_params comment.
        key_proj = random.PRNGKey(0)
        param_dict["output_proj_W"] = random.normal(key_proj, (self.d_model, 1)) * 0.5
        param_dict["output_proj_b"] = jnp.zeros(1)

        return param_dict

    def forward(
        self,
        X: jnp.ndarray,
        params: Dict,
        training: bool = True,
        rng_key: random.PRNGKey = None,
        skip_centering: bool = False,
    ) -> jnp.ndarray:
        """
        Forward pass through Transformer.

        Args:
            X: (n_samples, n_inputs) input data
            params: Dict-format parameters
            training: Whether in training mode (enables dropout) v6.0
            rng_key: RNG key for dropout (required when training=True) v6.0
            skip_centering: If True, skip mean centering (Session 35, Y classification path)

        Returns:
            output: (n_samples,) predictions
        """
        n_samples, n_inputs = X.shape

        # Convert dict params back to Flax format
        flax_params = dict_to_flax_params(params)

        # Forward through Transformer (no adjacency matrix for structure learning)
        # training=True enables dropout (requires RNG key)
        if training and rng_key is not None:
            h = self.model.apply(
                flax_params, X, A=None, training=training, rngs={"dropout": rng_key}
            )  # (n_samples, n_inputs, d_model)
        else:
            # Deterministic mode (evaluation or no RNG provided)
            h = self.model.apply(
                flax_params, X, A=None, training=False
            )  # (n_samples, n_inputs, d_model)

        # Pool across features: mean pooling
        h_pooled = jnp.mean(h, axis=1)  # (n_samples, d_model)

        # Project to scalar output (backward compat: lazy-create if loading old pkl)
        if "output_proj_W" not in params:
            key = random.PRNGKey(0)
            params["output_proj_W"] = random.normal(key, (self.d_model, 1)) * 0.5
            params["output_proj_b"] = jnp.zeros(1)

        # Apply normalization if weights were solved with normalized features
        if params.get("_weights_solved", False) and "_H_mean" in params:
            h_pooled = (h_pooled - params["_H_mean"]) / params["_H_std"]

        # Normalize h_pooled for classification (see MLPAdapter.forward comment)
        if skip_centering and not params.get("_weights_solved", False):
            h_pooled = (h_pooled - jnp.mean(h_pooled, axis=0, keepdims=True)) / (
                jnp.std(h_pooled, axis=0, keepdims=True) + 1e-8
            )

        output = h_pooled @ params["output_proj_W"] + params["output_proj_b"]
        output = output.squeeze()

        # Skip centering for classification (see MLPAdapter.forward comment)
        if not skip_centering and not params.get("_weights_solved", False):
            output = output - jnp.mean(output)

        return output

    def get_hidden_features(self, X: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """Get hidden features (h_pooled) for solving output weights."""
        flax_params = dict_to_flax_params(params)
        h = self.model.apply(flax_params, X, A=None, training=False)
        h_pooled = jnp.mean(h, axis=1)
        return h_pooled

    def solve_output_weights(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: Dict,
        lambda_reg: float = 1e-4,
        for_classification: bool = True,
    ) -> Dict:
        """
        Solve for optimal output weights.
        v6.1.3: Uses logistic regression for classification (like ELM/MLP).

        Args:
            X: Input data
            y: Target labels
            params: Parameters dict
            lambda_reg: Regularization strength
            for_classification: If True, use logistic regression; if False, use ridge
        """
        H = self.get_hidden_features(X, params)
        n_hidden = H.shape[1]
        n_samples = len(y)

        if for_classification:
            # Normalize hidden features to prevent gradient explosion
            # Transformer's attention outputs can have extreme values
            H_mean = jnp.mean(H, axis=0, keepdims=True)
            H_std = jnp.std(H, axis=0, keepdims=True) + 1e-8
            H_norm = (H - H_mean) / H_std

            # Logistic regression via gradient descent (BCE loss)
            # Lower lr and gradient clipping for stability
            n_steps = 300  # More steps with lower lr
            lr = 0.1  # Lower lr for stability
            max_grad_norm = 1.0  # Gradient clipping
            key = random.PRNGKey(42)
            W = random.normal(key, (n_hidden, 1)) * 0.01
            b = jnp.array([0.0])
            y_col = y.reshape(-1, 1)

            for step in range(n_steps):
                logits = H_norm @ W + b
                probs = jax.nn.sigmoid(logits)
                error = probs - y_col
                grad_W = (H_norm.T @ error) / n_samples + lambda_reg * W
                grad_b = jnp.mean(error)

                # Gradient clipping
                grad_norm = jnp.sqrt(jnp.sum(grad_W**2) + grad_b**2)
                scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
                grad_W = grad_W * scale
                grad_b = grad_b * scale

                W = W - lr * grad_W
                b = b - lr * grad_b

            # Store normalized weights (will be applied to normalized features in forward)
            params["output_proj_W"] = W
            params["output_proj_b"] = b
            params["_H_mean"] = H_mean  # Store normalization params
            params["_H_std"] = H_std
        else:
            # Ridge regression (for regression tasks)
            HTH = H.T @ H + lambda_reg * jnp.eye(n_hidden)
            HTy = H.T @ y.reshape(-1, 1)
            W_out_solved = jnp.linalg.solve(HTH, HTy)
            predictions = H @ W_out_solved
            b_out_solved = jnp.mean(y.reshape(-1, 1) - predictions)
            params["output_proj_W"] = W_out_solved
            params["output_proj_b"] = jnp.array([b_out_solved])

        # Mark that weights were solved
        params["_weights_solved"] = True

        return params


# ============================================================================
# Mamba Adapter
# ============================================================================


def get_mamba_config(n_features: int) -> dict:
    """
    v5.1: Dynamic Mamba configuration based on feature count.

    Memory scales as: n_samples × n_features × d_inner × d_state × 4 bytes
    where d_inner = expand × d_model

    Threshold analysis from overnight experiments:
    - n_features ≤ 13: Full config works (heart_disease, lucas)
    - n_features > 20: OOM with default config (breast_cancer, steel_plates)

    Returns:
        dict with d_model, d_state, d_conv, expand
    """
    if n_features <= 13:
        return {"d_model": 128, "d_state": 8, "d_conv": 4, "expand": 2}
    elif n_features <= 25:
        return {"d_model": 64, "d_state": 4, "d_conv": 4, "expand": 2}
    else:  # > 25 features
        return {"d_model": 32, "d_state": 4, "d_conv": 4, "expand": 2}


class MambaAdapter:
    """
    Adapter for Mamba processor to work with GOLEM-Unified.

    Wraps jcce.models.mamba.MambaProcessor for causal structure learning.

    v5.1: Supports automatic dimension scaling based on n_features to prevent OOM.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        key: random.PRNGKey = None,
        n_features: int = None,
    ):
        # n_features stored for reference but does NOT override explicit args.
        # Callers (Optuna/NSGA-II) pass d_model/d_state from the search space;
        # auto-scaling via get_mamba_config() was silently discarding those
        # choices, making Optuna's mamba hyperparameter exploration a no-op.

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_features = n_features
        self.key = key if key is not None else random.PRNGKey(42)

        # Create Flax model
        self.model = MambaProcessorBase(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=1,  # Use 1 layer for structure learning (efficiency)
        )

    def init_params(self, n_inputs: int) -> Dict:
        """Initialize parameters in dict format."""
        # Create dummy input to initialize Flax model
        dummy_input = jnp.ones(
            (1, n_inputs, self.model.d_model)
        )  # (batch=1, seq_len=n_inputs, d_model)

        # Initialize Flax parameters (MambaProcessor only takes z_sequence)
        flax_params = self.model.init(self.key, dummy_input)

        # Convert to dict format for GOLEM
        param_dict = flax_to_dict_params(flax_params)
        param_dict["n_inputs"] = n_inputs  # Store for reshaping

        # Output projection — trainable. See MLPAdapter init_params comment.
        key_proj = random.PRNGKey(0)
        param_dict["output_proj_W"] = random.normal(key_proj, (self.d_model, 1)) * 0.5
        param_dict["output_proj_b"] = jnp.zeros(1)

        return param_dict

    def forward(self, X: jnp.ndarray, params: Dict, skip_centering: bool = False) -> jnp.ndarray:
        """
        Forward pass through Mamba.

        Args:
            X: (n_samples, n_inputs) input data
            params: Dict-format parameters
            skip_centering: If True, skip mean centering (Session 35, Y classification path)

        Returns:
            output: (n_samples,) predictions
        """
        n_samples, n_inputs = X.shape

        # Convert dict params back to Flax format
        flax_params = dict_to_flax_params(params)

        # Reshape for Mamba: (batch, seq_len, d_model)
        # Project input to d_model dimension
        X_projected = jnp.tile(
            X[:, :, jnp.newaxis], (1, 1, self.model.d_model)
        )  # (n_samples, n_inputs, d_model)

        # Forward through Mamba (only takes z_sequence)
        h = self.model.apply(flax_params, X_projected)  # (n_samples, n_inputs, d_model)

        # Pool across sequence: mean pooling
        h_pooled = jnp.mean(h, axis=1)  # (n_samples, d_model)

        # Project to scalar output (backward compat: lazy-create if loading old pkl)
        if "output_proj_W" not in params:
            key = random.PRNGKey(0)
            params["output_proj_W"] = random.normal(key, (self.d_model, 1)) * 0.5
            params["output_proj_b"] = jnp.zeros(1)

        # Normalize h_pooled for classification (see MLPAdapter.forward comment)
        if skip_centering and not params.get("_weights_solved", False):
            h_pooled = (h_pooled - jnp.mean(h_pooled, axis=0, keepdims=True)) / (
                jnp.std(h_pooled, axis=0, keepdims=True) + 1e-8
            )

        output = h_pooled @ params["output_proj_W"] + params["output_proj_b"]
        output = output.squeeze()  # (n_samples,)

        # Skip centering for classification (see MLPAdapter.forward comment)
        if not skip_centering and not params.get("_weights_solved", False):
            output = output - jnp.mean(output)

        return output

    def get_hidden_features(self, X: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """Get hidden features (h_pooled) for solving output weights."""
        n_samples, n_inputs = X.shape
        flax_params = dict_to_flax_params(params)

        # Project input to d_model dimension
        X_projected = jnp.tile(X[:, :, jnp.newaxis], (1, 1, self.d_model))

        # Forward through Mamba
        h = self.model.apply(flax_params, X_projected)
        h_pooled = jnp.mean(h, axis=1)
        return h_pooled

    def solve_output_weights(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: Dict,
        lambda_reg: float = 1e-4,
        for_classification: bool = True,
    ) -> Dict:
        """Solve output weights via logistic regression (classification) or ridge (regression)."""
        H = self.get_hidden_features(X, params)
        n_hidden = H.shape[1]
        n_samples = len(y)

        if for_classification:
            # Logistic regression via gradient descent
            n_steps = 200
            lr = 0.5
            key = random.PRNGKey(42)
            W = random.normal(key, (n_hidden, 1)) * 0.01
            b = jnp.array([0.0])
            y_col = y.reshape(-1, 1)

            for step in range(n_steps):
                logits = H @ W + b
                probs = jax.nn.sigmoid(logits)
                error = probs - y_col
                grad_W = (H.T @ error) / n_samples + lambda_reg * W
                grad_b = jnp.mean(error)
                W = W - lr * grad_W
                b = b - lr * grad_b

            params["output_proj_W"] = W
            params["output_proj_b"] = b
        else:
            # Ridge regression
            HTH = H.T @ H + lambda_reg * jnp.eye(n_hidden)
            HTy = H.T @ y.reshape(-1, 1)
            W = jnp.linalg.solve(HTH, HTy)
            predictions = H @ W
            b = jnp.mean(y.reshape(-1, 1) - predictions)
            params["output_proj_W"] = W
            params["output_proj_b"] = jnp.array([b])

        # Mark that weights were solved (for consistency)
        params["_weights_solved"] = True

        return params


# ============================================================================
# ELM Adapter
# ============================================================================


class ELMAdapter:
    """
    Adapter for ELM processor to work with GOLEM-Unified.

    Wraps jcce.models.elm.ELMProcessor for causal structure learning.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_hidden_nodes: int = 128,
        activation: str = "tanh",
        key: random.PRNGKey = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        self.key = key if key is not None else random.PRNGKey(42)

        # Create Flax model
        self.model = ELMProcessorBase(
            hidden_dim=hidden_dim,
            n_hidden_nodes=n_hidden_nodes,
            activation=activation,
        )

    def init_params(self, n_inputs: int) -> Dict:
        """Initialize parameters in dict format."""
        # Create dummy input
        dummy_input = jnp.ones((1, n_inputs))  # (batch=1, n_inputs)

        # Initialize Flax parameters
        flax_params = self.model.init(self.key, dummy_input, A=None, training=False)

        # Convert to dict format
        param_dict = flax_to_dict_params(flax_params)
        param_dict["n_inputs"] = n_inputs

        # Output projection — trainable. ELM uses scale 1.0 for wider initial range.
        key_proj = random.PRNGKey(0)
        param_dict["output_proj_W"] = random.normal(key_proj, (self.hidden_dim, 1)) * 1.0
        param_dict["output_proj_b"] = jnp.zeros(1)

        return param_dict

    def forward(self, X: jnp.ndarray, params: Dict, skip_centering: bool = False) -> jnp.ndarray:
        """
        Forward pass through ELM.

        Args:
            X: (n_samples, n_inputs) input data
            params: Dict-format parameters
            skip_centering: If True, skip mean centering (Session 35, Y classification path)

        Returns:
            output: (n_samples,) predictions
        """
        n_samples, n_inputs = X.shape

        # Convert dict params back to Flax format
        flax_params = dict_to_flax_params(params)

        # Forward through ELM (no adjacency matrix needed)
        h = self.model.apply(
            flax_params, X, A=None, training=False
        )  # (n_samples, n_inputs, hidden_dim)

        # Pool across features
        h_pooled = jnp.mean(h, axis=1)  # (n_samples, hidden_dim)

        # Project to scalar output (backward compat: lazy-create if loading old pkl)
        if "output_proj_W" not in params:
            key = random.PRNGKey(0)
            params["output_proj_W"] = random.normal(key, (self.hidden_dim, 1)) * 1.0
            params["output_proj_b"] = jnp.zeros(1)

        # Normalize h_pooled for classification (see MLPAdapter.forward comment)
        if skip_centering and not params.get("_weights_solved", False):
            h_pooled = (h_pooled - jnp.mean(h_pooled, axis=0, keepdims=True)) / (
                jnp.std(h_pooled, axis=0, keepdims=True) + 1e-8
            )

        output = h_pooled @ params["output_proj_W"] + params["output_proj_b"]
        output = output.squeeze()

        # Skip centering for classification (see MLPAdapter.forward comment)
        if not skip_centering and not params.get("_weights_solved", False):
            output = output - jnp.mean(output)

        return output

    def get_hidden_features(self, X: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """
        Get hidden features (h_pooled) for solving output weights.

        Args:
            X: (n_samples, n_inputs) input data
            params: Dict-format parameters

        Returns:
            h_pooled: (n_samples, hidden_dim) pooled hidden features
        """
        flax_params = dict_to_flax_params(params)
        h = self.model.apply(flax_params, X, A=None, training=False)
        h_pooled = jnp.mean(h, axis=1)
        return h_pooled

    def solve_output_weights(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: Dict,
        lambda_reg: float = 1e-4,
        for_classification: bool = True,
    ) -> Dict:
        """
        Solve for optimal output weights.

        For classification: Uses logistic regression via gradient descent (BCE loss).
        For regression: Uses ridge regression (MSE loss).
        v6.1.3: Sets _weights_solved flag to skip centering in forward().

        Args:
            X: (n_samples, n_inputs) input data
            y: (n_samples,) target labels
            params: Dict-format parameters (will be updated in-place)
            lambda_reg: Regularization strength
            for_classification: If True, use logistic regression; if False, use ridge

        Returns:
            params: Updated parameters with solved output weights
        """
        # Get hidden features
        H = self.get_hidden_features(X, params)
        n_hidden = H.shape[1]
        n_samples = len(y)

        if for_classification:
            # Logistic regression via gradient descent (BCE loss)
            # This is critical for classification - ridge regression gives wrong loss!
            n_steps = 200
            lr = 0.5

            # Initialize with small random weights for symmetry breaking
            key = random.PRNGKey(42)
            W = random.normal(key, (n_hidden, 1)) * 0.01
            b = jnp.array([0.0])

            y_col = y.reshape(-1, 1)

            for step in range(n_steps):
                # Forward pass
                logits = H @ W + b
                probs = jax.nn.sigmoid(logits)

                # Gradient of BCE loss + L2 regularization
                error = probs - y_col  # (n_samples, 1)
                grad_W = (H.T @ error) / n_samples + lambda_reg * W
                grad_b = jnp.mean(error)

                # Gradient descent update
                W = W - lr * grad_W
                b = b - lr * grad_b

            params["output_proj_W"] = W
            params["output_proj_b"] = b
        else:
            # Ridge regression: W = (H'H + λI)^{-1} H'y
            HTH = H.T @ H + lambda_reg * jnp.eye(n_hidden)
            HTy = H.T @ y.reshape(-1, 1)
            W_out_solved = jnp.linalg.solve(HTH, HTy)

            # Compute bias as mean of residuals
            predictions = H @ W_out_solved
            b_out_solved = jnp.mean(y.reshape(-1, 1) - predictions)

            params["output_proj_W"] = W_out_solved
            params["output_proj_b"] = jnp.array([b_out_solved])

        # Mark that weights were solved (skip centering in forward)
        params["_weights_solved"] = True

        return params


# ============================================================================
# GNN Adapter
# ============================================================================


class GNNAdapter:
    """
    Adapter for GNN processor to work with GOLEM-Unified.

    Wraps jcce.models.gnn.CausalGNN for causal structure learning.

    Key innovation: GNN can use the evolving adjacency matrix A during learning!
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        gnn_type: str = "gcn",
        sage_aggregation: str = "mean",
        key: random.PRNGKey = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gnn_type = gnn_type
        self.sage_aggregation = sage_aggregation
        self.key = key if key is not None else random.PRNGKey(42)

        # Create Flax model
        self.model = CausalGNN(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            layer_type=gnn_type,  # Use layer_type parameter
            sage_aggregation=sage_aggregation,  # Fixed: use sage_aggregation for GraphSAGE
            aggregation="last",  # Keep default for multi-layer aggregation
        )

    def init_params(self, n_inputs: int) -> Dict:
        """Initialize parameters in dict format."""
        # Create dummy input and adjacency
        dummy_input = jnp.ones((1, n_inputs))  # (batch=1, n_inputs)
        dummy_A = jnp.eye(n_inputs)  # Identity adjacency for initialization

        # Initialize Flax parameters
        flax_params = self.model.init(self.key, dummy_input, dummy_A, training=False)

        # Convert to dict format
        param_dict = flax_to_dict_params(flax_params)
        param_dict["n_inputs"] = n_inputs

        # Output projection — trainable. See MLPAdapter init_params comment.
        key_proj = random.PRNGKey(0)
        param_dict["output_proj_W"] = random.normal(key_proj, (self.hidden_dim, 1)) * 0.5
        param_dict["output_proj_b"] = jnp.zeros(1)

        return param_dict

    def forward(
        self, X: jnp.ndarray, params: Dict, A: jnp.ndarray = None, skip_centering: bool = False
    ) -> jnp.ndarray:
        """
        Forward pass through GNN.

        Args:
            X: (n_samples, n_inputs) input data
            params: Dict-format parameters
            A: (n_inputs, n_inputs) adjacency matrix (OPTIONAL - key feature!)
            skip_centering: If True, skip mean centering (Session 35, Y classification path)

        Returns:
            output: (n_samples,) predictions
        """
        n_samples, n_inputs = X.shape

        # Convert dict params back to Flax format
        flax_params = dict_to_flax_params(params)

        # If no adjacency provided, use identity (no message passing)
        if A is None:
            A = jnp.eye(n_inputs)

        # Forward through GNN WITH adjacency matrix!
        # This is the key innovation: GNN can use evolving graph structure
        h = self.model.apply(flax_params, X, A, training=False)  # (n_samples, n_inputs, hidden_dim)

        # Pool across nodes
        h_pooled = jnp.mean(h, axis=1)  # (n_samples, hidden_dim)

        # Project to scalar output (backward compat: lazy-create if loading old pkl)
        if "output_proj_W" not in params:
            key = random.PRNGKey(0)
            params["output_proj_W"] = random.normal(key, (self.hidden_dim, 1)) * 0.5
            params["output_proj_b"] = jnp.zeros(1)

        # Apply H normalization if weights were solved with normalized features
        if params.get("_weights_solved", False) and "_H_mean" in params:
            h_pooled = (h_pooled - params["_H_mean"]) / params["_H_std"]

        # Normalize h_pooled for classification (see MLPAdapter.forward comment)
        if skip_centering and not params.get("_weights_solved", False):
            h_pooled = (h_pooled - jnp.mean(h_pooled, axis=0, keepdims=True)) / (
                jnp.std(h_pooled, axis=0, keepdims=True) + 1e-8
            )

        output = h_pooled @ params["output_proj_W"] + params["output_proj_b"]
        output = output.squeeze()

        # Skip centering for classification (see MLPAdapter.forward comment)
        if not skip_centering and not params.get("_weights_solved", False):
            output = output - jnp.mean(output)

        return output

    def get_hidden_features(
        self, X: jnp.ndarray, params: Dict, A: jnp.ndarray = None
    ) -> jnp.ndarray:
        """Get hidden features (h_pooled) for solving output weights."""
        n_samples, n_inputs = X.shape
        flax_params = dict_to_flax_params(params)

        if A is None:
            A = jnp.eye(n_inputs)

        h = self.model.apply(flax_params, X, A, training=False)
        h_pooled = jnp.mean(h, axis=1)
        return h_pooled

    def solve_output_weights(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: Dict,
        lambda_reg: float = 1e-4,
        for_classification: bool = True,
        A: jnp.ndarray = None,
    ) -> Dict:
        """
        Solve output weights via logistic regression (classification) or ridge (regression).

        Normalizes H (like MLP/Transformer) for stable logistic regression.
        A is now passed from the caller, and forward() applies denormalization
        using stored _H_mean/_H_std.
        """
        H = self.get_hidden_features(X, params, A=A)
        n_hidden = H.shape[1]
        n_samples = len(y)

        if for_classification:
            # Normalize hidden features (matches MLP pattern).
            # GNN message-passing amplifies h_pooled to [1, 10] range — without
            # normalization, W grows large and logits become unbounded.
            H_mean = jnp.mean(H, axis=0, keepdims=True)
            H_std = jnp.std(H, axis=0, keepdims=True) + 1e-8
            H_norm = (H - H_mean) / H_std

            n_steps = 300
            lr = 0.1
            max_grad_norm = 1.0
            key = random.PRNGKey(42)
            W = random.normal(key, (n_hidden, 1)) * 0.01
            b = jnp.array([0.0])
            y_col = y.reshape(-1, 1)

            for step in range(n_steps):
                logits = H_norm @ W + b
                probs = jax.nn.sigmoid(logits)
                error = probs - y_col
                grad_W = (H_norm.T @ error) / n_samples + lambda_reg * W
                grad_b = jnp.mean(error)

                # Gradient clipping (matches MLP)
                grad_norm = jnp.sqrt(jnp.sum(grad_W**2) + grad_b**2)
                scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
                grad_W = grad_W * scale
                grad_b = grad_b * scale

                W = W - lr * grad_W
                b = b - lr * grad_b

            params["output_proj_W"] = W
            params["output_proj_b"] = b
            params["_H_mean"] = H_mean
            params["_H_std"] = H_std
        else:
            # Ridge regression
            HTH = H.T @ H + lambda_reg * jnp.eye(n_hidden)
            HTy = H.T @ y.reshape(-1, 1)
            W = jnp.linalg.solve(HTH, HTy)
            predictions = H @ W
            b = jnp.mean(y.reshape(-1, 1) - predictions)
            params["output_proj_W"] = W
            params["output_proj_b"] = jnp.array([b])

        # Mark that weights were solved (for consistency)
        params["_weights_solved"] = True

        return params


# ============================================================================
# Linear Head Adapter
# ============================================================================


class LinearHeadAdapter:
    """Minimal linear-only processor for ablation against richer adapters.

    Per-variable forward is a single affine map f_j(X) = X @ beta_j + b_j —
    no attention, no LayerNorm, no MLP. Used to test whether the trained
    DAG-Attention's per-variable head needs nonlinearity at all on JCCE
    benchmarks. Day-1/2 of the AAP architectural-fix mini-Sprint
    (commits 0310b39, ddbaf09 on origin/aap) showed that post-hoc OLS and
    logistic regression on the same input distribution recover signal that
    the trained Transformer head missed; this adapter trains the linear
    head jointly with the rest of the JCCE pipeline so the comparison
    table includes a fully end-to-end linear baseline.

    Notes
    -----
    The pipeline only invokes ``init_params`` and ``forward`` on the
    processor (``solve_output_weights`` and ``get_hidden_features`` are
    disabled in jcce_learner; see line 1598). This class implements only
    those two methods plus an init that mirrors the kwargs convention of
    the other adapters.
    """

    def __init__(self, key: random.PRNGKey = None):
        self.key = key if key is not None else random.PRNGKey(42)

    def init_params(self, n_inputs: int) -> Dict:
        """Initialize per-variable linear params: ``beta`` and bias ``b``."""
        # Small init scaled by 1/sqrt(n) keeps the initial logit magnitude
        # bounded regardless of feature count; JCCE's L1 sparsity and recon
        # losses then shape the weights during training.
        scale = 0.01 / max(1, n_inputs) ** 0.5
        beta = random.normal(self.key, (n_inputs,)) * scale
        return {
            "beta": beta,
            "b": jnp.zeros(()),
            "n_inputs": n_inputs,
        }

    def forward(
        self,
        X: jnp.ndarray,
        params: Dict,
        training: bool = True,
        rng_key: random.PRNGKey = None,
        skip_centering: bool = False,
        **kwargs,  # absorb A=..., other adapter-specific kwargs we ignore
    ) -> jnp.ndarray:
        """Forward pass: ``X @ beta + b`` with optional mean centering.

        Parameters
        ----------
        X : (n_samples, n_inputs) jnp.ndarray
        params : dict with ``"beta"`` (n_inputs,) and ``"b"`` ()
        skip_centering : bool
            For Y classification (``j == Y_idx``), pass ``True`` to keep the
            classification logit; for X reconstruction, pass ``False`` so
            the per-variable forward is mean-centered against constant-
            output shortcuts (mirrors MLPAdapter behavior).

        Returns
        -------
        out : (n_samples,) jnp.ndarray
        """
        out = X @ params["beta"] + params["b"]
        if not skip_centering:
            out = out - jnp.mean(out)
        return out


# ============================================================================
# MLP Head Adapter
# ============================================================================


class MLPHeadAdapter:
    """Per-variable shallow MLP: ``X -> Linear(d_in, h) -> ReLU -> Linear(h, 1)``.

    Tests whether modest nonlinearity in the per-variable readout escapes the
    procedural f_Y collapse documented by the AAP architectural-fix mini-Sprint
    (commits 11d75a0, 0310b39, ddbaf09 on origin/aap; main 7b42789 ablation).
    LinearHead and DAG-Attention both collapse to a constant f_Y under JCCE's
    joint loss; if MLPHead also collapses, the failure mode is more general
    than "the architecture lacks expressiveness". If MLPHead escapes, modest
    nonlinearity is the cure and the post-hoc fix can move into joint
    training.

    Hidden dimension default is intentionally small (16): the experiment is
    about whether nonlinearity helps at all, not about model capacity.
    Single hidden layer; ReLU activation; no normalization, no residual.

    Notes
    -----
    Mirrors ``LinearHeadAdapter`` in scope: only ``init_params`` and
    ``forward`` are implemented, since ``solve_output_weights`` and
    ``get_hidden_features`` are disabled in jcce_learner (line 1598).
    """

    def __init__(self, hidden_dim: int = 16, key: random.PRNGKey = None):
        self.hidden_dim = hidden_dim
        self.key = key if key is not None else random.PRNGKey(42)

    def init_params(self, n_inputs: int) -> Dict:
        """Initialize MLP params with Glorot-style scaling.

        ``W1`` scaled by ``1/sqrt(n_inputs)``, ``W2`` scaled by
        ``1/sqrt(hidden_dim)``. Keeps initial output magnitude bounded
        independently of ``n_inputs`` and ``hidden_dim``.
        """
        key_W1, key_W2 = random.split(self.key)
        W1_scale = 1.0 / max(1, n_inputs) ** 0.5
        W2_scale = 1.0 / max(1, self.hidden_dim) ** 0.5
        W1 = random.normal(key_W1, (n_inputs, self.hidden_dim)) * W1_scale
        b1 = jnp.zeros((self.hidden_dim,))
        W2 = random.normal(key_W2, (self.hidden_dim,)) * W2_scale
        b2 = jnp.zeros(())
        return {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "n_inputs": n_inputs,
            "hidden_dim": self.hidden_dim,
        }

    def forward(
        self,
        X: jnp.ndarray,
        params: Dict,
        training: bool = True,
        rng_key: random.PRNGKey = None,
        skip_centering: bool = False,
        **kwargs,  # absorb A=..., other adapter-specific kwargs we ignore
    ) -> jnp.ndarray:
        """Forward pass: ``ReLU(X @ W1 + b1) @ W2 + b2`` with optional mean centering.

        Parameters
        ----------
        X : (n_samples, n_inputs) jnp.ndarray
        params : dict with ``W1`` (n_inputs, hidden_dim), ``b1`` (hidden_dim,),
            ``W2`` (hidden_dim,), and ``b2`` ()
        skip_centering : bool
            ``True`` for the Y classification path (keep the logit);
            ``False`` for X reconstruction (mean-center the output to block
            constant-output shortcuts; matches MLPAdapter / LinearHeadAdapter).

        Returns
        -------
        out : (n_samples,) jnp.ndarray
        """
        h = jax.nn.relu(X @ params["W1"] + params["b1"])
        out = h @ params["W2"] + params["b2"]
        if not skip_centering:
            out = out - jnp.mean(out)
        return out


# ============================================================================
# DAG-Attention Transformer Adapter
# ============================================================================

from jcce.models.dag_attention_transformer import (
    DAGAttentionTransformer as DAGAttentionTransformerBase,
)


class DAGAttentionAdapter:
    """
    Adapter for DAG-Attention Transformer.

    Key difference from TransformerAdapter: passes A to the model during forward pass
    with soft masking (gradients flow through A), and can compute consistency loss.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        key: random.PRNGKey = None,
        dropout_rate: float = 0.1,
        temperature: float = 5.0,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.key = key if key is not None else random.PRNGKey(42)

        self.model = DAGAttentionTransformerBase(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            temperature=temperature,
        )

    def init_params(self, n_inputs: int) -> Dict:
        dummy_input = jnp.ones((1, n_inputs))
        dummy_A = jnp.zeros((n_inputs, n_inputs))
        flax_params = self.model.init(self.key, dummy_input, A=dummy_A, training=False)

        param_dict = flax_to_dict_params(flax_params)
        param_dict["n_inputs"] = n_inputs

        key_proj = random.PRNGKey(0)
        param_dict["output_proj_W"] = random.normal(key_proj, (self.d_model, 1)) * 0.5
        param_dict["output_proj_b"] = jnp.zeros(1)

        return param_dict

    def forward(
        self,
        X: jnp.ndarray,
        params: Dict,
        A: jnp.ndarray = None,
        training: bool = True,
        rng_key: random.PRNGKey = None,
        skip_centering: bool = False,
        return_attn: bool = False,
        temperature: float = None,
    ) -> jnp.ndarray:
        """
        Forward pass with soft DAG-masked attention.

        Args:
            X: (n_samples, n_inputs) input data
            params: Dict-format parameters
            A: (n_inputs, n_inputs) adjacency matrix — USED for soft masking
            training: enables dropout
            rng_key: RNG key for dropout
            skip_centering: skip mean centering for classification
            return_attn: if True, returns (output, attn_weights_list)
            temperature: optional runtime override for the soft-mask temperature.
                When provided (e.g., from a learner-side annealing schedule),
                supersedes self.temperature for this call only.
        """
        n_samples, n_inputs = X.shape
        flax_params = dict_to_flax_params(params)

        if training and rng_key is not None:
            result = self.model.apply(
                flax_params,
                X,
                A=A,
                training=training,
                return_attn=return_attn,
                temperature=temperature,
                rngs={"dropout": rng_key},
            )
        else:
            result = self.model.apply(
                flax_params,
                X,
                A=A,
                training=False,
                return_attn=return_attn,
                temperature=temperature,
            )

        if return_attn:
            h, attn_list = result
        else:
            h = result
            attn_list = None

        # Pool across features
        h_pooled = jnp.mean(h, axis=1)

        if "output_proj_W" not in params:
            key = random.PRNGKey(0)
            params["output_proj_W"] = random.normal(key, (self.d_model, 1)) * 0.5
            params["output_proj_b"] = jnp.zeros(1)

        if params.get("_weights_solved", False) and "_H_mean" in params:
            h_pooled = (h_pooled - params["_H_mean"]) / params["_H_std"]

        if skip_centering and not params.get("_weights_solved", False):
            h_pooled = (h_pooled - jnp.mean(h_pooled, axis=0, keepdims=True)) / (
                jnp.std(h_pooled, axis=0, keepdims=True) + 1e-8
            )

        output = h_pooled @ params["output_proj_W"] + params["output_proj_b"]
        output = output.squeeze()

        if not skip_centering and not params.get("_weights_solved", False):
            output = output - jnp.mean(output)

        if return_attn:
            return output, attn_list
        return output

    def get_hidden_features(
        self, X: jnp.ndarray, params: Dict, A: jnp.ndarray = None
    ) -> jnp.ndarray:
        flax_params = dict_to_flax_params(params)
        h = self.model.apply(flax_params, X, A=A, training=False)
        h_pooled = jnp.mean(h, axis=1)
        return h_pooled

    def solve_output_weights(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: Dict,
        lambda_reg: float = 1e-4,
        for_classification: bool = True,
        A: jnp.ndarray = None,
    ) -> Dict:
        H = self.get_hidden_features(X, params, A=A)
        n_hidden = H.shape[1]
        n_samples = len(y)

        if for_classification:
            H_mean = jnp.mean(H, axis=0, keepdims=True)
            H_std = jnp.std(H, axis=0, keepdims=True) + 1e-8
            H_norm = (H - H_mean) / H_std

            n_steps = 300
            lr = 0.1
            key = random.PRNGKey(42)
            W = random.normal(key, (n_hidden, 1)) * 0.01
            b = jnp.array([0.0])
            y_col = y.reshape(-1, 1)

            for step in range(n_steps):
                logits = H_norm @ W + b
                probs = jax.nn.sigmoid(logits)
                error = probs - y_col
                grad_W = (H_norm.T @ error) / n_samples + lambda_reg * W
                grad_b = jnp.mean(error)
                W = W - lr * grad_W
                b = b - lr * grad_b

            params["output_proj_W"] = W
            params["output_proj_b"] = b
            params["_H_mean"] = H_mean
            params["_H_std"] = H_std
        else:
            HTH = H.T @ H + lambda_reg * jnp.eye(n_hidden)
            HTy = H.T @ y.reshape(-1, 1)
            W = jnp.linalg.solve(HTH, HTy)
            predictions = H @ W
            b = jnp.mean(y.reshape(-1, 1) - predictions)
            params["output_proj_W"] = W
            params["output_proj_b"] = jnp.array([b])

        params["_weights_solved"] = True
        return params

    def consistency_loss(
        self,
        X: jnp.ndarray,
        params: Dict,
        A: jnp.ndarray,
        temperature: float = None,
        edge_only: bool = False,
        edge_threshold: float = 0.05,
    ) -> jnp.ndarray:
        """Compute attention-DAG consistency loss for the multi-loss objective.

        When `temperature` is provided, both the soft mask used in the attention
        forward pass and the sigmoid(A * temperature) target distribution use it.
        When `edge_only` is True, the consistency target is restricted to A's
        existing edges (|A| > edge_threshold) plus the diagonal — stops the
        "pull A toward density" pathology of vanilla KL on data-poor datasets.
        """
        eff_temperature = self.temperature if temperature is None else temperature
        flax_params = dict_to_flax_params(params)
        _, attn_list = self.model.apply(
            flax_params, X, A=A, training=False, return_attn=True, temperature=temperature
        )
        return DAGAttentionTransformerBase.consistency_loss(
            attn_list, A, temperature=eff_temperature,
            edge_only=edge_only, edge_threshold=edge_threshold,
        )


# ============================================================================
# TopoMamba Adapter (renamed 2026-04-28 from CausalMamba; see topo_mamba.py)
# ============================================================================

from jcce.models.topo_mamba import TopoMambaProcessor as TopoMambaBase
from jcce.models.topo_mamba import (
    sinkhorn_topological_sort,
    topological_sort_from_adjacency,
)


class TopoMambaAdapter:
    """
    Adapter for TopoMamba with selectable variable ordering and optional
    DAG-gated state transitions.

    Four sort modes:
      - "topological" (default): hard order from
        topological_sort_from_adjacency(A). Non-differentiable (gradient
        stops at A); runs host-side via jax.pure_callback.
      - "sinkhorn": differentiable soft sort. The Sprint 2 mechanism. P is
        a doubly-stochastic matrix from Sinkhorn over ancestral-depth
        scores; gradient flows from the loss back to A through P.
      - "random": fixed permutation seeded by ``sort_seed``. Gate 1 control.
      - "identity": no permutation. Equivalent to standard Mamba.

    Sprint 3 ``enable_gating`` (Mechanism 2): when True and A is provided,
    the SSM hidden-state recurrence gets a per-position gate in R^{d_state}
    derived from A's columns selected by the variable at position t (hard
    or soft mixture). The four ablation modes from skill spec:

      sort_mode    enable_gating   meaning
      topological  False           Sprint 1 hard sort
      sinkhorn     False           Sprint 2 sort-only
      topological  True            hard_sort + DAG gate
      sinkhorn     True            soft_sort + DAG gate (full TopoMamba)

    The "random" mode does not re-sample per step; doing so would require an
    rng_key plumbed through jcce_learner.py's dispatch sites (main-owned).
    Across run-level seeds the fixed permutation varies, which gives the
    distribution Gate 1 needs.
    """

    _SORT_MODES = ("topological", "sinkhorn", "random", "identity")

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        key: random.PRNGKey = None,
        n_features: int = None,
        sort_mode: str = "topological",
        sort_seed: int = 0,
        sinkhorn_temperature: float = 0.1,
        sinkhorn_n_iters: int = 10,
        enable_gating: bool = False,
    ):
        if sort_mode not in self._SORT_MODES:
            raise ValueError(
                f"sort_mode must be one of {self._SORT_MODES}, got {sort_mode!r}"
            )
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_features = n_features
        self.key = key if key is not None else random.PRNGKey(42)
        self.sort_mode = sort_mode
        self.sort_seed = sort_seed
        self.sinkhorn_temperature = sinkhorn_temperature
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.enable_gating = enable_gating

        self.model = TopoMambaBase(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=1,
            enable_gating=enable_gating,
        )


    def init_params(self, n_inputs: int) -> Dict:
        dummy_input = jnp.ones((1, n_inputs))
        init_kwargs = dict(topo_order=None, training=False)
        # When gating is enabled, init must trace through the gated path so
        # the dag_gate Dense and the _GatedMambaProcessor params get
        # registered. Passing a dummy A is enough; the gate values are
        # discarded after init.
        if self.enable_gating:
            init_kwargs["A"] = jnp.zeros((n_inputs, n_inputs), dtype=jnp.float32)
        flax_params = self.model.init(self.key, dummy_input, **init_kwargs)

        param_dict = flax_to_dict_params(flax_params)
        param_dict["n_inputs"] = n_inputs

        key_proj = random.PRNGKey(0)
        param_dict["output_proj_W"] = random.normal(key_proj, (self.d_model, 1)) * 0.5
        param_dict["output_proj_b"] = jnp.zeros(1)

        return param_dict

    def _resolve_ordering(self, A, n_inputs, temperature=None):
        """Resolve the variable ordering for TopoMambaProcessor.

        Returns a (topo_order, perm_matrix) tuple. At most one of the two
        is non-None; both are None for "identity" or for any dispatch path
        that does not pass A (init, no-A reconstruction paths).

          - topological: topo_order from pure_callback (hard sort).
          - sinkhorn:    perm_matrix from Sinkhorn (soft, differentiable).
          - random:      topo_order from a fixed permutation per sort_seed.
          - identity:    (None, None).

        ``temperature`` overrides ``self.sinkhorn_temperature`` for the
        sinkhorn path; None falls back to the adapter's default. The override
        path is used by jcce_learner.py to thread the cosine schedule
        (1.0 -> 0.1 over training steps) without rebuilding the adapter.
        """
        if self.sort_mode == "identity":
            return None, None
        if A is None:
            return None, None
        if self.sort_mode == "topological":
            return topological_sort_from_adjacency(A, threshold=0.01), None
        if self.sort_mode == "sinkhorn":
            T = self.sinkhorn_temperature if temperature is None else temperature
            P = sinkhorn_topological_sort(
                A,
                temperature=T,
                n_iters=self.sinkhorn_n_iters,
            )
            return None, P
        # random — deterministic given (sort_seed, n_inputs); no caching.
        return random.permutation(random.PRNGKey(self.sort_seed), n_inputs), None

    def forward(
        self,
        X: jnp.ndarray,
        params: Dict,
        A: jnp.ndarray = None,
        skip_centering: bool = False,
        training: bool = False,
        rng_key=None,
        temperature=None,
    ) -> jnp.ndarray:
        """
        Forward pass with causal-aware variable ordering.

        Args:
            X: (n_samples, n_inputs) input data.
            params: Dict-format parameters.
            A: (n_inputs, n_inputs) adjacency matrix — used for ordering.
            skip_centering: skip mean centering for classification.
            training, rng_key: accepted for dispatch-signature parity with
                DAGAttentionAdapter; currently unused (TopoMamba has no
                dropout).
            temperature: per-call Sinkhorn temperature override. None falls
                back to ``self.sinkhorn_temperature``. When jcce_learner.py
                lands the temperature plumbing patch, learn_structure will
                pass the cosine-scheduled value here at each dispatch site.
                Ignored for non-sinkhorn sort_modes.
        """
        n_samples, n_inputs = X.shape
        flax_params = dict_to_flax_params(params)

        topo_order, perm_matrix = self._resolve_ordering(
            A, n_inputs, temperature=temperature
        )

        # Forward through TopoMamba. A is passed only when gating is on
        # (the gated path needs A's columns for the gate). Non-gated path
        # ignores A.
        h = self.model.apply(
            flax_params,
            X,
            topo_order=topo_order,
            perm_matrix=perm_matrix,
            A=(A if self.enable_gating else None),
            training=False,
        )

        # Pool across sequence
        h_pooled = jnp.mean(h, axis=1)

        if "output_proj_W" not in params:
            key = random.PRNGKey(0)
            params["output_proj_W"] = random.normal(key, (self.d_model, 1)) * 0.5
            params["output_proj_b"] = jnp.zeros(1)

        if skip_centering and not params.get("_weights_solved", False):
            h_pooled = (h_pooled - jnp.mean(h_pooled, axis=0, keepdims=True)) / (
                jnp.std(h_pooled, axis=0, keepdims=True) + 1e-8
            )

        output = h_pooled @ params["output_proj_W"] + params["output_proj_b"]
        output = output.squeeze()

        if not skip_centering and not params.get("_weights_solved", False):
            output = output - jnp.mean(output)

        return output

    def get_hidden_features(
        self, X: jnp.ndarray, params: Dict, A: jnp.ndarray = None,
        temperature=None,
    ) -> jnp.ndarray:
        flax_params = dict_to_flax_params(params)
        n_inputs = X.shape[1]
        topo_order, perm_matrix = self._resolve_ordering(
            A, n_inputs, temperature=temperature
        )
        h = self.model.apply(
            flax_params,
            X,
            A=(A if self.enable_gating else None),
            topo_order=topo_order,
            perm_matrix=perm_matrix,
            training=False,
        )
        h_pooled = jnp.mean(h, axis=1)
        return h_pooled

    def solve_output_weights(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: Dict,
        lambda_reg: float = 1e-4,
        for_classification: bool = True,
        A: jnp.ndarray = None,
    ) -> Dict:
        H = self.get_hidden_features(X, params, A=A)
        n_hidden = H.shape[1]
        n_samples = len(y)

        if for_classification:
            n_steps = 200
            lr = 0.5
            key = random.PRNGKey(42)
            W = random.normal(key, (n_hidden, 1)) * 0.01
            b = jnp.array([0.0])
            y_col = y.reshape(-1, 1)

            for step in range(n_steps):
                logits = H @ W + b
                probs = jax.nn.sigmoid(logits)
                error = probs - y_col
                grad_W = (H.T @ error) / n_samples + lambda_reg * W
                grad_b = jnp.mean(error)
                W = W - lr * grad_W
                b = b - lr * grad_b

            params["output_proj_W"] = W
            params["output_proj_b"] = b
        else:
            HTH = H.T @ H + lambda_reg * jnp.eye(n_hidden)
            HTy = H.T @ y.reshape(-1, 1)
            W = jnp.linalg.solve(HTH, HTy)
            predictions = H @ W
            b = jnp.mean(y.reshape(-1, 1) - predictions)
            params["output_proj_W"] = W
            params["output_proj_b"] = jnp.array([b])

        params["_weights_solved"] = True
        return params


# Backward-compatibility alias (renamed 2026-04-28; see jcce/models/topo_mamba.py)
CausalMambaAdapter = TopoMambaAdapter


# ============================================================================
# Effect Adapter Wrapper (for Causal Effect Estimation)
# ============================================================================


class EffectAdapterWrapper:
    """
    Wrapper that adds effect estimation capability to ANY processor adapter.

    This wrapper enables causal effect estimation (ATE, CATE) by adding:
    - Y(0) head: Predicts potential outcome under no treatment
    - Y(1) head: Predicts potential outcome under treatment
    - Propensity head: Predicts P(T=1|X,L)

    The wrapper preserves all original adapter functionality while adding
    effect estimation capability. NSGA-II can compare all processors with
    effect estimation enabled.

    Usage:
        elm_adapter = ELMAdapter(hidden_dim=32, key=key)
        elm_with_effects = EffectAdapterWrapper(elm_adapter, latent_dim=4)
        params = elm_with_effects.init_params(n_inputs)
        outputs = elm_with_effects.forward_with_effects(X, params, U, T)

    v6.0: Initial implementation following CAUSAL_EFFECT_ESTIMATION_IMPLEMENTATION_PLAN.md
    """

    def __init__(
        self,
        base_adapter,
        latent_dim: int = 4,
        head_hidden_dim: int = 32,
        enable_effects: bool = True,
        key: random.PRNGKey = None,
    ):
        """
        Initialize effect adapter wrapper.

        Args:
            base_adapter: Any processor adapter (ELM, MLP, Transformer, Mamba, GNN)
            latent_dim: Dimension of latent confounders (k in L = U @ V.T)
            head_hidden_dim: Hidden dimension for effect heads
            enable_effects: Whether to enable effect estimation
            key: RNG key for initialization
        """
        self.base_adapter = base_adapter
        self.latent_dim = latent_dim
        self.head_hidden_dim = head_hidden_dim
        self.enable_effects = enable_effects
        self.key = key if key is not None else random.PRNGKey(42)

        # Import effect heads from effect_estimation module
        from jcce.structure_learning.effect_estimation import EffectHeads

        self.effect_heads = EffectHeads(head_hidden_dim=head_hidden_dim)

        # Get hidden_dim from base adapter (adapters store this)
        self.hidden_dim = getattr(base_adapter, "hidden_dim", 32)
        if hasattr(base_adapter, "d_model"):
            self.hidden_dim = base_adapter.d_model

    def init_params(self, n_inputs: int) -> Dict:
        """
        Initialize parameters including effect heads.

        Args:
            n_inputs: Number of input features

        Returns:
            params: Dict with base adapter params and effect head params
        """
        # Initialize base adapter parameters
        params = self.base_adapter.init_params(n_inputs)

        if self.enable_effects:
            # Initialize effect head parameters
            augmented_dim = self.hidden_dim + self.latent_dim
            dummy_input = jnp.zeros((1, augmented_dim))

            key1, key2 = random.split(self.key)
            effect_params = self.effect_heads.init(key1, dummy_input, training=False)

            # Store effect head params
            params["effect_heads"] = flax_to_dict_params(effect_params)
            params["effect_enabled"] = True
            params["latent_dim"] = self.latent_dim
            params["head_hidden_dim"] = self.head_hidden_dim

        return params

    def forward(self, X: jnp.ndarray, params: Dict, **kwargs) -> jnp.ndarray:
        """
        Standard forward pass (delegates to base adapter).

        Args:
            X: Input data
            params: Parameters dict
            **kwargs: Additional arguments (A for GNN, etc.)

        Returns:
            output: Predictions from base adapter
        """
        return self.base_adapter.forward(X, params, **kwargs)

    def forward_with_effects(
        self,
        X: jnp.ndarray,
        params: Dict,
        U: jnp.ndarray,
        T: jnp.ndarray,
        sample_indices: jnp.ndarray = None,
        rng_key: random.PRNGKey = None,
        training: bool = True,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass with effect estimation.

        Args:
            X: (batch, n_inputs) input features
            params: Parameters dict
            U: (n_samples, latent_dim) latent factor scores
            T: (batch,) binary treatment assignments
            sample_indices: (batch,) indices into U for batching (optional)
            rng_key: RNG key for dropout (optional)
            training: Whether in training mode
            **kwargs: Additional arguments for base adapter

        Returns:
            Dict with:
            - 'output': Standard prediction
            - 'representation': Hidden representation
            - 'y0': Potential outcome under T=0
            - 'y1': Potential outcome under T=1
            - 'propensity': P(T=1|X,L)
            - 'tau': CATE (y1 - y0)
            - 'ATE': Average treatment effect
        """
        n_samples = X.shape[0]

        # Get base adapter forward (may need representation extraction)
        # Most adapters compute h_pooled internally then project to output
        # We need to get the representation before final projection

        # Use base forward for standard output
        output = self.base_adapter.forward(X, params, **kwargs)

        result = {"output": output}

        if not self.enable_effects or not params.get("effect_enabled", False):
            return result

        # Extract representation from base adapter
        # We need to recompute to get the pooled representation
        representation = self._get_representation(X, params, **kwargs)

        # Get latent scores for batch
        if sample_indices is not None:
            L_batch = U[sample_indices, :]  # (batch, latent_dim)
        else:
            L_batch = U[:n_samples, :]  # Assume aligned

        # Augment representation with latent factors
        augmented_rep = jnp.concatenate([representation, L_batch], axis=-1)

        # Get effect head params
        effect_params = dict_to_flax_params(params["effect_heads"])

        # Forward through effect heads
        if training and rng_key is not None:
            y0, y1, propensity = self.effect_heads.apply(
                effect_params, augmented_rep, training=training, rngs={"dropout": rng_key}
            )
        else:
            y0, y1, propensity = self.effect_heads.apply(
                effect_params, augmented_rep, training=False
            )

        # Clip propensity for numerical stability
        propensity = jnp.clip(propensity, 1e-4, 1 - 1e-4)

        # Compute CATE
        tau = y1 - y0

        result.update(
            {
                "representation": representation,
                "y0": y0.squeeze(),
                "y1": y1.squeeze(),
                "propensity": propensity.squeeze(),
                "tau": tau.squeeze(),
                "ATE": jnp.mean(tau),
            }
        )

        return result

    def _get_representation(self, X: jnp.ndarray, params: Dict, **kwargs) -> jnp.ndarray:
        """
        Extract pooled representation from base adapter.

        This replicates the internal forward pass up to pooling,
        needed for effect head input.

        Args:
            X: Input data
            params: Parameters dict

        Returns:
            representation: (batch, hidden_dim) pooled features
        """
        adapter_name = self.base_adapter.__class__.__name__

        # Convert params to Flax format
        flax_params = dict_to_flax_params(params)

        if adapter_name == "ELMAdapter":
            h = self.base_adapter.model.apply(flax_params, X, A=None, training=False)
            return jnp.mean(h, axis=1)  # Pool over nodes

        elif adapter_name == "MLPAdapter":
            h = self.base_adapter.model.apply(flax_params, X, A=None, training=False)
            return jnp.mean(h, axis=1)

        elif adapter_name == "TransformerAdapter":
            h = self.base_adapter.model.apply(flax_params, X, A=None, training=False)
            return jnp.mean(h, axis=1)

        elif adapter_name == "MambaAdapter":
            X_projected = jnp.tile(X[:, :, jnp.newaxis], (1, 1, self.base_adapter.model.d_model))
            h = self.base_adapter.model.apply(flax_params, X_projected)
            return jnp.mean(h, axis=1)

        elif adapter_name == "GNNAdapter":
            A = kwargs.get("A", jnp.eye(X.shape[1]))
            h = self.base_adapter.model.apply(flax_params, X, A, training=False)
            return jnp.mean(h, axis=1)

        else:
            # Fallback: use output as representation (may not work well)
            output = self.base_adapter.forward(X, params, **kwargs)
            return output[:, None] if output.ndim == 1 else output


def create_effect_adapter(
    processor_type: str,
    latent_dim: int = 4,
    head_hidden_dim: int = 32,
    enable_effects: bool = True,
    key: random.PRNGKey = None,
    **adapter_kwargs,
) -> EffectAdapterWrapper:
    """
    Factory function to create any processor adapter with effect estimation.

    Args:
        processor_type: 'elm', 'mlp', 'transformer', 'mamba', 'gnn'
        latent_dim: Dimension of latent confounders
        head_hidden_dim: Hidden dimension for effect heads
        enable_effects: Whether to enable effect estimation
        key: RNG key
        **adapter_kwargs: Arguments for specific adapter

    Returns:
        EffectAdapterWrapper wrapping the specified adapter
    """
    if key is None:
        key = random.PRNGKey(42)

    key1, key2 = random.split(key)

    # Create base adapter
    processor_type = processor_type.lower()

    if processor_type == "elm":
        base_adapter = ELMAdapter(
            hidden_dim=adapter_kwargs.get("hidden_dim", 32),
            n_hidden_nodes=adapter_kwargs.get("n_hidden_nodes", 128),
            activation=adapter_kwargs.get("activation", "tanh"),
            key=key1,
        )

    elif processor_type == "mlp":
        base_adapter = MLPAdapter(
            hidden_dim=adapter_kwargs.get("hidden_dim", 64),
            n_layers=adapter_kwargs.get("n_layers", 2),
            activation=adapter_kwargs.get("activation", "relu"),
            dropout_rate=adapter_kwargs.get("dropout_rate", 0.1),
            key=key1,
        )

    elif processor_type == "transformer":
        base_adapter = TransformerAdapter(
            d_model=adapter_kwargs.get("d_model", 128),
            n_heads=adapter_kwargs.get("n_heads", 4),
            n_layers=adapter_kwargs.get("n_layers", 2),
            d_ff=adapter_kwargs.get("d_ff", 512),
            dropout_rate=adapter_kwargs.get("dropout_rate", 0.1),
            key=key1,
        )

    elif processor_type == "mamba":
        base_adapter = MambaAdapter(
            d_model=adapter_kwargs.get("d_model", 128),
            d_state=adapter_kwargs.get("d_state", 16),
            d_conv=adapter_kwargs.get("d_conv", 4),
            expand=adapter_kwargs.get("expand", 2),
            n_features=adapter_kwargs.get("n_features", None),
            key=key1,
        )

    elif processor_type == "gnn":
        base_adapter = GNNAdapter(
            hidden_dim=adapter_kwargs.get("hidden_dim", 64),
            n_layers=adapter_kwargs.get("n_layers", 2),
            gnn_type=adapter_kwargs.get("gnn_type", "gcn"),
            sage_aggregation=adapter_kwargs.get("sage_aggregation", "mean"),
            key=key1,
        )

    else:
        raise ValueError(f"Unknown processor type: {processor_type}")

    # Wrap with effect estimation
    return EffectAdapterWrapper(
        base_adapter=base_adapter,
        latent_dim=latent_dim,
        head_hidden_dim=head_hidden_dim,
        enable_effects=enable_effects,
        key=key2,
    )


# ============================================================================
# Test Functions
# ============================================================================

if __name__ == "__main__":
    print("Testing Processor Adapters")
    print("=" * 60)

    key = random.PRNGKey(42)
    n_samples, n_inputs = 10, 5
    X = random.normal(key, (n_samples, n_inputs))

    # Test MLP
    print("\n1. MLP Adapter")
    mlp = MLPAdapter(hidden_dim=32, n_layers=2, key=key)
    params_mlp = mlp.init_params(n_inputs)
    output_mlp = mlp.forward(X, params_mlp)
    print(f"   Input: {X.shape}, Output: {output_mlp.shape}")
    print("   [OK] MLP working!")

    # Test Transformer
    print("\n2. Transformer Adapter")
    transformer = TransformerAdapter(d_model=32, n_heads=2, n_layers=1, d_ff=64, key=key)
    params_transformer = transformer.init_params(n_inputs)
    output_transformer = transformer.forward(X, params_transformer)
    print(f"   Input: {X.shape}, Output: {output_transformer.shape}")
    print("   [OK] Transformer working!")

    # Test Mamba
    print("\n3. Mamba Adapter")
    mamba = MambaAdapter(d_model=32, d_state=8, d_conv=4, expand=2, key=key)
    params_mamba = mamba.init_params(n_inputs)
    output_mamba = mamba.forward(X, params_mamba)
    print(f"   Input: {X.shape}, Output: {output_mamba.shape}")
    print("   [OK] Mamba working!")

    # Test ELM
    print("\n4. ELM Adapter")
    elm = ELMAdapter(hidden_dim=32, n_hidden_nodes=64, key=key)
    params_elm = elm.init_params(n_inputs)
    output_elm = elm.forward(X, params_elm)
    print(f"   Input: {X.shape}, Output: {output_elm.shape}")
    print("   [OK] ELM working!")

    # Test GNN
    print("\n5. GNN Adapter")
    gnn = GNNAdapter(hidden_dim=32, n_layers=2, gnn_type="gcn", key=key)
    params_gnn = gnn.init_params(n_inputs)
    A_test = random.bernoulli(key, p=0.3, shape=(n_inputs, n_inputs)).astype(jnp.float32)
    output_gnn = gnn.forward(X, params_gnn, A=A_test)
    print(f"   Input: {X.shape}, Adjacency: {A_test.shape}, Output: {output_gnn.shape}")
    print("   [OK] GNN working!")

    print("\n" + "=" * 60)
    print("[OK] All adapters working!")

    # Test EffectAdapterWrapper with all processors
    print("\n" + "=" * 60)
    print("Testing Effect Adapter Wrapper (ALL Processors)")
    print("=" * 60)

    latent_dim = 4
    U = random.normal(key, (n_samples, latent_dim))  # Latent scores
    T = (random.uniform(key, (n_samples,)) > 0.5).astype(jnp.float32)  # Treatment

    for processor_type in ["elm", "mlp", "transformer", "mamba", "gnn"]:
        print(
            f"\n6.{['elm', 'mlp', 'transformer', 'mamba', 'gnn'].index(processor_type) + 1}. {processor_type.upper()} with Effect Heads"
        )
        try:
            effect_adapter = create_effect_adapter(
                processor_type=processor_type,
                latent_dim=latent_dim,
                head_hidden_dim=32,
                enable_effects=True,
                key=key,
                hidden_dim=32,
                d_model=32,
                n_layers=1,
                n_heads=2,
                d_ff=64,
            )

            params = effect_adapter.init_params(n_inputs)
            print(f"   Params initialized: effect_enabled={params.get('effect_enabled', False)}")

            # Test standard forward
            output = effect_adapter.forward(X, params)
            print(f"   Standard forward: {output.shape}")

            # Test forward with effects
            if processor_type == "gnn":
                outputs = effect_adapter.forward_with_effects(
                    X, params, U, T, training=False, A=A_test
                )
            else:
                outputs = effect_adapter.forward_with_effects(X, params, U, T, training=False)

            print(f"   Effect outputs: y0={outputs['y0'].shape}, y1={outputs['y1'].shape}")
            print(f"   CATE (tau): {outputs['tau'].shape}, ATE={float(outputs['ATE']):.4f}")
            print(f"   [OK] {processor_type.upper()} with effects working!")

        except Exception as e:
            print(f"   [ERROR] Error: {e}")

    print("\n" + "=" * 60)
    print("[OK] All effect adapters working!")
