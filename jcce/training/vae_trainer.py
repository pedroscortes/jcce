"""
VAE training utilities.

Provides training step, training loop, and utilities for baseline VAE.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from jcce.models.vae import BaselineVAE, elbo_loss


class TrainState(train_state.TrainState):
    """
    Extended train state with additional fields for VAE training.

    Attributes:
        key: JAX random key for sampling
    """

    key: jax.random.PRNGKey


def create_train_state(
    model: BaselineVAE,
    learning_rate: float,
    key: jax.random.PRNGKey,
    input_shape: tuple,
) -> TrainState:
    """
    Create initial training state.

    Args:
        model: VAE model
        learning_rate: Learning rate for Adam optimizer
        key: JAX random key
        input_shape: Shape of input (batch_size, input_dim)

    Returns:
        state: Initialized training state

    Example:
        >>> model = BaselineVAE(latent_dim=10, output_dim=20)
        >>> key = jax.random.PRNGKey(0)
        >>> state = create_train_state(model, lr=1e-3, key=key, input_shape=(32, 20))
    """
    key, init_key, dropout_key = jax.random.split(key, 3)

    # Initialize model parameters
    dummy_input = jnp.ones(input_shape)
    variables = model.init(init_key, dummy_input, dropout_key, training=True)

    # Create optimizer
    tx = optax.adam(learning_rate)

    # Create train state
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, key=key)

    return state


@jax.jit
def train_step(state: TrainState, batch: jnp.ndarray, beta: float = 1.0) -> tuple[TrainState, dict]:
    """
    Single training step for VAE.

    Args:
        state: Current training state
        batch: (batch_size, input_dim) batch of observations
        beta: Weight for KL term (β-VAE)

    Returns:
        state: Updated training state
        metrics: Dictionary with loss components

    Example:
        >>> # Assuming state is initialized
        >>> batch = jnp.ones((32, 20))
        >>> new_state, metrics = train_step(state, batch)
        >>> print(metrics.keys())
        dict_keys(['total_loss', 'reconstruction_loss', 'kl_divergence'])
    """
    key, subkey = jax.random.split(state.key)

    def loss_fn(params):
        # Forward pass
        x_recon, info = state.apply_fn({"params": params}, batch, subkey, training=True)

        # Compute ELBO loss
        loss, metrics = elbo_loss(
            x=batch,
            x_reconstructed=x_recon,
            mu=info["mu"],
            log_var=info["log_var"],
            beta=beta,
            reconstruction_type="mse",
        )

        return loss, metrics

    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    # Update random key
    state = state.replace(key=key)

    return state, metrics


def train_epoch(
    state: TrainState,
    train_data: jnp.ndarray,
    batch_size: int,
    beta: float = 1.0,
) -> tuple[TrainState, dict]:
    """
    Train for one epoch.

    Args:
        state: Training state
        train_data: (n_samples, input_dim) training data
        batch_size: Batch size
        beta: Weight for KL term

    Returns:
        state: Updated training state
        epoch_metrics: Average metrics over epoch

    Example:
        >>> # Assuming state is initialized
        >>> train_data = jnp.ones((1000, 20))
        >>> state, metrics = train_epoch(state, train_data, batch_size=32)
    """
    n_samples = train_data.shape[0]
    n_batches = n_samples // batch_size

    # Shuffle data
    key, shuffle_key = jax.random.split(state.key)
    perm = jax.random.permutation(shuffle_key, n_samples)
    train_data = train_data[perm]

    state = state.replace(key=key)

    # Accumulate metrics
    total_metrics = {"total_loss": 0.0, "reconstruction_loss": 0.0, "kl_divergence": 0.0}

    for i in range(n_batches):
        batch = train_data[i * batch_size : (i + 1) * batch_size]
        state, metrics = train_step(state, batch, beta=beta)

        # Accumulate
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    # Average over batches
    epoch_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return state, epoch_metrics


def train_vae(
    model: BaselineVAE,
    train_data: jnp.ndarray,
    val_data: jnp.ndarray | None = None,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    key: Optional[jax.random.PRNGKey] = None,
    verbose: bool = True,
) -> tuple[TrainState, dict]:
    """
    Train VAE for multiple epochs.

    Args:
        model: VAE model
        train_data: (n_train, input_dim) training data
        val_data: (n_val, input_dim) validation data (optional)
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        beta: Weight for KL term (β-VAE)
        key: JAX random key
        verbose: Whether to print progress

    Returns:
        state: Final training state
        history: Dictionary with training history

    Example:
        >>> model = BaselineVAE(latent_dim=10, output_dim=20)
        >>> train_data = jnp.ones((1000, 20))
        >>> state, history = train_vae(model, train_data, n_epochs=10)
    """
    # Create key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize training state
    input_shape = (batch_size, train_data.shape[1])
    state = create_train_state(model, learning_rate, key, input_shape)

    # Training history
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
    }

    if val_data is not None:
        history["val_loss"] = []
        history["val_recon"] = []
        history["val_kl"] = []

    # Training loop
    for epoch in range(n_epochs):
        # Train for one epoch
        state, train_metrics = train_epoch(state, train_data, batch_size, beta=beta)

        # Record training metrics
        history["train_loss"].append(float(train_metrics["total_loss"]))
        history["train_recon"].append(float(train_metrics["reconstruction_loss"]))
        history["train_kl"].append(float(train_metrics["kl_divergence"]))

        # Validation (optional)
        if val_data is not None:
            val_metrics = evaluate_vae(state, val_data, batch_size, beta=beta)
            history["val_loss"].append(float(val_metrics["total_loss"]))
            history["val_recon"].append(float(val_metrics["reconstruction_loss"]))
            history["val_kl"].append(float(val_metrics["kl_divergence"]))

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            msg = f"Epoch {epoch + 1}/{n_epochs} - "
            msg += f"Loss: {train_metrics['total_loss']:.4f} "
            msg += f"(Recon: {train_metrics['reconstruction_loss']:.4f}, "
            msg += f"KL: {train_metrics['kl_divergence']:.4f})"

            if val_data is not None:
                msg += f" | Val Loss: {val_metrics['total_loss']:.4f}"

            print(msg)

    return state, history


@jax.jit
def evaluate_batch(
    state: TrainState, batch: jnp.ndarray, key: jax.random.PRNGKey, beta: float = 1.0
) -> dict:
    """
    Evaluate VAE on a single batch.

    Args:
        state: Training state
        batch: (batch_size, input_dim) batch
        key: JAX random key
        beta: Weight for KL term

    Returns:
        metrics: Dictionary with loss components
    """
    # Forward pass (no training mode, use mean)
    x_recon, info = state.apply_fn({"params": state.params}, batch, key, training=False)

    # Compute loss
    loss, metrics = elbo_loss(
        x=batch,
        x_reconstructed=x_recon,
        mu=info["mu"],
        log_var=info["log_var"],
        beta=beta,
        reconstruction_type="mse",
    )

    return metrics


def evaluate_vae(state: TrainState, data: jnp.ndarray, batch_size: int, beta: float = 1.0) -> dict:
    """
    Evaluate VAE on dataset.

    Args:
        state: Training state
        data: (n_samples, input_dim) data
        batch_size: Batch size
        beta: Weight for KL term

    Returns:
        metrics: Average metrics over dataset
    """
    n_samples = data.shape[0]
    n_batches = n_samples // batch_size

    total_metrics = {"total_loss": 0.0, "reconstruction_loss": 0.0, "kl_divergence": 0.0}

    key = jax.random.PRNGKey(0)  # Fixed key for evaluation

    for i in range(n_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        key, subkey = jax.random.split(key)
        metrics = evaluate_batch(state, batch, subkey, beta=beta)

        for k in total_metrics:
            total_metrics[k] += metrics[k]

    # Average over batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return avg_metrics
