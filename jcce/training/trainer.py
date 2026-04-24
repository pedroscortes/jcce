"""
Training utilities for CausalVAE experiments.

This module provides training loops and optimization for all 4 experimental variants.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Any, Optional
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Extended train state with batch statistics."""
    key: jax.random.PRNGKey


def create_train_state(
    model,
    key: jax.random.PRNGKey,
    learning_rate: float,
    x_sample: jnp.ndarray,
    A: jnp.ndarray,
    topo_order: jnp.ndarray,
    warmup_epochs: int = 0,
    total_epochs: int = 100,
    max_grad_norm: float = 1.0,
) -> TrainState:
    """
    Create initial training state with advanced optimizer.

    Args:
        model: UnifiedCausalVAE instance
        key: Random key for initialization
        learning_rate: Peak learning rate
        x_sample: Sample input for initialization
        A: Adjacency matrix
        topo_order: Topological ordering
        warmup_epochs: Number of warmup epochs (0 = no warmup)
        total_epochs: Total training epochs (for cosine schedule)
        max_grad_norm: Maximum gradient norm for clipping (0 = no clipping)

    Returns:
        TrainState with initialized parameters and optimizer
    """
    key_init, key_forward = jax.random.split(key)

    # Initialize model parameters
    variables = model.init(key_init, x_sample, A, topo_order, key_forward, training=True)

    # Create optimizer with gradient clipping and optional scheduling
    optimizer_chain = []

    # 1. Gradient clipping (prevents exploding gradients)
    if max_grad_norm > 0:
        optimizer_chain.append(optax.clip_by_global_norm(max_grad_norm))

    # 2. Adam optimizer
    if warmup_epochs > 0:
        # Learning rate schedule with warmup + cosine decay
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_epochs,
            decay_steps=total_epochs,
            end_value=learning_rate * 0.01,  # Decay to 1% of peak
        )
        optimizer_chain.append(optax.adam(learning_rate=schedule))
    else:
        # Constant learning rate
        optimizer_chain.append(optax.adam(learning_rate))

    # Chain optimizers
    tx = optax.chain(*optimizer_chain)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        key=key_forward
    )


def kl_divergence_with_free_bits(
    mu: jnp.ndarray,
    log_var: jnp.ndarray,
    free_bits_lambda: float = 0.0,
) -> jnp.ndarray:
    """
    Compute KL divergence with optional free bits constraint.

    Free bits prevents posterior collapse by enforcing a minimum KL per dimension.
    This is a standard technique to stabilize VAE training.

    Mathematical formulation:
        kl_per_dim[i] = -0.5 * sum_batch(1 + log_var[:,i] - mu[:,i]^2 - exp(log_var[:,i]))
        kl_clamped[i] = max(kl_per_dim[i], free_bits_lambda)
        kl_total = sum_dims(kl_clamped)

    Args:
        mu: (batch, latent_dim) - Mean of q(ε|X)
        log_var: (batch, latent_dim) - Log variance of q(ε|X)
        free_bits_lambda: Minimum KL per dimension (0.0 = disabled)

    Returns:
        kl_loss: Scalar KL divergence

    References:
        Kingma et al. "Improved Variational Inference with Inverse Autoregressive Flow" (2016)

    Example:
        >>> mu = jnp.zeros((32, 10))
        >>> log_var = jnp.ones((32, 10)) * -10  # Collapsed variance
        >>> kl_standard = -0.5 * jnp.sum(1 + log_var - mu**2 - jnp.exp(log_var))
        >>> print(f"Standard KL: {kl_standard:.6f}")  # Very small (collapsed)
        >>> kl_free_bits = kl_divergence_with_free_bits(mu, log_var, free_bits_lambda=0.5)
        >>> print(f"Free bits KL: {kl_free_bits:.6f}")  # >= 0.5 * 10 = 5.0
    """
    # Standard KL divergence per sample
    kl_per_sample = -0.5 * jnp.sum(
        1 + log_var - mu**2 - jnp.exp(log_var),
        axis=-1
    )
    kl_standard = jnp.mean(kl_per_sample)

    # Free bits: enforce minimum KL per dimension
    # KL per dimension (sum over batch, keep dimensions separate)
    kl_per_dim = -0.5 * jnp.sum(
        1 + log_var - mu**2 - jnp.exp(log_var),
        axis=0  # Sum over batch dimension
    )

    # Apply free bits constraint
    # This says: "Don't penalize KL below free_bits_lambda per dimension"
    kl_per_dim_clamped = jnp.maximum(kl_per_dim, free_bits_lambda)
    kl_free_bits = jnp.sum(kl_per_dim_clamped)

    # Return free bits version if lambda > 0, otherwise standard
    # Use where instead of if to keep it JAX-friendly
    return jnp.where(free_bits_lambda > 0.0, kl_free_bits, kl_standard)


def compute_vae_loss(
    params: Dict,
    apply_fn,
    x: jnp.ndarray,
    A: jnp.ndarray,
    topo_order: jnp.ndarray,
    key: jax.random.PRNGKey,
    beta: float = 1.0,
    free_bits_lambda: float = 0.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute VAE loss (reconstruction + KL divergence).

    Loss = E[||x - x_recon||²] + β * KL(q(ε|x) || p(ε))

    where:
    - Reconstruction: MSE between input and reconstruction
    - KL: KL divergence between posterior and N(0,I) prior (with optional free bits)
    - β: Weight for KL term (β-VAE)

    Args:
        params: Model parameters
        apply_fn: Model apply function
        x: Input observations (batch_size, output_dim)
        A: Adjacency matrix (latent_dim, latent_dim)
        topo_order: Topological ordering (latent_dim,)
        key: Random key for sampling
        beta: Weight for KL term (default=1.0)
        free_bits_lambda: Free bits constraint (0.0 = disabled)

    Returns:
        loss: Total loss
        metrics: Dictionary with loss components

    Example:
        >>> # Standard VAE loss
        >>> loss, metrics = compute_vae_loss(params, apply_fn, x, A, topo, key, beta=1.0)
        >>>
        >>> # With free bits to prevent collapse
        >>> loss, metrics = compute_vae_loss(params, apply_fn, x, A, topo, key,
        ...                                  beta=1.0, free_bits_lambda=0.5)
    """
    # Forward pass
    x_recon, info = apply_fn(
        {'params': params},
        x, A, topo_order, key,
        training=True
    )

    # Reconstruction loss (MSE)
    recon_loss = jnp.mean((x - x_recon) ** 2)

    # KL divergence with optional free bits
    mu_epsilon = info["mu_epsilon"]
    log_var_epsilon = info["log_var_epsilon"]

    kl_div = kl_divergence_with_free_bits(
        mu_epsilon, log_var_epsilon, free_bits_lambda
    )

    # Total loss
    loss = recon_loss + beta * kl_div

    metrics = {
        "loss": loss,
        "recon_loss": recon_loss,
        "kl_div": kl_div,
        "beta": beta,
    }

    return loss, metrics


@jax.jit
def train_step(
    state: TrainState,
    x: jnp.ndarray,
    A: jnp.ndarray,
    topo_order: jnp.ndarray,
    beta: float = 1.0,
    free_bits_lambda: float = 0.0,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Single training step.

    Args:
        state: Current training state
        x: Batch of observations
        A: Adjacency matrix
        topo_order: Topological ordering
        beta: KL weight
        free_bits_lambda: Free bits constraint (0.0 = disabled)

    Returns:
        new_state: Updated training state
        metrics: Training metrics
    """
    # Split key for this step
    key, new_key = jax.random.split(state.key)

    # Compute loss and gradients
    (loss, metrics), grads = jax.value_and_grad(compute_vae_loss, has_aux=True)(
        state.params,
        state.apply_fn,
        x, A, topo_order, key, beta, free_bits_lambda
    )

    # Update parameters
    new_state = state.apply_gradients(grads=grads, key=new_key)

    return new_state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    x: jnp.ndarray,
    A: jnp.ndarray,
    topo_order: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> Dict[str, jnp.ndarray]:
    """
    Single evaluation step.

    Args:
        state: Training state
        x: Batch of observations
        A: Adjacency matrix
        topo_order: Topological ordering
        key: Random key

    Returns:
        metrics: Evaluation metrics
    """
    # Forward pass in eval mode (no dropout, deterministic)
    x_recon, info = state.apply_fn(
        {'params': state.params},
        x, A, topo_order, key,
        training=False
    )

    # Compute metrics
    recon_loss = jnp.mean((x - x_recon) ** 2)

    mu_epsilon = info["mu_epsilon"]
    log_var_epsilon = info["log_var_epsilon"]

    kl_div = -0.5 * jnp.sum(
        1 + log_var_epsilon - mu_epsilon**2 - jnp.exp(log_var_epsilon),
        axis=-1
    )
    kl_div = jnp.mean(kl_div)

    loss = recon_loss + kl_div

    return {
        "loss": loss,
        "recon_loss": recon_loss,
        "kl_div": kl_div,
    }


def get_beta_schedule(epoch: int, total_epochs: int, beta_final: float = 1.0,
                      warmup_epochs: int = 0, beta_start: float = 0.0) -> float:
    """
    Compute beta (KL weight) with optional annealing to prevent posterior collapse.

    Posterior collapse is a common problem where the decoder ignores the latent code
    and KL divergence goes to zero. This function implements LINEAR warmup from
    beta_start to beta_final over warmup_epochs.

    Key insight from review: SLOW warmup (50 epochs, not 10-20) allows model to
    learn reconstruction first, then gradually adds disentanglement pressure.

    Combined with Free Bits, this dual defense prevents posterior collapse:
    - Free Bits: enforces minimum KL per dimension (hard constraint)
    - β annealing: gradually increases KL pressure (soft warmup)

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs (unused, kept for compatibility)
        beta_final: Final β value (default 1.0 for standard VAE, >1 for β-VAE)
        warmup_epochs: Number of epochs to anneal β (0 = no annealing, constant β)
        beta_start: Initial β value (default 0.0 when using Free Bits)

    Returns:
        beta: KL weight for this epoch

    Example:
        >>> # Slow warmup (recommended): 0.0 → 1.0 over 50 epochs
        >>> get_beta_schedule(0, 100, beta_final=1.0, warmup_epochs=50, beta_start=0.0)
        0.0
        >>> get_beta_schedule(25, 100, beta_final=1.0, warmup_epochs=50, beta_start=0.0)
        0.5
        >>> get_beta_schedule(50, 100, beta_final=1.0, warmup_epochs=50, beta_start=0.0)
        1.0
        >>> get_beta_schedule(100, 100, beta_final=1.0, warmup_epochs=50, beta_start=0.0)
        1.0

    Note:
        Starting at β=0.0 is now SAFE when using Free Bits (free_bits_lambda > 0).
        Free Bits enforces minimum KL, preventing immediate collapse even with β=0.
    """
    if warmup_epochs == 0:
        # No annealing, constant β
        return beta_final
    else:
        # Linear annealing: beta_start → beta_final over warmup_epochs
        progress = min(1.0, epoch / warmup_epochs)
        return beta_start + (beta_final - beta_start) * progress


def train_epoch(
    state: TrainState,
    train_data: jnp.ndarray,
    A: jnp.ndarray,
    topo_order: jnp.ndarray,
    batch_size: int,
    beta: float = 1.0,
    free_bits_lambda: float = 0.0,
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        state: Training state
        train_data: Full training dataset (n_samples, output_dim)
        A: Adjacency matrix
        topo_order: Topological ordering
        batch_size: Batch size
        beta: KL weight for this epoch (can be annealed)
        free_bits_lambda: Free bits constraint (0.0 = disabled)

    Returns:
        state: Updated training state
        metrics: Average metrics over epoch
    """
    n_samples = train_data.shape[0]
    n_batches = n_samples // batch_size

    # Shuffle data
    key, shuffle_key = jax.random.split(state.key)
    perm = jax.random.permutation(shuffle_key, n_samples)
    train_data = train_data[perm]

    # Update state key
    state = state.replace(key=key)

    # Accumulate metrics
    epoch_metrics = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_div": 0.0,
    }

    for i in range(n_batches):
        batch = train_data[i * batch_size:(i + 1) * batch_size]
        state, batch_metrics = train_step(
            state, batch, A, topo_order, beta, free_bits_lambda
        )

        # Accumulate
        for key in epoch_metrics:
            epoch_metrics[key] += float(batch_metrics[key])

    # Average over batches
    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches

    return state, epoch_metrics


def evaluate(
    state: TrainState,
    eval_data: jnp.ndarray,
    A: jnp.ndarray,
    topo_order: jnp.ndarray,
    batch_size: int,
) -> Dict[str, float]:
    """
    Evaluate on dataset.

    Args:
        state: Training state
        eval_data: Evaluation dataset
        A: Adjacency matrix
        topo_order: Topological ordering
        batch_size: Batch size

    Returns:
        metrics: Average metrics over dataset
    """
    n_samples = eval_data.shape[0]
    n_batches = n_samples // batch_size

    # Accumulate metrics
    eval_metrics = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl_div": 0.0,
    }

    key = jax.random.PRNGKey(0)  # Fixed key for eval

    for i in range(n_batches):
        batch = eval_data[i * batch_size:(i + 1) * batch_size]
        batch_metrics = eval_step(state, batch, A, topo_order, key)

        # Accumulate
        for key_name in eval_metrics:
            eval_metrics[key_name] += float(batch_metrics[key_name])

    # Average over batches
    for key_name in eval_metrics:
        eval_metrics[key_name] /= n_batches

    return eval_metrics
