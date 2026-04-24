"""
VAE Encoder network.

Maps observations X to parameters of the approximate posterior q(ε|X).
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    """
    VAE Encoder: X → parameters of q(ε|X)

    For the baseline VAE, this outputs parameters for the latent distribution.
    For CausalVAE, this will output parameters for EPSILON (exogenous noise).

    Attributes:
        latent_dim: Dimension of latent space
        hidden_dims: Sequence of hidden layer dimensions
    """

    latent_dim: int
    hidden_dims: Sequence[int] = (128, 64)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode observations to latent distribution parameters.

        Args:
            x: (batch_size, input_dim) observations

        Returns:
            mu: (batch_size, latent_dim) mean of latent distribution
            log_var: (batch_size, latent_dim) log variance of latent distribution

        Example:
            >>> encoder = Encoder(latent_dim=10, hidden_dims=(128, 64))
            >>> x = jnp.ones((32, 20))  # batch_size=32, input_dim=20
            >>> variables = encoder.init(jax.random.PRNGKey(0), x)
            >>> mu, log_var = encoder.apply(variables, x)
            >>> print(mu.shape, log_var.shape)
            (32, 10) (32, 10)
        """
        h = x

        # Hidden layers with ReLU activation
        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = nn.relu(h)

        # Output layers (linear, no activation)
        mu = nn.Dense(self.latent_dim)(h)
        log_var = nn.Dense(self.latent_dim)(h)

        return mu, log_var


def reparameterize(mu: jnp.ndarray, log_var: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    """
    Reparameterization trick for VAE.

    Samples from q(z|x) = N(μ, σ²I) using:
        z = μ + σ ⊙ ε, where ε ~ N(0, I)

    This allows gradients to flow through the sampling operation.

    Args:
        mu: (batch_size, latent_dim) mean
        log_var: (batch_size, latent_dim) log variance
        key: JAX random key

    Returns:
        z: (batch_size, latent_dim) sampled latent variables

    Example:
        >>> mu = jnp.zeros((32, 10))
        >>> log_var = jnp.zeros((32, 10))  # var = 1
        >>> key = jax.random.PRNGKey(0)
        >>> z = reparameterize(mu, log_var, key)
        >>> print(z.shape)
        (32, 10)
    """
    std = jnp.exp(0.5 * log_var)
    eps = jax.random.normal(key, mu.shape)
    return mu + std * eps
