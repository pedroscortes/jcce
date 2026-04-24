"""
Baseline VAE implementation.

Combines encoder and decoder with ELBO loss.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from jcce.models.encoder import Encoder, reparameterize
from jcce.models.decoder import Decoder


class BaselineVAE(nn.Module):
    """
    Standard Variational Autoencoder (Baseline model).

    This is the baseline for comparison - a simple VAE with factorized
    Gaussian prior p(z) = N(0, I) and no causal structure.

    Attributes:
        latent_dim: Dimension of latent space
        output_dim: Dimension of output (observation space)
        encoder_hidden_dims: Hidden layer sizes for encoder
        decoder_hidden_dims: Hidden layer sizes for decoder
    """

    latent_dim: int
    output_dim: int
    encoder_hidden_dims: Sequence[int] = (128, 64)
    decoder_hidden_dims: Sequence[int] = (64, 128)

    def setup(self):
        """Initialize encoder and decoder."""
        self.encoder = Encoder(
            latent_dim=self.latent_dim, hidden_dims=self.encoder_hidden_dims
        )
        self.decoder = Decoder(
            output_dim=self.output_dim, hidden_dims=self.decoder_hidden_dims
        )

    def __call__(
        self, x: jnp.ndarray, key: jax.random.PRNGKey, *, training: bool = True
    ) -> tuple[jnp.ndarray, dict]:
        """
        Forward pass through VAE.

        Args:
            x: (batch_size, input_dim) observations
            key: JAX random key for sampling
            training: Whether in training mode (always sample) or eval mode

        Returns:
            x_reconstructed: (batch_size, output_dim) reconstructed observations
            info: Dictionary with 'mu', 'log_var', 'z' for analysis

        Example:
            >>> vae = BaselineVAE(latent_dim=10, output_dim=20)
            >>> x = jnp.ones((32, 20))
            >>> key = jax.random.PRNGKey(0)
            >>> variables = vae.init(key, x, key, training=True)
            >>> x_recon, info = vae.apply(variables, x, key, training=True)
            >>> print(x_recon.shape, info['z'].shape)
            (32, 20) (32, 10)
        """
        # Encode: x → (μ, log_var)
        mu, log_var = self.encoder(x)

        # Sample: z ~ q(z|x) using reparameterization trick
        if training:
            z = reparameterize(mu, log_var, key)
        else:
            # In eval mode, use mean (deterministic)
            z = mu

        # Decode: z → x_reconstructed
        x_reconstructed = self.decoder(z)

        # Return reconstruction and auxiliary info
        info = {"mu": mu, "log_var": log_var, "z": z}

        return x_reconstructed, info

    def encode(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode observations to latent distribution parameters.

        Args:
            x: (batch_size, input_dim) observations

        Returns:
            mu: (batch_size, latent_dim)
            log_var: (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent variables to observations.

        Args:
            z: (batch_size, latent_dim) latent variables

        Returns:
            x_reconstructed: (batch_size, output_dim)
        """
        return self.decoder(z)


def reconstruction_loss(
    x: jnp.ndarray, x_reconstructed: jnp.ndarray, loss_type: str = "mse"
) -> jnp.ndarray:
    """
    Compute reconstruction loss.

    Args:
        x: (batch_size, input_dim) original observations
        x_reconstructed: (batch_size, input_dim) reconstructed observations
        loss_type: 'mse' for continuous data, 'bce' for binary data

    Returns:
        loss: Scalar reconstruction loss (averaged over batch)

    Example:
        >>> x = jnp.ones((32, 20))
        >>> x_recon = jnp.ones((32, 20)) * 0.9
        >>> loss = reconstruction_loss(x, x_recon, loss_type='mse')
        >>> print(loss.shape)
        ()
    """
    if loss_type == "mse":
        # Mean Squared Error (for continuous data)
        # Sum over features, mean over batch
        return jnp.mean(jnp.sum((x - x_reconstructed) ** 2, axis=-1))
    elif loss_type == "bce":
        # Binary Cross-Entropy (for binary data)
        # Assumes x_reconstructed are logits
        # Sum over features, mean over batch
        return jnp.mean(
            jnp.sum(
                jax.nn.sigmoid_binary_cross_entropy(x_reconstructed, x), axis=-1
            )
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def kl_divergence(mu: jnp.ndarray, log_var: jnp.ndarray) -> jnp.ndarray:
    """
    Compute KL divergence KL(q(z|x) || p(z)) where p(z) = N(0, I).

    Using the analytical form:
        KL(N(μ, σ²) || N(0, I)) = 0.5 * Σ(σ² + μ² - 1 - log(σ²))

    Args:
        mu: (batch_size, latent_dim) mean of q(z|x)
        log_var: (batch_size, latent_dim) log variance of q(z|x)

    Returns:
        kl: Scalar KL divergence (averaged over batch)

    Example:
        >>> mu = jnp.zeros((32, 10))
        >>> log_var = jnp.zeros((32, 10))  # N(0, 1) should have KL = 0
        >>> kl = kl_divergence(mu, log_var)
        >>> print(jnp.abs(kl) < 1e-5)
        True
    """
    # KL(q || p) = 0.5 * Σ(exp(log_var) + mu^2 - 1 - log_var)
    # Sum over latent dimensions, mean over batch
    kl = 0.5 * jnp.sum(jnp.exp(log_var) + mu**2 - 1.0 - log_var, axis=-1)
    return jnp.mean(kl)


def elbo_loss(
    x: jnp.ndarray,
    x_reconstructed: jnp.ndarray,
    mu: jnp.ndarray,
    log_var: jnp.ndarray,
    beta: float = 1.0,
    reconstruction_type: str = "mse",
) -> tuple[jnp.ndarray, dict]:
    """
    Compute Evidence Lower Bound (ELBO) loss for VAE.

    ELBO = E[log p(x|z)] - β * KL(q(z|x) || p(z))

    We minimize the negative ELBO:
        Loss = Reconstruction Loss + β * KL Divergence

    Args:
        x: (batch_size, input_dim) original observations
        x_reconstructed: (batch_size, input_dim) reconstructed observations
        mu: (batch_size, latent_dim) mean of q(z|x)
        log_var: (batch_size, latent_dim) log variance of q(z|x)
        beta: Weight for KL term (β-VAE), default=1.0 for standard VAE
        reconstruction_type: 'mse' or 'bce'

    Returns:
        loss: Scalar total loss
        metrics: Dictionary with 'reconstruction_loss' and 'kl_divergence'

    Example:
        >>> x = jnp.ones((32, 20))
        >>> x_recon = jnp.ones((32, 20)) * 0.9
        >>> mu = jnp.zeros((32, 10))
        >>> log_var = jnp.zeros((32, 10))
        >>> loss, metrics = elbo_loss(x, x_recon, mu, log_var)
        >>> print(loss.shape)
        ()
    """
    recon_loss = reconstruction_loss(x, x_reconstructed, loss_type=reconstruction_type)
    kl_loss = kl_divergence(mu, log_var)

    total_loss = recon_loss + beta * kl_loss

    metrics = {
        "reconstruction_loss": recon_loss,
        "kl_divergence": kl_loss,
        "total_loss": total_loss,
    }

    return total_loss, metrics
