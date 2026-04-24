"""
CausalVAE: VAE with implicit causal layer.

This integrates the causal layer between encoder and decoder:
    X → Encoder → ε → CausalLayer(ε, A) → z → Decoder → X_reconstructed
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from jcce.models.encoder import Encoder, reparameterize
from jcce.models.decoder import Decoder
from jcce.models.causal_layer import implicit_causal_layer


class CausalVAE(nn.Module):
    """
    Causal Variational Autoencoder.

    Unlike the baseline VAE which learns q(z|x) directly, CausalVAE:
    1. Encodes to exogenous noise: q(ε|x)
    2. Applies causal structure: z = (I - A)^{-1} ε
    3. Decodes from causal factors: p(x|z)

    This enforces the causal structure A_true in the latent space.

    Attributes:
        latent_dim: Dimension of latent space (number of causal factors)
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
        self,
        x: jnp.ndarray,
        A_true: jnp.ndarray,
        key: jax.random.PRNGKey,
        *,
        training: bool = True,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Forward pass through CausalVAE.

        Args:
            x: (batch_size, input_dim) observations
            A_true: (latent_dim, latent_dim) ground-truth adjacency matrix
            key: JAX random key for sampling
            training: Whether in training mode (sample) or eval mode (use mean)

        Returns:
            x_reconstructed: (batch_size, output_dim) reconstructed observations
            info: Dictionary with 'mu_epsilon', 'log_var_epsilon', 'epsilon', 'z'

        Example:
            >>> causal_vae = CausalVAE(latent_dim=10, output_dim=20)
            >>> x = jnp.ones((32, 20))
            >>> A = jnp.zeros((10, 10))  # No causal structure for example
            >>> key = jax.random.PRNGKey(0)
            >>> variables = causal_vae.init(key, x, A, key, training=True)
            >>> x_recon, info = causal_vae.apply(variables, x, A, key, training=True)
            >>> print(x_recon.shape, info['z'].shape)
            (32, 20) (32, 10)
        """
        # 1. Encode: x → (μ_ε, log_var_ε)
        #    Note: We encode to EPSILON (exogenous noise), not directly to z
        mu_epsilon, log_var_epsilon = self.encoder(x)

        # 2. Sample exogenous noise: ε ~ q(ε|x)
        if training:
            epsilon = reparameterize(mu_epsilon, log_var_epsilon, key)
        else:
            # In eval mode, use mean (deterministic)
            epsilon = mu_epsilon

        # 3. Apply causal structure: z = (I - A)^{-1} ε
        #    This is where the causal DAG structure is enforced
        z = implicit_causal_layer(epsilon, A_true)

        # 4. Decode: z → x_reconstructed
        x_reconstructed = self.decoder(z)

        # Return reconstruction and auxiliary info
        info = {
            "mu_epsilon": mu_epsilon,
            "log_var_epsilon": log_var_epsilon,
            "epsilon": epsilon,
            "z": z,
        }

        return x_reconstructed, info

    def encode(
        self, x: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode observations to exogenous noise distribution parameters.

        Args:
            x: (batch_size, input_dim) observations

        Returns:
            mu_epsilon: (batch_size, latent_dim)
            log_var_epsilon: (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode causal factors to observations.

        Args:
            z: (batch_size, latent_dim) causal factors

        Returns:
            x_reconstructed: (batch_size, output_dim)
        """
        return self.decoder(z)

    def sample_epsilon_and_get_z(
        self,
        x: jnp.ndarray,
        A_true: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode to epsilon and transform to z via causal layer.

        Useful for analysis and visualization.

        Args:
            x: (batch_size, input_dim) observations
            A_true: (latent_dim, latent_dim) adjacency matrix
            key: JAX random key

        Returns:
            epsilon: (batch_size, latent_dim) sampled exogenous noise
            z: (batch_size, latent_dim) causal factors
        """
        mu_epsilon, log_var_epsilon = self.encoder(x)
        epsilon = reparameterize(mu_epsilon, log_var_epsilon, key)
        z = implicit_causal_layer(epsilon, A_true)
        return epsilon, z


def causal_vae_elbo_loss(
    x: jnp.ndarray,
    x_reconstructed: jnp.ndarray,
    mu_epsilon: jnp.ndarray,
    log_var_epsilon: jnp.ndarray,
    beta: float = 1.0,
    reconstruction_type: str = "mse",
) -> tuple[jnp.ndarray, dict]:
    """
    Compute ELBO loss for CausalVAE.

    The loss is the same as standard VAE, but we're computing KL divergence
    for the exogenous noise distribution q(ε|x) instead of q(z|x).

    ELBO = E[log p(x|z)] - β * KL(q(ε|x) || p(ε))

    where:
    - p(ε) = N(0, I) is the prior on exogenous noise
    - q(ε|x) = N(μ_ε, σ²_ε I) is the approximate posterior
    - z = (I - A)^{-1} ε is deterministic given ε

    Args:
        x: (batch_size, input_dim) original observations
        x_reconstructed: (batch_size, input_dim) reconstructed observations
        mu_epsilon: (batch_size, latent_dim) mean of q(ε|x)
        log_var_epsilon: (batch_size, latent_dim) log variance of q(ε|x)
        beta: Weight for KL term (β-VAE), default=1.0
        reconstruction_type: 'mse' or 'bce'

    Returns:
        loss: Scalar total loss
        metrics: Dictionary with 'reconstruction_loss' and 'kl_divergence'

    Example:
        >>> x = jnp.ones((32, 20))
        >>> x_recon = jnp.ones((32, 20)) * 0.9
        >>> mu_eps = jnp.zeros((32, 10))
        >>> log_var_eps = jnp.zeros((32, 10))
        >>> loss, metrics = causal_vae_elbo_loss(x, x_recon, mu_eps, log_var_eps)
        >>> print(loss.shape)
        ()
    """
    # Import here to avoid circular dependency
    from jcce.models.vae import reconstruction_loss, kl_divergence

    # Reconstruction loss
    recon_loss = reconstruction_loss(x, x_reconstructed, loss_type=reconstruction_type)

    # KL divergence: KL(q(ε|x) || p(ε)) where p(ε) = N(0, I)
    # This is exactly the same formula as for standard VAE
    kl_loss = kl_divergence(mu_epsilon, log_var_epsilon)

    total_loss = recon_loss + beta * kl_loss

    metrics = {
        "reconstruction_loss": recon_loss,
        "kl_divergence": kl_loss,
        "total_loss": total_loss,
    }

    return total_loss, metrics
