"""
VAE Decoder network.

Maps latent variables z to reconstructed observations p(x|z).
"""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


class Decoder(nn.Module):
    """
    VAE Decoder: z → reconstructed X

    Maps latent variables (either z directly for baseline VAE, or z from causal layer
    for CausalVAE) back to observation space.

    Attributes:
        output_dim: Dimension of output (observation space)
        hidden_dims: Sequence of hidden layer dimensions (reversed from encoder)
    """

    output_dim: int
    hidden_dims: Sequence[int] = (64, 128)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent variables to reconstructed observations.

        Args:
            z: (batch_size, latent_dim) latent variables

        Returns:
            x_reconstructed: (batch_size, output_dim) reconstructed observations

        Example:
            >>> decoder = Decoder(output_dim=20, hidden_dims=(64, 128))
            >>> z = jnp.ones((32, 10))  # batch_size=32, latent_dim=10
            >>> variables = decoder.init(jax.random.PRNGKey(0), z)
            >>> x_recon = decoder.apply(variables, z)
            >>> print(x_recon.shape)
            (32, 20)
        """
        h = z

        # Hidden layers with ReLU activation
        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = nn.relu(h)

        # Output layer (linear, no activation for continuous data)
        # Note: For binary data, you might want sigmoid activation
        x_reconstructed = nn.Dense(self.output_dim)(h)

        return x_reconstructed
