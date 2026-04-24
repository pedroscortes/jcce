"""
Epsilon Projection layer for Architecture v2.

Projects processor output (or encoder output if no processor) to epsilon parameters.
This is where the KL gradients enter the processing pipeline in v2.
"""

import jax.numpy as jnp
from flax import linen as nn


class EpsilonProjection(nn.Module):
    """
    Projects features to epsilon distribution parameters.

    In Architecture v2, this sits AFTER the processor, so it receives gradients
    from BOTH reconstruction loss (via z) and KL loss (directly on mu_epsilon, log_var_epsilon).

    h_proc → (μ_ε, log_var_ε)

    Attributes:
        latent_dim: Dimension of latent space (and epsilon)
        input_dim: Dimension of input features (processor output or encoder output)
    """

    latent_dim: int
    input_dim: int

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Project features to epsilon parameters.

        Args:
            h: (batch_size, input_dim) features from processor or encoder

        Returns:
            mu_epsilon: (batch_size, latent_dim) mean of epsilon distribution
            log_var_epsilon: (batch_size, latent_dim) log variance of epsilon distribution

        Example:
            >>> proj = EpsilonProjection(latent_dim=10, input_dim=128)
            >>> h = jnp.ones((32, 128))
            >>> variables = proj.init(jax.random.PRNGKey(0), h)
            >>> mu_eps, log_var_eps = proj.apply(variables, h)
            >>> print(mu_eps.shape, log_var_eps.shape)
            (32, 10) (32, 10)
        """
        # Project to epsilon parameters (linear, no activation)
        mu_epsilon = nn.Dense(self.latent_dim)(h)
        log_var_epsilon = nn.Dense(self.latent_dim)(h)

        return mu_epsilon, log_var_epsilon
