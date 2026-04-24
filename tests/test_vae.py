"""
Tests for VAE components (encoder, decoder, training).

Run with: uv run pytest tests/test_vae.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from jcce.data.synthetic_dataset import create_simple_dataset
from jcce.models.decoder import Decoder
from jcce.models.encoder import Encoder, reparameterize
from jcce.models.vae import (
    BaselineVAE,
    elbo_loss,
    kl_divergence,
    reconstruction_loss,
)
from jcce.training import create_train_state, train_step, train_vae


class TestEncoder:
    """Tests for Encoder network."""

    def test_encoder_output_shape(self):
        """Verify encoder outputs correct shapes."""
        encoder = Encoder(latent_dim=10, hidden_dims=(128, 64))
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))  # batch_size=32, input_dim=20

        variables = encoder.init(key, x)
        mu, log_var = encoder.apply(variables, x)

        assert mu.shape == (32, 10), "mu should be (batch_size, latent_dim)"
        assert log_var.shape == (32, 10), "log_var should be (batch_size, latent_dim)"

    def test_encoder_deterministic(self):
        """Verify same input produces same output."""
        encoder = Encoder(latent_dim=10, hidden_dims=(128, 64))
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (16, 20))

        variables = encoder.init(key, x)
        mu1, log_var1 = encoder.apply(variables, x)
        mu2, log_var2 = encoder.apply(variables, x)

        assert jnp.allclose(mu1, mu2), "Encoder should be deterministic"
        assert jnp.allclose(log_var1, log_var2), "Encoder should be deterministic"


class TestDecoder:
    """Tests for Decoder network."""

    def test_decoder_output_shape(self):
        """Verify decoder outputs correct shape."""
        decoder = Decoder(output_dim=20, hidden_dims=(64, 128))
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (32, 10))  # batch_size=32, latent_dim=10

        variables = decoder.init(key, z)
        x_recon = decoder.apply(variables, z)

        assert x_recon.shape == (32, 20), "Output should be (batch_size, output_dim)"

    def test_decoder_deterministic(self):
        """Verify same input produces same output."""
        decoder = Decoder(output_dim=20, hidden_dims=(64, 128))
        key = jax.random.PRNGKey(42)
        z = jax.random.normal(key, (16, 10))

        variables = decoder.init(key, z)
        x_recon1 = decoder.apply(variables, z)
        x_recon2 = decoder.apply(variables, z)

        assert jnp.allclose(x_recon1, x_recon2), "Decoder should be deterministic"


class TestReparameterization:
    """Tests for reparameterization trick."""

    def test_reparameterize_shape(self):
        """Verify reparameterize outputs correct shape."""
        mu = jnp.zeros((32, 10))
        log_var = jnp.zeros((32, 10))
        key = jax.random.PRNGKey(0)

        z = reparameterize(mu, log_var, key)

        assert z.shape == (32, 10), "Output should match input shape"

    def test_reparameterize_standard_normal(self):
        """Verify reparameterize with μ=0, σ=1 gives standard normal samples."""
        mu = jnp.zeros((1000, 10))
        log_var = jnp.zeros((1000, 10))  # log(1) = 0, so σ = 1
        key = jax.random.PRNGKey(42)

        z = reparameterize(mu, log_var, key)

        # Check empirical mean and std
        empirical_mean = jnp.mean(z, axis=0)
        empirical_std = jnp.std(z, axis=0)

        assert jnp.allclose(empirical_mean, 0.0, atol=0.1), "Mean should be ~0"
        assert jnp.allclose(empirical_std, 1.0, atol=0.1), "Std should be ~1"

    def test_reparameterize_different_seeds(self):
        """Verify different seeds produce different samples."""
        mu = jnp.zeros((32, 10))
        log_var = jnp.zeros((32, 10))

        z1 = reparameterize(mu, log_var, jax.random.PRNGKey(0))
        z2 = reparameterize(mu, log_var, jax.random.PRNGKey(1))

        assert not jnp.allclose(z1, z2), "Different seeds should give different samples"


class TestLossFunctions:
    """Tests for VAE loss functions."""

    def test_reconstruction_loss_mse(self):
        """Verify MSE reconstruction loss."""
        x = jnp.ones((32, 20))
        x_recon = jnp.ones((32, 20)) * 0.9

        loss = reconstruction_loss(x, x_recon, loss_type="mse")

        expected_loss = jnp.mean(jnp.sum((x - x_recon) ** 2, axis=-1))
        assert jnp.allclose(loss, expected_loss), "MSE loss should match formula"

    def test_reconstruction_loss_perfect(self):
        """Verify zero loss for perfect reconstruction."""
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 20))
        loss = reconstruction_loss(x, x, loss_type="mse")

        assert jnp.abs(loss) < 1e-6, "Perfect reconstruction should have zero loss"

    def test_kl_divergence_standard_normal(self):
        """Verify KL divergence is zero for N(0,1)."""
        mu = jnp.zeros((32, 10))
        log_var = jnp.zeros((32, 10))  # log(1) = 0

        kl = kl_divergence(mu, log_var)

        assert jnp.abs(kl) < 1e-5, "KL(N(0,1) || N(0,1)) should be zero"

    def test_kl_divergence_positive(self):
        """Verify KL divergence is non-negative."""
        key = jax.random.PRNGKey(42)
        mu = jax.random.normal(key, (32, 10))
        log_var = jax.random.normal(key, (32, 10))

        kl = kl_divergence(mu, log_var)

        assert kl >= 0.0, "KL divergence should always be non-negative"

    def test_elbo_loss(self):
        """Verify ELBO loss computation."""
        x = jnp.ones((32, 20))
        x_recon = jnp.ones((32, 20)) * 0.9
        mu = jnp.zeros((32, 10))
        log_var = jnp.zeros((32, 10))

        loss, metrics = elbo_loss(x, x_recon, mu, log_var, beta=1.0)

        assert "reconstruction_loss" in metrics
        assert "kl_divergence" in metrics
        assert "total_loss" in metrics
        assert jnp.allclose(loss, metrics["total_loss"]), "Loss should match total_loss in metrics"


class TestBaselineVAE:
    """Tests for complete VAE model."""

    def test_vae_forward_pass(self):
        """Verify VAE forward pass."""
        vae = BaselineVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))

        key_init, key_forward = jax.random.split(key)
        variables = vae.init(key_init, x, key_forward, training=True)

        x_recon, info = vae.apply(variables, x, key_forward, training=True)

        assert x_recon.shape == (32, 20), "Reconstruction should match input shape"
        assert "mu" in info and "log_var" in info and "z" in info
        assert info["z"].shape == (32, 10), "Latent should be (batch, latent_dim)"

    def test_vae_encode_decode(self):
        """Verify encode and decode methods."""
        vae = BaselineVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))

        key_init, key_forward = jax.random.split(key)
        variables = vae.init(key_init, x, key_forward, training=True)

        # Encode
        mu, log_var = vae.apply(variables, x, method=vae.encode)
        assert mu.shape == (32, 10)
        assert log_var.shape == (32, 10)

        # Decode
        z = jax.random.normal(key, (32, 10))
        x_recon = vae.apply(variables, z, method=vae.decode)
        assert x_recon.shape == (32, 20)

    def test_vae_training_mode(self):
        """Verify training vs eval mode differences."""
        vae = BaselineVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (32, 20))

        key_init, key_forward = jax.random.split(key)
        variables = vae.init(key_init, x, key_forward, training=True)

        # Training mode - uses sampling
        x_recon1, _ = vae.apply(variables, x, key_forward, training=True)

        # Eval mode - uses mean (deterministic)
        x_recon2, _ = vae.apply(variables, x, key_forward, training=False)
        x_recon3, _ = vae.apply(variables, x, key_forward, training=False)

        # Eval mode should be deterministic
        assert jnp.allclose(x_recon2, x_recon3), "Eval mode should be deterministic"


class TestVAETraining:
    """Tests for VAE training."""

    def test_create_train_state(self):
        """Verify training state creation."""
        model = BaselineVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)

        state = create_train_state(model, learning_rate=1e-3, key=key, input_shape=(32, 20))

        assert state.params is not None
        assert state.tx is not None
        assert state.step == 0

    def test_train_step(self):
        """Verify single training step."""
        model = BaselineVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)

        state = create_train_state(model, learning_rate=1e-3, key=key, input_shape=(32, 20))

        # Generate a batch
        batch = jax.random.normal(key, (32, 20))

        # Training step
        new_state, metrics = train_step(state, batch, beta=1.0)

        assert new_state.step == 1, "Step should increment"
        assert "total_loss" in metrics
        assert metrics["total_loss"] >= 0.0, "Loss should be non-negative"

    def test_train_vae_reduces_loss(self):
        """Verify training reduces loss over time."""
        # Create simple synthetic data
        X, A = create_simple_dataset(num_nodes=10, n_samples=500, graph_type="chain")

        # Create VAE
        model = BaselineVAE(latent_dim=10, output_dim=10)

        # Train for a few epochs
        key = jax.random.PRNGKey(42)
        state, history = train_vae(
            model,
            train_data=X,
            val_data=None,
            n_epochs=20,
            batch_size=32,
            learning_rate=1e-3,
            key=key,
            verbose=False,
        )

        # Check that loss decreased
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]

        assert final_loss < initial_loss, "Training should reduce loss"
        assert len(history["train_loss"]) == 20, "Should have 20 epochs of history"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
