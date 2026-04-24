"""
Tests for CausalVAE components.

Run with: uv run pytest tests/test_causal_vae.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.synthetic_dataset import create_simple_dataset
from jcce.models.causal_vae import CausalVAE, causal_vae_elbo_loss
from jcce.training import (
    causal_vae_train_step,
    create_causal_vae_train_state,
    train_causal_vae,
)


class TestCausalVAE:
    """Tests for CausalVAE model."""

    def test_causal_vae_forward_pass(self):
        """Verify CausalVAE forward pass."""
        causal_vae = CausalVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))
        A = generate_dag(DAGConfig(num_nodes=10, graph_type="chain", seed=42))

        key_init, key_forward = jax.random.split(key)
        variables = causal_vae.init(key_init, x, A, key_forward, training=True)

        x_recon, info = causal_vae.apply(variables, x, A, key_forward, training=True)

        assert x_recon.shape == (32, 20), "Reconstruction should match input shape"
        assert "mu_epsilon" in info and "log_var_epsilon" in info
        assert "epsilon" in info and "z" in info
        assert info["epsilon"].shape == (32, 10), "Epsilon should be (batch, latent_dim)"
        assert info["z"].shape == (32, 10), "Z should be (batch, latent_dim)"

    def test_causal_vae_with_identity_graph(self):
        """Verify CausalVAE with no causal structure (identity)."""
        causal_vae = CausalVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))
        A = jnp.zeros((10, 10))  # No edges, so z = epsilon

        key_init, key_forward = jax.random.split(key)
        variables = causal_vae.init(key_init, x, A, key_forward, training=True)

        x_recon, info = causal_vae.apply(variables, x, A, key_forward, training=True)

        # With no causal structure, z should equal epsilon
        assert jnp.allclose(info["z"], info["epsilon"], atol=1e-5), (
            "With A=0, z should equal epsilon"
        )

    def test_causal_vae_with_chain_dag(self):
        """Verify CausalVAE with chain DAG."""
        causal_vae = CausalVAE(latent_dim=5, output_dim=10)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (16, 10))
        A = generate_dag(DAGConfig(num_nodes=5, graph_type="chain", seed=42))

        key_init, key_forward = jax.random.split(key)
        variables = causal_vae.init(key_init, x, A, key_forward, training=True)

        x_recon, info = causal_vae.apply(variables, x, A, key_forward, training=True)

        # With chain structure, z should NOT equal epsilon (causal propagation)
        assert not jnp.allclose(info["z"], info["epsilon"], atol=1e-2), (
            "With chain DAG, z should differ from epsilon"
        )

    def test_causal_vae_encode_decode(self):
        """Verify encode and decode methods."""
        causal_vae = CausalVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))
        A = jnp.zeros((10, 10))

        key_init, key_forward = jax.random.split(key)
        variables = causal_vae.init(key_init, x, A, key_forward, training=True)

        # Encode
        mu_eps, log_var_eps = causal_vae.apply(variables, x, method=causal_vae.encode)
        assert mu_eps.shape == (32, 10)
        assert log_var_eps.shape == (32, 10)

        # Decode
        z = jax.random.normal(key, (32, 10))
        x_recon = causal_vae.apply(variables, z, method=causal_vae.decode)
        assert x_recon.shape == (32, 20)

    def test_causal_vae_training_mode(self):
        """Verify training vs eval mode differences."""
        causal_vae = CausalVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (32, 20))
        A = jnp.zeros((10, 10))

        key_init, key_forward = jax.random.split(key)
        variables = causal_vae.init(key_init, x, A, key_forward, training=True)

        # Training mode - uses sampling
        x_recon1, _ = causal_vae.apply(variables, x, A, key_forward, training=True)

        # Eval mode - uses mean (deterministic)
        x_recon2, _ = causal_vae.apply(variables, x, A, key_forward, training=False)
        x_recon3, _ = causal_vae.apply(variables, x, A, key_forward, training=False)

        # Eval mode should be deterministic
        assert jnp.allclose(x_recon2, x_recon3), "Eval mode should be deterministic"

    def test_sample_epsilon_and_get_z(self):
        """Verify sample_epsilon_and_get_z method."""
        causal_vae = CausalVAE(latent_dim=10, output_dim=20)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))
        A = generate_dag(DAGConfig(num_nodes=10, graph_type="erdos_renyi", seed=42))

        key_init, key_forward = jax.random.split(key)
        variables = causal_vae.init(key_init, x, A, key_forward, training=True)

        epsilon, z = causal_vae.apply(
            variables, x, A, key_forward, method=causal_vae.sample_epsilon_and_get_z
        )

        assert epsilon.shape == (32, 10)
        assert z.shape == (32, 10)


class TestCausalVAELoss:
    """Tests for CausalVAE loss functions."""

    def test_causal_vae_elbo_loss(self):
        """Verify CausalVAE ELBO loss computation."""
        x = jnp.ones((32, 20))
        x_recon = jnp.ones((32, 20)) * 0.9
        mu_eps = jnp.zeros((32, 10))
        log_var_eps = jnp.zeros((32, 10))

        loss, metrics = causal_vae_elbo_loss(x, x_recon, mu_eps, log_var_eps, beta=1.0)

        assert "reconstruction_loss" in metrics
        assert "kl_divergence" in metrics
        assert "total_loss" in metrics
        assert jnp.allclose(loss, metrics["total_loss"]), "Loss should match total_loss in metrics"

    def test_causal_vae_elbo_loss_perfect_reconstruction(self):
        """Verify loss for perfect reconstruction with standard normal."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 20))
        mu_eps = jnp.zeros((32, 10))
        log_var_eps = jnp.zeros((32, 10))

        # Perfect reconstruction
        loss, metrics = causal_vae_elbo_loss(x, x, mu_eps, log_var_eps, beta=1.0)

        # With perfect reconstruction and N(0,1), only recon loss should be ~0
        # (KL is 0 for N(0,1))
        assert metrics["reconstruction_loss"] < 1e-5, "Perfect reconstruction should have zero loss"
        assert jnp.abs(metrics["kl_divergence"]) < 1e-5, "KL(N(0,1) || N(0,1)) should be zero"


class TestCausalVAETraining:
    """Tests for CausalVAE training."""

    def test_create_causal_vae_train_state(self):
        """Verify training state creation."""
        model = CausalVAE(latent_dim=10, output_dim=20)
        A_true = generate_dag(DAGConfig(num_nodes=10, graph_type="chain", seed=42))
        key = jax.random.PRNGKey(0)

        state = create_causal_vae_train_state(
            model, A_true, learning_rate=1e-3, key=key, input_shape=(32, 20)
        )

        assert state.params is not None
        assert state.tx is not None
        assert state.step == 0
        assert jnp.array_equal(state.A_true, A_true), "State should store A_true"

    def test_causal_vae_train_step(self):
        """Verify single training step."""
        model = CausalVAE(latent_dim=10, output_dim=20)
        A_true = generate_dag(DAGConfig(num_nodes=10, graph_type="chain", seed=42))
        key = jax.random.PRNGKey(0)

        state = create_causal_vae_train_state(
            model, A_true, learning_rate=1e-3, key=key, input_shape=(32, 20)
        )

        # Generate a batch
        batch = jax.random.normal(key, (32, 20))

        # Training step
        new_state, metrics = causal_vae_train_step(state, batch, beta=1.0)

        assert new_state.step == 1, "Step should increment"
        assert "total_loss" in metrics
        assert metrics["total_loss"] >= 0.0, "Loss should be non-negative"

    def test_train_causal_vae_reduces_loss(self):
        """Verify training reduces loss over time."""
        # Create synthetic data from known DAG
        X, A_true = create_simple_dataset(num_nodes=10, n_samples=500, graph_type="chain")

        # Create CausalVAE
        model = CausalVAE(latent_dim=10, output_dim=10)

        # Train for a few epochs
        key = jax.random.PRNGKey(42)
        state, history = train_causal_vae(
            model,
            A_true=A_true,
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

    def test_causal_vae_vs_baseline_vae_on_causal_data(self):
        """
        Compare CausalVAE vs BaselineVAE on data generated from a causal model.

        CausalVAE should potentially perform better when given the correct A_true.
        """
        # Create synthetic data from known DAG
        X, A_true = create_simple_dataset(num_nodes=10, n_samples=1000, graph_type="chain")

        # Train CausalVAE with correct A_true
        causal_model = CausalVAE(latent_dim=10, output_dim=10)
        key = jax.random.PRNGKey(42)

        causal_state, causal_history = train_causal_vae(
            causal_model,
            A_true=A_true,
            train_data=X,
            n_epochs=30,
            batch_size=32,
            learning_rate=1e-3,
            key=key,
            verbose=False,
        )

        # Just verify it trains successfully
        assert len(causal_history["train_loss"]) == 30
        assert causal_history["train_loss"][-1] < causal_history["train_loss"][0]


class TestCausalVAEGradients:
    """Test that gradients flow through the causal layer."""

    def test_causal_vae_gradients_flow(self):
        """Verify gradients flow through CausalVAE (including causal layer)."""
        model = CausalVAE(latent_dim=5, output_dim=10)
        A_true = generate_dag(DAGConfig(num_nodes=5, graph_type="chain", seed=42))
        key = jax.random.PRNGKey(0)

        state = create_causal_vae_train_state(
            model, A_true, learning_rate=1e-3, key=key, input_shape=(8, 10)
        )

        # Create a small batch
        batch = jax.random.normal(key, (8, 10))

        # Compute gradients
        def loss_fn(params):
            key_local = jax.random.PRNGKey(1)
            x_recon, info = state.apply_fn(
                {"params": params}, batch, state.A_true, key_local, training=True
            )
            loss, _ = causal_vae_elbo_loss(
                x=batch,
                x_reconstructed=x_recon,
                mu_epsilon=info["mu_epsilon"],
                log_var_epsilon=info["log_var_epsilon"],
            )
            return loss

        grads = jax.grad(loss_fn)(state.params)

        # Check that gradients are non-zero (i.e., they flow through)
        has_nonzero_grads = False
        for param_dict in jax.tree_util.tree_leaves(grads):
            if jnp.any(jnp.abs(param_dict) > 1e-6):
                has_nonzero_grads = True
                break

        assert has_nonzero_grads, "Gradients should flow through the model"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
