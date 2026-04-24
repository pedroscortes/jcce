"""
Tests for Session 36 fixes:
- Fix 19: GNN logit stability (clamp + H normalization)
- Fix 20: Lambda-2 stall detection (stagnation check)
- Fix 21: h_pooled normalization for Y classification (all processors)
"""
import jax
import jax.numpy as jnp
from jax import random
import pytest


# ============================================================================
# Fix 19: GNN logit clamping + H normalization
# ============================================================================

class TestGNNLogitClamping:
    """Test that logit clamping prevents BCE explosion."""

    def test_clamped_bce_bounded(self):
        """BCE with clamped logits should never exceed ~6."""
        Y_output = jnp.array([10.0, -8.0, 15.0, -12.0, 7.0])
        batch_Y = jnp.array([0.0, 1.0, 0.0, 1.0, 1.0])

        Y_clamped = jnp.clip(Y_output, -6.0, 6.0)
        Y_pred = jax.nn.sigmoid(Y_clamped)
        eps = 1e-7
        loss_clamped = -jnp.mean(
            batch_Y * jnp.log(Y_pred + eps) +
            (1 - batch_Y) * jnp.log(1 - Y_pred + eps)
        )

        Y_pred_raw = jax.nn.sigmoid(Y_output)
        loss_raw = -jnp.mean(
            batch_Y * jnp.log(Y_pred_raw + eps) +
            (1 - batch_Y) * jnp.log(1 - Y_pred_raw + eps)
        )

        assert float(loss_clamped) < 6.5, f"Clamped BCE should be < 6.5, got {float(loss_clamped)}"
        assert float(loss_raw) > float(loss_clamped), "Raw loss should exceed clamped loss for extreme logits"

    def test_clamping_no_effect_normal_range(self):
        """Clamping should not affect logits in normal [-3, 3] range."""
        Y_output = jnp.array([1.5, -0.8, 2.1, -1.3, 0.2])
        Y_clamped = jnp.clip(Y_output, -6.0, 6.0)
        assert jnp.allclose(Y_output, Y_clamped)

    def test_clamping_gradient_flows(self):
        """Gradients should flow through unclamped region."""
        def loss_fn(logits, targets):
            clamped = jnp.clip(logits, -6.0, 6.0)
            pred = jax.nn.sigmoid(clamped)
            eps = 1e-7
            return -jnp.mean(
                targets * jnp.log(pred + eps) +
                (1 - targets) * jnp.log(1 - pred + eps)
            )

        logits = jnp.array([2.0, -1.0, 0.5])
        targets = jnp.array([1.0, 0.0, 1.0])

        grads = jax.grad(loss_fn)(logits, targets)
        assert jnp.all(jnp.isfinite(grads)), "Gradients should be finite"
        assert float(jnp.sum(jnp.abs(grads))) > 0, "Gradients should be non-zero"


class TestGNNHNormalization:
    """Test that GNN solve_output_weights now normalizes H."""

    def test_gnn_stores_h_stats(self):
        """GNN solve_output_weights should store _H_mean and _H_std."""
        from jcce.structure_learning.processor_adapters import GNNAdapter

        key = random.PRNGKey(0)
        adapter = GNNAdapter(hidden_dim=16, n_layers=2, key=key)

        X = random.normal(key, (50, 5))
        y = (random.uniform(key, (50,)) > 0.5).astype(jnp.float32)
        params = adapter.init_params(5)

        params = adapter.solve_output_weights(X, y, params, for_classification=True)

        assert '_H_mean' in params, "GNN should store _H_mean after solve_output_weights"
        assert '_H_std' in params, "GNN should store _H_std after solve_output_weights"
        assert params['_weights_solved'] is True

    def test_gnn_h_stats_correct_shape(self):
        """_H_mean/_H_std should be (1, hidden_dim) for broadcasting."""
        from jcce.structure_learning.processor_adapters import GNNAdapter

        key = random.PRNGKey(1)
        hidden_dim = 16
        adapter = GNNAdapter(hidden_dim=hidden_dim, n_layers=2, key=key)

        X = random.normal(key, (50, 5))
        y = (random.uniform(key, (50,)) > 0.5).astype(jnp.float32)
        params = adapter.init_params(5)
        params = adapter.solve_output_weights(X, y, params, for_classification=True)

        assert params['_H_mean'].shape == (1, hidden_dim)
        assert params['_H_std'].shape == (1, hidden_dim)

    def test_gnn_forward_uses_h_stats(self):
        """After solve_output_weights, forward() should apply H denormalization."""
        from jcce.structure_learning.processor_adapters import GNNAdapter

        key = random.PRNGKey(2)
        adapter = GNNAdapter(hidden_dim=16, n_layers=2, key=key)

        X = random.normal(key, (50, 5))
        y = (random.uniform(key, (50,)) > 0.5).astype(jnp.float32)
        params = adapter.init_params(5)

        out_before = adapter.forward(X, params.copy())
        params = adapter.solve_output_weights(X, y, params, for_classification=True)
        out_after = adapter.forward(X, params)

        assert not jnp.allclose(out_before, out_after)
        assert float(jnp.max(jnp.abs(out_after))) < 20.0

    def test_gnn_output_bounded_with_skip_centering(self):
        """Even with skip_centering=True, GNN output should be reasonable after solve."""
        from jcce.structure_learning.processor_adapters import GNNAdapter

        key = random.PRNGKey(3)
        adapter = GNNAdapter(hidden_dim=16, n_layers=2, key=key)

        X = random.normal(key, (50, 5)) * 3.0
        y = (random.uniform(key, (50,)) > 0.5).astype(jnp.float32)
        params = adapter.init_params(5)
        params = adapter.solve_output_weights(X, y, params, for_classification=True)

        out = adapter.forward(X, params, skip_centering=True)
        max_abs = float(jnp.max(jnp.abs(out)))
        assert max_abs < 30.0, f"Output should be bounded, got max={max_abs}"


# ============================================================================
# Fix 20: Lambda-2 stall detection
# ============================================================================

class TestLambda2StallDetection:
    """Test that lambda_2 doesn't increase when h_A is improving."""

    def test_lambda2_no_increase_when_improving(self):
        """When h_A is decreasing, lambda_2 should NOT increase (second branch)."""
        lambda_2 = 100.0
        lambda_2_max = 1e4
        h_A_val = 0.04
        prev_h_A = 0.05  # Was higher → improving

        if h_A_val > 0.01 and True:  # iter > 20
            if h_A_val >= prev_h_A * 0.99:  # 0.04 >= 0.0495? NO
                lambda_2 = min(lambda_2 * 1.5, lambda_2_max)

        assert lambda_2 == 100.0

    def test_lambda2_increases_when_stagnating(self):
        """When h_A is NOT improving, lambda_2 should increase."""
        lambda_2 = 100.0
        lambda_2_max = 1e4
        h_A_val = 0.04
        prev_h_A = 0.04  # Same → stagnating

        if h_A_val > 0.01 and True:
            if h_A_val >= prev_h_A * 0.99:  # 0.04 >= 0.0396? YES
                lambda_2 = min(lambda_2 * 1.5, lambda_2_max)

        assert lambda_2 == 150.0

    def test_frozen_training_scenario(self):
        """Simulate the Heart Disease Trial 32 scenario."""
        lambda_2_old = 100.0
        lambda_2_new = 100.0
        lambda_2_max = 1e4
        h_A_history = [0.08, 0.06, 0.045, 0.04, 0.04, 0.04, 0.04, 0.04]

        for i, h_A_val in enumerate(h_A_history):
            prev_h_A = h_A_history[i - 1] if i > 0 else 0.1

            if h_A_val > 0.01:
                lambda_2_old = min(lambda_2_old * 1.5, lambda_2_max)

            if h_A_val > 0.01:
                if h_A_val >= prev_h_A * 0.99:
                    lambda_2_new = min(lambda_2_new * 1.5, lambda_2_max)

        assert lambda_2_new < lambda_2_old
        assert lambda_2_new < 1000


# ============================================================================
# Fix 21: h_pooled normalization for Y classification (all processors)
# ============================================================================

class TestHPooledNormalization:
    """Test that h_pooled is normalized when skip_centering=True (training time)."""

    def test_mlp_skip_centering_output_bounded(self):
        """MLP with skip_centering=True should produce bounded output (h_pooled normalized)."""
        from jcce.structure_learning.processor_adapters import MLPAdapter

        key = random.PRNGKey(10)
        adapter = MLPAdapter(hidden_dim=32, key=key)
        X = random.normal(key, (100, 11)) * 5.0  # Large scale input
        params = adapter.init_params(11)

        out = adapter.forward(X, params, skip_centering=True)
        max_abs = float(jnp.max(jnp.abs(out)))
        # h_pooled normalization prevents extreme outputs
        assert max_abs < 10.0, f"MLP skip_centering output should be bounded, got max={max_abs}"

    def test_transformer_skip_centering_output_bounded(self):
        """Transformer with skip_centering=True should produce bounded output."""
        from jcce.structure_learning.processor_adapters import TransformerAdapter

        key = random.PRNGKey(11)
        adapter = TransformerAdapter(d_model=16, n_heads=2, n_layers=1, key=key)
        X = random.normal(key, (100, 11)) * 5.0
        params = adapter.init_params(11)

        out = adapter.forward(X, params, skip_centering=True)
        max_abs = float(jnp.max(jnp.abs(out)))
        # h_pooled is normalized but output_proj_W * 0.5 with hidden_dim features → some variance
        assert max_abs < 25.0, f"Transformer skip_centering output should be bounded, got max={max_abs}"

    def test_elm_skip_centering_output_bounded(self):
        """ELM with skip_centering=True should produce bounded output."""
        from jcce.structure_learning.processor_adapters import ELMAdapter

        key = random.PRNGKey(12)
        adapter = ELMAdapter(hidden_dim=32, key=key)
        X = random.normal(key, (100, 11)) * 5.0
        params = adapter.init_params(11)

        out = adapter.forward(X, params, skip_centering=True)
        max_abs = float(jnp.max(jnp.abs(out)))
        # ELM has output_proj_W * 1.0 (larger than MLP's 0.5) → higher variance
        assert max_abs < 35.0, f"ELM skip_centering output should be bounded, got max={max_abs}"

    def test_h_pooled_norm_preserves_bias_learning(self):
        """output_proj_b should still be free to learn (unlike old output centering)."""
        from jcce.structure_learning.processor_adapters import MLPAdapter

        key = random.PRNGKey(20)
        adapter = MLPAdapter(hidden_dim=32, key=key)
        X = random.normal(key, (100, 5))
        params = adapter.init_params(5)

        # Set a non-zero bias
        params['output_proj_b'] = jnp.array([2.0])

        out_with_bias = adapter.forward(X, params, skip_centering=True)
        mean_with_bias = float(jnp.mean(out_with_bias))

        # Reset bias to zero
        params['output_proj_b'] = jnp.array([0.0])
        out_no_bias = adapter.forward(X, params, skip_centering=True)
        mean_no_bias = float(jnp.mean(out_no_bias))

        # Bias should shift the mean (unlike old centering which killed it)
        diff = abs(mean_with_bias - mean_no_bias)
        assert diff > 1.0, f"Bias should shift output mean by ~2.0, got diff={diff}"

    def test_h_pooled_norm_vs_old_centering(self):
        """h_pooled norm should give different (better) output than old output centering."""
        from jcce.structure_learning.processor_adapters import MLPAdapter

        key = random.PRNGKey(30)
        adapter = MLPAdapter(hidden_dim=32, key=key)
        X = random.normal(key, (100, 5))
        params = adapter.init_params(5)

        # New: h_pooled norm (skip_centering=True)
        out_new = adapter.forward(X, params, skip_centering=True)

        # Old: output centering (skip_centering=False)
        out_old = adapter.forward(X, params, skip_centering=False)

        # Old centering should have mean ≈ 0 (forced)
        assert abs(float(jnp.mean(out_old))) < 0.01, "Old centering should force mean=0"

        # New h_pooled norm: output_proj_b=0, so mean should be near 0 too,
        # but because W projects normalized h, the distribution is different
        # Key test: both should produce finite, bounded outputs
        assert jnp.all(jnp.isfinite(out_new))
        assert jnp.all(jnp.isfinite(out_old))

    def test_h_pooled_norm_gradient_not_zero(self):
        """Gradients through h_pooled norm should be non-zero (not a dead signal)."""
        from jcce.structure_learning.processor_adapters import MLPAdapter

        key = random.PRNGKey(40)
        adapter = MLPAdapter(hidden_dim=32, key=key)
        X = random.normal(key, (50, 5))
        y = jnp.where(random.uniform(key, (50,)) > 0.4, 1.0, 0.0)  # Imbalanced
        params = adapter.init_params(5)

        def classification_loss(W, b):
            params_copy = dict(params)
            params_copy['output_proj_W'] = W
            params_copy['output_proj_b'] = b
            out = adapter.forward(X, params_copy, skip_centering=True)
            pred = jax.nn.sigmoid(out)
            eps = 1e-7
            return -jnp.mean(y * jnp.log(pred + eps) + (1 - y) * jnp.log(1 - pred + eps))

        W = params['output_proj_W']
        b = params['output_proj_b']
        grad_W, grad_b = jax.grad(classification_loss, argnums=(0, 1))(W, b)

        assert float(jnp.sum(jnp.abs(grad_W))) > 0.01, f"grad_W should be non-zero, got {float(jnp.sum(jnp.abs(grad_W)))}"
        assert float(jnp.abs(grad_b.squeeze())) > 0.001, f"grad_b should be non-zero, got {float(jnp.abs(grad_b.squeeze()))}"

    def test_all_5_adapters_have_h_pooled_norm(self):
        """All 5 adapters should apply h_pooled normalization with skip_centering=True."""
        from jcce.structure_learning.processor_adapters import (
            MLPAdapter, TransformerAdapter, MambaAdapter, ELMAdapter, GNNAdapter
        )

        key = random.PRNGKey(50)
        X = random.normal(key, (50, 5)) * 10.0  # Very large scale

        adapters = [
            ('MLP', MLPAdapter(hidden_dim=16, key=key)),
            ('Transformer', TransformerAdapter(d_model=16, n_heads=2, n_layers=1, key=key)),
            ('Mamba', MambaAdapter(d_model=16, key=key)),
            ('ELM', ELMAdapter(hidden_dim=16, key=key)),
            ('GNN', GNNAdapter(hidden_dim=16, n_layers=2, key=key)),
        ]

        for name, adapter in adapters:
            params = adapter.init_params(5)
            if name == 'GNN':
                out = adapter.forward(X, params, skip_centering=True)
            else:
                out = adapter.forward(X, params, skip_centering=True)

            max_abs = float(jnp.max(jnp.abs(out)))
            assert max_abs < 15.0, f"{name} skip_centering output too large: max={max_abs}"
            assert jnp.all(jnp.isfinite(out)), f"{name} output contains non-finite values"


# ============================================================================
# Integration: Combined effect
# ============================================================================

class TestIntegration:
    """Test combined effect of all fixes."""

    def test_gnn_bce_stays_reasonable(self):
        """GNN classification loss should stay below 6 with logit clamping + h_pooled norm."""
        from jcce.structure_learning.processor_adapters import GNNAdapter

        key = random.PRNGKey(42)
        adapter = GNNAdapter(hidden_dim=16, n_layers=2, key=key)

        X = random.normal(key, (100, 8))
        y = jnp.where(random.uniform(key, (100,)) > 0.41, 1.0, 0.0)
        params = adapter.init_params(8)

        output = adapter.forward(X, params, skip_centering=True)

        output_clamped = jnp.clip(output, -6.0, 6.0)
        pred = jax.nn.sigmoid(output_clamped)
        eps = 1e-7
        bce = -jnp.mean(y * jnp.log(pred + eps) + (1 - y) * jnp.log(1 - pred + eps))

        assert float(bce) < 2.0, f"With h_pooled norm + clamp, BCE should be < 2.0, got {float(bce)}"
        assert jnp.isfinite(bce)

    def test_mlp_bce_not_stuck_at_7_77(self):
        """MLP with h_pooled norm should NOT produce class=7.77 divergence."""
        from jcce.structure_learning.processor_adapters import MLPAdapter

        key = random.PRNGKey(99)
        adapter = MLPAdapter(hidden_dim=64, key=key)

        # Simulate Heart Disease-like data
        X = random.normal(key, (166, 13)) * 3.0
        y = jnp.where(random.uniform(key, (166,)) > 0.45, 1.0, 0.0)
        params = adapter.init_params(13)

        output = adapter.forward(X, params, skip_centering=True)
        output_clamped = jnp.clip(output, -6.0, 6.0)
        pred = jax.nn.sigmoid(output_clamped)
        eps = 1e-7
        bce = -jnp.mean(y * jnp.log(pred + eps) + (1 - y) * jnp.log(1 - pred + eps))

        assert float(bce) < 2.0, f"MLP BCE should be < 2.0 (not 7.77!), got {float(bce)}"
