"""Tests for the classification plateau fix.

Verifies that freezing log_var_recon (removing uncertainty weighting) allows
classification loss to drop below ln(2) = 0.6931 during training.
"""

import jax
import jax.numpy as jnp
import optax
from jax import random, value_and_grad

from jcce.structure_learning.jcce_learner import (
    extract_trainable_params,
    merge_trained_params,
)
from jcce.structure_learning.processor_adapters import ELMAdapter


def _make_data(key, n_vars=11, n_samples=300):
    """Generate classification data with known Y dependence."""
    k1, k2 = random.split(key)
    X = random.normal(k1, (n_samples, n_vars))
    logits = 0.8 * X[:, 0] + 0.5 * X[:, 3] + 0.3 * X[:, 7] - 0.6 * X[:, 2]
    Y = (logits > 0).astype(jnp.float32)
    return X, Y


def _train_loop(X, Y, n_vars, use_uncertainty_weighting, n_steps=150):
    """Run a mini GOLEM-like training loop and return classification loss history."""
    Y_idx = n_vars
    n_total = n_vars + 1
    key = random.PRNGKey(123)
    k1, k2 = random.split(key)

    processor = ELMAdapter(hidden_dim=32, n_hidden_nodes=64, activation="tanh", key=k1)
    orig_params = [processor.init_params(n_vars) for _ in range(n_total)]
    trainable = extract_trainable_params(orig_params)

    A_direct = random.normal(k2, (n_total, n_total)) * 0.1
    A_direct = A_direct.at[jnp.arange(n_total), jnp.arange(n_total)].set(0.0)
    A_direct = A_direct.at[Y_idx, :].set(0.0)

    params = {
        "A_direct": A_direct,
        "processor_params": trainable,
    }
    if use_uncertainty_weighting:
        params["log_var_recon"] = jnp.array(0.0)

    w_recon, w_class = 1.0, 0.5
    lambda_1 = 0.02

    def loss_fn(params):
        A_curr = params["A_direct"]
        proc_merged = merge_trained_params(orig_params, params["processor_params"])
        n_v = n_vars

        # Reconstruction
        recon_losses = []
        for j in range(n_v):
            w_j = jnp.abs(A_curr[:n_v, j]).at[j].set(0.0)
            out = processor.forward(X * w_j[jnp.newaxis, :], proc_merged[j])
            recon_losses.append(jnp.mean((X[:, j] - out) ** 2))
        recon_loss = jnp.mean(jnp.stack(recon_losses))

        # Classification
        w_Y = jnp.abs(A_curr[:n_v, Y_idx]) + 0.01
        Y_out = processor.forward(X * w_Y[jnp.newaxis, :], proc_merged[Y_idx])
        Y_pred = jax.nn.sigmoid(Y_out)
        eps = 1e-7
        class_loss = -jnp.mean(Y * jnp.log(Y_pred + eps) + (1 - Y) * jnp.log(1 - Y_pred + eps))

        if use_uncertainty_weighting:
            log_var_r = params["log_var_recon"]
            prec = jnp.exp(-log_var_r)
            w_struct = w_recon * (0.5 * prec * recon_loss + 0.5 * log_var_r)
        else:
            w_struct = w_recon * recon_loss

        w_cls = w_class * class_loss
        sparsity = lambda_1 * jnp.sum(jnp.abs(A_curr))
        total = w_struct + w_cls + sparsity

        return total, class_loss

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.001))
    opt_state = optimizer.init(params)

    class_history = []
    for step in range(n_steps):
        (loss, class_loss), grads = value_and_grad(loss_fn, has_aux=True)(params)

        # Freeze log_var_recon gradient (matching the fix)
        if use_uncertainty_weighting:
            grads = {**grads, "log_var_recon": jnp.zeros_like(grads["log_var_recon"])}

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params["A_direct"] = params["A_direct"].at[Y_idx, :].set(0.0)

        class_history.append(float(class_loss))

    return class_history


class TestClassPlateauFix:
    def test_without_uw_classification_improves(self):
        """Without uncertainty weighting, classification loss drops well below ln(2)."""
        X, Y = _make_data(random.PRNGKey(42))
        history = _train_loop(X, Y, n_vars=11, use_uncertainty_weighting=False)

        assert history[-1] < 0.55, (
            f"Classification should improve without UW, got {history[-1]:.4f}"
        )

    def test_with_frozen_uw_classification_improves(self):
        """With frozen log_var_recon (gradient zeroed), classification also improves."""
        X, Y = _make_data(random.PRNGKey(42))
        history = _train_loop(X, Y, n_vars=11, use_uncertainty_weighting=True)

        # Since gradient is zeroed, log_var_r stays at 0, precision=1,
        # and the formula is equivalent to no UW (just with 0.5 multiplier)
        assert history[-1] < 0.60, (
            f"Classification should improve with frozen UW, got {history[-1]:.4f}"
        )

    def test_classification_decreases_monotonically(self):
        """Classification loss should generally trend downward."""
        X, Y = _make_data(random.PRNGKey(42))
        history = _train_loop(X, Y, n_vars=11, use_uncertainty_weighting=False)

        # Check that the second half average is lower than first half
        mid = len(history) // 2
        first_half = sum(history[:mid]) / mid
        second_half = sum(history[mid:]) / (len(history) - mid)
        assert second_half < first_half, (
            f"Classification should improve over time: first={first_half:.4f}, second={second_half:.4f}"
        )

    def test_fix_matches_no_uw_baseline(self):
        """Frozen UW should produce similar results to no UW."""
        X, Y = _make_data(random.PRNGKey(42))
        history_no_uw = _train_loop(X, Y, n_vars=11, use_uncertainty_weighting=False)
        history_frozen = _train_loop(X, Y, n_vars=11, use_uncertainty_weighting=True)

        # Both should reach similar final classification loss
        # (not identical because the 0.5 multiplier changes the effective weight)
        diff = abs(history_no_uw[-1] - history_frozen[-1])
        assert diff < 0.15, (
            f"Frozen UW and no UW should be similar: {history_no_uw[-1]:.4f} vs {history_frozen[-1]:.4f}"
        )

    def test_classification_below_random_chance(self):
        """Classification should get below random chance (ln(2)=0.6931)."""
        X, Y = _make_data(random.PRNGKey(42))
        history = _train_loop(X, Y, n_vars=11, use_uncertainty_weighting=False)

        # Should break below ln(2) within 50 steps
        broke_through = any(h < 0.6931 for h in history[:50])
        assert broke_through, (
            f"Classification should break through ln(2) within 50 steps, min={min(history[:50]):.4f}"
        )
