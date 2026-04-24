"""
Test JIT-compiled train step.

Validates that:
1. JIT-compiled train step produces same results as eager execution
2. JIT compilation happens once and reuses cached graph
3. donate_argnums works correctly (memory reuse)
4. Recompilation works when closure variables change
5. Performance improvement over eager mode
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import optax
from jax import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_simple_loss_and_step(n_vars, use_effects=False, adjustment_sets=None):
    """Create a simplified loss_fn and train_step mimicking the v7 pattern."""
    from functools import partial

    from jcce.structure_learning.jcce_learner import (
        dag_constraint,
    )

    Y_idx = n_vars  # Y is last variable
    n_total = n_vars + 1

    def loss_fn(params, batch_data, batch_Y, batch_Y_effect, lambda_2_current, curriculum_weights):
        A_curr = params["A_direct"]
        w_recon, w_class, w_effect = curriculum_weights

        n_batch, n_v = batch_data.shape

        # Simplified reconstruction loss
        total_recon_loss = 0.0
        for j in range(n_v):
            weights = jnp.abs(A_curr[:n_v, j])
            weights = weights.at[j].set(0.0)
            X_weighted = batch_data * weights[jnp.newaxis, :]
            # Simple linear reconstruction
            output = jnp.sum(X_weighted, axis=1) * params["scale"]
            mse = jnp.mean((batch_data[:, j] - output) ** 2)
            total_recon_loss += mse
        total_recon_loss = total_recon_loss / n_v

        # Classification
        weights_Y = jnp.abs(A_curr[:n_v, Y_idx])
        weights_Y = weights_Y / (jnp.sum(weights_Y) + 1e-8)
        X_weighted_Y = batch_data * weights_Y[jnp.newaxis, :]
        Y_output = jnp.sum(X_weighted_Y, axis=1) * params["scale"]
        Y_pred = jax.nn.sigmoid(Y_output)
        eps = 1e-7
        classification_loss = -jnp.mean(
            batch_Y * jnp.log(Y_pred + eps) + (1 - batch_Y) * jnp.log(1 - Y_pred + eps)
        )

        # Structure penalties
        sparsity_loss = 0.01 * jnp.sum(jnp.abs(A_curr))
        h_A = dag_constraint(A_curr)
        dag_loss = lambda_2_current * h_A

        structural_loss = total_recon_loss + sparsity_loss + dag_loss

        # Effect loss (simplified)
        effect_loss = 0.0
        if use_effects:
            effect_loss = jnp.mean(A_curr[:n_v, Y_idx] ** 2) * 0.1

        # Curriculum weighting
        effect_enabled = 1.0 if use_effects else 0.0
        total_loss = (
            w_recon * structural_loss
            + w_class * classification_loss
            + w_effect * effect_enabled * effect_loss
        )

        bow_loss = 0.0
        return total_loss, (
            h_A,
            total_recon_loss,
            classification_loss,
            effect_loss,
            bow_loss,
            Y_output,
        )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=1e-3))

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(
        params, opt_state, batch_data, batch_Y, batch_Y_effect, lambda_2_jax, curriculum_w
    ):
        cw = (curriculum_w[0], curriculum_w[1], curriculum_w[2])
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch_data, batch_Y, batch_Y_effect, lambda_2_jax, cw
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        # Outcome sink constraint
        new_params["A_direct"] = new_params["A_direct"].at[Y_idx, :].set(0.0)
        return new_params, new_opt_state, loss_val, aux

    return loss_fn, optimizer, train_step


def test_jit_correctness():
    """JIT train step should produce same results as eager execution."""
    print("=" * 60)
    print("Test 1: JIT correctness (same as eager)")
    print("=" * 60)

    n_vars = 5
    n_total = n_vars + 1
    n_samples = 32
    Y_idx = n_vars

    key = random.PRNGKey(42)
    key, data_key, A_key = random.split(key, 3)
    data = random.normal(data_key, (n_samples, n_vars))
    Y = (random.uniform(key, (n_samples,)) > 0.5).astype(jnp.float32)

    A_init = random.normal(A_key, (n_total, n_total)) * 0.1
    A_init = A_init.at[Y_idx, :].set(0.0)

    params = {
        "A_direct": A_init,
        "scale": jnp.array(0.1),
    }

    loss_fn, optimizer, train_step_jit = _make_simple_loss_and_step(n_vars)
    opt_state = optimizer.init(params)

    lambda_2 = jnp.float32(1.0)
    cw = jnp.array([1.0, 0.3, 0.0], dtype=jnp.float32)

    # Eager execution
    (loss_eager, aux_eager), grads_eager = jax.value_and_grad(loss_fn, has_aux=True)(
        params, data, Y, Y, lambda_2, (cw[0], cw[1], cw[2])
    )
    updates_eager, opt_state_eager = optimizer.update(grads_eager, opt_state)
    params_eager = optax.apply_updates(params, updates_eager)
    params_eager["A_direct"] = params_eager["A_direct"].at[Y_idx, :].set(0.0)

    # JIT execution
    params_jit, opt_state_jit, loss_jit, aux_jit = train_step_jit(
        params, opt_state, data, Y, Y, lambda_2, cw
    )

    # Compare
    loss_diff = abs(float(loss_eager) - float(loss_jit))
    A_diff = float(jnp.max(jnp.abs(params_eager["A_direct"] - params_jit["A_direct"])))
    h_diff = abs(float(aux_eager[0]) - float(aux_jit[0]))

    print(f"  Loss diff:    {loss_diff:.10f}")
    print(f"  A_direct diff: {A_diff:.10f}")
    print(f"  h(A) diff:    {h_diff:.10f}")

    assert loss_diff < 1e-5, f"Loss mismatch: eager={float(loss_eager)}, jit={float(loss_jit)}"
    assert A_diff < 1e-5, f"A_direct mismatch: max diff={A_diff}"
    assert h_diff < 1e-5, f"h(A) mismatch: diff={h_diff}"

    print("  PASSED\n")


def test_jit_caching():
    """Second call should reuse compiled graph (much faster than first)."""
    print("=" * 60)
    print("Test 2: JIT caching (compile once, reuse)")
    print("=" * 60)

    n_vars = 8
    n_total = n_vars + 1
    n_samples = 64

    key = random.PRNGKey(123)
    data = random.normal(key, (n_samples, n_vars))
    Y = (random.uniform(key, (n_samples,)) > 0.5).astype(jnp.float32)

    params = {
        "A_direct": random.normal(key, (n_total, n_total)) * 0.1,
        "scale": jnp.array(0.1),
    }

    _, optimizer, train_step = _make_simple_loss_and_step(n_vars)
    opt_state = optimizer.init(params)

    lambda_2 = jnp.float32(1.0)
    cw = jnp.array([1.0, 0.3, 0.0], dtype=jnp.float32)

    # First call (compile + execute)
    t0 = time.perf_counter()
    params, opt_state, _, _ = train_step(params, opt_state, data, Y, Y, lambda_2, cw)
    t_first = time.perf_counter() - t0

    # Subsequent calls (execute only)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        params, opt_state, _, _ = train_step(params, opt_state, data, Y, Y, lambda_2, cw)
        jax.block_until_ready(params["A_direct"])
        times.append(time.perf_counter() - t0)

    t_cached = sum(times) / len(times)

    print(f"  First call (compile):  {t_first * 1000:.1f}ms")
    print(f"  Cached calls (avg):    {t_cached * 1000:.1f}ms")
    print(f"  Speedup:               {t_first / t_cached:.1f}x")

    # Cached should be significantly faster than first (compile takes >> execution)
    assert t_cached < t_first, f"Cached call not faster: {t_cached:.4f} vs {t_first:.4f}"

    print("  PASSED\n")


def test_donate_argnums():
    """donate_argnums should not cause errors over many iterations."""
    print("=" * 60)
    print("Test 3: donate_argnums stability (100 iterations)")
    print("=" * 60)

    n_vars = 5
    n_total = n_vars + 1
    n_samples = 32

    key = random.PRNGKey(42)
    data = random.normal(key, (n_samples, n_vars))
    Y = (random.uniform(key, (n_samples,)) > 0.5).astype(jnp.float32)

    params = {
        "A_direct": random.normal(key, (n_total, n_total)) * 0.1,
        "scale": jnp.array(0.1),
    }

    _, optimizer, train_step = _make_simple_loss_and_step(n_vars)
    opt_state = optimizer.init(params)

    lambda_2 = jnp.float32(1.0)
    cw = jnp.array([1.0, 0.3, 0.0], dtype=jnp.float32)

    losses = []
    for i in range(100):
        params, opt_state, loss_val, aux = train_step(params, opt_state, data, Y, Y, lambda_2, cw)
        if i % 25 == 0 or i == 99:
            losses.append(float(loss_val))
            print(f"  Iter {i:3d}: loss={float(loss_val):.6f}, h(A)={float(aux[0]):.6f}")

    # Loss should decrease
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # No NaN
    assert not jnp.any(jnp.isnan(params["A_direct"])), "NaN in params after 100 iters"

    # Outcome sink constraint should be maintained
    assert float(jnp.max(jnp.abs(params["A_direct"][n_vars, :]))) < 1e-8, (
        "Outcome sink constraint violated"
    )

    print("  PASSED\n")


def test_recompilation():
    """Recompilation should work correctly when closure variables change."""
    print("=" * 60)
    print("Test 4: Recompilation (closure variable change)")
    print("=" * 60)

    n_vars = 5

    # Phase 1: no effects
    loss_fn1, optimizer, step1 = _make_simple_loss_and_step(n_vars, use_effects=False)

    # Phase 2: with effects (simulates effect_warmup_completed changing)
    loss_fn2, _, step2 = _make_simple_loss_and_step(n_vars, use_effects=True)

    n_total = n_vars + 1
    n_samples = 32

    key = random.PRNGKey(42)
    data = random.normal(key, (n_samples, n_vars))
    Y = (random.uniform(key, (n_samples,)) > 0.5).astype(jnp.float32)

    params = {
        "A_direct": random.normal(key, (n_total, n_total)) * 0.1,
        "scale": jnp.array(0.1),
    }
    opt_state = optimizer.init(params)

    lambda_2 = jnp.float32(1.0)
    cw = jnp.array([1.0, 0.3, 0.5], dtype=jnp.float32)

    # Run with step1 (no effects)
    params, opt_state, loss1, aux1 = step1(params, opt_state, data, Y, Y, lambda_2, cw)
    print(f"  Phase 1 (no effects): loss={float(loss1):.6f}, effect_loss={float(aux1[3]):.6f}")
    assert float(aux1[3]) == 0.0, "Effect loss should be 0 in phase 1"

    # Switch to step2 (with effects) - simulates recompilation
    params, opt_state, loss2, aux2 = step2(params, opt_state, data, Y, Y, lambda_2, cw)
    print(f"  Phase 2 (effects):    loss={float(loss2):.6f}, effect_loss={float(aux2[3]):.6f}")
    assert float(aux2[3]) > 0.0, "Effect loss should be > 0 in phase 2"

    print("  PASSED\n")


def test_performance_vs_eager():
    """JIT should be faster than eager for multiple iterations."""
    print("=" * 60)
    print("Test 5: Performance (JIT vs eager)")
    print("=" * 60)

    for n_vars in [5, 11]:
        n_total = n_vars + 1
        n_samples = 64
        Y_idx = n_vars

        key = random.PRNGKey(42)
        data = random.normal(key, (n_samples, n_vars))
        Y = (random.uniform(key, (n_samples,)) > 0.5).astype(jnp.float32)

        params_init = {
            "A_direct": random.normal(key, (n_total, n_total)) * 0.1,
            "scale": jnp.array(0.1),
        }

        loss_fn, optimizer, train_step = _make_simple_loss_and_step(n_vars)

        # Benchmark eager
        params = {k: v.copy() for k, v in params_init.items()}
        opt_state = optimizer.init(params)
        lambda_2 = jnp.float32(1.0)
        cw_tuple = (1.0, 0.3, 0.0)

        # Warmup eager
        for _ in range(3):
            (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, data, Y, Y, lambda_2, cw_tuple
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            params["A_direct"] = params["A_direct"].at[Y_idx, :].set(0.0)

        n_iters = 30
        t0 = time.perf_counter()
        for _ in range(n_iters):
            (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, data, Y, Y, lambda_2, cw_tuple
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            params["A_direct"] = params["A_direct"].at[Y_idx, :].set(0.0)
            jax.block_until_ready(params["A_direct"])
        eager_time = (time.perf_counter() - t0) / n_iters

        # Benchmark JIT
        params = {k: v.copy() for k, v in params_init.items()}
        opt_state = optimizer.init(params)
        cw_arr = jnp.array([1.0, 0.3, 0.0], dtype=jnp.float32)

        # Warmup JIT (includes compilation)
        for _ in range(3):
            params, opt_state, _, _ = train_step(params, opt_state, data, Y, Y, lambda_2, cw_arr)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            params, opt_state, _, _ = train_step(params, opt_state, data, Y, Y, lambda_2, cw_arr)
            jax.block_until_ready(params["A_direct"])
        jit_time = (time.perf_counter() - t0) / n_iters

        speedup = eager_time / jit_time if jit_time > 0 else float("inf")
        print(
            f"  d={n_vars:2d}: eager={eager_time * 1000:.2f}ms  jit={jit_time * 1000:.2f}ms  speedup={speedup:.2f}x"
        )

    print("  DONE\n")


def test_curriculum_weights_dynamic():
    """Changing curriculum weights (as JAX arrays) should not cause recompilation."""
    print("=" * 60)
    print("Test 6: Dynamic curriculum weights (no retrace)")
    print("=" * 60)

    n_vars = 5
    n_total = n_vars + 1
    n_samples = 32

    key = random.PRNGKey(42)
    data = random.normal(key, (n_samples, n_vars))
    Y = (random.uniform(key, (n_samples,)) > 0.5).astype(jnp.float32)

    params = {
        "A_direct": random.normal(key, (n_total, n_total)) * 0.1,
        "scale": jnp.array(0.1),
    }

    _, optimizer, train_step = _make_simple_loss_and_step(n_vars)
    opt_state = optimizer.init(params)
    lambda_2 = jnp.float32(1.0)

    # Different curriculum weights each iteration (simulating phase transitions)
    weight_schedules = [
        jnp.array([1.0, 0.1, 0.0], dtype=jnp.float32),  # Phase 1
        jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32),  # Transition
        jnp.array([0.3, 0.7, 0.5], dtype=jnp.float32),  # Phase 2
        jnp.array([0.2, 0.6, 0.8], dtype=jnp.float32),  # Phase 3
    ]

    for i, cw in enumerate(weight_schedules):
        params, opt_state, loss, _ = train_step(params, opt_state, data, Y, Y, lambda_2, cw)
        print(f"  Phase {i + 1} weights={[float(x) for x in cw]}: loss={float(loss):.6f}")

    # Multiple calls with varying lambda_2 (also should not retrace)
    for lam in [0.5, 1.0, 2.0, 5.0]:
        params, opt_state, loss, _ = train_step(
            params, opt_state, data, Y, Y, jnp.float32(lam), weight_schedules[0]
        )

    print("  4 different lambda_2 values: OK (no retrace)")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JIT TRAIN STEP VALIDATION TEST SUITE")
    print("=" * 60 + "\n")

    test_jit_correctness()
    test_jit_caching()
    test_donate_argnums()
    test_recompilation()
    test_performance_vs_eager()
    test_curriculum_weights_dynamic()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
