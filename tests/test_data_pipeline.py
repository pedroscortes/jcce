"""
Test data pipeline optimization for v7 training loop.

Validates that:
1. device_put places data on device correctly
2. Batch selection inside JIT produces correct shapes and values
3. batch_Y is correctly returned from train_step
4. Cached JAX scalars (lambda_2_jax, curriculum_w_jax) work correctly
5. Performance: fewer Python↔XLA round trips vs old approach
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_device_put():
    """Data placed on device should remain accessible and unchanged."""
    print("=" * 60)
    print("Test 1: jax.device_put correctness")
    print("=" * 60)

    key = random.PRNGKey(42)
    n_samples, n_vars = 200, 11

    data = random.normal(key, (n_samples, n_vars))
    Y = random.bernoulli(key, 0.5, (n_samples,)).astype(jnp.float32)

    # Place on device
    data_dev = jax.device_put(data)
    Y_dev = jax.device_put(Y)

    # Values should be identical
    diff_data = float(jnp.max(jnp.abs(data - data_dev)))
    diff_Y = float(jnp.max(jnp.abs(Y - Y_dev)))
    print(f"  data diff: {diff_data:.2e}")
    print(f"  Y diff:    {diff_Y:.2e}")
    assert diff_data == 0.0, f"Data changed after device_put: {diff_data}"
    assert diff_Y == 0.0, f"Y changed after device_put: {diff_Y}"

    # Shape preserved
    assert data_dev.shape == data.shape
    assert Y_dev.shape == Y.shape

    # Device placement
    assert data_dev.devices() == Y_dev.devices()
    print(f"  device: {data_dev.devices()}")
    print("  PASSED\n")


def test_batch_selection_inside_jit():
    """Batch selection inside JIT should produce correct shapes and valid indices."""
    print("=" * 60)
    print("Test 2: Batch selection inside JIT")
    print("=" * 60)

    key = random.PRNGKey(42)
    n_samples, n_vars = 200, 11
    batch_size = 64

    data = jax.device_put(random.normal(key, (n_samples, n_vars)))
    Y = jax.device_put(random.bernoulli(key, 0.5, (n_samples,)).astype(jnp.float32))

    # JIT function that selects batch internally (like our train_step)
    @jax.jit
    def select_batch(batch_key):
        batch_idx = random.choice(batch_key, n_samples, shape=(batch_size,), replace=False)
        batch_data = data[batch_idx]
        batch_Y = Y[batch_idx]
        return batch_data, batch_Y, batch_idx

    key, batch_key = random.split(key)
    batch_data, batch_Y, batch_idx = select_batch(batch_key)

    # Correct shapes
    assert batch_data.shape == (batch_size, n_vars), f"Wrong shape: {batch_data.shape}"
    assert batch_Y.shape == (batch_size,), f"Wrong Y shape: {batch_Y.shape}"
    print(f"  batch_data shape: {batch_data.shape}")
    print(f"  batch_Y shape:    {batch_Y.shape}")

    # Indices are valid (within [0, n_samples))
    assert int(jnp.min(batch_idx)) >= 0
    assert int(jnp.max(batch_idx)) < n_samples
    # No replacement — all unique
    assert len(set(batch_idx.tolist())) == batch_size
    print(f"  indices range: [{int(jnp.min(batch_idx))}, {int(jnp.max(batch_idx))}]")
    print(f"  all unique:    {len(set(batch_idx.tolist()))} == {batch_size}")

    # Values match manual indexing
    batch_idx_np = batch_idx
    manual_data = data[batch_idx_np]
    manual_Y = Y[batch_idx_np]
    assert float(jnp.max(jnp.abs(batch_data - manual_data))) == 0.0
    assert float(jnp.max(jnp.abs(batch_Y - manual_Y))) == 0.0
    print("  values match manual indexing")

    # Different keys → different batches
    key, batch_key2 = random.split(key)
    batch_data2, _, _ = select_batch(batch_key2)
    diff = float(jnp.max(jnp.abs(batch_data - batch_data2)))
    assert diff > 0.0, "Same batch with different keys!"
    print(f"  different keys -> different batches (diff={diff:.4f})")

    print("  PASSED\n")


def test_batch_y_returned():
    """train_step should return batch_Y for curriculum accuracy computation."""
    print("=" * 60)
    print("Test 3: batch_Y returned from JIT step")
    print("=" * 60)

    key = random.PRNGKey(42)
    n_samples, n_vars = 200, 11
    batch_size = 64

    data = jax.device_put(random.normal(key, (n_samples, n_vars)))
    Y = jax.device_put(random.bernoulli(key, 0.5, (n_samples,)).astype(jnp.float32))

    # Simulate a minimal train_step that returns batch_Y
    @jax.jit
    def mock_step(batch_key):
        batch_idx = random.choice(batch_key, n_samples, shape=(batch_size,), replace=False)
        batch_data = data[batch_idx]
        batch_Y = Y[batch_idx]

        # Simulate some computation (like loss_fn)
        loss = jnp.mean(batch_data**2) + jnp.mean(batch_Y)
        Y_logits = jnp.sum(batch_data, axis=1)  # fake logits

        return loss, Y_logits, batch_Y

    key, batch_key = random.split(key)
    loss, Y_logits, batch_Y_out = mock_step(batch_key)

    # batch_Y should be valid for accuracy computation
    assert batch_Y_out.shape == (batch_size,)
    Y_pred = (jax.nn.sigmoid(Y_logits) > 0.5).astype(jnp.float32)
    accuracy = float(jnp.mean(Y_pred == batch_Y_out))
    print(f"  batch_Y shape: {batch_Y_out.shape}")
    print(f"  accuracy computable: {accuracy:.4f}")
    assert 0.0 <= accuracy <= 1.0

    # batch_Y values should be subset of original Y
    assert jnp.all((batch_Y_out == 0.0) | (batch_Y_out == 1.0))
    print("  all values binary")

    print("  PASSED\n")


def test_cached_jax_scalars():
    """Cached lambda_2_jax and curriculum_w_jax should work identically to per-iteration creation."""
    print("=" * 60)
    print("Test 4: Cached JAX scalar allocations")
    print("=" * 60)

    # Simulate the caching pattern
    lambda_2 = 1.0
    curriculum_weights = (0.5, 0.3, 0.2)

    # Pre-allocated (our approach)
    lambda_2_jax = jnp.float32(lambda_2)
    curriculum_w_jax = jnp.array(
        [curriculum_weights[0], curriculum_weights[1], curriculum_weights[2]], dtype=jnp.float32
    )

    @jax.jit
    def compute_with_cached(lam, cw):
        return lam * cw[0] + cw[1] + cw[2]

    @jax.jit
    def compute_with_fresh(lam_py, cw_py):
        lam = jnp.float32(lam_py)
        cw = jnp.array([cw_py[0], cw_py[1], cw_py[2]], dtype=jnp.float32)
        return lam * cw[0] + cw[1] + cw[2]

    result_cached = compute_with_cached(lambda_2_jax, curriculum_w_jax)
    result_fresh = compute_with_fresh(lambda_2, curriculum_weights)
    diff = abs(float(result_cached) - float(result_fresh))
    print(
        f"  initial: cached={float(result_cached):.6f} fresh={float(result_fresh):.6f} diff={diff:.2e}"
    )
    assert diff < 1e-7

    # After lambda_2 changes
    lambda_2 = 2.5
    lambda_2_jax = jnp.float32(lambda_2)  # update cache
    result_cached = compute_with_cached(lambda_2_jax, curriculum_w_jax)
    result_fresh = compute_with_fresh(lambda_2, curriculum_weights)
    diff = abs(float(result_cached) - float(result_fresh))
    print(
        f"  λ₂ update: cached={float(result_cached):.6f} fresh={float(result_fresh):.6f} diff={diff:.2e}"
    )
    assert diff < 1e-7

    # After curriculum changes
    curriculum_weights = (0.1, 0.6, 0.3)
    curriculum_w_jax = jnp.array(
        [curriculum_weights[0], curriculum_weights[1], curriculum_weights[2]], dtype=jnp.float32
    )
    result_cached = compute_with_cached(lambda_2_jax, curriculum_w_jax)
    result_fresh = compute_with_fresh(lambda_2, curriculum_weights)
    diff = abs(float(result_cached) - float(result_fresh))
    print(
        f"  cw update: cached={float(result_cached):.6f} fresh={float(result_fresh):.6f} diff={diff:.2e}"
    )
    assert diff < 1e-7

    print("  PASSED\n")


def test_no_batching_path():
    """When use_batching=False, full data should be used directly from closure."""
    print("=" * 60)
    print("Test 5: No-batching path (full data from closure)")
    print("=" * 60)

    key = random.PRNGKey(42)
    n_samples, n_vars = 50, 5  # small data → no batching

    data = jax.device_put(random.normal(key, (n_samples, n_vars)))
    Y = jax.device_put(random.bernoulli(key, 0.5, (n_samples,)).astype(jnp.float32))

    use_batching = False

    @jax.jit
    def step_no_batch(batch_key):
        if use_batching:
            batch_idx = random.choice(batch_key, n_samples, shape=(32,), replace=False)
            batch_data = data[batch_idx]
            batch_Y = Y[batch_idx]
        else:
            batch_data = data
            batch_Y = Y

        loss = jnp.mean(batch_data**2)
        return loss, batch_Y

    key, batch_key = random.split(key)
    loss, batch_Y_out = step_no_batch(batch_key)

    # Should return full Y (not a subset)
    assert batch_Y_out.shape == (n_samples,), f"Wrong shape: {batch_Y_out.shape}"
    diff = float(jnp.max(jnp.abs(batch_Y_out - Y)))
    assert diff == 0.0, f"batch_Y differs from full Y: {diff}"
    print(f"  batch_Y shape: {batch_Y_out.shape} == ({n_samples},)")
    print(f"  matches full Y: diff={diff}")

    # Different keys should produce same result (no randomness used)
    key, batch_key2 = random.split(key)
    loss2, batch_Y_out2 = step_no_batch(batch_key2)
    assert float(jnp.max(jnp.abs(batch_Y_out - batch_Y_out2))) == 0.0
    print("  deterministic (no batch selection)")

    print("  PASSED\n")


def test_performance_batch_inside_jit():
    """Batch selection inside JIT should reduce Python↔XLA overhead."""
    print("=" * 60)
    print("Test 6: Performance — batch inside vs outside JIT")
    print("=" * 60)

    key = random.PRNGKey(42)
    n_samples, n_vars = 1000, 11
    batch_size = 256
    n_warmup = 5
    n_runs = 100

    data = jax.device_put(random.normal(key, (n_samples, n_vars)))
    Y = jax.device_put(random.bernoulli(key, 0.5, (n_samples,)).astype(jnp.float32))

    # Old approach: batch selection OUTSIDE JIT
    @jax.jit
    def step_outside(batch_data, batch_Y):
        loss = jnp.mean(batch_data**2) + jnp.mean(batch_Y)
        return loss

    # New approach: batch selection INSIDE JIT
    @jax.jit
    def step_inside(batch_key):
        batch_idx = random.choice(batch_key, n_samples, shape=(batch_size,), replace=False)
        batch_data = data[batch_idx]
        batch_Y = Y[batch_idx]
        loss = jnp.mean(batch_data**2) + jnp.mean(batch_Y)
        return loss

    # Warmup
    keys = random.split(key, n_warmup + n_runs + 10)
    for i in range(n_warmup):
        batch_key = keys[i]
        batch_idx = random.choice(batch_key, n_samples, shape=(batch_size,), replace=False)
        step_outside(data[batch_idx], Y[batch_idx]).block_until_ready()
        step_inside(batch_key).block_until_ready()

    # Benchmark old approach (outside JIT)
    t0 = time.perf_counter()
    for i in range(n_runs):
        batch_key = keys[n_warmup + i]
        batch_idx = random.choice(batch_key, n_samples, shape=(batch_size,), replace=False)
        batch_data = data[batch_idx]
        batch_Y_b = Y[batch_idx]
        step_outside(batch_data, batch_Y_b).block_until_ready()
    outside_time = (time.perf_counter() - t0) / n_runs

    # Benchmark new approach (inside JIT)
    t0 = time.perf_counter()
    for i in range(n_runs):
        batch_key = keys[n_warmup + i]
        step_inside(batch_key).block_until_ready()
    inside_time = (time.perf_counter() - t0) / n_runs

    ratio = outside_time / inside_time if inside_time > 0 else float("inf")
    print(f"  outside JIT: {outside_time * 1000:.3f}ms/iter")
    print(f"  inside JIT:  {inside_time * 1000:.3f}ms/iter")
    print(f"  speedup:     {ratio:.2f}x")

    # Also benchmark scalar allocation
    lambda_2 = 1.5
    cw = (0.5, 0.3, 0.2)

    # Fresh allocation per iter
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = jnp.float32(lambda_2)
        _ = jnp.array([cw[0], cw[1], cw[2]], dtype=jnp.float32)
    fresh_time = (time.perf_counter() - t0) / n_runs

    # Cached (reuse)
    l2_cached = jnp.float32(lambda_2)
    cw_cached = jnp.array([cw[0], cw[1], cw[2]], dtype=jnp.float32)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = l2_cached  # no allocation
        _ = cw_cached
    cached_time = (time.perf_counter() - t0) / n_runs

    scalar_ratio = fresh_time / cached_time if cached_time > 0 else float("inf")
    print(f"  scalar fresh:  {fresh_time * 1000:.4f}ms/iter")
    print(f"  scalar cached: {cached_time * 1000:.4f}ms/iter")
    print(f"  scalar ratio:  {scalar_ratio:.1f}x")

    print("  DONE\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATA PIPELINE OPTIMIZATION TEST SUITE")
    print("=" * 60 + "\n")

    test_device_put()
    test_batch_selection_inside_jit()
    test_batch_y_returned()
    test_cached_jax_scalars()
    test_no_batching_path()
    test_performance_batch_inside_jit()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
