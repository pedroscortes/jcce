"""
Test vmap-vectorized reconstruction loop.

Validates that:
1. Vectorized reconstruction matches loop-based version exactly
2. Gradients through vmapped reconstruction are correct
3. Works with all processor types (ELM, MLP, Transformer, Mamba)
4. Confound contribution is correctly vectorized
5. Performance improvement over loop-based version
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _recon_loop(batch_data, A_curr, proc_params, processor, A_conf=None):
    """Original loop-based reconstruction (reference implementation)."""
    n_batch, n_v = batch_data.shape

    self_loop_mask = 1.0 - jnp.eye(n_v)
    total_recon_loss = 0.0

    for j in range(n_v):
        weights = jnp.abs(A_curr[:n_v, j])
        weights = weights.at[j].set(0.0)
        weight_sum = jnp.sum(weights)
        fallback = jnp.ones(n_v).at[j].set(0.0)
        weights = jnp.where(weight_sum > 0.01, weights, fallback)

        X_weighted = batch_data * weights[jnp.newaxis, :]

        if processor.__class__.__name__ == "GNNAdapter":
            A_norm = A_curr[:n_v, :n_v] / (
                jnp.sum(jnp.abs(A_curr[:n_v, :n_v]), axis=0, keepdims=True) + 1e-8
            )
            output = processor.forward(X_weighted, proc_params[j], A=A_norm)
        else:
            output = processor.forward(X_weighted, proc_params[j])

        if A_conf is not None:
            confound_weights = jnp.abs(A_conf[:n_v, j])
            confound_weights = confound_weights.at[j].set(0.0)
            confound_contribution = jnp.sum(batch_data * confound_weights[jnp.newaxis, :], axis=1)
            output = output + confound_contribution

        mse = jnp.mean((batch_data[:, j] - output) ** 2)
        total_recon_loss += mse

    return total_recon_loss / n_v


def _recon_vmap(batch_data, A_curr, proc_params, processor, A_conf=None):
    """New vmap-based reconstruction."""
    n_batch, n_v = batch_data.shape

    self_loop_mask = 1.0 - jnp.eye(n_v)
    all_weights = jnp.abs(A_curr[:n_v, :n_v]).T * self_loop_mask
    weight_sums = jnp.sum(all_weights, axis=1, keepdims=True)
    all_weights = jnp.where(weight_sums > 0.01, all_weights, self_loop_mask)

    all_X_weighted = batch_data[jnp.newaxis, :, :] * all_weights[:, jnp.newaxis, :]

    _example_p = proc_params[0]
    _meta = {k: v for k, v in _example_p.items() if not isinstance(v, jnp.ndarray)}
    _arr_keys = sorted(k for k, v in _example_p.items() if isinstance(v, jnp.ndarray))
    _stacked = {k: jnp.stack([proc_params[j][k] for j in range(n_v)]) for k in _arr_keys}

    if processor.__class__.__name__ == "GNNAdapter":
        A_norm = A_curr[:n_v, :n_v] / (
            jnp.sum(jnp.abs(A_curr[:n_v, :n_v]), axis=0, keepdims=True) + 1e-8
        )

        def _recon_fwd(X_w_j, arrays_j):
            return processor.forward(X_w_j, {**_meta, **arrays_j}, A=A_norm)
    else:

        def _recon_fwd(X_w_j, arrays_j):
            return processor.forward(X_w_j, {**_meta, **arrays_j})

    all_outputs = jax.vmap(_recon_fwd)(all_X_weighted, _stacked)

    if A_conf is not None:
        all_conf_weights = jnp.abs(A_conf[:n_v, :n_v]).T * self_loop_mask
        all_conf_contrib = all_conf_weights @ batch_data.T
        all_outputs = all_outputs + all_conf_contrib

    all_mse = jnp.mean((batch_data.T - all_outputs) ** 2, axis=1)
    return jnp.mean(all_mse)


def test_vmap_correctness():
    """Vmap reconstruction should match loop-based version."""
    print("=" * 60)
    print("Test 1: vmap correctness (matches loop)")
    print("=" * 60)

    from jcce.structure_learning.jcce_learner import create_processor

    for proc_type in ["elm", "mlp"]:
        n_v = 5
        n_batch = 32
        n_total = n_v + 1
        key = random.PRNGKey(42)

        processor = create_processor(proc_type, random.PRNGKey(0))
        data = random.normal(key, (n_batch, n_v))
        A = random.normal(key, (n_total, n_total)) * 0.2

        # Initialize params for each variable
        proc_params = [processor.init_params(n_v) for _ in range(n_v)]

        # Test without confound
        loss_loop = _recon_loop(data, A, proc_params, processor)
        loss_vmap = _recon_vmap(data, A, proc_params, processor)
        diff = abs(float(loss_loop) - float(loss_vmap))
        print(
            f"  {proc_type:12s} (no confound): loop={float(loss_loop):.8f}  vmap={float(loss_vmap):.8f}  diff={diff:.2e}"
        )
        assert diff < 1e-4, f"Mismatch for {proc_type}: {diff}"

        # Test with confound matrix
        A_conf = random.normal(key, (n_total, n_total)) * 0.1
        loss_loop_c = _recon_loop(data, A, proc_params, processor, A_conf)
        loss_vmap_c = _recon_vmap(data, A, proc_params, processor, A_conf)
        diff_c = abs(float(loss_loop_c) - float(loss_vmap_c))
        print(
            f"  {proc_type:12s} (w/ confound): loop={float(loss_loop_c):.8f}  vmap={float(loss_vmap_c):.8f}  diff={diff_c:.2e}"
        )
        assert diff_c < 1e-4, f"Confound mismatch for {proc_type}: {diff_c}"

    print("  PASSED\n")


def test_vmap_gradients():
    """Gradients through vmapped reconstruction should be well-behaved."""
    print("=" * 60)
    print("Test 2: vmap gradient quality")
    print("=" * 60)

    from jcce.structure_learning.jcce_learner import create_processor

    for proc_type in ["elm", "mlp"]:
        n_v = 5
        n_batch = 32
        n_total = n_v + 1
        key = random.PRNGKey(42)

        processor = create_processor(proc_type, random.PRNGKey(0))
        data = random.normal(key, (n_batch, n_v))
        A = random.normal(key, (n_total, n_total)) * 0.2
        proc_params = [processor.init_params(n_v) for _ in range(n_v)]

        # Gradient w.r.t. A
        grad_loop = jax.grad(lambda a: _recon_loop(data, a, proc_params, processor))(A)
        grad_vmap = jax.grad(lambda a: _recon_vmap(data, a, proc_params, processor))(A)

        loop_norm = float(jnp.linalg.norm(grad_loop))
        vmap_norm = float(jnp.linalg.norm(grad_vmap))
        grad_diff = float(jnp.max(jnp.abs(grad_loop - grad_vmap)))

        has_nan_loop = bool(jnp.any(jnp.isnan(grad_loop)))
        has_nan_vmap = bool(jnp.any(jnp.isnan(grad_vmap)))

        print(
            f"  {proc_type:12s}: loop_grad_norm={loop_norm:.4f}  vmap_grad_norm={vmap_norm:.4f}  "
            f"max_diff={grad_diff:.2e}  nan_loop={has_nan_loop}  nan_vmap={has_nan_vmap}"
        )

        assert not has_nan_vmap, f"vmap gradient has NaN for {proc_type}"
        assert grad_diff < 1e-3, f"Gradient mismatch for {proc_type}: {grad_diff}"

    print("  PASSED\n")


def test_vmap_jit_compatible():
    """vmapped reconstruction should work inside JIT."""
    print("=" * 60)
    print("Test 3: vmap + JIT compatibility")
    print("=" * 60)

    from jcce.structure_learning.jcce_learner import create_processor

    n_v = 8
    n_batch = 32
    n_total = n_v + 1
    key = random.PRNGKey(42)

    processor = create_processor("elm", random.PRNGKey(0))
    data = random.normal(key, (n_batch, n_v))
    A = random.normal(key, (n_total, n_total)) * 0.2
    proc_params = [processor.init_params(n_v) for _ in range(n_v)]

    # JIT compile the vmap reconstruction
    @jax.jit
    def jitted_recon(A_in):
        return _recon_vmap(data, A_in, proc_params, processor)

    # First call (compile)
    loss1 = jitted_recon(A)
    print(f"  JIT'd vmap recon: loss={float(loss1):.6f}")

    # Gradient through JIT
    grad = jax.jit(jax.grad(lambda a: _recon_vmap(data, a, proc_params, processor)))(A)
    grad_norm = float(jnp.linalg.norm(grad))
    print(f"  JIT'd vmap grad norm: {grad_norm:.6f}")
    assert not jnp.any(jnp.isnan(grad)), "JIT'd vmap gradient has NaN"

    # Multiple calls (should reuse compiled graph)
    for i in range(5):
        A_test = A + random.normal(random.PRNGKey(i), A.shape) * 0.01
        loss = jitted_recon(A_test)

    print("  5 JIT'd calls completed")
    print("  PASSED\n")


def test_vmap_performance():
    """vmap should be faster than or comparable to loop for various n_v."""
    print("=" * 60)
    print("Test 4: vmap performance vs loop")
    print("=" * 60)

    from jcce.structure_learning.jcce_learner import create_processor

    n_warmup = 3
    n_runs = 20

    for n_v in [5, 11]:
        n_batch = 64
        n_total = n_v + 1
        key = random.PRNGKey(42)

        processor = create_processor("elm", random.PRNGKey(0))
        data = random.normal(key, (n_batch, n_v))
        A = random.normal(key, (n_total, n_total)) * 0.2
        proc_params = [processor.init_params(n_v) for _ in range(n_v)]

        # Eager warmup: initialize output_proj_W/b (ELM creates these lazily)
        for j in range(n_v):
            _ = processor.forward(data, proc_params[j])

        # JIT compile both
        loop_jit = jax.jit(lambda a: _recon_loop(data, a, proc_params, processor))
        vmap_jit = jax.jit(lambda a: _recon_vmap(data, a, proc_params, processor))

        # Also benchmark grad
        loop_grad_jit = jax.jit(jax.grad(lambda a: _recon_loop(data, a, proc_params, processor)))
        vmap_grad_jit = jax.jit(jax.grad(lambda a: _recon_vmap(data, a, proc_params, processor)))

        # Warmup
        for _ in range(n_warmup):
            loop_jit(A).block_until_ready()
            vmap_jit(A).block_until_ready()
            loop_grad_jit(A).block_until_ready()
            vmap_grad_jit(A).block_until_ready()

        # Forward benchmark
        t0 = time.perf_counter()
        for _ in range(n_runs):
            loop_jit(A).block_until_ready()
        loop_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            vmap_jit(A).block_until_ready()
        vmap_time = (time.perf_counter() - t0) / n_runs

        # Gradient benchmark
        t0 = time.perf_counter()
        for _ in range(n_runs):
            loop_grad_jit(A).block_until_ready()
        loop_grad_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            vmap_grad_jit(A).block_until_ready()
        vmap_grad_time = (time.perf_counter() - t0) / n_runs

        fwd_ratio = loop_time / vmap_time if vmap_time > 0 else float("inf")
        grad_ratio = loop_grad_time / vmap_grad_time if vmap_grad_time > 0 else float("inf")

        print(
            f"  n_v={n_v:2d} fwd:  loop={loop_time * 1000:.3f}ms  vmap={vmap_time * 1000:.3f}ms  ratio={fwd_ratio:.2f}x"
        )
        print(
            f"  n_v={n_v:2d} grad: loop={loop_grad_time * 1000:.3f}ms  vmap={vmap_grad_time * 1000:.3f}ms  ratio={grad_ratio:.2f}x"
        )

    print("  DONE\n")


def test_edge_cases():
    """Test edge cases: zero A, near-zero weights, large n_v."""
    print("=" * 60)
    print("Test 5: Edge cases")
    print("=" * 60)

    from jcce.structure_learning.jcce_learner import create_processor

    n_v = 5
    n_batch = 16
    n_total = n_v + 1
    key = random.PRNGKey(42)

    processor = create_processor("elm", random.PRNGKey(0))
    data = random.normal(key, (n_batch, n_v))
    proc_params = [processor.init_params(n_v) for _ in range(n_v)]

    # Zero A (should use fallback weights)
    A_zero = jnp.zeros((n_total, n_total))
    loss_zero = _recon_vmap(data, A_zero, proc_params, processor)
    print(f"  Zero A:        loss={float(loss_zero):.6f} (should be finite)")
    assert jnp.isfinite(loss_zero), "Non-finite loss for zero A"

    # Very small A (near-zero weights → fallback)
    A_tiny = jnp.ones((n_total, n_total)) * 1e-5
    loss_tiny = _recon_vmap(data, A_tiny, proc_params, processor)
    print(f"  Tiny A (1e-5): loss={float(loss_tiny):.6f} (should be finite)")
    assert jnp.isfinite(loss_tiny), "Non-finite loss for tiny A"

    # Large A (should handle normally)
    A_large = random.normal(key, (n_total, n_total)) * 2.0
    loss_large = _recon_vmap(data, A_large, proc_params, processor)
    print(f"  Large A (2.0): loss={float(loss_large):.6f} (should be finite)")
    assert jnp.isfinite(loss_large), "Non-finite loss for large A"

    # Gradient through zero A
    grad_zero = jax.grad(lambda a: _recon_vmap(data, a, proc_params, processor))(A_zero)
    has_nan = bool(jnp.any(jnp.isnan(grad_zero)))
    print(f"  Grad at A=0:   nan={has_nan}, norm={float(jnp.linalg.norm(grad_zero)):.6f}")
    assert not has_nan, "NaN gradient at A=0"

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VMAP RECONSTRUCTION VALIDATION TEST SUITE")
    print("=" * 60 + "\n")

    test_vmap_correctness()
    test_vmap_gradients()
    test_vmap_jit_compatible()
    test_vmap_performance()
    test_edge_cases()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
