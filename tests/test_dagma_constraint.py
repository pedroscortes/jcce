"""
Test DAGMA DAG constraint vs matrix exponential.

Validates that:
1. DAGMA correctly identifies DAGs (h=0) and cycles (h>0)
2. DAGMA and expm agree on DAG/non-DAG classification
3. DAGMA gradients are well-behaved
4. DAGMA works inside JIT compilation
5. DAGMA is faster than expm for various d
"""

import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jcce.structure_learning.jcce_learner import (
    compute_dag_constraint_checkpointed,
    dag_constraint,
)


def test_dag_detection():
    """DAGMA should return h~0 for DAGs and h>0 for cyclic graphs."""
    print("=" * 60)
    print("Test 1: DAG detection (DAG vs cycle)")
    print("=" * 60)

    # DAG: strictly lower triangular
    A_dag = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.0],
        ]
    )
    h_dag = dag_constraint(A_dag)
    print(f"  DAG (lower triangular): h = {h_dag:.8f}")
    assert h_dag < 1e-4, f"DAG should have h~0, got {h_dag}"

    # Empty graph (trivially DAG)
    A_empty = jnp.zeros((4, 4))
    h_empty = dag_constraint(A_empty)
    print(f"  Empty graph:            h = {h_empty:.8f}")
    assert h_empty < 1e-6, f"Empty graph should have h=0, got {h_empty}"

    # 2-cycle
    A_cycle2 = jnp.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    h_cycle2 = dag_constraint(A_cycle2)
    print(f"  2-cycle (0↔1):          h = {h_cycle2:.4f}")
    assert h_cycle2 > 0.01, f"Cycle should have h>0, got {h_cycle2}"

    # 3-cycle
    A_cycle3 = jnp.array(
        [
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.0],
        ]
    )
    h_cycle3 = dag_constraint(A_cycle3)
    print(f"  3-cycle (0→1→2→0):      h = {h_cycle3:.4f}")
    assert h_cycle3 > 0.01, f"Cycle should have h>0, got {h_cycle3}"

    print("  PASSED\n")


def test_agreement_with_expm():
    """DAGMA and expm should agree on DAG vs non-DAG classification."""
    print("=" * 60)
    print("Test 2: DAGMA vs expm agreement")
    print("=" * 60)

    key = jax.random.PRNGKey(42)

    for d in [5, 11, 20, 30]:
        # Generate random DAG (lower triangular with random permutation)
        key, subkey = jax.random.split(key)
        L = jnp.tril(jax.random.normal(subkey, (d, d)) * 0.3, k=-1)
        # Random permutation to make it non-obvious
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, d)
        A_dag = L[perm][:, perm]

        h_dagma = dag_constraint(A_dag)
        h_expm = compute_dag_constraint_checkpointed(A_dag)

        # Both should be near zero for DAGs
        dagma_is_dag = float(h_dagma) < 0.1
        expm_is_dag = float(h_expm) < 0.1

        print(
            f"  d={d:2d} DAG:   DAGMA h={float(h_dagma):.6f}  expm h={float(h_expm):.6f}  agree={dagma_is_dag == expm_is_dag}"
        )
        assert dagma_is_dag == expm_is_dag, f"Disagreement on DAG at d={d}"

        # Generate cyclic graph
        key, subkey = jax.random.split(key)
        A_cycle = jax.random.normal(subkey, (d, d)) * 0.3

        h_dagma_c = dag_constraint(A_cycle)
        h_expm_c = compute_dag_constraint_checkpointed(A_cycle)

        dagma_is_cyclic = float(h_dagma_c) > 0.1
        expm_is_cyclic = float(h_expm_c) > 0.1

        print(
            f"  d={d:2d} cycle: DAGMA h={float(h_dagma_c):.4f}  expm h={float(h_expm_c):.4f}  agree={dagma_is_cyclic == expm_is_cyclic}"
        )
        assert dagma_is_cyclic == expm_is_cyclic, f"Disagreement on cycle at d={d}"

    print("  PASSED\n")


def test_gradient_quality():
    """DAGMA gradients should be well-behaved (no NaN/Inf, reasonable magnitude)."""
    print("=" * 60)
    print("Test 3: Gradient quality")
    print("=" * 60)

    key = jax.random.PRNGKey(123)

    for d in [5, 11, 30]:
        key, subkey = jax.random.split(key)
        A = jax.random.normal(subkey, (d, d)) * 0.2

        grad_dagma = jax.grad(dag_constraint)(A)
        grad_expm = jax.grad(compute_dag_constraint_checkpointed)(A)

        dagma_has_nan = bool(jnp.any(jnp.isnan(grad_dagma)))
        dagma_has_inf = bool(jnp.any(jnp.isinf(grad_dagma)))
        expm_has_nan = bool(jnp.any(jnp.isnan(grad_expm)))
        expm_has_inf = bool(jnp.any(jnp.isinf(grad_expm)))

        dagma_norm = float(jnp.linalg.norm(grad_dagma))
        expm_norm = float(jnp.linalg.norm(grad_expm))

        print(
            f"  d={d:2d}: DAGMA grad norm={dagma_norm:.4f} (nan={dagma_has_nan}, inf={dagma_has_inf})"
        )
        print(f"        expm  grad norm={expm_norm:.4f} (nan={expm_has_nan}, inf={expm_has_inf})")

        assert not dagma_has_nan, f"DAGMA gradient has NaN at d={d}"
        assert not dagma_has_inf, f"DAGMA gradient has Inf at d={d}"

    print("  PASSED\n")


def test_jit_compatibility():
    """DAGMA should work correctly inside jax.jit."""
    print("=" * 60)
    print("Test 4: JIT compatibility")
    print("=" * 60)

    @jax.jit
    def loss_fn(A):
        h = dag_constraint(A)
        sparsity = jnp.sum(jnp.abs(A))
        return h + 0.01 * sparsity

    A = jnp.zeros((11, 11))
    A = A.at[1, 0].set(0.5)
    A = A.at[2, 1].set(0.3)

    # First call (compile)
    loss = loss_fn(A)
    print(f"  JIT'd loss (DAG): {float(loss):.6f}")

    # Gradient through JIT
    grad = jax.grad(loss_fn)(A)
    print(f"  JIT'd grad norm:  {float(jnp.linalg.norm(grad)):.6f}")
    assert not jnp.any(jnp.isnan(grad)), "JIT'd gradient has NaN"

    # Multiple calls (should reuse compiled graph)
    for i in range(5):
        A_test = A + jax.random.normal(jax.random.PRNGKey(i), A.shape) * 0.01
        loss = loss_fn(A_test)

    print("  5 JIT'd calls completed")
    print("  PASSED\n")


def test_performance_comparison():
    """DAGMA should be faster or comparable to expm, especially for larger d."""
    print("=" * 60)
    print("Test 5: Performance comparison (DAGMA vs expm)")
    print("=" * 60)

    n_warmup = 3
    n_runs = 20

    for d in [11, 30, 50]:
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (d, d)) * 0.2

        # JIT compile both
        dagma_jit = jax.jit(dag_constraint)
        expm_jit = jax.jit(compute_dag_constraint_checkpointed)

        # Warmup
        for _ in range(n_warmup):
            dagma_jit(A).block_until_ready()
            expm_jit(A).block_until_ready()

        # Benchmark DAGMA
        t0 = time.perf_counter()
        for _ in range(n_runs):
            dagma_jit(A).block_until_ready()
        dagma_time = (time.perf_counter() - t0) / n_runs

        # Benchmark expm
        t0 = time.perf_counter()
        for _ in range(n_runs):
            expm_jit(A).block_until_ready()
        expm_time = (time.perf_counter() - t0) / n_runs

        speedup = expm_time / dagma_time if dagma_time > 0 else float("inf")
        print(
            f"  d={d:2d}: DAGMA={dagma_time * 1000:.3f}ms  expm={expm_time * 1000:.3f}ms  speedup={speedup:.2f}x"
        )

    # Also benchmark gradients
    print("\n  Gradient computation:")
    for d in [11, 30]:
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (d, d)) * 0.2

        dagma_grad_jit = jax.jit(jax.grad(dag_constraint))
        expm_grad_jit = jax.jit(jax.grad(compute_dag_constraint_checkpointed))

        # Warmup
        for _ in range(n_warmup):
            dagma_grad_jit(A).block_until_ready()
            expm_grad_jit(A).block_until_ready()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            dagma_grad_jit(A).block_until_ready()
        dagma_grad_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            expm_grad_jit(A).block_until_ready()
        expm_grad_time = (time.perf_counter() - t0) / n_runs

        speedup = expm_grad_time / dagma_grad_time if dagma_grad_time > 0 else float("inf")
        print(
            f"  d={d:2d}: DAGMA grad={dagma_grad_time * 1000:.3f}ms  expm grad={expm_grad_time * 1000:.3f}ms  speedup={speedup:.2f}x"
        )

    print("  DONE\n")


def test_constraint_in_optimization():
    """Simulate a mini training loop to verify DAGMA drives h→0."""
    print("=" * 60)
    print("Test 6: DAGMA in optimization loop (h should decrease)")
    print("=" * 60)

    import optax

    d = 11
    key = jax.random.PRNGKey(99)
    # Start with random (cyclic) matrix
    A = jax.random.normal(key, (d, d)) * 0.3

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(A)

    lambda_dag = 1.0

    def loss_fn(A_param):
        h = dag_constraint(A_param)
        sparsity = 0.01 * jnp.sum(jnp.abs(A_param))
        return lambda_dag * h + sparsity

    h_values = []
    for step in range(100):
        loss, grads = jax.value_and_grad(loss_fn)(A)
        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)

        if step % 20 == 0 or step == 99:
            h_val = float(dag_constraint(A))
            h_values.append(h_val)
            print(f"  Step {step:3d}: h(A) = {h_val:.6f}")

    assert h_values[-1] < h_values[0], "h should decrease during optimization"
    assert h_values[-1] < 0.1, f"h should converge near 0, got {h_values[-1]}"
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DAGMA CONSTRAINT VALIDATION TEST SUITE")
    print("=" * 60 + "\n")

    test_dag_detection()
    test_agreement_with_expm()
    test_gradient_quality()
    test_jit_compatibility()
    test_performance_comparison()
    test_constraint_in_optimization()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
