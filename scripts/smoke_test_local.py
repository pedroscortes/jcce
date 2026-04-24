#!/usr/bin/env python3
"""Local smoke test: structural DML vs DragonNet vs PCGrad on synthetic data.

Runs on CPU. Uses a small synthetic dataset with KNOWN ground-truth causal effects
to compare the three configurations. This gives conclusive results without GPU.

Usage:
    python scripts/smoke_test_local.py
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jcce.structure_learning.jcce_learner import create_processor, learn_structure


def generate_synthetic_data(n=500, d=8, seed=42):
    """Generate linear SEM data with known causal effects."""
    rng = np.random.RandomState(seed)

    # True DAG: X0 -> X2, X1 -> X2, X2 -> X4, X3 -> X4, X0 -> Y, X2 -> Y, X4 -> Y
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)  # X0: root
    X[:, 1] = rng.randn(n)  # X1: root
    X[:, 2] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.randn(n) * 0.5  # X2 = f(X0, X1)
    X[:, 3] = rng.randn(n)  # X3: root
    X[:, 4] = 0.4 * X[:, 2] + 0.6 * X[:, 3] + rng.randn(n) * 0.5  # X4 = f(X2, X3)
    X[:, 5] = rng.randn(n)  # X5: noise
    X[:, 6] = rng.randn(n)  # X6: noise
    X[:, 7] = rng.randn(n)  # X7: noise

    # Y = 0.7*X0 + 0.4*X2 + 0.3*X4 + noise
    # True parents of Y: X0 (ATE~0.7), X2 (ATE~0.4), X4 (ATE~0.3)
    Y = 0.7 * X[:, 0] + 0.4 * X[:, 2] + 0.3 * X[:, 4] + rng.randn(n) * 0.3
    Y_binary = (Y > np.median(Y)).astype(np.float32)

    true_effects = {"X0": 0.7, "X2": 0.4, "X4": 0.3}
    true_parents = [0, 2, 4]

    return X.astype(np.float32), Y_binary, true_effects, true_parents


def run_config(
    name, X, Y, key, use_structural_dml=False, use_pcgrad=False, use_dragonnet=True, max_iter=80
):
    """Run one configuration and return results."""
    print(f"\n{'=' * 60}")
    print(f"  CONFIG: {name}")
    print(f"  structural_dml={use_structural_dml}, pcgrad={use_pcgrad}, dragonnet={use_dragonnet}")
    print(f"{'=' * 60}")

    t0 = time.time()

    Y_col = Y.reshape(-1, 1)
    Y_idx = X.shape[1]  # Y is appended as last variable

    try:
        # Create processor first
        proc_key, learn_key = random.split(key)
        processor = create_processor("elm", proc_key)

        A_est, processor_out, proc_params, metrics = learn_structure(
            data=jnp.array(X),
            Y=jnp.array(Y_col),
            Y_idx=Y_idx,
            processor=processor,
            key=learn_key,
            processor_type="elm",  # Fastest processor
            max_iter=max_iter,
            lr=0.003,
            lambda_1=0.05,
            lambda_2_init=0.01,
            lambda_class=1.0,
            patience=20,
            verbose=1,
            use_amortized_effects=True,
            use_dragonnet=use_dragonnet and not use_structural_dml,
            use_structural_dml=use_structural_dml,
            use_pcgrad=use_pcgrad,
            lambda_effect=0.5,
            effect_warmup_iter=10,
            use_latent_confounders=False,
            task="classification",
        )

        elapsed = time.time() - t0

        # Extract results
        A_np = np.array(A_est)
        n_features = X.shape[1]

        # Get causal effects
        causal_effects = metrics.get("causal_effects", {})
        h_A = metrics.get("final_h_A", float("inf"))
        bacc = metrics.get("balanced_accuracy", 0)
        n_edges = metrics.get("n_edges", 0)

        print("\n  Results:")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    h(A): {h_A:.6f}")
        print(f"    BAcc: {bacc:.3f}")
        print(f"    Edges: {n_edges}")

        # Report X->Y effects
        print("\n    Causal effects (X->Y):")
        for key_name, val in sorted(causal_effects.items()):
            if "->Y" in key_name:
                ate = val.get("ate", val) if isinstance(val, dict) else val
                print(f"      {key_name}: ATE={float(ate):.4f}")

        # Edge weights to Y
        print("\n    Edge weights to Y:")
        for i in range(n_features):
            w = float(A_np[i, Y_idx]) if A_np.shape[1] > Y_idx else 0
            if abs(w) > 0.001:
                print(f"      X{i} -> Y: {w:.4f}")

        return {
            "name": name,
            "time": elapsed,
            "h_A": h_A,
            "bacc": bacc,
            "n_edges": n_edges,
            "causal_effects": causal_effects,
            "A_est": A_np,
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return {"name": name, "error": str(e)}


def main():
    print("LOCAL SMOKE TEST: DragonNet vs Structural DML vs PCGrad")
    print("=" * 60)
    print(f"Device: {jax.devices()}")
    print("Running on CPU — results are conclusive, just slower\n")

    # Generate data
    X, Y, true_effects, true_parents = generate_synthetic_data(n=500, d=8)
    print(f"Data: n={X.shape[0]}, d={X.shape[1]}")
    print(f"True parents of Y: {true_parents}")
    print(f"True effects: {true_effects}")

    key = random.PRNGKey(42)
    max_iter = 80  # Enough for convergence on CPU

    # Config 1: DragonNet (current default)
    k1, k2, k3, k4 = random.split(key, 4)
    r1 = run_config(
        "DragonNet (baseline)",
        X,
        Y,
        k1,
        use_structural_dml=False,
        use_pcgrad=False,
        max_iter=max_iter,
    )

    # Config 2: Structural DML (new)
    r2 = run_config(
        "Structural DML", X, Y, k2, use_structural_dml=True, use_pcgrad=False, max_iter=max_iter
    )

    # Config 3: PCGrad + DragonNet
    r3 = run_config(
        "PCGrad + DragonNet", X, Y, k3, use_structural_dml=False, use_pcgrad=True, max_iter=max_iter
    )

    # Config 4: Structural DML + PCGrad (best of both)
    r4 = run_config(
        "Structural DML + PCGrad",
        X,
        Y,
        k4,
        use_structural_dml=True,
        use_pcgrad=True,
        max_iter=max_iter,
    )

    # Summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("True effects: X0->Y=0.7, X2->Y=0.4, X4->Y=0.3\n")
    print(f"{'Config':<30s} {'Time':>6s} {'h(A)':>10s} {'BAcc':>6s} {'Edges':>6s}")
    print("-" * 65)

    for r in [r1, r2, r3, r4]:
        if "error" in r:
            print(f"{r['name']:<30s} FAILED: {r['error']}")
            continue
        print(
            f"{r['name']:<30s} {r['time']:>5.0f}s {r['h_A']:>10.6f} {r['bacc']:>6.3f} {r['n_edges']:>6d}"
        )

    print(f"\n{'Config':<30s} {'X0->Y':>8s} {'X2->Y':>8s} {'X4->Y':>8s}")
    print("-" * 60)
    for r in [r1, r2, r3, r4]:
        if "error" in r:
            continue
        ce = r.get("causal_effects", {})
        x0 = ce.get("X0->Y", {})
        x2 = ce.get("X2->Y", {})
        x4 = ce.get("X4->Y", {})
        ate0 = float(x0.get("ate", x0) if isinstance(x0, dict) else x0) if x0 else 0
        ate2 = float(x2.get("ate", x2) if isinstance(x2, dict) else x2) if x2 else 0
        ate4 = float(x4.get("ate", x4) if isinstance(x4, dict) else x4) if x4 else 0
        print(f"{r['name']:<30s} {ate0:>+8.4f} {ate2:>+8.4f} {ate4:>+8.4f}")

    print(f"\n{'TRUE':30s} {'0.7000':>8s} {'0.4000':>8s} {'0.3000':>8s}")


if __name__ == "__main__":
    main()
