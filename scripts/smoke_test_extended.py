#!/usr/bin/env python3
"""Extended smoke test: structural DML + PCGrad across multiple configs.

Tests the winning config (structural DML + PCGrad) against DragonNet baseline
with more iterations, multiple processors, and multiple seeds.

Usage:
    uv run python scripts/smoke_test_extended.py
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
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)
    X[:, 1] = rng.randn(n)
    X[:, 2] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.randn(n) * 0.5
    X[:, 3] = rng.randn(n)
    X[:, 4] = 0.4 * X[:, 2] + 0.6 * X[:, 3] + rng.randn(n) * 0.5
    X[:, 5] = rng.randn(n)
    X[:, 6] = rng.randn(n)
    X[:, 7] = rng.randn(n)
    Y = 0.7 * X[:, 0] + 0.4 * X[:, 2] + 0.3 * X[:, 4] + rng.randn(n) * 0.3
    Y_binary = (Y > np.median(Y)).astype(np.float32)
    return X.astype(np.float32), Y_binary


def run_one(X, Y, key, processor_type, max_iter, use_structural_dml, use_pcgrad):
    """Run one config, return results dict."""
    proc_key, learn_key = random.split(key)
    processor = create_processor(processor_type, proc_key)
    Y_col = Y.reshape(-1, 1)
    Y_idx = X.shape[1]

    t0 = time.time()
    try:
        A_est, _, _, metrics = learn_structure(
            data=jnp.array(X),
            Y=jnp.array(Y_col),
            Y_idx=Y_idx,
            processor=processor,
            key=learn_key,
            processor_type=processor_type,
            max_iter=max_iter,
            lr=0.003,
            lambda_1=0.05,
            lambda_2_init=0.01,
            lambda_class=1.0,
            patience=25,
            verbose=0,
            use_amortized_effects=True,
            use_dragonnet=not use_structural_dml,
            use_structural_dml=use_structural_dml,
            use_pcgrad=use_pcgrad,
            lambda_effect=0.5,
            effect_warmup_iter=10,
            use_latent_confounders=False,
            task="classification",
        )
        elapsed = time.time() - t0
        ce = metrics.get("causal_effects", {})

        def get_ate(key):
            v = ce.get(key, {})
            return float(v.get("ate", v) if isinstance(v, dict) else v) if v else 0.0

        return {
            "time": elapsed,
            "h_A": float(metrics.get("final_h_A", 99)),
            "bacc": float(metrics.get("balanced_accuracy", 0)),
            "n_edges": int(metrics.get("n_edges", 0)),
            "mb": metrics.get("markov_blanket", []),
            "ate_x0": get_ate("X0->Y"),
            "ate_x2": get_ate("X2->Y"),
            "ate_x4": get_ate("X4->Y"),
        }
    except Exception as e:
        return {"time": time.time() - t0, "error": str(e)}


def main():
    print("EXTENDED SMOKE TEST: Structural DML + PCGrad")
    print(f"Device: {jax.devices()}")
    print("=" * 80)

    X, Y = generate_synthetic_data(n=500, d=8, seed=42)
    true_ates = {"X0": 0.7, "X2": 0.4, "X4": 0.3}
    print(f"Data: n={X.shape[0]}, d={X.shape[1]}")
    print(f"True ATEs: {true_ates}\n")

    # Test matrix
    configs = [
        # (name, processor, max_iter, structural_dml, pcgrad)
        # 2x2 comparison: DragonNet vs StructDML × with/without PCGrad
        ("DragonNet/elm/200it", "elm", 200, False, False),
        ("DragonNet+PCGrad/elm/200it", "elm", 200, False, True),
        ("StructDML/elm/200it", "elm", 200, True, False),
        ("StructDML+PCGrad/elm/200it", "elm", 200, True, True),
        # Processor diversity (winner config only)
        ("StructDML+PCGrad/mlp/200it", "mlp", 200, True, True),
        ("StructDML+PCGrad/transformer/200it", "transformer", 200, True, True),
    ]

    # Run with 3 seeds each
    seeds = [42, 123, 456]
    all_results = {}

    for name, proc, max_iter, sdml, pcg in configs:
        print(f"\n--- {name} ---")
        seed_results = []
        for seed in seeds:
            key = random.PRNGKey(seed)
            r = run_one(X, Y, key, proc, max_iter, sdml, pcg)
            if "error" in r:
                print(f"  seed={seed}: FAILED ({r['error']})")
            else:
                print(
                    f"  seed={seed}: BAcc={r['bacc']:.3f} h(A)={r['h_A']:.4f} "
                    f"ATEs=[{r['ate_x0']:.3f},{r['ate_x2']:.3f},{r['ate_x4']:.3f}] "
                    f"MB={r['mb']} ({r['time']:.0f}s)"
                )
                seed_results.append(r)
        all_results[name] = seed_results

    # Summary table
    print(f"\n\n{'=' * 90}")
    print("SUMMARY (mean ± std across 3 seeds)")
    print(f"{'=' * 90}")
    print(
        f"{'Config':<35s} {'BAcc':>10s} {'h(A)':>10s} {'X0->Y':>10s} {'X2->Y':>10s} {'X4->Y':>10s} {'Time':>6s}"
    )
    print("-" * 90)

    for name, results in all_results.items():
        if not results:
            print(f"{name:<35s} ALL FAILED")
            continue
        baccs = [r["bacc"] for r in results]
        has = [r["h_A"] for r in results]
        a0 = [r["ate_x0"] for r in results]
        a2 = [r["ate_x2"] for r in results]
        a4 = [r["ate_x4"] for r in results]
        times = [r["time"] for r in results]
        print(
            f"{name:<35s} {np.mean(baccs):>5.3f}±{np.std(baccs):.3f} "
            f"{np.mean(has):>5.4f}±{np.std(has):.4f} "
            f"{np.mean(a0):>+5.3f}±{np.std(a0):.3f} "
            f"{np.mean(a2):>+5.3f}±{np.std(a2):.3f} "
            f"{np.mean(a4):>+5.3f}±{np.std(a4):.3f} "
            f"{np.mean(times):>5.0f}s"
        )

    print(f"\n{'TRUE':35s} {'':>10s} {'':>10s} {'+0.700':>10s} {'+0.400':>10s} {'+0.300':>10s}")


if __name__ == "__main__":
    main()
