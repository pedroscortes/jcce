#!/usr/bin/env python3
"""Smoke test at d=20: verify structural DML scales beyond d=8."""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jcce.structure_learning.jcce_learner import create_processor, learn_structure


def generate_d20_data(n=800, seed=42):
    """Generate d=20 linear SEM with known effects."""
    rng = np.random.RandomState(seed)
    d = 20
    X = np.zeros((n, d))

    # Roots
    for i in [0, 1, 3, 5, 7, 10, 15]:
        X[:, i] = rng.randn(n)

    # Intermediate
    X[:, 2] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.randn(n) * 0.5
    X[:, 4] = 0.4 * X[:, 2] + 0.6 * X[:, 3] + rng.randn(n) * 0.5
    X[:, 6] = 0.3 * X[:, 5] + rng.randn(n) * 0.5
    X[:, 8] = 0.5 * X[:, 7] + rng.randn(n) * 0.5
    X[:, 9] = 0.4 * X[:, 8] + rng.randn(n) * 0.5
    X[:, 11] = 0.3 * X[:, 10] + rng.randn(n) * 0.5
    X[:, 12] = 0.5 * X[:, 11] + 0.2 * X[:, 4] + rng.randn(n) * 0.5
    X[:, 13] = 0.4 * X[:, 12] + rng.randn(n) * 0.5
    X[:, 14] = 0.3 * X[:, 13] + rng.randn(n) * 0.5
    X[:, 16] = 0.5 * X[:, 15] + rng.randn(n) * 0.5
    X[:, 17] = 0.4 * X[:, 16] + rng.randn(n) * 0.5
    X[:, 18] = 0.3 * X[:, 17] + 0.2 * X[:, 9] + rng.randn(n) * 0.5
    X[:, 19] = 0.5 * X[:, 18] + rng.randn(n) * 0.5

    # Y = 0.6*X0 + 0.4*X4 + 0.3*X12 + 0.2*X18 + noise
    Y = 0.6 * X[:, 0] + 0.4 * X[:, 4] + 0.3 * X[:, 12] + 0.2 * X[:, 18] + rng.randn(n) * 0.3
    Y_binary = (Y > np.median(Y)).astype(np.float32)

    true_effects = {"X0": 0.6, "X4": 0.4, "X12": 0.3, "X18": 0.2}
    return X.astype(np.float32), Y_binary, true_effects


def run_one(X, Y, key, name, use_structural_dml, use_pcgrad, max_iter=150):
    proc_key, learn_key = random.split(key)
    processor = create_processor("elm", proc_key)
    Y_col = Y.reshape(-1, 1)
    t0 = time.time()
    try:
        A_est, _, _, metrics = learn_structure(
            data=jnp.array(X),
            Y=jnp.array(Y_col),
            Y_idx=X.shape[1],
            processor=processor,
            key=learn_key,
            processor_type="elm",
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
            effect_warmup_iter=15,
            use_latent_confounders=False,
            task="classification",
        )
        elapsed = time.time() - t0
        ce = metrics.get("causal_effects", {})

        def get_ate(k):
            v = ce.get(k, {})
            return float(v.get("ate", v) if isinstance(v, dict) else v) if v else 0.0

        return {
            "time": elapsed,
            "h_A": float(metrics.get("final_h_A", 99)),
            "bacc": float(metrics.get("balanced_accuracy", 0)),
            "n_edges": int(metrics.get("n_edges", 0)),
            "mb": metrics.get("markov_blanket", []),
            "ate_x0": get_ate("X0->Y"),
            "ate_x4": get_ate("X4->Y"),
            "ate_x12": get_ate("X12->Y"),
            "ate_x18": get_ate("X18->Y"),
        }
    except Exception as e:
        return {"time": time.time() - t0, "error": str(e)}


def main():
    print("SMOKE TEST d=20: Structural DML scaling")
    print(f"Device: {jax.devices()}")
    print("=" * 70)

    X, Y, true_effects = generate_d20_data(n=800)
    print(f"Data: n={X.shape[0]}, d={X.shape[1]}")
    print("True parents: X0(0.6), X4(0.4), X12(0.3), X18(0.2)\n")

    configs = [
        ("DragonNet/elm", False, False),
        ("StructDML/elm", True, False),
        ("StructDML+PCGrad/elm", True, True),
    ]

    seeds = [42, 123]
    all_results = {}

    for name, sdml, pcg in configs:
        print(f"\n--- {name} ---")
        seed_results = []
        for seed in seeds:
            key = random.PRNGKey(seed)
            r = run_one(X, Y, key, name, sdml, pcg, max_iter=150)
            if "error" in r:
                print(f"  seed={seed}: FAILED ({r['error']})")
            else:
                print(
                    f"  seed={seed}: BAcc={r['bacc']:.3f} h(A)={r['h_A']:.4f} "
                    f"ATEs=[{r['ate_x0']:.3f},{r['ate_x4']:.3f},{r['ate_x12']:.3f},{r['ate_x18']:.3f}] "
                    f"MB={r['mb'][:6]}... ({r['time']:.0f}s)"
                )
                seed_results.append(r)
        all_results[name] = seed_results

    print(f"\n\n{'=' * 80}")
    print("SUMMARY d=20 (mean across seeds)")
    print(f"{'=' * 80}")
    print(
        f"{'Config':<25s} {'BAcc':>8s} {'X0(0.6)':>8s} {'X4(0.4)':>8s} {'X12(0.3)':>9s} {'X18(0.2)':>9s} {'Time':>6s}"
    )
    print("-" * 75)
    for name, results in all_results.items():
        if not results:
            print(f"{name:<25s} FAILED")
            continue
        print(
            f"{name:<25s} "
            f"{np.mean([r['bacc'] for r in results]):>8.3f} "
            f"{np.mean([r['ate_x0'] for r in results]):>+8.3f} "
            f"{np.mean([r['ate_x4'] for r in results]):>+8.3f} "
            f"{np.mean([r['ate_x12'] for r in results]):>+9.3f} "
            f"{np.mean([r['ate_x18'] for r in results]):>+9.3f} "
            f"{np.mean([r['time'] for r in results]):>5.0f}s"
        )
    print(f"{'TRUE':<25s} {'':>8s} {'+0.600':>8s} {'+0.400':>8s} {'+0.300':>9s} {'+0.200':>9s}")


if __name__ == "__main__":
    main()
