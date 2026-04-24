#!/usr/bin/env python3
"""Smoke test at d=30: verify structural DML at breast-cancer dimensionality."""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jcce.structure_learning.jcce_learner import create_processor, learn_structure


def generate_d30_data(n=1000, seed=42):
    """Generate d=30 linear SEM with known effects."""
    rng = np.random.RandomState(seed)
    d = 30
    X = np.zeros((n, d))

    # Roots (10 independent)
    for i in [0, 1, 3, 5, 7, 10, 15, 20, 24, 28]:
        X[:, i] = rng.randn(n)

    # Chain 1: 0→2→4→6
    X[:, 2] = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.randn(n) * 0.5
    X[:, 4] = 0.4 * X[:, 2] + 0.6 * X[:, 3] + rng.randn(n) * 0.5
    X[:, 6] = 0.3 * X[:, 4] + 0.4 * X[:, 5] + rng.randn(n) * 0.5

    # Chain 2: 7→8→9→11→12
    X[:, 8] = 0.5 * X[:, 7] + rng.randn(n) * 0.5
    X[:, 9] = 0.4 * X[:, 8] + rng.randn(n) * 0.5
    X[:, 11] = 0.3 * X[:, 10] + 0.3 * X[:, 9] + rng.randn(n) * 0.5
    X[:, 12] = 0.5 * X[:, 11] + rng.randn(n) * 0.5

    # Chain 3: 15→16→17→18→19
    X[:, 13] = 0.4 * X[:, 12] + rng.randn(n) * 0.5
    X[:, 14] = 0.3 * X[:, 13] + rng.randn(n) * 0.5
    X[:, 16] = 0.5 * X[:, 15] + rng.randn(n) * 0.5
    X[:, 17] = 0.4 * X[:, 16] + rng.randn(n) * 0.5
    X[:, 18] = 0.3 * X[:, 17] + 0.2 * X[:, 9] + rng.randn(n) * 0.5
    X[:, 19] = 0.5 * X[:, 18] + rng.randn(n) * 0.5

    # Chain 4: 20→21→22→23
    X[:, 21] = 0.4 * X[:, 20] + rng.randn(n) * 0.5
    X[:, 22] = 0.3 * X[:, 21] + rng.randn(n) * 0.5
    X[:, 23] = 0.5 * X[:, 22] + rng.randn(n) * 0.5

    # Chain 5: 24→25→26→27
    X[:, 25] = 0.4 * X[:, 24] + rng.randn(n) * 0.5
    X[:, 26] = 0.3 * X[:, 25] + rng.randn(n) * 0.5
    X[:, 27] = 0.5 * X[:, 26] + rng.randn(n) * 0.5
    X[:, 29] = 0.3 * X[:, 28] + rng.randn(n) * 0.5

    # Y = 0.5*X0 + 0.3*X6 + 0.4*X12 + 0.2*X22 + 0.15*X27 + noise
    Y = (
        0.5 * X[:, 0]
        + 0.3 * X[:, 6]
        + 0.4 * X[:, 12]
        + 0.2 * X[:, 22]
        + 0.15 * X[:, 27]
        + rng.randn(n) * 0.3
    )
    Y_binary = (Y > np.median(Y)).astype(np.float32)

    true_effects = {"X0": 0.5, "X6": 0.3, "X12": 0.4, "X22": 0.2, "X27": 0.15}
    return X.astype(np.float32), Y_binary, true_effects


def run_one(X, Y, key, name, use_structural_dml, use_pcgrad, max_iter=100):
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
            "ate_x0": get_ate("X0->Y"),
            "ate_x6": get_ate("X6->Y"),
            "ate_x12": get_ate("X12->Y"),
            "ate_x22": get_ate("X22->Y"),
            "ate_x27": get_ate("X27->Y"),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"time": time.time() - t0, "error": str(e)}


def main():
    print("SMOKE TEST d=30: Structural DML at breast-cancer scale")
    print(f"Device: {jax.devices()}")
    print("=" * 80)

    X, Y, true_effects = generate_d30_data(n=1000)
    print(f"Data: n={X.shape[0]}, d={X.shape[1]}")
    print("True parents: X0(0.5), X6(0.3), X12(0.4), X22(0.2), X27(0.15)\n")

    configs = [
        ("DragonNet", False, False),
        ("StructDML+PCGrad", True, True),
    ]

    seeds = [42, 123]

    for name, sdml, pcg in configs:
        print(f"\n--- {name} ---")
        for seed in seeds:
            key = random.PRNGKey(seed)
            r = run_one(X, Y, key, name, sdml, pcg, max_iter=100)
            if "error" in r:
                print(f"  seed={seed}: FAILED ({r['error']})")
            else:
                print(
                    f"  seed={seed}: BAcc={r['bacc']:.3f} h(A)={r['h_A']:.4f} edges={r['n_edges']} "
                    f"ATEs=[{r['ate_x0']:.3f},{r['ate_x6']:.3f},{r['ate_x12']:.3f},"
                    f"{r['ate_x22']:.3f},{r['ate_x27']:.3f}] ({r['time']:.0f}s)"
                )

    print("\nTRUE: X0=0.500, X6=0.300, X12=0.400, X22=0.200, X27=0.150")


if __name__ == "__main__":
    main()
