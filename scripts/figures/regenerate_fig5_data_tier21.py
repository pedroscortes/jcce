"""Regenerate the data needed for Fig 5 on the Tier-21 redesigned synthetic.

Tier-21 (test_synthetic_jpc_v2.py) reports BAcc + |abs_dT| per cell but does NOT
save struct_corr. Fig 5 needs per-cell (test_bacc, struct_corr). This script
re-runs the Tier-21 linear_d20 config with struct_corr computation added and
saves to results/server/fig5_tier21_data.json for the plotter.

Configurations: linear ER-SCM, d=20, n=2000 (the Tier-21 headline config) ×
3 processors (linear_head, mlp_head, dag_transformer) × 3 fixes (no_fix,
post_hoc, warm_start) × 5 seeds = 45 cells × ~15s CPU each = ~11 min total.

Usage (jcce main venv):
    python scripts/figures/regenerate_fig5_data_tier21.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "theory"))

from test_synthetic_jpc_v2 import (
    generate_clean_synthetic, train_jcce, jcce_test_bacc, post_hoc_fix,
)

OUT_PATH = REPO_ROOT / "results" / "server" / "fig5_tier21_data.json"


def struct_corr(A_learned: np.ndarray, A_Y_true_vec: np.ndarray, n_features: int) -> float:
    """Pearson correlation between learned |A[:, Y_idx]| and ground-truth |A_true[:, Y]|."""
    Y_idx = n_features
    learned_Y_col = np.abs(A_learned[:n_features, Y_idx])
    true_Y_col = np.abs(A_Y_true_vec)
    if learned_Y_col.std() < 1e-8 or true_Y_col.std() < 1e-8:
        return float("nan")
    r, _ = pearsonr(learned_Y_col, true_Y_col)
    return float(r)


def run_one(seed: int, processor: str, fix: str, max_iter: int = 150,
             warm_iters: int = 50):
    X, Y, T_idx, A_Y_true_vec, _ = generate_clean_synthetic(
        d=20, n=2000, seed=seed, scm_type="linear", noise_scale=0.5,
    )
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y).astype(int)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=seed,
    )
    n_features = X_tr.shape[1]

    if fix == "warm_start":
        A, proc, params = train_jcce(X_tr, Y_tr, T_idx, processor, seed,
                                       max_iter=max_iter, warm_iters=warm_iters)
        bacc = jcce_test_bacc(proc, A, params, X_tr, Y_tr, X_te, Y_te)
    else:
        A, proc, params = train_jcce(X_tr, Y_tr, T_idx, processor, seed,
                                       max_iter=max_iter, warm_iters=0)
        if fix == "no_fix":
            bacc = jcce_test_bacc(proc, A, params, X_tr, Y_tr, X_te, Y_te)
        elif fix == "post_hoc":
            bacc = post_hoc_fix(A, X_tr, Y_tr, X_te, Y_te)
        else:
            raise ValueError(f"unknown fix: {fix}")

    sc = struct_corr(A, A_Y_true_vec, n_features)
    return {
        "seed": seed,
        "processor": processor,
        "fix": fix,
        "test_bacc": float(bacc),
        "struct_corr": sc,
    }


def main():
    print("Regenerating Fig 5 data on Tier-21 (linear, d=20, n=2000)")
    procs = ["linear_head", "mlp_head", "dag_transformer"]
    fixes = ["no_fix", "post_hoc", "warm_start"]
    seeds = list(range(5))

    rows = []
    total = len(procs) * len(fixes) * len(seeds)
    n = 0
    t_start = time.time()
    for proc in procs:
        for fix in fixes:
            for seed in seeds:
                n += 1
                t0 = time.time()
                try:
                    r = run_one(seed, proc, fix)
                    rows.append(r)
                    print(f"  [{n:>2d}/{total}] {proc:>16s} + {fix:>10s} + seed={seed}: "
                          f"BAcc={r['test_bacc']:.3f}, struct_corr={r['struct_corr']:+.3f}  "
                          f"({time.time()-t0:.0f}s)")
                except Exception as exc:
                    print(f"  [{n:>2d}/{total}] {proc:>16s} + {fix:>10s} + seed={seed}: "
                          f"ERROR {type(exc).__name__}: {exc}")

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
