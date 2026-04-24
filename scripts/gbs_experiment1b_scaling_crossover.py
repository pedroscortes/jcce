#!/usr/bin/env python3
"""
GBS Experiment 1b: Kernel Scaling Crossover Analysis

Experiment 1 showed:
  - d=11: classical wins (SHD ARI=0.646, GBS ARI=0.145)
  - d=30: GBS wins (GBS ARI=0.751, Jaccard ARI=0.612)

This experiment investigates the CROSSOVER POINT by testing intermediate
dimensions: d=13, 15, 18, 20, 25.

Also tests order 2 vs order 3 features to understand which contributes
more at different scales.

Reference: Extension of Roadmap Step 1.3
"""

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Number of distinct clusters")

sys.path.insert(0, ".")


# Import generators from Experiment 1
from gbs_experiment1_discriminative import (
    compute_distance_matrix,
    evaluate_distance_method,
    evaluate_kernel_method,
    generate_dag_families,
    robust_dequantized_kernel_matrix,
)

from jcce.gbs.gbs_utils import (
    encode_dag_to_gbs,
    frobenius_distance,
    jaccard_edge_distance,
    shd,
    spectral_distance,
)


def run_scaling_experiment(dimensions: list, n_per_family: int = 30, seed: int = 42):
    """Run the discriminative experiment across multiple dimensions."""
    print(f"\n{'=' * 80}")
    print("  GBS Kernel Scaling Crossover Analysis")
    print(f"  Dimensions: {dimensions}")
    print(f"  DAGs per family: {n_per_family} × 3 = {n_per_family * 3} total")
    print(f"{'=' * 80}")

    all_results = {}

    for d in dimensions:
        print(f"\n{'─' * 70}")
        print(f"  d = {d}")
        print(f"{'─' * 70}")

        # Generate DAGs
        t0 = time.time()
        dags, labels, family_names = generate_dag_families(d, n_per_family, seed)
        print(f"  Generated {len(dags)} DAGs in {time.time() - t0:.2f}s")

        # Edge density stats
        for fam_idx, fam_name in enumerate(family_names):
            fam_dags = [dags[i] for i in range(len(dags)) if labels[i] == fam_idx]
            densities = [np.sum(np.abs(dag) > 0.1) / (d * (d - 1)) for dag in fam_dags]
            print(f"    {fam_name}: density={np.mean(densities):.3f}")

        # Encode for GBS
        W_list = [encode_dag_to_gbs(dag, scale=0.9) for dag in dags]

        results = {}

        # GBS order 2
        t0 = time.time()
        K2 = robust_dequantized_kernel_matrix(W_list, max_order=2, normalize=True)
        t2 = time.time() - t0
        res2 = evaluate_kernel_method(K2, labels, seed=seed)
        results["GBS-deq (o2)"] = {**res2, "time": t2}
        print(f"  GBS o2: ARI={res2['ari']:.4f} ({t2:.1f}s)")

        # GBS order 3
        t0 = time.time()
        K3 = robust_dequantized_kernel_matrix(W_list, max_order=3, normalize=True)
        t3 = time.time() - t0
        res3 = evaluate_kernel_method(K3, labels, seed=seed)
        results["GBS-deq (o3)"] = {**res3, "time": t3}
        print(f"  GBS o3: ARI={res3['ari']:.4f} ({t3:.1f}s)")

        # Classical methods
        for name, metric_fn in [
            ("SHD", lambda a, b: float(shd(a, b))),
            ("Frobenius", frobenius_distance),
            ("Spectral", spectral_distance),
            ("Jaccard", jaccard_edge_distance),
        ]:
            t0 = time.time()
            D = compute_distance_matrix(dags, metric_fn)
            tc = time.time() - t0
            res = evaluate_distance_method(D, labels, seed=seed)
            results[name] = {**res, "time": tc}
            print(f"  {name}: ARI={res['ari']:.4f} ({tc:.1f}s)")

        all_results[d] = results

    return all_results


def analyze_crossover(all_results: dict):
    """Analyze the crossover point where GBS becomes advantageous."""
    print(f"\n{'=' * 80}")
    print("  CROSSOVER ANALYSIS")
    print(f"{'=' * 80}")

    dims = sorted(all_results.keys())
    methods = ["GBS-deq (o2)", "GBS-deq (o3)", "SHD", "Frobenius", "Spectral", "Jaccard"]

    # Table: ARI by dimension
    print(f"\n  {'Method':<18}", end="")
    for d in dims:
        print(f" | {'d=' + str(d):>7}", end="")
    print()
    print(f"  {'-' * (18 + len(dims) * 11)}")

    for m in methods:
        print(f"  {m:<18}", end="")
        for d in dims:
            ari = all_results[d][m]["ari"]
            print(f" | {ari:>7.4f}", end="")
        print()

    # Find crossover point: where GBS o3 beats best classical
    print("\n  Crossover detection:")
    for d in dims:
        gbs_ari = all_results[d]["GBS-deq (o3)"]["ari"]
        best_classical = max(
            all_results[d][m]["ari"] for m in ["SHD", "Frobenius", "Spectral", "Jaccard"]
        )
        best_classical_name = max(
            ["SHD", "Frobenius", "Spectral", "Jaccard"], key=lambda m: all_results[d][m]["ari"]
        )
        delta = gbs_ari - best_classical
        winner = "GBS" if delta > 0 else "Classical"
        print(
            f"    d={d:>2}: GBS o3={gbs_ari:.4f} vs {best_classical_name}={best_classical:.4f} "
            f"→ delta={delta:+.4f} → {winner}"
        )

    # Order 2 vs Order 3 analysis
    print("\n  Order 2 vs Order 3 (GBS advantage of higher order):")
    for d in dims:
        o2 = all_results[d]["GBS-deq (o2)"]["ari"]
        o3 = all_results[d]["GBS-deq (o3)"]["ari"]
        delta = o3 - o2
        print(f"    d={d:>2}: o2={o2:.4f}, o3={o3:.4f}, delta={delta:+.4f}")

    # Timing analysis
    print("\n  Timing (seconds for kernel matrix):")
    print(f"  {'Method':<18}", end="")
    for d in dims:
        print(f" | {'d=' + str(d):>7}", end="")
    print()
    print(f"  {'-' * (18 + len(dims) * 11)}")
    for m in methods:
        print(f"  {m:<18}", end="")
        for d in dims:
            t = all_results[d][m]["time"]
            print(f" | {t:>6.1f}s", end="")
        print()


def main():
    print("GBS Experiment 1b: Kernel Scaling Crossover Analysis")
    print("=" * 50)

    # Test dimensions between d=11 (classical wins) and d=30 (GBS wins)
    # Use fewer DAGs per family (30 instead of 50) for speed
    dimensions = [11, 13, 15, 18, 20, 25, 30]

    all_results = run_scaling_experiment(dimensions, n_per_family=30, seed=42)
    analyze_crossover(all_results)

    print(f"\n{'=' * 80}")
    print("  Experiment 1b complete.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
