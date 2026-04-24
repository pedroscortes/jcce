#!/usr/bin/env python3
"""
GBS Analysis on Real v16 Pareto DAGs (LUCAS + Heart Disease)

Applies GBS dequantized kernel + Hellinger uncertainty analysis to
real Pareto-optimal DAGs from completed v16 NSGA-II experiments.

Analyses:
  1. GBS kernel matrix (dequantized, orders 1-3)
  2. Classical distance matrices (SHD, Frobenius, Jaccard, Spectral)
  3. Hellinger uncertainty metrics (diameter, dispersion, coverage)
  4. Kernel-based clustering vs processor type
  5. Cross-kernel correlation (Mantel-like)
  6. Edge stability from Pareto front

Reference: Roadmap Step 1.3b (Experiment 1b) + Step 2.2 (Experiment 4)
"""

import pickle
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, ".")

from jcce.gbs.gbs_utils import (
    dequantized_kernel_matrix,
    encode_dag_to_gbs,
    frobenius_distance,
    jaccard_edge_distance,
    shd,
    spectral_distance,
)
from jcce.gbs.pareto_uncertainty import (
    dequantized_hellinger_matrix,
    hellinger_coverage,
    hellinger_diameter,
    mean_hellinger_dispersion,
    pareto_uncertainty_report,
)


def load_pareto_dags(pkl_path):
    """Load Pareto-optimal adjacency matrices from B6.pkl."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    sols = data["pareto_solutions"]
    dags = []
    metadata = []

    for sol in sols:
        A = np.array(sol["A_weights"])  # continuous weights
        dags.append(A)
        metadata.append(
            {
                "processor": sol.get("processor_type", "?"),
                "bacc": sol.get("balanced_accuracy", 0),
                "n_edges": sol.get("n_edges", 0),
                "h_A": sol.get("h_A", 0),
                "mb": sol.get("markov_blanket", []),
            }
        )

    return dags, metadata, data


def adaptive_threshold(A):
    """Match v7/v16 thresholding: max(0.03 * max_weight, 0.01)."""
    return max(0.03 * np.max(np.abs(A)), 0.01)


def compute_classical_distances(dags, threshold=None):
    """Compute pairwise distance matrices for classical metrics.

    Uses adaptive threshold per DAG pair (min of both) if threshold is None.
    """
    n = len(dags)
    D_shd = np.zeros((n, n))
    D_frob = np.zeros((n, n))
    D_jaccard = np.zeros((n, n))
    D_spectral = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Use min threshold of the pair so edges present in either DAG are counted
            t = (
                threshold
                if threshold is not None
                else min(adaptive_threshold(dags[i]), adaptive_threshold(dags[j]))
            )
            D_shd[i, j] = D_shd[j, i] = shd(dags[i], dags[j], t)
            D_frob[i, j] = D_frob[j, i] = frobenius_distance(dags[i], dags[j])
            D_jaccard[i, j] = D_jaccard[j, i] = jaccard_edge_distance(dags[i], dags[j], t)
            D_spectral[i, j] = D_spectral[j, i] = spectral_distance(dags[i], dags[j])

    return {
        "SHD": D_shd,
        "Frobenius": D_frob,
        "Jaccard": D_jaccard,
        "Spectral": D_spectral,
    }


def kernel_to_distance(K):
    """Convert kernel matrix to distance: d(i,j) = sqrt(K(i,i) + K(j,j) - 2K(i,j))."""
    diag = np.diag(K)
    D_sq = diag[:, None] + diag[None, :] - 2 * K
    D_sq = np.maximum(D_sq, 0)  # numerical safety
    return np.sqrt(D_sq)


def compute_edge_stability(dags, threshold=None):
    """Compute edge frequency across Pareto front.

    Uses adaptive threshold matching v7/v16: max(0.03 * max_weight, 0.01).
    """
    d = dags[0].shape[0]
    n = len(dags)
    freq = np.zeros((d, d))
    for A in dags:
        if threshold is None:
            # Adaptive threshold matching jcce_learner v11.1
            max_w = np.max(np.abs(A))
            t = max(0.03 * max_w, 0.01)
        else:
            t = threshold
        freq += (np.abs(A) > t).astype(float)
    freq /= n
    return freq


def mantel_correlation(D1, D2):
    """Mantel test: Pearson correlation between upper-triangular distance vectors."""
    n = D1.shape[0]
    if n < 3:
        return float("nan"), 1.0  # Need at least 3 solutions for pairwise correlation
    idx = np.triu_indices(n, k=1)
    v1 = D1[idx]
    v2 = D2[idx]
    if len(v1) < 2 or np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
        return float("nan"), 1.0  # Degenerate case
    r, p = stats.pearsonr(v1, v2)
    return r, p


def analyze_dataset(dataset_name, pkl_path):
    """Run full GBS + classical analysis on a dataset's Pareto front."""
    print(f"\n{'=' * 80}")
    print(f"GBS ANALYSIS: {dataset_name.upper()}")
    print(f"{'=' * 80}")

    # Load DAGs
    dags, meta, raw_data = load_pareto_dags(pkl_path)
    n = len(dags)
    d = dags[0].shape[0]

    print(f"\nDataset: {dataset_name}")
    print(f"Pareto solutions: {n}")
    print(f"Dimensions: {d}×{d} (d={d - 1} features + Y)")
    procs = {}
    for m in meta:
        procs[m["processor"]] = procs.get(m["processor"], 0) + 1
    print(f"Processors: {procs}")
    baccs = [m["bacc"] for m in meta]
    print(f"BAcc range: {min(baccs):.3f} – {max(baccs):.3f}")
    edges = [m["n_edges"] for m in meta]
    print(f"Edge range: {min(edges)} – {max(edges)}")

    # =========================================================================
    # 1. Classical distance matrices
    # =========================================================================
    print("\n--- Classical Distances ---")
    t0 = time.time()
    classical = compute_classical_distances(dags)
    print(f"  Computed in {time.time() - t0:.2f}s")

    for name, D in classical.items():
        idx = np.triu_indices(n, k=1)
        vals = D[idx]
        print(
            f"  {name:12s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
            f"min={vals.min():.3f}, max={vals.max():.3f}"
        )

    # =========================================================================
    # 2. GBS dequantized kernel matrices (orders 1, 2, 3)
    # =========================================================================
    print("\n--- GBS Dequantized Kernel ---")
    gbs_distances = {}

    # Encode DAGs to GBS graph matrices (symmetrize + spectral normalize)
    W_list = [encode_dag_to_gbs(A, scale=0.9) for A in dags]

    for order in [1, 2, 3]:
        t0 = time.time()
        K = dequantized_kernel_matrix(W_list, max_order=order, normalize=True)
        elapsed = time.time() - t0

        D_gbs = kernel_to_distance(K)
        gbs_distances[f"GBS-o{order}"] = D_gbs

        idx = np.triu_indices(n, k=1)
        k_vals = K[idx]
        d_vals = D_gbs[idx]
        print(
            f"  Order {order}: kernel mean={k_vals.mean():.4f}, "
            f"dist mean={d_vals.mean():.4f}, dist std={d_vals.std():.4f}  "
            f"({elapsed:.2f}s)"
        )

    # =========================================================================
    # 3. Hellinger uncertainty metrics
    # =========================================================================
    print("\n--- Hellinger Uncertainty (GBS-derived) ---")
    for order in [2, 3]:
        t0 = time.time()
        H = dequantized_hellinger_matrix(dags, max_order=order)
        elapsed = time.time() - t0

        diam = hellinger_diameter(H)
        disp = mean_hellinger_dispersion(H)
        cov = hellinger_coverage(H)

        print(f"  Order {order} ({elapsed:.2f}s):")
        print(f"    Diameter:   {diam:.4f}")
        print(f"    Dispersion: {disp:.4f}")
        print(f"    Coverage:   {cov:.4f}")

    # Full report at order 3
    report = pareto_uncertainty_report(dags, max_order=3)
    print("\n  Full report (order 3):")
    print(f"    n_dags:     {report['n_dags']}")
    print(f"    dimension:  {report['d']}")
    print(f"    diameter:   {report['diameter']:.4f}")
    print(f"    dispersion: {report['mean_dispersion']:.4f}")
    print(f"    coverage:   {report['coverage']:.4f}")

    # =========================================================================
    # 4. Cross-kernel correlations (Mantel-like)
    # =========================================================================
    print("\n--- Cross-Kernel Correlations ---")
    all_distances = {**classical, **gbs_distances}

    # Add Hellinger as a distance
    H3 = dequantized_hellinger_matrix(dags, max_order=3)
    all_distances["Hellinger"] = H3

    names = list(all_distances.keys())
    print(f"  {'':15s}", end="")
    for name in names:
        print(f"{name:>12s}", end="")
    print()

    corr_matrix = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        print(f"  {n1:15s}", end="")
        for j, n2 in enumerate(names):
            r, p = mantel_correlation(all_distances[n1], all_distances[n2])
            corr_matrix[i, j] = r
            if i == j:
                print(f"{'1.000':>12s}", end="")
            else:
                sig = "*" if p < 0.01 else " "
                print(f"{r:>11.3f}{sig}", end="")
        print()

    # =========================================================================
    # 5. Edge stability
    # =========================================================================
    print("\n--- Edge Stability (Pareto Front) ---")
    freq = compute_edge_stability(dags)  # adaptive threshold per DAG

    # Stability thresholds
    for thresh in [0.5, 0.8, 0.9, 1.0]:
        n_stable = np.sum(freq >= thresh)
        print(f"  Edges with P ≥ {thresh:.1f}: {n_stable}")

    # Most stable edges (top 10)
    edge_list = []
    for i in range(d):
        for j in range(d):
            if freq[i, j] > 0:
                edge_list.append((i, j, freq[i, j]))
    edge_list.sort(key=lambda x: -x[2])

    print("\n  Top 10 most stable edges:")
    print(f"  {'From':>6s} → {'To':>4s}  {'Freq':>6s}")
    for i, j, f in edge_list[:10]:
        print(f"  {i:>6d} → {j:>4d}  {f:>6.2f}")

    # =========================================================================
    # 6. Processor-specific analysis
    # =========================================================================
    unique_procs = list(set(m["processor"] for m in meta))
    if len(unique_procs) > 1:
        print("\n--- Processor-Specific Analysis ---")
        # Mean within-processor vs between-processor distance (GBS order 3)
        D = gbs_distances["GBS-o3"]
        within = []
        between = []
        for i in range(n):
            for j in range(i + 1, n):
                if meta[i]["processor"] == meta[j]["processor"]:
                    within.append(D[i, j])
                else:
                    between.append(D[i, j])
        if within and between:
            print("  GBS-o3 distance:")
            print(f"    Within-processor:  {np.mean(within):.4f} ± {np.std(within):.4f}")
            print(f"    Between-processor: {np.mean(between):.4f} ± {np.std(between):.4f}")
            ratio = np.mean(between) / (np.mean(within) + 1e-10)
            print(f"    Ratio (between/within): {ratio:.2f}x")

        # Per-processor stats
        for proc in unique_procs:
            idx = [i for i, m in enumerate(meta) if m["processor"] == proc]
            proc_baccs = [meta[i]["bacc"] for i in idx]
            proc_edges = [meta[i]["n_edges"] for i in idx]
            print(f"\n  {proc} ({len(idx)} solutions):")
            print(f"    BAcc: {np.mean(proc_baccs):.3f} ± {np.std(proc_baccs):.3f}")
            print(f"    Edges: {np.mean(proc_edges):.1f} ± {np.std(proc_edges):.1f}")
    else:
        print("\n--- Processor Analysis ---")
        print(f"  Single processor ({unique_procs[0]}) — no between-processor comparison")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {dataset_name.upper()}")
    print(f"{'=' * 80}")

    # Structural diversity interpretation
    H3_report = pareto_uncertainty_report(dags, max_order=3)
    diam = H3_report["diameter"]
    disp = H3_report["mean_dispersion"]

    if diam < 0.3:
        diversity = "LOW — Pareto front is structurally homogeneous"
    elif diam < 0.7:
        diversity = "MODERATE — some structural variation across solutions"
    else:
        diversity = "HIGH — structurally diverse Pareto front"

    print(f"  Structural diversity: {diversity}")
    print(f"  Hellinger diameter: {diam:.4f}, dispersion: {disp:.4f}")

    # GBS vs classical agreement
    r_gbs_shd, _ = mantel_correlation(gbs_distances["GBS-o3"], classical["SHD"])
    r_hell_shd, _ = mantel_correlation(H3, classical["SHD"])
    print(f"  GBS-SHD correlation: r={r_gbs_shd:.3f}")
    print(f"  Hellinger-SHD correlation: r={r_hell_shd:.3f}")

    if abs(r_gbs_shd) < 0.5:
        print("  → GBS captures structural info BEYOND edge counting (low SHD correlation)")
    else:
        print("  → GBS correlates with SHD (similar structural signal)")

    return {
        "dataset": dataset_name,
        "n_solutions": n,
        "dimension": d,
        "classical_distances": classical,
        "gbs_distances": gbs_distances,
        "hellinger_report": H3_report,
        "edge_stability": freq,
        "metadata": meta,
    }


def main():
    print("GBS ANALYSIS ON REAL v16 PARETO DAGS")
    print("=" * 80)
    print("Experiment 1b (kernel on real DAGs) + Experiment 4 (Hellinger uncertainty)")
    print()

    results = {}

    import os

    dataset_names = [
        "lucas",
        "heart_disease",
        "breast_cancer",
        "asia",
        "diabetes",
        "sachs",
        "child",
        "insurance",
        "neuropathic_pain",
        "alarm",
    ]
    for dataset in dataset_names:
        # Try NSGA2 (ablation) first, then Optuna
        path = f"results/ablation_v16/{dataset}/B6.pkl"
        if not os.path.exists(path):
            path = f"results/optuna_v16/{dataset}/optuna_seed42.pkl"
        if not os.path.exists(path):
            print(f"\n  SKIP {dataset}: no pkl found")
            continue
        results[dataset] = analyze_dataset(dataset, path)

    # =========================================================================
    # Cross-dataset comparison
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'=' * 80}")

    print(
        f"\n{'Dataset':15s} {'d':>3s} {'#Sol':>5s} {'BAcc':>12s} {'Edges':>12s} "
        f"{'H-diam':>8s} {'H-disp':>8s} {'H-cov':>8s} {'Processors':>15s}"
    )
    print("-" * 95)

    for name, r in results.items():
        meta = r["metadata"]
        baccs = [m["bacc"] for m in meta]
        edges = [m["n_edges"] for m in meta]
        procs = set(m["processor"] for m in meta)
        hr = r["hellinger_report"]
        print(
            f"{name:15s} {r['dimension']:>3d} {r['n_solutions']:>5d} "
            f"{np.mean(baccs):.3f}±{np.std(baccs):.3f} "
            f"{np.mean(edges):>5.1f}±{np.std(edges):<5.1f} "
            f"{hr['diameter']:>8.4f} {hr['mean_dispersion']:>8.4f} {hr['coverage']:>8.4f} "
            f"{','.join(sorted(procs)):>15s}"
        )

    # Save results
    output_path = "results/gbs_real_pareto_analysis.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
