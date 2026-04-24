#!/usr/bin/env python3
"""
GBS Experiment 3: Hafnian Ablation (CRITICAL)

Tests whether the GBS-specific hafnian weighting provides genuine value
for Markov Blanket detection compared to simpler subgraph sampling methods.

DeepSeek's key question: "Is hafnian-weighted sampling better than
uniform subgraph sampling for finding MB members?"

Methods compared:
  1. GBS dequantized co-occurrence (hafnian-weighted marginal click probs)
  2. Uniform subgraph sampling (random subsets, no weighting)
  3. Degree-weighted sampling (probability proportional to node degrees)
  4. Thresholded partial correlation (classical baseline)

Ground truth: Known MB from synthetic DAGs (BnLearn-style).
Metrics: MB precision, recall, F1.

Reference: Roadmap Step 3.4 (prompts/gbs_mb_roadmap.md)
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from jcce.gbs.gbs_utils import (
    dequantized_cooccurrence,
    encode_dag_to_gbs,
)

# =============================================================================
# Ground-truth MB extraction
# =============================================================================


def true_markov_blanket(A: np.ndarray, target: int, threshold: float = 0.1) -> set:
    """
    Extract ground-truth Markov blanket from a DAG adjacency matrix.

    MB(target) = parents(target) ∪ children(target) ∪ spouses(target)
    where spouses are other parents of target's children.

    Args:
        A: (d, d) directed adjacency matrix. A[i,j] > 0 means i → j.
        target: target variable index.
        threshold: minimum weight to count as an edge.

    Returns:
        Set of variable indices in the Markov blanket of target.
    """
    d = A.shape[0]
    A_bin = (np.abs(A) > threshold).astype(int)

    parents = set(np.where(A_bin[:, target] > 0)[0])
    children = set(np.where(A_bin[target, :] > 0)[0])

    # Spouses: other parents of target's children
    spouses = set()
    for child in children:
        child_parents = set(np.where(A_bin[:, child] > 0)[0])
        spouses |= child_parents

    mb = (parents | children | spouses) - {target}
    return mb


# =============================================================================
# Co-occurrence methods
# =============================================================================


def gbs_cooccurrence_method(A: np.ndarray, scale: float = 0.9) -> np.ndarray:
    """
    Method 1: GBS dequantized co-occurrence.

    Uses exact marginal click probabilities from the Gaussian state encoding.
    This is the GBS-specific method we want to evaluate.
    """
    W = encode_dag_to_gbs(A, scale=scale)
    return dequantized_cooccurrence(W)


def uniform_subgraph_cooccurrence(
    A: np.ndarray,
    n_samples: int = 10000,
    k_range: tuple = (2, 6),
    seed: int = 42,
) -> np.ndarray:
    """
    Method 2: Uniform random subgraph sampling.

    Samples random subsets of variables uniformly, then counts co-occurrences.
    This is the simplest possible baseline — no structure-aware weighting.

    Args:
        A: (d, d) adjacency matrix (used only for dimension).
        n_samples: number of random subsets to draw.
        k_range: (min_size, max_size) of subsets.
        seed: random seed.

    Returns:
        C: (d, d) co-occurrence matrix, values in [0, 1].
    """
    d = A.shape[0]
    rng = np.random.default_rng(seed)

    C = np.zeros((d, d))
    for _ in range(n_samples):
        k = rng.integers(k_range[0], min(k_range[1] + 1, d + 1))
        subset = rng.choice(d, size=k, replace=False)
        for i in subset:
            for j in subset:
                C[i, j] += 1

    C /= n_samples
    return C


def degree_weighted_cooccurrence(
    A: np.ndarray,
    n_samples: int = 10000,
    k_range: tuple = (2, 6),
    seed: int = 42,
) -> np.ndarray:
    """
    Method 3: Degree-weighted subgraph sampling.

    Variables with more edges are more likely to be included in subsets.
    More informative than uniform but still doesn't use hafnian structure.

    Args:
        A: (d, d) adjacency matrix.
        n_samples: number of random subsets.
        k_range: (min_size, max_size) of subsets.
        seed: random seed.

    Returns:
        C: (d, d) co-occurrence matrix, values in [0, 1].
    """
    d = A.shape[0]
    A_sym = (np.abs(A) + np.abs(A).T) / 2.0
    rng = np.random.default_rng(seed)

    # Degree = sum of edge weights
    degrees = A_sym.sum(axis=1) + 1e-6  # avoid zero
    probs = degrees / degrees.sum()

    C = np.zeros((d, d))
    for _ in range(n_samples):
        k = rng.integers(k_range[0], min(k_range[1] + 1, d + 1))
        k = min(k, d)
        subset = rng.choice(d, size=k, replace=False, p=probs)
        for i in subset:
            for j in subset:
                C[i, j] += 1

    C /= n_samples
    return C


def partial_corr_cooccurrence(
    A: np.ndarray,
    n_data: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Method 4: Thresholded partial correlation (classical baseline).

    Generates data from a linear SEM with the given DAG, computes partial
    correlations, and returns the absolute partial correlation matrix.

    Args:
        A: (d, d) adjacency matrix (used as SEM weights).
        n_data: number of data samples to generate.
        seed: random seed.

    Returns:
        C: (d, d) absolute partial correlation matrix.
    """
    d = A.shape[0]
    rng = np.random.default_rng(seed)

    # Generate data from linear SEM: X = A^T X + eps
    X = np.zeros((n_data, d))
    # Topological order: find a valid ordering
    order = _topological_sort(A)

    for sample_idx in range(n_data):
        noise = rng.standard_normal(d)
        x = np.zeros(d)
        for node in order:
            parent_vals = A[:, node] @ x
            x[node] = parent_vals + noise[node]
        X[sample_idx] = x

    # Partial correlation via precision matrix
    cov = np.cov(X, rowvar=False)
    cov += np.eye(d) * 1e-6
    from scipy import linalg

    prec = linalg.inv(cov)

    diag = np.sqrt(np.abs(np.diag(prec)))
    C = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                C[i, j] = abs(prec[i, j] / (diag[i] * diag[j] + 1e-10))

    return C


def _topological_sort(A: np.ndarray, threshold: float = 0.1) -> list:
    """Return nodes in topological order (parents before children)."""
    d = A.shape[0]
    A_bin = (np.abs(A) > threshold).astype(int)
    in_degree = A_bin.sum(axis=0)
    order = []
    remaining = set(range(d))

    while remaining:
        # Find nodes with no incoming edges from remaining nodes
        ready = [
            n
            for n in remaining
            if all(A_bin[p, n] == 0 or p not in remaining for p in range(d) if p != n)
        ]
        if not ready:
            # Cycle or numerical issue — just add remaining in order
            order.extend(sorted(remaining))
            break
        # Pick node with smallest in-degree
        node = min(ready, key=lambda n: in_degree[n])
        order.append(node)
        remaining.remove(node)

    return order


# =============================================================================
# MB prediction from co-occurrence
# =============================================================================


def predict_mb_from_cooccurrence(
    C: np.ndarray,
    target: int,
    threshold: str | float = "auto",
    max_mb_size: int | None = None,
) -> set:
    """
    Predict Markov blanket from a co-occurrence matrix.

    Variables with high co-occurrence with the target are predicted as MB members.

    Args:
        C: (d, d) co-occurrence or similarity matrix.
        target: target variable index.
        threshold: 'auto' (mean + 1 std), 'top_k' (top max_mb_size), or float.
        max_mb_size: maximum MB size (for 'top_k' mode).

    Returns:
        Set of predicted MB variable indices.
    """
    d = C.shape[0]
    scores = C[target, :].copy()
    scores[target] = 0  # exclude self

    if threshold == "auto":
        nonzero = scores[scores > 0]
        if len(nonzero) == 0:
            return set()
        thresh = np.mean(nonzero) + np.std(nonzero)
        predicted = set(np.where(scores > thresh)[0])
    elif threshold == "top_k":
        if max_mb_size is None:
            max_mb_size = max(3, d // 3)
        top_indices = np.argsort(scores)[::-1][:max_mb_size]
        predicted = set(top_indices[scores[top_indices] > 0])
    else:
        predicted = set(np.where(scores > float(threshold))[0])

    return predicted - {target}


def compute_mb_metrics(predicted: set, true_mb: set) -> dict:
    """
    Compute MB detection metrics.

    Returns:
        dict with 'precision', 'recall', 'f1', 'predicted_size', 'true_size'.
    """
    if len(predicted) == 0 and len(true_mb) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "predicted_size": 0, "true_size": 0}

    tp = len(predicted & true_mb)
    fp = len(predicted - true_mb)
    fn = len(true_mb - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_size": len(predicted),
        "true_size": len(true_mb),
    }


# =============================================================================
# Synthetic DAG generators
# =============================================================================


def make_chain_dag(d: int, rng=None) -> np.ndarray:
    """Chain: 0→1→2→...→d-1"""
    if rng is None:
        rng = np.random.default_rng(42)
    A = np.zeros((d, d))
    for i in range(d - 1):
        A[i, i + 1] = rng.uniform(0.5, 1.5)
    return A


def make_fork_dag(d: int, rng=None) -> np.ndarray:
    """Fork: node 0 is parent of all others (star with v-structures)."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = np.zeros((d, d))
    for i in range(1, d):
        A[0, i] = rng.uniform(0.5, 1.5)
    # Add some v-structures: nodes 1,2 → node d-1
    if d > 3:
        A[1, d - 1] = rng.uniform(0.5, 1.5)
        A[2, d - 1] = rng.uniform(0.5, 1.5)
    return A


def make_collider_dag(d: int, rng=None) -> np.ndarray:
    """
    DAG with multiple v-structures (colliders).

    Structure: d//2 sources → central collider node ← d//2 sources
    Plus: collider → remaining nodes
    This is the hardest case for GBS since spouses have no direct edge.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    A = np.zeros((d, d))
    collider = d // 2
    # First half: sources → collider
    for i in range(collider):
        A[i, collider] = rng.uniform(0.5, 1.5)
    # Second half: collider → sinks
    for i in range(collider + 1, d):
        A[collider, i] = rng.uniform(0.5, 1.5)
    return A


def make_random_dag(d: int, edge_prob: float = 0.3, rng=None) -> np.ndarray:
    """Random sparse DAG."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                A[i, j] = rng.uniform(0.3, 1.5)
    return A


# =============================================================================
# Main experiment
# =============================================================================


def run_ablation_for_dag(
    A: np.ndarray,
    dag_name: str,
    targets: list[int] | None = None,
    seed: int = 42,
):
    """
    Run hafnian ablation for a single DAG across multiple target variables.

    Returns per-method average metrics.
    """
    d = A.shape[0]
    if targets is None:
        # Test all nodes with non-empty MB
        targets = []
        for t in range(d):
            mb = true_markov_blanket(A, t)
            if len(mb) > 0:
                targets.append(t)
        if len(targets) > 10:
            # Subsample if too many
            rng = np.random.default_rng(seed)
            targets = sorted(rng.choice(targets, size=10, replace=False).tolist())

    if not targets:
        print(f"  {dag_name}: no valid targets (all MB empty)")
        return None

    # Compute co-occurrence matrices for all methods
    print(f"\n  {dag_name} (d={d}, {len(targets)} targets):")

    methods = {}

    t0 = time.time()
    C_gbs = gbs_cooccurrence_method(A)
    t_gbs = time.time() - t0
    methods["GBS-deq"] = (C_gbs, t_gbs)

    t0 = time.time()
    C_unif = uniform_subgraph_cooccurrence(A, seed=seed)
    t_unif = time.time() - t0
    methods["Uniform"] = (C_unif, t_unif)

    t0 = time.time()
    C_deg = degree_weighted_cooccurrence(A, seed=seed)
    t_deg = time.time() - t0
    methods["Degree-wtd"] = (C_deg, t_deg)

    t0 = time.time()
    C_pcorr = partial_corr_cooccurrence(A, seed=seed)
    t_pcorr = time.time() - t0
    methods["Partial-corr"] = (C_pcorr, t_pcorr)

    # Evaluate each method on each target
    results = {name: {"precision": [], "recall": [], "f1": []} for name in methods}

    for target in targets:
        true_mb = true_markov_blanket(A, target)
        if len(true_mb) == 0:
            continue

        for name, (C, _) in methods.items():
            # Use top-k threshold with true MB size as guide
            pred = predict_mb_from_cooccurrence(
                C,
                target,
                threshold="top_k",
                max_mb_size=max(len(true_mb) + 2, d // 3),
            )
            metrics = compute_mb_metrics(pred, true_mb)
            results[name]["precision"].append(metrics["precision"])
            results[name]["recall"].append(metrics["recall"])
            results[name]["f1"].append(metrics["f1"])

    # Print per-method averages
    print(f"    {'Method':<15} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Time':>8}")
    print(f"    {'-' * 47}")
    summary = {}
    for name in methods:
        if results[name]["f1"]:
            avg_p = np.mean(results[name]["precision"])
            avg_r = np.mean(results[name]["recall"])
            avg_f1 = np.mean(results[name]["f1"])
            _, t = methods[name]
            print(f"    {name:<15} {avg_p:>7.3f} {avg_r:>7.3f} {avg_f1:>7.3f} {t:>7.3f}s")
            summary[name] = {"precision": avg_p, "recall": avg_r, "f1": avg_f1, "time": t}

    return summary


def main():
    print("=" * 70)
    print("  GBS Experiment 3: Hafnian Ablation")
    print("=" * 70)
    print("Does GBS (hafnian-weighted) co-occurrence outperform")
    print("uniform/degree-weighted sampling for Markov blanket detection?")

    rng = np.random.default_rng(42)
    all_results = {}

    # Test on multiple DAG structures and dimensions
    test_configs = [
        # (name, dimension, generator, num_instances)
        ("Chain-8", 8, make_chain_dag, 5),
        ("Fork-8", 8, make_fork_dag, 5),
        ("Collider-8", 8, make_collider_dag, 5),
        ("Random-8", 8, lambda d, rng=None: make_random_dag(d, 0.3, rng), 5),
        ("Chain-11", 11, make_chain_dag, 5),
        ("Fork-11", 11, make_fork_dag, 5),
        ("Collider-11", 11, make_collider_dag, 5),
        ("Random-11", 11, lambda d, rng=None: make_random_dag(d, 0.3, rng), 5),
        ("Random-13", 13, lambda d, rng=None: make_random_dag(d, 0.25, rng), 3),
        ("Random-20", 20, lambda d, rng=None: make_random_dag(d, 0.15, rng), 3),
    ]

    for config_name, d, gen_fn, n_instances in test_configs:
        print(f"\n{'─' * 70}")
        print(f"  Configuration: {config_name}")
        print(f"{'─' * 70}")

        config_results = {
            name: {"precision": [], "recall": [], "f1": [], "time": []}
            for name in ["GBS-deq", "Uniform", "Degree-wtd", "Partial-corr"]
        }

        for inst in range(n_instances):
            seed = 42 + inst * 100
            inst_rng = np.random.default_rng(seed)
            A = gen_fn(d, rng=inst_rng)

            summary = run_ablation_for_dag(A, f"{config_name}_inst{inst}", seed=seed)
            if summary:
                for name, metrics in summary.items():
                    config_results[name]["precision"].append(metrics["precision"])
                    config_results[name]["recall"].append(metrics["recall"])
                    config_results[name]["f1"].append(metrics["f1"])
                    config_results[name]["time"].append(metrics["time"])

        all_results[config_name] = config_results

    # ==========================================================================
    # Summary table
    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Hafnian Ablation Results")
    print(f"{'=' * 70}")

    method_names = ["GBS-deq", "Uniform", "Degree-wtd", "Partial-corr"]

    print(f"\n{'Config':<15}", end="")
    for name in method_names:
        print(f" | {name:>12}", end="")
    print()
    print(f"{'':15}", end="")
    for _ in method_names:
        print(f" | {'F1':>12}", end="")
    print()
    print("-" * (15 + len(method_names) * 16))

    for config_name in all_results:
        row = f"{config_name:<15}"
        for name in method_names:
            f1_vals = all_results[config_name][name]["f1"]
            if f1_vals:
                row += f" | {np.mean(f1_vals):>12.3f}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    # Overall averages
    print("-" * (15 + len(method_names) * 16))
    row = f"{'OVERALL':<15}"
    for name in method_names:
        all_f1 = []
        for config_name in all_results:
            all_f1.extend(all_results[config_name][name]["f1"])
        if all_f1:
            row += f" | {np.mean(all_f1):>12.3f}"
        else:
            row += f" | {'N/A':>12}"
    print(row)

    # Detailed breakdown: which structures does GBS win on?
    print(f"\n{'=' * 70}")
    print("  GBS vs Uniform: Per-Structure Comparison")
    print(f"{'=' * 70}")
    print(f"{'Config':<15} {'GBS F1':>8} {'Unif F1':>8} {'Delta':>8} {'Winner':>10}")
    print("-" * 55)

    gbs_wins = 0
    total = 0
    for config_name in all_results:
        gbs_f1 = all_results[config_name]["GBS-deq"]["f1"]
        unif_f1 = all_results[config_name]["Uniform"]["f1"]
        if gbs_f1 and unif_f1:
            g = np.mean(gbs_f1)
            u = np.mean(unif_f1)
            delta = g - u
            winner = "GBS" if delta > 0.01 else ("Uniform" if delta < -0.01 else "Tie")
            print(f"{config_name:<15} {g:>8.3f} {u:>8.3f} {delta:>+8.3f} {winner:>10}")
            if delta > 0.01:
                gbs_wins += 1
            total += 1

    print(f"\nGBS wins on {gbs_wins}/{total} configurations")

    # GO/NO-GO decision
    print(f"\n{'=' * 70}")
    print("  GO/NO-GO: Hafnian Ablation Decision")
    print(f"{'=' * 70}")

    all_gbs_f1 = []
    all_unif_f1 = []
    for config_name in all_results:
        all_gbs_f1.extend(all_results[config_name]["GBS-deq"]["f1"])
        all_unif_f1.extend(all_results[config_name]["Uniform"]["f1"])

    if all_gbs_f1 and all_unif_f1:
        gbs_mean = np.mean(all_gbs_f1)
        unif_mean = np.mean(all_unif_f1)
        delta = gbs_mean - unif_mean

        print(f"  GBS-deq overall F1:  {gbs_mean:.3f}")
        print(f"  Uniform overall F1:  {unif_mean:.3f}")
        print(f"  Delta (GBS - Unif):  {delta:+.3f}")
        print()

        if delta > 0.05:
            print("  VERDICT: GBS hafnian weighting provides GENUINE value")
            print("  → Proceed with GBS-MB integration (Application B)")
        elif delta > 0.0:
            print("  VERDICT: GBS provides MARGINAL advantage")
            print("  → GBS-MB may not justify complexity; focus on Application A/D")
        else:
            print("  VERDICT: Uniform sampling matches or beats GBS")
            print("  → Hafnian weighting adds NO value for MB detection")
            print("  → Frame as characterized negative result")
            print("  → Focus on Application A (kernel) and D (Hellinger)")


if __name__ == "__main__":
    main()
