#!/usr/bin/env python3
"""
GBS Experiment 2: Spouse Detection (Collider Test + Moralization)

Tests Gemini's key insight: GBS systematically fails on v-structures (A→C←B)
because spouses A and B have zero partial correlation. Moralization adds the
missing A-B edge, turning the v-structure into a clique that GBS can detect.

Methods compared:
  1. GBS-raw: dequantized co-occurrence on encode_dag_to_gbs (symmetrized DAG)
  2. GBS-moralized: dequantized co-occurrence on encode_moralized_to_gbs
  3. Partial-corr: partial correlation from linear SEM data (classical baseline)

Metrics:
  - Spouse detection rate: fraction of true spouse pairs detected
  - Full MB precision, recall, F1
  - Per-relationship breakdown: parents, children, spouses

Varies:
  - Number of v-structures (1 to 6)
  - Dimension (d=8, 11, 15, 20)
  - Edge strength (weak=0.3-0.5, medium=0.5-1.0, strong=1.0-2.0)

Reference: Roadmap Step 3.3 / Experiment 2 (prompts/gbs_mb_roadmap.md)
"""

import sys

import numpy as np
from scipy import linalg

sys.path.insert(0, ".")

from jcce.gbs.gbs_utils import (
    dequantized_cooccurrence,
    encode_dag_to_gbs,
    encode_moralized_to_gbs,
)

# =============================================================================
# DAG generators with controlled v-structures
# =============================================================================


def make_vstruct_dag(
    d: int,
    n_vstructs: int = 3,
    edge_range: tuple = (0.5, 1.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a DAG with a controlled number of v-structures.

    Layout:
      - First n_vstructs*2 nodes are "source" nodes (paired as spouses)
      - Next n_vstructs nodes are "collider" nodes (children of spouse pairs)
      - Remaining nodes form a chain from the last collider

    V-structure k: source[2k] → collider[k] ← source[2k+1]
    Spouses source[2k] and source[2k+1] are NOT directly connected.

    Args:
        d: total number of variables (must be >= 3*n_vstructs + 1).
        n_vstructs: number of v-structures to include.
        edge_range: (min_weight, max_weight) for edge strengths.
        rng: random number generator.

    Returns:
        A: (d, d) adjacency matrix with exactly n_vstructs v-structures.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    min_d = 3 * n_vstructs
    if d < min_d:
        raise ValueError(f"Need d >= {min_d} for {n_vstructs} v-structures, got d={d}")

    A = np.zeros((d, d))

    # Create v-structures
    source_nodes = list(range(2 * n_vstructs))
    collider_nodes = list(range(2 * n_vstructs, 3 * n_vstructs))

    for k in range(n_vstructs):
        parent_a = source_nodes[2 * k]
        parent_b = source_nodes[2 * k + 1]
        child = collider_nodes[k]

        # parent_a → child
        A[parent_a, child] = rng.uniform(*edge_range)
        # parent_b → child
        A[parent_b, child] = rng.uniform(*edge_range)

    # Connect remaining nodes in a chain from the last collider
    remaining = list(range(3 * n_vstructs, d))
    if remaining:
        # Last collider → first remaining
        A[collider_nodes[-1], remaining[0]] = rng.uniform(*edge_range)
        for i in range(len(remaining) - 1):
            A[remaining[i], remaining[i + 1]] = rng.uniform(*edge_range)

    # Add some extra edges (not creating new v-structures)
    # Connect some sources to remaining nodes for richer structure
    for src in source_nodes[: min(3, len(source_nodes))]:
        if remaining:
            target = rng.choice(remaining)
            if A[src, target] == 0:
                A[src, target] = rng.uniform(*edge_range)

    return A


def extract_spouse_pairs(A: np.ndarray, threshold: float = 0.1) -> list[tuple]:
    """
    Extract all spouse pairs from a DAG.

    Spouses are pairs of nodes that are co-parents of the same child
    but have no direct edge between them.

    Returns:
        List of (parent_a, parent_b, child) tuples.
    """
    d = A.shape[0]
    A_bin = (np.abs(A) > threshold).astype(int)
    spouse_triples = []

    for child in range(d):
        parents = list(np.where(A_bin[:, child] > 0)[0])
        if len(parents) < 2:
            continue
        for i, pa in enumerate(parents):
            for pb in parents[i + 1 :]:
                # Check no direct edge between spouses
                if A_bin[pa, pb] == 0 and A_bin[pb, pa] == 0:
                    spouse_triples.append((pa, pb, child))

    return spouse_triples


def true_markov_blanket(A: np.ndarray, target: int, threshold: float = 0.1) -> dict:
    """
    Extract ground-truth Markov blanket with per-relationship breakdown.

    Returns dict with 'parents', 'children', 'spouses', 'all' sets.
    """
    d = A.shape[0]
    A_bin = (np.abs(A) > threshold).astype(int)

    parents = set(np.where(A_bin[:, target] > 0)[0])
    children = set(np.where(A_bin[target, :] > 0)[0])

    spouses = set()
    for child in children:
        child_parents = set(np.where(A_bin[:, child] > 0)[0])
        spouses |= child_parents
    spouses -= {target}
    spouses -= parents  # don't double-count nodes that are both parent and spouse

    return {
        "parents": parents,
        "children": children,
        "spouses": spouses,
        "all": (parents | children | spouses) - {target},
    }


# =============================================================================
# Co-occurrence methods
# =============================================================================


def gbs_raw_cooccurrence(A: np.ndarray) -> np.ndarray:
    """GBS on raw symmetrized DAG (no moralization)."""
    W = encode_dag_to_gbs(A, scale=0.9)
    return dequantized_cooccurrence(W)


def gbs_moralized_cooccurrence(A: np.ndarray) -> np.ndarray:
    """GBS on moralized graph (Gemini's fix for v-structures)."""
    W = encode_moralized_to_gbs(A, scale=0.9)
    return dequantized_cooccurrence(W)


def partial_corr_from_sem(A: np.ndarray, n_data: int = 2000, seed: int = 42) -> np.ndarray:
    """Generate data from linear SEM and compute partial correlations."""
    d = A.shape[0]
    rng = np.random.default_rng(seed)

    # Topological sort
    order = _topological_sort(A)

    X = np.zeros((n_data, d))
    for sample_idx in range(n_data):
        noise = rng.standard_normal(d)
        x = np.zeros(d)
        for node in order:
            parent_vals = A[:, node] @ x
            x[node] = parent_vals + noise[node]
        X[sample_idx] = x

    # Partial correlation via precision matrix
    cov = np.cov(X, rowvar=False) + np.eye(d) * 1e-6
    prec = linalg.inv(cov)
    diag = np.sqrt(np.abs(np.diag(prec)))

    C = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                C[i, j] = abs(prec[i, j] / (diag[i] * diag[j] + 1e-10))
    return C


def _topological_sort(A: np.ndarray, threshold: float = 0.1) -> list:
    """Return nodes in topological order."""
    d = A.shape[0]
    A_bin = (np.abs(A) > threshold).astype(int)
    in_degree = A_bin.sum(axis=0).copy()
    order = []
    remaining = set(range(d))

    while remaining:
        ready = [n for n in remaining if sum(A_bin[p, n] for p in remaining if p != n) == 0]
        if not ready:
            order.extend(sorted(remaining))
            break
        node = min(ready, key=lambda n: in_degree[n])
        order.append(node)
        remaining.remove(node)

    return order


# =============================================================================
# Spouse detection evaluation
# =============================================================================


def evaluate_spouse_detection(
    C: np.ndarray,
    spouse_triples: list[tuple],
    threshold: str = "auto",
) -> dict:
    """
    Evaluate whether a co-occurrence/similarity matrix detects spouse pairs.

    For each spouse triple (pa, pb, child), check if C[pa, pb] is above
    threshold, indicating the method "sees" the spouse relationship.

    Args:
        C: (d, d) co-occurrence or similarity matrix.
        spouse_triples: list of (parent_a, parent_b, child) tuples.
        threshold: 'auto' (median of non-zero off-diag), 'mean', or float.

    Returns:
        dict with 'detection_rate', 'detected', 'total', 'scores'.
    """
    if not spouse_triples:
        return {"detection_rate": 0.0, "detected": 0, "total": 0, "scores": []}

    d = C.shape[0]

    # Determine threshold
    if threshold == "auto":
        off_diag = C[np.triu_indices(d, k=1)]
        nonzero = off_diag[off_diag > 0]
        thresh = np.median(nonzero) if len(nonzero) > 0 else 0.0
    elif threshold == "mean":
        off_diag = C[np.triu_indices(d, k=1)]
        thresh = np.mean(off_diag)
    else:
        thresh = float(threshold)

    detected = 0
    scores = []
    for pa, pb, child in spouse_triples:
        score = C[pa, pb]
        scores.append(score)
        if score > thresh:
            detected += 1

    return {
        "detection_rate": detected / len(spouse_triples) if spouse_triples else 0.0,
        "detected": detected,
        "total": len(spouse_triples),
        "scores": scores,
        "threshold": thresh,
    }


def evaluate_mb_with_breakdown(
    C: np.ndarray,
    target: int,
    mb_info: dict,
    max_mb_size: int | None = None,
) -> dict:
    """
    Evaluate MB detection with per-relationship-type breakdown.

    Uses top-k scoring: predict the top max_mb_size variables as MB members.

    Returns dict with overall and per-type (parents, children, spouses) metrics.
    """
    d = C.shape[0]
    scores = C[target, :].copy()
    scores[target] = 0

    true_mb = mb_info["all"]
    if not true_mb:
        return {"overall_f1": 0.0, "parent_recall": 0.0, "child_recall": 0.0, "spouse_recall": 0.0}

    if max_mb_size is None:
        max_mb_size = max(len(true_mb) + 2, d // 3)

    top_indices = np.argsort(scores)[::-1][:max_mb_size]
    predicted = set(top_indices[scores[top_indices] > 0])
    predicted -= {target}

    # Overall metrics
    tp = len(predicted & true_mb)
    fp = len(predicted - true_mb)
    fn = len(true_mb - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-type recall
    parent_recall = (
        len(predicted & mb_info["parents"]) / len(mb_info["parents"]) if mb_info["parents"] else 1.0
    )
    child_recall = (
        len(predicted & mb_info["children"]) / len(mb_info["children"])
        if mb_info["children"]
        else 1.0
    )
    spouse_recall = (
        len(predicted & mb_info["spouses"]) / len(mb_info["spouses"]) if mb_info["spouses"] else 1.0
    )

    return {
        "overall_f1": f1,
        "overall_precision": precision,
        "overall_recall": recall,
        "parent_recall": parent_recall,
        "child_recall": child_recall,
        "spouse_recall": spouse_recall,
        "predicted_size": len(predicted),
        "true_size": len(true_mb),
    }


# =============================================================================
# Main experiment
# =============================================================================


def run_vstruct_experiment(
    d: int,
    n_vstructs_list: list[int],
    edge_ranges: dict[str, tuple],
    n_instances: int = 5,
    base_seed: int = 42,
):
    """
    Run spouse detection across varying v-structure counts and edge strengths.
    """
    print(f"\n{'=' * 70}")
    print(f"  Spouse Detection Experiment  (d={d})")
    print(f"{'=' * 70}")

    methods = ["GBS-raw", "GBS-moral", "Partial-corr"]

    all_results = {}

    for strength_name, edge_range in edge_ranges.items():
        print(f"\n{'─' * 70}")
        print(f"  Edge strength: {strength_name} {edge_range}")
        print(f"{'─' * 70}")

        for n_vs in n_vstructs_list:
            if 3 * n_vs > d:
                continue

            config_key = f"d={d}, k={n_vs}, {strength_name}"
            spouse_rates = {m: [] for m in methods}
            mb_f1s = {m: [] for m in methods}
            spouse_recalls = {m: [] for m in methods}

            for inst in range(n_instances):
                seed = base_seed + inst * 100 + n_vs * 7
                rng = np.random.default_rng(seed)

                A = make_vstruct_dag(d, n_vstructs=n_vs, edge_range=edge_range, rng=rng)
                spouse_triples = extract_spouse_pairs(A)

                if not spouse_triples:
                    continue

                # Compute co-occurrence matrices
                C_raw = gbs_raw_cooccurrence(A)
                C_moral = gbs_moralized_cooccurrence(A)
                C_pcorr = partial_corr_from_sem(A, n_data=2000, seed=seed)

                co_matrices = {
                    "GBS-raw": C_raw,
                    "GBS-moral": C_moral,
                    "Partial-corr": C_pcorr,
                }

                # Spouse detection
                for mname, C in co_matrices.items():
                    result = evaluate_spouse_detection(C, spouse_triples)
                    spouse_rates[mname].append(result["detection_rate"])

                # MB metrics for collider nodes (most interesting targets)
                collider_nodes = list(range(2 * n_vs, 3 * n_vs))
                for target in collider_nodes:
                    mb_info = true_markov_blanket(A, target)
                    if not mb_info["all"]:
                        continue
                    for mname, C in co_matrices.items():
                        res = evaluate_mb_with_breakdown(C, target, mb_info)
                        mb_f1s[mname].append(res["overall_f1"])
                        spouse_recalls[mname].append(res["spouse_recall"])

            # Print results for this config
            print(f"\n  k={n_vs} v-structures ({len(spouse_rates['GBS-raw'])} instances):")
            print(f"    {'Method':<15} {'Spouse Det':>10} {'Spouse Rec':>10} {'MB F1':>8}")
            print(f"    {'-' * 48}")

            for mname in methods:
                sr = np.mean(spouse_rates[mname]) if spouse_rates[mname] else 0.0
                sp_rec = np.mean(spouse_recalls[mname]) if spouse_recalls[mname] else 0.0
                f1 = np.mean(mb_f1s[mname]) if mb_f1s[mname] else 0.0
                print(f"    {mname:<15} {sr:>10.3f} {sp_rec:>10.3f} {f1:>8.3f}")

            all_results[config_key] = {
                mname: {
                    "spouse_detection": np.mean(spouse_rates[mname])
                    if spouse_rates[mname]
                    else 0.0,
                    "spouse_recall": np.mean(spouse_recalls[mname])
                    if spouse_recalls[mname]
                    else 0.0,
                    "mb_f1": np.mean(mb_f1s[mname]) if mb_f1s[mname] else 0.0,
                }
                for mname in methods
            }

    return all_results


def main():
    print("=" * 70)
    print("  GBS Experiment 2: Spouse Detection + Moralization")
    print("=" * 70)
    print("Gemini's insight: GBS fails on v-structures because spouses")
    print("(co-parents A,B in A→C←B) have no direct edge.")
    print("Moralization adds the A-B edge → GBS can detect the clique.")
    print()
    print("Hypothesis:")
    print("  - GBS-raw: FAILS on spouse detection")
    print("  - GBS-moralized: SUCCEEDS on spouse detection")
    print("  - Partial-corr: Detects parents/children but NOT spouses")
    print("    (A⊥B|{everything else} in a v-structure)")

    edge_ranges = {
        "weak": (0.3, 0.5),
        "medium": (0.5, 1.0),
        "strong": (1.0, 2.0),
    }

    all_experiment_results = {}

    for d in [8, 11, 15, 20]:
        max_vs = min(6, d // 3)
        n_vstructs_list = list(range(1, max_vs + 1))

        results = run_vstruct_experiment(
            d=d,
            n_vstructs_list=n_vstructs_list,
            edge_ranges=edge_ranges,
            n_instances=5,
        )
        all_experiment_results.update(results)

    # ==========================================================================
    # Summary tables
    # ==========================================================================
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Spouse Detection Rates")
    print(f"{'=' * 70}")

    methods = ["GBS-raw", "GBS-moral", "Partial-corr"]

    print(f"\n{'Config':<30}", end="")
    for m in methods:
        print(f" | {m:>12}", end="")
    print()
    print("-" * (30 + len(methods) * 16))

    agg = {m: {"spouse_det": [], "spouse_rec": [], "mb_f1": []} for m in methods}

    for config_key in sorted(all_experiment_results.keys()):
        row = f"{config_key:<30}"
        for m in methods:
            v = all_experiment_results[config_key][m]["spouse_detection"]
            row += f" | {v:>12.3f}"
            agg[m]["spouse_det"].append(v)
            agg[m]["spouse_rec"].append(all_experiment_results[config_key][m]["spouse_recall"])
            agg[m]["mb_f1"].append(all_experiment_results[config_key][m]["mb_f1"])
        print(row)

    print("-" * (30 + len(methods) * 16))
    row = f"{'OVERALL (spouse detection)':<30}"
    for m in methods:
        row += f" | {np.mean(agg[m]['spouse_det']):>12.3f}"
    print(row)

    # Spouse recall summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Spouse Recall in MB Detection")
    print(f"{'=' * 70}")

    print(f"\n{'Config':<30}", end="")
    for m in methods:
        print(f" | {m:>12}", end="")
    print()
    print("-" * (30 + len(methods) * 16))

    for config_key in sorted(all_experiment_results.keys()):
        row = f"{config_key:<30}"
        for m in methods:
            v = all_experiment_results[config_key][m]["spouse_recall"]
            row += f" | {v:>12.3f}"
        print(row)

    print("-" * (30 + len(methods) * 16))
    row = f"{'OVERALL (spouse recall)':<30}"
    for m in methods:
        row += f" | {np.mean(agg[m]['spouse_rec']):>12.3f}"
    print(row)

    # MB F1 summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Overall MB F1 (collider nodes)")
    print(f"{'=' * 70}")

    row_f1 = f"{'OVERALL MB F1':<30}"
    for m in methods:
        row_f1 += f" | {np.mean(agg[m]['mb_f1']):>12.3f}"
    print(row_f1)

    # Verdict
    print(f"\n{'=' * 70}")
    print("  VERDICT: Moralization Effect")
    print(f"{'=' * 70}")

    raw_det = np.mean(agg["GBS-raw"]["spouse_det"])
    moral_det = np.mean(agg["GBS-moral"]["spouse_det"])
    pcorr_det = np.mean(agg["Partial-corr"]["spouse_det"])
    delta_moral = moral_det - raw_det

    print(f"  GBS-raw spouse detection:       {raw_det:.3f}")
    print(f"  GBS-moralized spouse detection:  {moral_det:.3f}")
    print(f"  Partial-corr spouse detection:   {pcorr_det:.3f}")
    print(f"  Moralization improvement:        {delta_moral:+.3f}")
    print()

    if delta_moral > 0.1:
        print("  CONFIRMED: Moralization significantly improves spouse detection.")
        if moral_det > raw_det * 1.5:
            print(
                f"  Moralization lifts detection by {delta_moral / raw_det * 100:.0f}%"
                if raw_det > 0.01
                else "  Moralization enables detection from near-zero baseline"
            )
    elif delta_moral > 0.0:
        print("  MARGINAL: Moralization provides small improvement.")
    else:
        print("  UNEXPECTED: Moralization does not improve spouse detection.")

    raw_sr = np.mean(agg["GBS-raw"]["spouse_rec"])
    moral_sr = np.mean(agg["GBS-moral"]["spouse_rec"])
    pcorr_sr = np.mean(agg["Partial-corr"]["spouse_rec"])

    print(f"\n  GBS-raw spouse recall (in MB):       {raw_sr:.3f}")
    print(f"  GBS-moralized spouse recall (in MB):  {moral_sr:.3f}")
    print(f"  Partial-corr spouse recall (in MB):   {pcorr_sr:.3f}")

    if moral_sr > pcorr_sr:
        print("\n  GBS-moralized BEATS partial-corr on spouse recall!")
        print("  → Moralization is the key: it gives GBS the spouse information")
        print("    that partial correlation cannot provide (A⊥B|rest in v-structures)")
    elif moral_sr > raw_sr:
        print("\n  Moralization helps but partial-corr still better.")
        print("  → GBS-moralized is a viable complement to partial-corr")


if __name__ == "__main__":
    main()
