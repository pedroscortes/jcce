#!/usr/bin/env python3
"""
DAG Plotting for JCCE v16 Ablation Results.

Loads v16 B6 result pkl files and generates publication-quality DAG
visualizations for multiple Pareto-optimal solutions (best accuracy,
sparse, dense, etc.), matching the v14 multi-DAG approach.

Usage:
    uv run python scripts/plot_pareto_dags.py                          # all datasets, diverse solutions
    uv run python scripts/plot_pareto_dags.py --dataset lucas           # specific dataset
    uv run python scripts/plot_pareto_dags.py --sol 0 3 5 10            # specific solution indices
    uv run python scripts/plot_pareto_dags.py --file results/ablation_v16/lucas/B6.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from jcce.visualization.dag_viz import visualize_jcce_dag

# =============================================================================
# Configuration
# =============================================================================

RESULTS_PATH = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_PATH / "dags" / "v16_ablation"
VIZ_THRESHOLD = 0.01

OUTCOME_NAMES = {
    "lucas": "Lung_Cancer",
    "heart_disease": "Heart_Disease",
    "breast_cancer": "Malignant",
    "diabetes": "Diabetes",
}


# =============================================================================
# Feature Name Shortening
# =============================================================================


def shorten_feature_names(names: List[str], dataset_name: str) -> List[str]:
    if dataset_name != "breast_cancer":
        return names
    short = []
    for name in names:
        if name.startswith("mean "):
            short.append("M_" + name[5:].replace(" ", "_"))
        elif name.startswith("worst "):
            short.append("W_" + name[6:].replace(" ", "_"))
        elif name.endswith(" error"):
            short.append(name[:-6].replace(" ", "_") + "_Err")
        else:
            short.append(name.replace(" ", "_"))
    return short


# =============================================================================
# MB Filtering
# =============================================================================


def filter_mb_no_isolated(
    A_direct: np.ndarray,
    A_confound: Optional[np.ndarray],
    mb_indices: List[int],
    outcome_idx: int,
    threshold: float = VIZ_THRESHOLD,
) -> List[int]:
    """Remove MB features with no visible edge to any other kept node."""
    keep = sorted(set(mb_indices + [outcome_idx]))
    changed = True
    while changed:
        changed = False
        new_keep = []
        for idx in keep:
            if idx == outcome_idx:
                new_keep.append(idx)
                continue
            has_edge = False
            for other in keep:
                if idx == other:
                    continue
                if abs(A_direct[idx, other]) >= threshold or abs(A_direct[other, idx]) >= threshold:
                    has_edge = True
                    break
                if A_confound is not None:
                    if (
                        abs(A_confound[idx, other]) >= threshold
                        or abs(A_confound[other, idx]) >= threshold
                    ):
                        has_edge = True
                        break
            if has_edge:
                new_keep.append(idx)
            else:
                changed = True
        keep = new_keep
    return keep


def build_filtered_matrices(
    A_direct: np.ndarray,
    A_confound: Optional[np.ndarray],
    keep_indices: List[int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build sub-matrices for the kept node indices."""
    n = len(keep_indices)
    A_f = np.zeros((n, n))
    for i, oi in enumerate(keep_indices):
        for j, oj in enumerate(keep_indices):
            A_f[i, j] = A_direct[oi, oj]

    A_c_f = None
    if A_confound is not None:
        A_c_f = np.zeros((n, n))
        for i, oi in enumerate(keep_indices):
            for j, oj in enumerate(keep_indices):
                A_c_f[i, j] = A_confound[oi, oj]

    return A_f, A_c_f


# =============================================================================
# Structure Metrics (for LUCAS)
# =============================================================================


def compute_structure_metrics(A_est, A_true, threshold=0.01):
    A_est_bin = (np.abs(A_est) >= threshold).astype(int)
    A_true_bin = (A_true != 0).astype(int)
    tp = int(np.sum((A_est_bin == 1) & (A_true_bin == 1)))
    fp = int(np.sum((A_est_bin == 1) & (A_true_bin == 0)))
    fn = int(np.sum((A_est_bin == 0) & (A_true_bin == 1)))
    shd = fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "shd": shd,
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_mb_metrics(discovered_mb: List[int], true_mb: List[int]):
    discovered = set(discovered_mb)
    true = set(true_mb)
    tp = len(discovered & true)
    fp = len(discovered - true)
    fn = len(true - discovered)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"mb_f1": f1, "mb_precision": precision, "mb_recall": recall}


# =============================================================================
# Diverse Solution Selection (like v14 get_diverse_solutions)
# =============================================================================


def get_diverse_solutions(solutions: list, feature_names: list, n_max: int = 5):
    """
    Select diverse Pareto solutions for plotting.
    Returns list of (sol_idx, solution_dict, label) tuples.
    """
    if not solutions:
        return []

    n_features = len(feature_names)
    outcome_idx = n_features

    sol_data = []
    for i, sol in enumerate(solutions):
        acc = sol.get("balanced_accuracy", sol.get("accuracy", 0))
        n_edges = sol.get("n_edges", 0)
        mb_size = len(sol.get("mb_indices", []))
        sol_data.append({"idx": i, "sol": sol, "bacc": acc, "n_edges": n_edges, "mb_size": mb_size})

    selected = []
    used_indices = set()

    by_acc = sorted(sol_data, key=lambda x: x["bacc"], reverse=True)

    # 1. Best accuracy
    best = by_acc[0]
    selected.append((best["idx"], best["sol"], "best_accuracy"))
    used_indices.add(best["idx"])

    # 2. Sparse accurate (good acc, fewest MB features, must have visible nodes)
    good_acc = [
        s
        for s in sol_data
        if s["bacc"] >= 0.60 and s["idx"] not in used_indices and s["mb_size"] >= 1
    ]
    if good_acc:
        for candidate in sorted(good_acc, key=lambda x: x["mb_size"]):
            if candidate["mb_size"] == best["mb_size"]:
                continue
            sol = candidate["sol"]
            mb_idx = [int(x) for x in sol.get("mb_indices", [])]
            A_est = np.array(sol["A_est"])
            A_w = np.array(sol["A_weights"]) if sol.get("A_weights") is not None else A_est
            A = A_est * A_w
            A_cb = (
                np.array(sol["A_confound"])
                if sol.get("A_confound") is not None
                else np.zeros_like(A)
            )
            A_cw = (
                np.array(sol.get("A_confound_weights"))
                if sol.get("A_confound_weights") is not None
                else A_cb
            )
            A_c = A_cb * A_cw if A_cw is not None else A_cb
            visible = filter_mb_no_isolated(A, A_c, mb_idx, outcome_idx, VIZ_THRESHOLD)
            if len(visible) > 1:
                selected.append((candidate["idx"], candidate["sol"], "sparse_accurate"))
                used_indices.add(candidate["idx"])
                break

    # 3. Most edges (most complex structure)
    remaining = [s for s in sol_data if s["idx"] not in used_indices and s["bacc"] >= 0.60]
    if remaining:
        complex_sol = max(remaining, key=lambda x: x["n_edges"])
        selected.append((complex_sol["idx"], complex_sol["sol"], "most_edges"))
        used_indices.add(complex_sol["idx"])

    # 4+. Fill with diverse solutions (different edge counts)
    remaining = [s for s in by_acc if s["idx"] not in used_indices]
    edge_counts = {sol_data[idx]["n_edges"] for idx, _, _ in selected}
    for s in remaining:
        if len(selected) >= n_max:
            break
        if s["n_edges"] not in edge_counts and s["bacc"] >= 0.60:
            selected.append((s["idx"], s["sol"], f"sol{s['idx']}_acc{s['bacc'] * 100:.0f}"))
            used_indices.add(s["idx"])
            edge_counts.add(s["n_edges"])

    return selected


# =============================================================================
# Single Solution Plot
# =============================================================================


def plot_solution(
    sol: dict,
    sol_idx: int,
    label: str,
    dataset_name: str,
    feature_names: List[str],
    ds_config: dict,
    output_format: str = "both",
    output_dir: Optional[Path] = None,
):
    """Plot a single Pareto solution's DAG."""
    # Prefer human-readable outcome name; fall back to config's target_name
    outcome_name = OUTCOME_NAMES.get(dataset_name, ds_config.get("target_name", "Y"))
    n_features = len(feature_names)
    outcome_idx = n_features

    # Get matrices — use binary structure (A_est, A_confound) masked by
    # continuous weights (A_weights, A_confound_weights) for edge magnitudes.
    # A_est is the thresholded DAG; A_weights is the dense continuous param.
    A_est = np.array(sol["A_est"])
    A_weights = np.array(sol["A_weights"]) if sol.get("A_weights") is not None else A_est
    A_direct = A_est * A_weights  # Only edges in the DAG, with learned weights

    # Resolve 2-cycles (bidirectional edges): keep only the stronger direction.
    # This is the minimal intervention — longer cycles are kept and reported
    # as a limitation of the soft DAGMA constraint.
    n = A_direct.shape[0]
    n_resolved = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A_direct[i, j]) > 0 and abs(A_direct[j, i]) > 0:
                if abs(A_direct[i, j]) >= abs(A_direct[j, i]):
                    A_direct[j, i] = 0.0
                else:
                    A_direct[i, j] = 0.0
                n_resolved += 1
    if n_resolved > 0:
        print(f"  [INFO] Resolved {n_resolved} bidirectional edges (kept stronger direction)")

    A_confound_bin = np.array(sol["A_confound"]) if sol.get("A_confound") is not None else None
    A_confound_weights = (
        np.array(sol.get("A_confound_weights"))
        if sol.get("A_confound_weights") is not None
        else None
    )
    if A_confound_bin is not None and A_confound_weights is not None:
        A_confound = A_confound_bin * A_confound_weights
    elif A_confound_bin is not None:
        A_confound = A_confound_bin
    else:
        A_confound = np.zeros_like(A_direct)

    mb_indices = [int(x) for x in sol.get("mb_indices", [])]
    original_mb_size = len(mb_indices)

    # Translate causal_effects keys from X{idx}->Y to FeatureName->OutcomeName
    # Prefer all-edges DML effects (validated) over raw GOLEM effects
    raw_effects = sol.get("dml_causal_effects", sol.get("causal_effects", {}))
    causal_effects = {}
    if raw_effects:
        for key, val in raw_effects.items():
            # Parse "X{i}->Y" format
            parts = key.split("->")
            if len(parts) == 2:
                src, tgt = parts
                # Map X{i} to feature name
                if src.startswith("X") and src[1:].isdigit():
                    src_idx = int(src[1:])
                    if src_idx < n_features:
                        src = feature_names[src_idx]
                # Map Y to outcome name
                if tgt == "Y":
                    tgt = outcome_name
                causal_effects[f"{src}->{tgt}"] = val

    # Filter isolated nodes
    keep_indices = filter_mb_no_isolated(
        A_direct, A_confound, mb_indices, outcome_idx, VIZ_THRESHOLD
    )
    n_visible = len(keep_indices) - 1

    if n_visible == 0:
        print(f"  [SKIP] Sol {sol_idx} ({label}): all {original_mb_size} MB features isolated")
        return None

    # Build filtered matrices
    A_filtered, A_confound_filtered = build_filtered_matrices(A_direct, A_confound, keep_indices)

    # Node names
    all_names = list(feature_names) + [outcome_name]
    filtered_names = [all_names[i] for i in keep_indices]

    feature_only = [n for n in filtered_names if n != outcome_name]
    short_features = shorten_feature_names(feature_only, dataset_name)
    viz_names = []
    feat_idx = 0
    for name in filtered_names:
        if name == outcome_name:
            viz_names.append(outcome_name)
        else:
            viz_names.append(short_features[feat_idx])
            feat_idx += 1

    outcome_local = filtered_names.index(outcome_name)

    # Output
    out_dir = output_dir if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{dataset_name}_sol{sol_idx}_{label}"
    output_path = str(out_dir / output_name)

    # Remap causal_effects to use shortened viz_names (for breast cancer etc.)
    viz_effects = {}
    if causal_effects:
        name_map = {}
        feat_idx = 0
        for orig_name in filtered_names:
            if orig_name == outcome_name:
                name_map[orig_name] = outcome_name
            else:
                name_map[orig_name] = short_features[feat_idx]
                feat_idx += 1
        for key, val in causal_effects.items():
            parts = key.split("->")
            if len(parts) == 2:
                src_mapped = name_map.get(parts[0], parts[0])
                tgt_mapped = name_map.get(parts[1], parts[1])
                viz_effects[f"{src_mapped}->{tgt_mapped}"] = val

    formats = ["png", "pdf"] if output_format == "both" else [output_format]
    for fmt in formats:
        visualize_jcce_dag(
            A_direct=A_filtered,
            A_confound=A_confound_filtered,
            feature_names=viz_names,
            outcome_idx=outcome_local,
            causal_effects=viz_effects if viz_effects else None,
            threshold=VIZ_THRESHOLD,
            layout="dot",
            color_scheme="academic",
            show_ate=True,
            title=None,
            output_path=output_path,
            output_format=fmt,
            dpi="300",
        )

    # Count edges
    n_keep = len(keep_indices)
    n_direct = sum(
        1
        for i in range(n_keep)
        for j in range(n_keep)
        if i != j and abs(A_filtered[i, j]) >= VIZ_THRESHOLD
    )
    # Count only confound edges that are actually visible (no directed edge between same nodes)
    n_confound_vis = 0
    for i in range(n_keep):
        for j in range(i + 1, n_keep):
            if A_confound_filtered is not None and abs(A_confound_filtered[i, j]) >= VIZ_THRESHOLD:
                has_directed = (
                    abs(A_filtered[i, j]) >= VIZ_THRESHOLD or abs(A_filtered[j, i]) >= VIZ_THRESHOLD
                )
                if not has_directed:
                    n_confound_vis += 1

    n_removed = original_mb_size - n_visible
    fmt_str = "+".join(formats)
    bacc = sol.get("balanced_accuracy", sol.get("accuracy", 0))
    proc = sol.get("processor_type", "?")

    print(f"\n  Generated: {output_name}.{{{fmt_str}}}")
    print(
        f"    Processor={proc} | BAcc={bacc:.4f} | MB={original_mb_size} | Visible={n_visible} | Removed={n_removed}"
    )
    print(f"    Directed edges: {n_direct} | Confounding: {n_confound_vis}")

    # Per-solution post-hoc results (if available)
    if "cv" in sol:
        cv = sol["cv"]
        print(
            f"    CV: BAcc={cv['balanced_acc_mean']:.4f}+-{cv['balanced_acc_std']:.4f} | "
            f"AUC={cv['roc_auc_mean']:.4f}+-{cv['roc_auc_std']:.4f}"
        )
    if "dml_n_significant" in sol:
        print(f"    DML: {sol['dml_n_significant']}/{sol['dml_n_parents']} parents significant")
    if "cf" in sol:
        cf = sol["cf"]
        print(
            f"    CF: {cf['n_valid']}/{cf['n_instances']} valid (sparsity={cf['avg_sparsity']:.1f})"
        )

    # Edges to outcome
    edges_to_outcome = []
    for i in range(n_keep):
        if i == outcome_local:
            continue
        w = A_filtered[i, outcome_local]
        if abs(w) >= VIZ_THRESHOLD:
            edges_to_outcome.append((viz_names[i], w))
    edges_to_outcome.sort(key=lambda x: abs(x[1]), reverse=True)
    if edges_to_outcome:
        print(f"    Direct causes of {outcome_name}:")
        for name, w in edges_to_outcome:
            print(f"      {name} -> {outcome_name}: {w:+.4f}")

    # LUCAS structure metrics
    if dataset_name == "lucas":
        gt_path = Path(__file__).parent.parent / "data" / "benchmarks" / "lucas" / "true_dag.npy"
        if gt_path.exists():
            A_true = np.load(str(gt_path))
            A_est_full = np.array(sol["A_est"])
            sm = compute_structure_metrics(A_est_full, A_true, VIZ_THRESHOLD)
            print(
                f"    SHD={sm['shd']} | Edge F1={sm['edge_f1']:.3f} | Prec={sm['edge_precision']:.3f} | Rec={sm['edge_recall']:.3f}"
            )

        true_mb = ds_config.get("true_mb")
        if true_mb:
            mm = compute_mb_metrics(mb_indices, true_mb)
            print(
                f"    MB F1={mm['mb_f1']:.3f} | Prec={mm['mb_precision']:.3f} | Rec={mm['mb_recall']:.3f}"
            )

    return output_path


# =============================================================================
# Process a Dataset
# =============================================================================


def process_dataset(
    result: dict,
    output_format: str = "both",
    sol_indices: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
):
    """Plot DAGs for a single dataset's v16 result."""
    dataset_name = result["dataset"]
    ds_config = result["dataset_config"]
    feature_names = ds_config["feature_names"]

    # Check for pareto_solutions (new format with all solutions' DAG data)
    pareto_sols = result.get("pareto_solutions", [])

    if pareto_sols:
        print(f"  {len(pareto_sols)} Pareto solutions available")

        if sol_indices is not None:
            # Plot specific solutions
            for idx in sol_indices:
                if idx < 0 or idx >= len(pareto_sols):
                    print(f"  [WARN] Index {idx} out of range (0-{len(pareto_sols) - 1})")
                    continue
                plot_solution(
                    pareto_sols[idx],
                    idx,
                    f"sol{idx}",
                    dataset_name,
                    feature_names,
                    ds_config,
                    output_format,
                    output_dir=output_dir,
                )
        else:
            # Auto-select diverse solutions
            diverse = get_diverse_solutions(pareto_sols, feature_names, n_max=5)
            print(f"  Plotting {len(diverse)} diverse solutions...")
            for sol_idx, sol, label in diverse:
                plot_solution(
                    sol,
                    sol_idx,
                    label,
                    dataset_name,
                    feature_names,
                    ds_config,
                    output_format,
                    output_dir=output_dir,
                )
    elif "model" in result and result["model"] is not None:
        # Fallback: only best solution available (old pkl format)
        print("  No pareto_solutions data — plotting best solution only")
        print("  (Re-run experiment with updated code to get all Pareto DAGs)")
        model = result["model"]
        # Inject top-level metrics into model dict for display
        model.setdefault("balanced_accuracy", result.get("best_training_balanced_acc", 0))
        model.setdefault("accuracy", result.get("best_training_balanced_acc", 0))
        plot_solution(
            model,
            0,
            "best_accuracy",
            dataset_name,
            feature_names,
            ds_config,
            output_format,
            output_dir=output_dir,
        )
    else:
        print("  [SKIP] No model artifacts in result")


# =============================================================================
# Discovery / CLI
# =============================================================================


def find_v16_results():
    """Find NSGA2 B6 results."""
    results_dir = RESULTS_PATH / "ablation_v16"
    if not results_dir.exists():
        return {}
    found = {}
    for ds_dir in results_dir.iterdir():
        if not ds_dir.is_dir():
            continue
        b6_pkl = ds_dir / "B6.pkl"
        if b6_pkl.exists():
            found[ds_dir.name] = b6_pkl
    return found


def find_optuna_results():
    """Find Optuna results."""
    results_dir = RESULTS_PATH / "optuna_v16"
    if not results_dir.exists():
        return {}
    found = {}
    for ds_dir in results_dir.iterdir():
        if not ds_dir.is_dir():
            continue
        # Pick the first pkl in each dataset dir
        pkls = sorted(ds_dir.glob("*.pkl"))
        if pkls:
            found[ds_dir.name] = pkls[0]
    return found


def main():
    parser = argparse.ArgumentParser(description="Plot DAGs from JCCE v16 results")
    parser.add_argument("--file", type=str, help="Path to specific v16 result pkl")
    parser.add_argument("--dataset", type=str, nargs="+", help="Dataset name(s)")
    parser.add_argument(
        "--sol", type=int, nargs="+", metavar="IDX", help="Specific solution indices to plot"
    )
    parser.add_argument("--format", type=str, default="both", choices=["png", "pdf", "svg", "both"])
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "nsga2", "optuna"],
        help="Which results to plot (default: all)",
    )
    args = parser.parse_args()

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        # Infer source from path
        if "optuna" in str(filepath).lower():
            out_dir = RESULTS_PATH / "dags" / "optuna"
        else:
            out_dir = RESULTS_PATH / "dags" / "nsga2"
        print(f"{'=' * 70}")
        print(f"  V16 DAG: {filepath.name}")
        print(f"{'=' * 70}")
        process_dataset(result, args.format, args.sol, output_dir=out_dir)
        print(f"\n  DAGs saved to: {out_dir}")
        return

    # --- NSGA2 results ---
    if args.source in ("all", "nsga2"):
        nsga2_results = find_v16_results()
        if nsga2_results:
            nsga2_dir = RESULTS_PATH / "dags" / "nsga2"
            datasets = args.dataset if args.dataset else sorted(nsga2_results.keys())
            for ds in datasets:
                if ds not in nsga2_results:
                    continue
                with open(nsga2_results[ds], "rb") as f:
                    result = pickle.load(f)
                print(f"\n{'=' * 70}")
                print(f"  NSGA2 DAG: {ds}")
                print(f"{'=' * 70}")
                process_dataset(result, args.format, args.sol, output_dir=nsga2_dir)
            print(f"\n  NSGA2 DAGs saved to: {nsga2_dir}")
        else:
            print("No NSGA2 B6 results found in results/ablation_v16/")

    # --- Optuna results ---
    if args.source in ("all", "optuna"):
        optuna_results = find_optuna_results()
        if optuna_results:
            optuna_dir = RESULTS_PATH / "dags" / "optuna"
            datasets = args.dataset if args.dataset else sorted(optuna_results.keys())
            for ds in datasets:
                if ds not in optuna_results:
                    continue
                with open(optuna_results[ds], "rb") as f:
                    result = pickle.load(f)
                print(f"\n{'=' * 70}")
                print(f"  Optuna DAG: {ds}")
                print(f"{'=' * 70}")
                process_dataset(result, args.format, args.sol, output_dir=optuna_dir)
            print(f"\n  Optuna DAGs saved to: {optuna_dir}")
        else:
            print("No Optuna results found in results/optuna_v16/")


if __name__ == "__main__":
    main()
