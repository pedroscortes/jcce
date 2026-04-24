#!/usr/bin/env python3
"""Apply the 7-step validation protocol to all structure-learning baselines.

For each baseline that produces a DAG (PC, GES, FCI, DAGMA, NOTEARS, GOLEM,
DirectLiNGAM, CASTLE), runs:
  1. Fixed-structure CV (freeze baseline DAG, retrain 5 processors)
  2. DML effect estimation (use baseline DAG's adjustment sets)
  3. Cinelli sensitivity analysis (on DML results)
  4. Refutation tests (4 tests per significant edge)

Bootstrap stability and LOVO are optional (expensive for slow methods).

Usage:
    uv run python scripts/run_baseline_validation.py --all
    uv run python scripts/run_baseline_validation.py --dataset lucas
    uv run python scripts/run_baseline_validation.py --dataset lucas --methods pc ges castle
"""

import argparse
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jcce.data.benchmark_loader import load_dataset
from jcce.structure_learning.experiment_runner import (
    run_posthoc_cv,
    run_posthoc_dml,
)

BASELINES_DIR = Path("results/baselines_unified")
OUTPUT_DIR = Path("results/baseline_validation")

ALL_DATASETS = [
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

# Structure-learning methods that produce a DAG
STRUCTURE_METHODS = ["pc", "ges", "fci", "dagma", "notears", "golem", "directlingam"]

# Edge threshold sweep values for continuous methods
THRESHOLD_SWEEP = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]


def extract_baseline_dag(baselines: dict, method: str, n_features: int) -> dict:
    """Extract DAG from a baseline result, returning None if not available."""
    summary_key = f"{method}_summary"
    summary = baselines.get(summary_key, {})

    if "error" in summary or not summary:
        return None

    # Get adjacency matrix
    A_est = summary.get("A_est")
    if A_est is None:
        return None

    A_est = np.array(A_est)

    # Get MB indices
    mb_indices = summary.get("mb_indices", [])

    # Structure metrics if available
    shd = summary.get("structure_metrics", {}).get("shd", -1)

    return {
        "method": method,
        "A_est": A_est,
        "mb_indices": mb_indices,
        "n_edges": int(np.sum(np.abs(A_est) > 0.1)),
        "shd": shd,
    }


def extract_castle_dag(baselines: dict, n_features: int) -> dict:
    """Extract DAG from CASTLE baseline."""
    castle = baselines.get("castle", {})
    if "error" in castle or not castle:
        return None

    A_est = castle.get("A_est")
    if A_est is None:
        return None

    A_est = np.array(A_est)
    mb_indices = castle.get("mb_indices", [])

    return {
        "method": "castle",
        "A_est": A_est,
        "mb_indices": mb_indices,
        "n_edges": int(np.sum(np.abs(A_est) > 0.1)),
        "shd": castle.get("structure_metrics", {}).get("shd", -1),
    }


def dag_to_enhanced_solution(dag_info: dict, n_features: int, threshold: float = 0.3) -> dict:
    """Convert a baseline DAG into the enhanced_solution format expected by
    the GOLEM-GA validation functions."""
    A = dag_info["A_est"]

    # Ensure A is (n_features, n_features) — some methods return (d+1, d+1)
    if A.shape[0] > n_features:
        A = A[:n_features, :n_features]

    # Threshold to get binary adjacency for MB extraction

    # For MB: use Y as last variable (index n_features) in augmented matrix
    # But baselines return d x d matrices without Y column
    # Use provided mb_indices or extract from the adjacency
    mb = dag_info.get("mb_indices", [])

    return {
        "metrics": {
            "structure_A_est": A.tolist(),
            "processor_type": "elm",  # default processor for CV
            "mb_indices": mb,
            "markov_blanket": mb,
            "n_edges": dag_info["n_edges"],
            "balanced_accuracy": 0.0,  # not applicable
            "lambda_1": 0.02,
            "lambda_2": 0.01,
            "lambda_class": 1.0,
            "lr": 0.001,
        },
        "A_est": A,
    }


def threshold_sweep_dag(
    A_continuous: np.ndarray, thresholds: list, true_dag: np.ndarray = None
) -> list:
    """Evaluate a continuous DAG at multiple thresholds.

    Returns list of dicts with threshold, n_edges, mb_size, and structure
    metrics (if true_dag provided).
    """
    results = []
    n = A_continuous.shape[0]

    for thresh in thresholds:
        A_binary = (np.abs(A_continuous) > thresh).astype(float)
        n_edges = int(np.sum(A_binary))

        # Simple MB: parents + children of last variable (assumes Y is target)
        # For a d x d matrix, we look at column/row patterns
        # This is approximate — full MB needs the augmented matrix
        mb_cols = np.where(np.abs(A_continuous[:, -1]) > thresh)[0]
        mb_rows = np.where(np.abs(A_continuous[-1, :]) > thresh)[0]
        mb = sorted(set(mb_cols.tolist() + mb_rows.tolist()) - {n - 1})

        entry = {
            "threshold": thresh,
            "n_edges": n_edges,
            "mb_size": len(mb),
            "mb_indices": mb,
        }

        if true_dag is not None:
            true_binary = (np.abs(true_dag) > 0).astype(float)
            # Ensure same shape
            min_n = min(A_binary.shape[0], true_binary.shape[0])
            A_b = A_binary[:min_n, :min_n]
            T_b = true_binary[:min_n, :min_n]
            tp = int(np.sum(A_b * T_b))
            fp = int(np.sum(A_b * (1 - T_b)))
            fn = int(np.sum((1 - A_b) * T_b))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            shd = fp + fn  # simplified SHD (no reversals)
            entry.update(
                {
                    "shd": shd,
                    "edge_f1": f1,
                    "edge_precision": prec,
                    "edge_recall": rec,
                }
            )

        results.append(entry)

    return results


def run_validation_for_baseline(
    dataset_name: str,
    method: str,
    dag_info: dict,
    X: np.ndarray,
    Y: np.ndarray,
    config: dict,
    run_cv: bool = True,
    run_dml: bool = True,
    run_threshold_sweep: bool = True,
    n_cv_folds: int = 5,
    n_dml_folds: int = 5,
    cv_max_iter: int = 50,
    verbose: bool = True,
) -> dict:
    """Run validation protocol on a single baseline DAG."""
    result = {
        "dataset": dataset_name,
        "method": method,
        "n_edges": dag_info["n_edges"],
        "mb_size": len(dag_info.get("mb_indices", [])),
        "mb_indices": dag_info.get("mb_indices", []),
        "shd": dag_info.get("shd", -1),
    }

    n_features = X.shape[1]
    Y_int = np.asarray(Y).astype(int)

    # Sample split: 70/30
    from sklearn.model_selection import train_test_split

    X_struct, X_effect, Y_struct, Y_effect = train_test_split(
        X, Y_int, test_size=0.3, random_state=42, stratify=Y_int
    )

    # Wrap DAG as enhanced_solution
    enhanced = [dag_to_enhanced_solution(dag_info, n_features)]

    # 1. Fixed-structure CV
    if run_cv:
        if verbose:
            print(f"    Running fixed-structure CV ({n_cv_folds}-fold)...")
        t0 = time.time()
        try:
            cv_results = run_posthoc_cv(
                enhanced_solutions=enhanced,
                X=X_effect,
                Y=Y_effect.astype(np.float32),
                n_folds=n_cv_folds,
                max_solutions=1,
                golem_max_iter=cv_max_iter,
                freeze_structure=True,
                verbose=False,
            )
            if cv_results:
                result["cv"] = cv_results[0]
                cv_bacc = cv_results[0].get("balanced_acc_mean", 0)
                if verbose:
                    print(f"      CV BAcc: {cv_bacc:.3f} ({time.time() - t0:.1f}s)")
            else:
                result["cv"] = {"error": "empty result"}
        except Exception as e:
            result["cv"] = {"error": str(e)}
            if verbose:
                print(f"      CV failed: {e}")

    # 2. DML effect estimation
    if run_dml:
        if verbose:
            print(f"    Running DML ({n_dml_folds}-fold cross-fitting)...")
        t0 = time.time()
        try:
            dml_results = run_posthoc_dml(
                enhanced_solutions=enhanced,
                X_effect=X_effect,
                Y_effect=Y_effect.astype(np.float32),
                feature_names=config.get("feature_names"),
                max_solutions=1,
                n_dml_folds=n_dml_folds,
                run_refutation=True,
                n_refutation_sims=50,
                verbose=False,
            )
            if dml_results:
                result["dml"] = dml_results[0]
                n_sig = dml_results[0].get("n_significant", 0)
                n_total = dml_results[0].get("n_parents_tested", 0)
                if verbose:
                    print(f"      DML: {n_sig}/{n_total} significant ({time.time() - t0:.1f}s)")
            else:
                result["dml"] = {"error": "empty result"}
        except Exception as e:
            result["dml"] = {"error": str(e)}
            if verbose:
                print(f"      DML failed: {e}")

    # 3. Threshold sweep (for continuous methods)
    if run_threshold_sweep and method in ["dagma", "notears", "golem", "directlingam", "castle"]:
        if verbose:
            print(f"    Running threshold sweep ({len(THRESHOLD_SWEEP)} thresholds)...")
        true_dag = config.get("true_dag")
        result["threshold_sweep"] = threshold_sweep_dag(
            dag_info["A_est"], THRESHOLD_SWEEP, true_dag
        )
        if verbose:
            best = max(result["threshold_sweep"], key=lambda x: x.get("edge_f1", 0))
            print(
                f"      Best threshold: {best['threshold']} "
                f"(F1={best.get('edge_f1', 0):.3f}, edges={best['n_edges']})"
            )

    return result


def run_dataset_validation(
    dataset_name: str,
    methods: list = None,
    run_cv: bool = True,
    run_dml: bool = True,
    verbose: bool = True,
) -> dict:
    """Run validation on all baselines for one dataset."""
    if methods is None:
        methods = STRUCTURE_METHODS + ["castle"]

    # Load baseline results
    pkl_path = BASELINES_DIR / f"{dataset_name}_baselines.pkl"
    if not pkl_path.exists():
        print(f"  SKIP: {pkl_path} not found")
        return {}

    with open(pkl_path, "rb") as f:
        baseline_data = pickle.load(f)

    baselines = baseline_data["baselines"]

    # Load dataset
    X, Y, config = load_dataset(dataset_name)
    n_features = X.shape[1]

    # Standardize
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_std = (X - mu) / sigma

    print(f"\n{'=' * 60}")
    print(f"  {dataset_name.upper()} (d={n_features}, n={X.shape[0]})")
    print(f"{'=' * 60}")

    results = {}

    for method in methods:
        # Extract DAG
        if method == "castle":
            dag_info = extract_castle_dag(baselines, n_features)
        else:
            dag_info = extract_baseline_dag(baselines, method, n_features)

        if dag_info is None:
            if verbose:
                print(f"  {method:15s}: no DAG available, skipping")
            continue

        print(
            f"\n  {method.upper()} (edges={dag_info['n_edges']}, "
            f"MB={len(dag_info.get('mb_indices', []))})"
        )

        try:
            result = run_validation_for_baseline(
                dataset_name=dataset_name,
                method=method,
                dag_info=dag_info,
                X=X_std,
                Y=np.asarray(Y),
                config=config,
                run_cv=run_cv,
                run_dml=run_dml,
                verbose=verbose,
            )
            results[method] = result
        except Exception as e:
            print(f"    FAILED: {e}")
            if verbose:
                traceback.print_exc()
            results[method] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run validation protocol on structure-learning baselines"
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Methods to validate (default: all). Options: {STRUCTURE_METHODS + ['castle']}",
    )
    parser.add_argument("--no-cv", action="store_true", help="Skip CV")
    parser.add_argument("--no-dml", action="store_true", help="Skip DML")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else [args.dataset or "lucas"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ds in datasets:
        results = run_dataset_validation(
            dataset_name=ds,
            methods=args.methods,
            run_cv=not args.no_cv,
            run_dml=not args.no_dml,
            verbose=args.verbose > 0,
        )
        all_results[ds] = results

        # Save per-dataset
        with open(OUTPUT_DIR / f"{ds}_validation.pkl", "wb") as f:
            pickle.dump(results, f)

    # Save combined
    with open(OUTPUT_DIR / "all_validation.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Print summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Baseline Validation Results")
    print(f"{'=' * 80}")
    print(
        f"{'Dataset':<20s} {'Method':<15s} {'CV BAcc':>10s} {'DML sig':>10s} "
        f"{'Edges':>8s} {'MB':>5s}"
    )
    print("-" * 70)
    for ds, ds_results in all_results.items():
        for method, res in ds_results.items():
            if isinstance(res, dict) and "error" not in res:
                cv_bacc = res.get("cv", {}).get("balanced_acc_mean", "---")
                if isinstance(cv_bacc, float):
                    cv_bacc = f"{cv_bacc:.3f}"
                dml = res.get("dml", {})
                n_sig = dml.get("n_significant", "---")
                n_total = dml.get("n_parents_tested", "?")
                dml_str = f"{n_sig}/{n_total}" if n_sig != "---" else "---"
                edges = res.get("n_edges", "?")
                mb = res.get("mb_size", "?")
                print(f"{ds:<20s} {method:<15s} {cv_bacc:>10s} {dml_str:>10s} {edges:>8} {mb:>5}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
