#!/usr/bin/env python
"""
Baseline comparison: NSGA-II vs Optuna on the same data.

Measures:
- Pareto hypervolume (dominated area in BAcc × Sparsity space)
- Wall-clock time
- Number of evaluations
- Edge stability across Pareto solutions
- Processor distribution

Usage:
    # Quick comparison on synthetic data
    uv run python scripts/compare_nsga2_vs_optuna.py --dataset synthetic --n-evals 10

    # LUCAS comparison (server)
    uv run python scripts/compare_nsga2_vs_optuna.py --dataset lucas --n-evals 50 --max-iter 100
"""

import argparse
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jcce.utils.metrics import compute_sid

# ============================================================================
# Hypervolume Computation (2D, O(n log n))
# ============================================================================


def compute_hypervolume_2d(
    points: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """
    Compute dominated hypervolume for 2D maximization objectives.

    Uses the standard staircase algorithm: sort by obj1 descending,
    sweep left-to-right accumulating rectangular slabs bounded by
    the running maximum of obj2.

    Args:
        points: (n, 2) array of objective values (higher = better)
        ref_point: (2,) reference point (lower bound)

    Returns:
        Dominated hypervolume area
    """
    if len(points) == 0:
        return 0.0

    # Filter out points dominated by ref point
    valid = np.all(points > ref_point, axis=1)
    points = points[valid]

    if len(points) == 0:
        return 0.0

    # Sort by first objective descending (sweep right-to-left in obj1)
    sorted_idx = np.argsort(-points[:, 0])
    points = points[sorted_idx]

    hv = 0.0
    max_y = ref_point[1]  # Running max of obj2 seen so far

    for i, p in enumerate(points):
        # Width: from this point's obj1 to the NEXT point's obj1 (or ref_point)
        if i + 1 < len(points):
            width = p[0] - points[i + 1, 0]
        else:
            width = p[0] - ref_point[0]

        # Update running max of obj2
        max_y = max(max_y, p[1])

        # Height: from ref_point[1] to running max of obj2
        height = max_y - ref_point[1]

        hv += width * height

    return hv


# ============================================================================
# Edge Stability
# ============================================================================


def compute_edge_stability(
    solutions: list,
    n_vars: int,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Compute edge frequency across Pareto solutions.

    P(Xi → Xj) = (1/|P|) × Σ I(|A_m[i,j]| > threshold)

    Args:
        solutions: List of enhanced_solution dicts
        n_vars: Number of variables (for full A size = n_vars+1)
        threshold: Edge weight threshold

    Returns:
        (n_vars+1, n_vars+1) stability matrix
    """
    n = n_vars + 1  # includes Y
    if not solutions:
        return np.zeros((n, n))

    stability = np.zeros((n, n))
    for sol in solutions:
        A = sol["metrics"].get("structure_A_est")
        if A is not None:
            A = np.array(A)
            if A.shape == (n, n):
                stability += (np.abs(A) > threshold).astype(float)

    return stability / len(solutions)


# ============================================================================
# Run Comparison
# ============================================================================


def run_nsga2(X, Y, n_vars, n_evals, max_iter, seed, verbose):
    """Run NSGA-II and return results."""
    import jax.numpy as jnp

    from jcce.structure_learning.nsga2_search import run_nsga2

    # Estimate pop_size and n_generations from n_evals
    pop_size = max(6, min(20, n_evals // 3))
    n_generations = max(1, n_evals // pop_size)

    if verbose:
        print(
            f"\nNSGA-II: pop={pop_size}, gen={n_generations}, "
            f"max_iter={max_iter} (~{pop_size * n_generations} evals)"
        )

    X_jax = jnp.array(X)
    Y_jax = jnp.array(Y)

    start = time.time()
    results = run_nsga2(
        X=X_jax,
        Y=Y_jax,
        n_vars=n_vars,
        pop_size=pop_size,
        n_generations=n_generations,
        stage1_max_iter=max_iter,
        jax_key_seed=seed,
        verbose=verbose,
        use_v7=True,
    )
    wall_time = time.time() - start

    enhanced = results.get("enhanced_solutions", [])
    n_actual_evals = len(results.get("evaluation_cache", {}))

    return {
        "method": "NSGA-II",
        "enhanced_solutions": enhanced,
        "wall_time": wall_time,
        "n_evals": n_actual_evals,
        "pareto_front": results.get("pareto_front", []),
    }


def run_optuna(X, Y, n_vars, n_evals, max_iter, seed, verbose):
    """Run Optuna and return results."""
    import jax.numpy as jnp

    from jcce.structure_learning.optuna_search import run_optuna_search

    X_jax = jnp.array(X)
    Y_jax = jnp.array(Y)

    if verbose:
        print(f"\nOptuna: n_trials={n_evals}, max_iter={max_iter}")

    start = time.time()
    results = run_optuna_search(
        X=X_jax,
        Y=Y_jax,
        n_vars=n_vars,
        n_trials=n_evals,
        max_iter=max_iter,
        use_v7=True,
        jax_key_seed=seed,
        verbose=verbose,
    )
    wall_time = time.time() - start

    return {
        "method": "Optuna",
        "enhanced_solutions": results["enhanced_solutions"],
        "wall_time": wall_time,
        "n_evals": results["n_trials_completed"],
        "n_pruned": results["n_trials_pruned"],
        "pareto_front": results["pareto_front"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare NSGA-II vs Optuna for JCCE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=[
            "lucas",
            "heart_disease",
            "breast_cancer",
            "diabetes",
            "sachs",
            "asia",
            "synthetic",
        ],
    )
    parser.add_argument("--n-evals", type=int, default=10, help="Total evaluations per method")
    parser.add_argument(
        "--max-iter", type=int, default=30, help="Max GOLEM iterations per evaluation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to file")
    parser.add_argument(
        "--nsga2-results",
        type=str,
        default=None,
        help="Load cached NSGA-II results from JSON file (skip re-running NSGA-II)",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Load data
    from scripts.run_optuna_pipeline import load_dataset

    X, Y, true_mb, true_graph, n_vars, feature_names = load_dataset(args.dataset)

    print("Comparing NSGA-II vs Optuna")
    print(f"  Dataset: {args.dataset} ({X.shape[0]} samples, {n_vars} features)")
    print(f"  Evaluations: {args.n_evals} per method")
    print(f"  Max iter: {args.max_iter}")

    # Reference point for hypervolume (worst feasible values)
    ref_point = np.array([0.0, 0.0])  # BAcc=0, Sparsity=0

    # Run / load methods
    if args.nsga2_results:
        print(f"\n  Loading cached NSGA-II results from {args.nsga2_results}")
        with open(args.nsga2_results, "r") as f:
            cached = json.load(f)
        nsga2_result = {
            "method": "NSGA-II",
            "enhanced_solutions": cached.get("enhanced_solutions", []),
            "wall_time": cached.get("wall_time", 0.0),
            "n_evals": cached.get("n_evals", len(cached.get("enhanced_solutions", []))),
            "pareto_front": cached.get("pareto_front", []),
        }
    else:
        nsga2_result = run_nsga2(X, Y, n_vars, args.n_evals, args.max_iter, args.seed, verbose)
    optuna_result = run_optuna(X, Y, n_vars, args.n_evals, args.max_iter, args.seed, verbose)

    # Compute metrics
    results = {}
    for res in [nsga2_result, optuna_result]:
        method = res["method"]
        sols = res["enhanced_solutions"]

        # Pareto objective points
        if sols:
            obj_points = np.array(
                [
                    [s["metrics"]["classification_balanced_accuracy"], s["metrics"]["mb_sparsity"]]
                    for s in sols
                ]
            )
        else:
            obj_points = np.zeros((0, 2))

        hv = compute_hypervolume_2d(obj_points, ref_point)
        stability = compute_edge_stability(sols, n_vars)

        # Processor distribution
        proc_dist = {}
        for s in sols:
            pt = s["metrics"].get("processor_type", "unknown")
            proc_dist[pt] = proc_dist.get(pt, 0) + 1

        # Structure metrics (SHD/F1/SID) vs ground truth
        best_shd, best_f1, best_sid = None, None, None
        if true_graph is not None and sols:
            from jcce.utils.metrics import compute_structure_metrics

            A_true_sub = np.array(true_graph)[:n_vars, :n_vars]
            for s in sols:
                A_est = s.get("metrics", {}).get("structure_A_est")
                if A_est is not None:
                    A_est_sub = np.array(A_est)[:n_vars, :n_vars]
                    sm = compute_structure_metrics(jnp.array(A_est_sub), jnp.array(A_true_sub))
                    if best_f1 is None or sm["f1"] > best_f1:
                        best_f1 = sm["f1"]
                        best_shd = sm["shd"]
                        best_sid = compute_sid(A_est_sub, A_true_sub)

        results[method] = {
            "n_pareto": len(sols),
            "n_evals": res["n_evals"],
            "wall_time": res["wall_time"],
            "hypervolume": hv,
            "mean_bacc": float(np.mean(obj_points[:, 0])) if len(obj_points) > 0 else 0.0,
            "best_bacc": float(np.max(obj_points[:, 0])) if len(obj_points) > 0 else 0.0,
            "mean_sparsity": float(np.mean(obj_points[:, 1])) if len(obj_points) > 0 else 0.0,
            "n_stable_edges": int(np.sum(stability > 0.8)),
            "processor_distribution": proc_dist,
            "best_f1": best_f1,
            "best_shd": best_shd,
            "best_sid": best_sid,
        }
        if "n_pruned" in res:
            results[method]["n_pruned"] = res["n_pruned"]

    # Print comparison
    print(f"\n{'=' * 70}")
    print(f"{'Metric':<30} {'NSGA-II':>18} {'Optuna':>18}")
    print(f"{'=' * 70}")
    for metric in [
        "n_pareto",
        "n_evals",
        "wall_time",
        "hypervolume",
        "best_bacc",
        "mean_bacc",
        "mean_sparsity",
        "n_stable_edges",
        "best_f1",
        "best_shd",
        "best_sid",
    ]:
        v1 = results["NSGA-II"].get(metric, "N/A")
        v2 = results["Optuna"].get(metric, "N/A")
        if v1 is None:
            v1_str = "N/A"
        elif isinstance(v1, float):
            v1_str = f"{v1:.4f}"
        else:
            v1_str = str(v1)
        if v2 is None:
            v2_str = "N/A"
        elif isinstance(v2, float):
            v2_str = f"{v2:.4f}"
        else:
            v2_str = str(v2)
        print(f"{metric:<30} {v1_str:>18} {v2_str:>18}")

    if results["Optuna"].get("n_pruned"):
        print(f"{'n_pruned':<30} {'N/A':>18} {results['Optuna']['n_pruned']:>18}")

    print(f"\n{'Processor Distribution':}")
    for method in ["NSGA-II", "Optuna"]:
        dist = results[method]["processor_distribution"]
        print(f"  {method}: {dict(dist)}")

    # Speedup
    t1 = results["NSGA-II"]["wall_time"]
    t2 = results["Optuna"]["wall_time"]
    hv1 = results["NSGA-II"]["hypervolume"]
    hv2 = results["Optuna"]["hypervolume"]

    print(f"\nWall-clock ratio: {t1 / t2:.2f}x" if t2 > 0 else "")
    print(f"Hypervolume ratio: {hv2 / hv1:.2f}x" if hv1 > 0 else "")

    # Save results
    if args.output:
        # Convert to JSON-serializable
        json_results = {}
        for k, v in results.items():
            json_results[k] = {kk: vv for kk, vv in v.items()}
        json_results["config"] = {
            "dataset": args.dataset,
            "n_evals": args.n_evals,
            "max_iter": args.max_iter,
            "seed": args.seed,
        }
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
