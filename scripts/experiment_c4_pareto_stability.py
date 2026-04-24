#!/usr/bin/env python
"""
Experiment C.4: Pareto Edge Stability Analysis.

Runs JCCE Optuna search on synthetic data with known ground truth, extracts
the Pareto front, computes edge stability at multiple thresholds, and shows
that Pareto-stable edges have higher precision than edges from the single
best solution.

Connects to Stability Selection theory (Meinshausen & Buhlmann, JRSSB 2010).

Usage:
    uv run python scripts/experiment_c4_pareto_stability.py --quick
    uv run python scripts/experiment_c4_pareto_stability.py \
        --n-trials 20 --n-vars 10 --n-samples 500 --max-iter 100
"""

import argparse
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np
import optuna
from jax import random
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jcce.analysis.hypervolume import compute_hypervolume_2d, extract_pareto_front_2d
from jcce.analysis.pareto_stability import (
    compare_stability_to_ground_truth,
    compute_edge_stability,
    get_stable_edges,
    stability_sensitivity_analysis,
)
from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import LinearSCM, SCMConfig
from jcce.structure_learning.optuna_search import (
    constraints_func,
    create_optuna_objective,
    extract_pareto_solutions,
    get_trial_artifacts,
)
from jcce.utils.metrics import compute_structure_metrics

# ============================================================================
# Data Generation
# ============================================================================


def generate_classification_data(
    n_vars,
    n_samples,
    graph_type,
    expected_degree,
    noise_scale,
    seed,
    min_y_parents=1,
    max_attempts=100,
):
    """Generate synthetic classification data with known ground truth.

    Uses rejection sampling to ensure Y (last node) has at least
    ``min_y_parents`` parents, avoiding degenerate DGPs where Y is
    pure noise.
    """
    Y_idx = n_vars  # last node

    for attempt in range(max_attempts):
        current_seed = seed + attempt
        dag_config = DAGConfig(
            num_nodes=n_vars + 1,
            graph_type=graph_type,
            expected_degree=expected_degree,
            seed=current_seed,
        )
        A = generate_dag(dag_config)
        A_np = np.array(A)

        n_parents_y = int(np.sum(np.abs(A_np[Y_idx, :]) > 1e-6))

        if n_parents_y >= min_y_parents:
            if attempt > 0:
                print(
                    f"  [DGP] Rejected {attempt} seed(s); "
                    f"seed={current_seed} gives Y {n_parents_y} parent(s)"
                )
            break
    else:
        raise RuntimeError(
            f"Could not find DAG with Y having >= {min_y_parents} parents "
            f"after {max_attempts} attempts (seeds {seed}-{seed + max_attempts - 1})"
        )

    scm_config = SCMConfig(noise_scale=noise_scale)
    scm = LinearSCM(A, scm_config)
    key = random.PRNGKey(current_seed + 1000)
    X_full = scm.sample(n_samples, key)

    Y_continuous = X_full[:, -1]
    Y_binary = (Y_continuous > jnp.median(Y_continuous)).astype(jnp.float32)
    X = X_full[:, :n_vars]
    A_true = np.array(A)

    return jnp.array(X, dtype=jnp.float32), Y_binary, A_true


# ============================================================================
# Analysis: Single Best vs Pareto Stability
# ============================================================================


def analyze_single_best(study, n_vars, A_true_full):
    """Compute structure metrics for the single best trial (highest BAcc).

    Returns dict with precision, recall, f1, shd for the best-accuracy solution.
    """
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("h_A", 1.0) < 0.1
        and t.values is not None
    ]
    if not completed:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": -1}

    best = max(completed, key=lambda t: t.values[0])
    artifacts = get_trial_artifacts(best.number)
    if artifacts is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": -1}

    A_est = np.array(artifacts["A_est"])
    A_true_np = np.array(A_true_full)

    # Compute edge-level metrics
    sm = compute_structure_metrics(
        jnp.array(A_est[:n_vars, :n_vars]),
        jnp.array(A_true_np[:n_vars, :n_vars]),
    )
    return {
        "precision": sm["precision"],
        "recall": sm["recall"],
        "f1": sm["f1"],
        "shd": sm["shd"],
        "bacc": best.values[0],
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="C.4: Pareto Edge Stability")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-vars", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--graph-types", nargs="+", default=["erdos_renyi"])
    parser.add_argument("--expected-degree", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--stability-thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument(
        "--frequency-cutoffs", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 5 trials, 20 iters, 5 vars"
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.max_iter = 20
        args.n_vars = 5
        args.frequency_cutoffs = [0.5, 0.8, 1.0]

    print("PARETO EDGE STABILITY ANALYSIS (C.4)")
    print("=" * 70)
    print(f"Data: {args.n_samples} samples, {args.n_vars} vars, noise={args.noise_scale}")
    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")

    all_results = []

    for gt_idx, graph_type in enumerate(args.graph_types):
        print(f"\n{'=' * 70}")
        print(f"Graph type: {graph_type}")

        # Generate data
        X, Y, A_true = generate_classification_data(
            args.n_vars,
            args.n_samples,
            graph_type,
            args.expected_degree,
            args.noise_scale,
            args.seed + gt_idx * 100,
        )
        A_true_X = A_true[: args.n_vars, : args.n_vars]
        n_edges = int(np.sum(np.abs(A_true_X) > 1e-6))
        print(f"Ground truth: {n_edges} edges among {args.n_vars} vars")

        # Run Optuna search
        print(f"\nRunning {args.n_trials} trials...")
        sampler = TPESampler(
            multivariate=True,
            group=True,
            seed=args.seed + gt_idx,
            n_startup_trials=min(5, args.n_trials),
            constraints_func=constraints_func,
            constant_liar=True,
        )
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=sampler,
        )

        objective = create_optuna_objective(
            X=X,
            Y=Y,
            n_vars=args.n_vars,
            max_iter=args.max_iter,
            use_v7=True,
            jax_key_seed=args.seed + gt_idx,
            verbose=False,
        )

        t0 = time.time()
        study.optimize(objective, n_trials=args.n_trials)
        elapsed = time.time() - t0

        enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)

        n_feasible = len(
            [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.user_attrs.get("h_A", 1.0) < 0.1
            ]
        )

        print(
            f"Completed in {elapsed:.1f}s: "
            f"{n_feasible} feasible, {len(enhanced_solutions)} Pareto solutions"
        )

        if not enhanced_solutions:
            print("  No feasible Pareto solutions found. Skipping analysis.")
            all_results.append(
                {
                    "graph_type": graph_type,
                    "n_edges": n_edges,
                    "n_feasible": n_feasible,
                    "n_pareto": 0,
                    "error": "no feasible solutions",
                }
            )
            continue

        # HV
        points = np.array([[s["objectives"][0], s["objectives"][1]] for s in enhanced_solutions])
        pareto = extract_pareto_front_2d(points)
        hv = compute_hypervolume_2d(pareto, ref_point=np.array([0.0, 0.0]))

        # --- Analysis 1: Single best vs Pareto stability ---
        single_best = analyze_single_best(study, args.n_vars, A_true)
        print(
            f"\n  Single best solution: F1={single_best['f1']:.3f} "
            f"Prec={single_best['precision']:.3f} "
            f"Rec={single_best['recall']:.3f} SHD={single_best['shd']}"
        )

        # --- Analysis 2: Edge stability at multiple thresholds ---
        print("\n  Pareto Edge Stability (min_frequency -> ground truth comparison):")
        print(f"  {'Threshold':>10} {'FreqCut':>8} {'#Stable':>8} {'Prec':>6} {'Rec':>6} {'F1':>6}")

        stability_rows = []
        for threshold in args.stability_thresholds:
            stab = compute_edge_stability(enhanced_solutions, args.n_vars, threshold=threshold)
            for freq in args.frequency_cutoffs:
                gt_result = compare_stability_to_ground_truth(stab, A_true, min_frequency=freq)
                row = {
                    "threshold": threshold,
                    "frequency": freq,
                    **gt_result,
                }
                stability_rows.append(row)
                print(
                    f"  {threshold:>10.2f} {freq:>8.2f} "
                    f"{gt_result['n_stable']:>8} "
                    f"{gt_result['precision']:>6.3f} "
                    f"{gt_result['recall']:>6.3f} "
                    f"{gt_result['f1']:>6.3f}"
                )

        # --- Analysis 3: Sensitivity analysis ---
        sensitivity = stability_sensitivity_analysis(
            enhanced_solutions,
            args.n_vars,
            thresholds=tuple(args.stability_thresholds),
            frequencies=tuple(args.frequency_cutoffs),
            true_graph=A_true,
        )

        # --- Analysis 4: Key comparison ---
        # Find best stability F1 across all threshold/freq combos
        best_stab = max(stability_rows, key=lambda r: r["f1"])

        print("\n  Key Comparison:")
        print(f"    Single best solution:  F1={single_best['f1']:.3f}")
        print(
            f"    Best Pareto-stable:    F1={best_stab['f1']:.3f} "
            f"(thresh={best_stab['threshold']}, freq={best_stab['frequency']})"
        )

        improvement = best_stab["f1"] - single_best["f1"]
        if improvement > 0:
            print(f"    -> Pareto stability IMPROVES F1 by +{improvement:.3f}")
        elif improvement < 0:
            print(f"    -> Single best has higher F1 by +{-improvement:.3f}")
        else:
            print("    -> Same F1")

        # Stable edges list
        stab_default = compute_edge_stability(enhanced_solutions, args.n_vars, threshold=0.3)
        stable_edges = get_stable_edges(stab_default, min_frequency=0.8)
        if stable_edges:
            print("\n  Highly stable edges (freq >= 0.8, thresh=0.3):")
            for i, j, freq in stable_edges[:10]:  # Top 10
                is_true = np.abs(A_true[i, j]) > 1e-6
                label = "TRUE" if is_true else "FALSE"
                print(f"    {j} -> {i}: freq={freq:.2f} [{label}]")

        result = {
            "graph_type": graph_type,
            "n_edges": n_edges,
            "n_feasible": n_feasible,
            "n_pareto": len(enhanced_solutions),
            "hv": hv,
            "time": elapsed,
            "single_best": single_best,
            "best_stability": {
                "f1": best_stab["f1"],
                "precision": best_stab["precision"],
                "recall": best_stab["recall"],
                "threshold": best_stab["threshold"],
                "frequency": best_stab["frequency"],
            },
            "f1_improvement": improvement,
            "stability_rows": stability_rows,
        }
        all_results.append(result)

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    for r in all_results:
        if "error" in r:
            print(f"  {r['graph_type']}: FAILED ({r['error']})")
        else:
            print(
                f"  {r['graph_type']}: SingleBest F1={r['single_best']['f1']:.3f} "
                f"ParetoStable F1={r['best_stability']['f1']:.3f} "
                f"(delta={r['f1_improvement']:+.3f})"
            )

    # Save results
    if args.output:
        # Convert non-serializable items
        output = {
            "args": vars(args),
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
