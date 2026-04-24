#!/usr/bin/env python
"""
Experiment C.5: Processor Analysis.

For each dataset, analyze which processor types dominate which Pareto regions.
Computes per-processor statistics: accuracy, sparsity preference, edge stability,
and whether different architectures discover different structures.

Usage:
    uv run python scripts/experiment_c5_processor_analysis.py --quick
    uv run python scripts/experiment_c5_processor_analysis.py \
        --n-trials 30 --n-vars 10 --n-samples 500 --max-iter 100
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import optuna
from jax import random
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jcce.analysis.pareto_stability import compute_edge_stability
from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import LinearSCM, SCMConfig
from jcce.structure_learning.optuna_search import (
    PROCESSOR_TYPES,
    constraints_func,
    create_optuna_objective,
    extract_pareto_solutions,
)
from jcce.utils.metrics import compute_structure_metrics

# ============================================================================
# Data Generation
# ============================================================================


def generate_classification_data(
    n_vars, n_samples, expected_degree, noise_scale, seed, min_y_parents=1, max_attempts=100
):
    """Generate synthetic classification data from a linear SEM.

    Uses rejection sampling to ensure Y (last node) has at least
    min_y_parents parents, avoiding degenerate DGPs where Y is pure noise.
    """
    Y_idx = n_vars  # Last node is Y

    for attempt in range(max_attempts):
        current_seed = seed + attempt
        dag_config = DAGConfig(
            num_nodes=n_vars + 1,
            graph_type="erdos_renyi",
            expected_degree=expected_degree,
            seed=current_seed,
        )
        A = generate_dag(dag_config)
        A_np = np.array(A)

        # Count parents of Y: A[Y_idx, j] != 0 means j -> Y
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
    A_true = np.array(A)[:n_vars, :n_vars]

    return jnp.array(X, dtype=jnp.float32), Y_binary, A_true


# ============================================================================
# Per-Processor Analysis
# ============================================================================


def analyze_per_processor(study, enhanced_solutions, n_vars, A_true):
    """Compute per-processor statistics from an Optuna study.

    Returns dict mapping processor_type -> stats dict.
    """
    n_total = n_vars + 1
    A_true_full = np.zeros((n_total, n_total))
    A_true_full[:n_vars, :n_vars] = A_true

    # Group trials by processor
    proc_trials = defaultdict(list)
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        pt = trial.user_attrs.get("processor_type", "unknown")
        proc_trials[pt].append(trial)

    # Group Pareto solutions by processor
    proc_pareto = defaultdict(list)
    for sol in enhanced_solutions:
        pt = sol["metrics"].get("processor_type", "unknown")
        proc_pareto[pt].append(sol)

    results = {}
    for pt in PROCESSOR_TYPES:
        trials = proc_trials.get(pt, [])
        pareto_sols = proc_pareto.get(pt, [])

        if not trials:
            results[pt] = {
                "n_trials": 0,
                "n_feasible": 0,
                "n_pareto": 0,
                "mean_bacc": 0.0,
                "mean_sparsity": 0.0,
                "mean_time": 0.0,
                "stability_f1": 0.0,
                "best_shd": -1,
                "best_f1": 0.0,
            }
            continue

        # Basic stats
        baccs = [t.values[0] for t in trials if t.values is not None]
        sparsities = [t.values[1] for t in trials if t.values is not None]
        times = [t.user_attrs.get("trial_time", 0.0) for t in trials]
        feasible = [t for t in trials if t.user_attrs.get("h_A", 1.0) < 0.1]

        # Edge stability for this processor's Pareto solutions
        if pareto_sols:
            stab = compute_edge_stability(pareto_sols, n_vars, threshold=0.1)
            from jcce.analysis.pareto_stability import compare_stability_to_ground_truth

            gt = compare_stability_to_ground_truth(stab, A_true_full, min_frequency=0.5)
            stab_f1 = gt["f1"]
        else:
            stab_f1 = 0.0

        # Best structure recovery
        best_shd = float("inf")
        best_f1 = 0.0
        for sol in pareto_sols:
            A_est = sol["metrics"].get("structure_A_est")
            if A_est is not None:
                sm = compute_structure_metrics(
                    jnp.array(np.array(A_est)[:n_vars, :n_vars]),
                    jnp.array(A_true),
                )
                if sm["shd"] < best_shd:
                    best_shd = sm["shd"]
                    best_f1 = sm["f1"]

        results[pt] = {
            "n_trials": len(trials),
            "n_feasible": len(feasible),
            "n_pareto": len(pareto_sols),
            "mean_bacc": float(np.mean(baccs)) if baccs else 0.0,
            "mean_sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
            "mean_time": float(np.mean(times)) if times else 0.0,
            "stability_f1": stab_f1,
            "best_shd": int(best_shd) if best_shd < float("inf") else -1,
            "best_f1": best_f1,
        }

    return results


def compute_structural_agreement(enhanced_solutions, n_vars):
    """Compute pairwise structural agreement between Pareto solutions.

    Agreement = fraction of edges that match (both present or both absent).
    Returns agreement matrix of shape (n_pareto, n_pareto).
    """
    n = len(enhanced_solutions)
    if n < 2:
        return np.ones((n, n))

    # Binarize all adjacency matrices
    binary_As = []
    for sol in enhanced_solutions:
        A = np.array(sol["metrics"].get("structure_A_est", np.zeros((n_vars + 1, n_vars + 1))))
        binary_As.append((np.abs(A[:n_vars, :n_vars]) > 0.3).astype(float))

    agreement = np.zeros((n, n))
    n_pairs = n_vars * (n_vars - 1)  # Exclude diagonal

    for i in range(n):
        for j in range(n):
            if i == j:
                agreement[i, j] = 1.0
            else:
                match = np.sum(binary_As[i] == binary_As[j]) - n_vars  # Exclude diagonal
                agreement[i, j] = match / n_pairs if n_pairs > 0 else 1.0

    return agreement


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="C.5: Processor Analysis")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-vars", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--expected-degree", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 10 trials, 20 iters, 5 vars"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Real dataset name (lucas, sachs, etc.)"
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 10
        args.max_iter = 20
        args.n_vars = 5

    print("PROCESSOR ANALYSIS (C.5)")
    print("=" * 70)

    if args.dataset:
        from jcce.data.benchmark_loader import load_dataset as load_benchmark

        X_np, Y_np, config = load_benchmark(args.dataset)
        X = jnp.array(X_np, dtype=jnp.float32)
        Y = jnp.array(Y_np, dtype=jnp.float32)
        args.n_vars = X.shape[1]
        A_dag = config.get("true_dag")
        if config.get("has_true_dag", False) and A_dag is not None:
            A_true = np.array(A_dag)[: args.n_vars, : args.n_vars]
        else:
            A_true = np.zeros((args.n_vars, args.n_vars))
        print(f"Dataset: {args.dataset} ({X.shape[0]} samples, {args.n_vars} vars)")
    else:
        X, Y, A_true = generate_classification_data(
            args.n_vars,
            args.n_samples,
            args.expected_degree,
            args.noise_scale,
            args.seed,
        )
        print(
            f"Data: {args.n_samples} samples, {args.n_vars} vars, "
            f"ER-{args.expected_degree}, noise={args.noise_scale}"
        )

    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")
    print(f"Processors: {', '.join(PROCESSOR_TYPES)}")
    n_edges = int(np.sum(np.abs(A_true) > 1e-6))
    print(f"Ground truth: {n_edges} edges")

    # Run Optuna search
    print(f"\nRunning {args.n_trials} trials...")
    sampler = TPESampler(
        multivariate=True,
        group=True,
        seed=args.seed,
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
        jax_key_seed=args.seed,
        verbose=True,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    elapsed = time.time() - t0

    enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)
    print(f"\nTotal time: {elapsed:.1f}s, Pareto solutions: {len(enhanced_solutions)}")

    # Per-processor analysis
    proc_stats = analyze_per_processor(study, enhanced_solutions, args.n_vars, A_true)

    # Summary table
    print(f"\n{'=' * 70}")
    print("PER-PROCESSOR ANALYSIS")
    print(
        f"\n{'Processor':<14} {'#Tried':>7} {'#Feas':>6} {'#Pareto':>8} "
        f"{'BAcc':>6} {'Sparse':>7} {'StabF1':>7} {'BestSHD':>8} {'Time':>6}"
    )
    print(
        f"{'-' * 14} {'-' * 7} {'-' * 6} {'-' * 8} "
        f"{'-' * 6} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 6}"
    )

    for pt in PROCESSOR_TYPES:
        s = proc_stats[pt]
        print(
            f"{pt:<14} {s['n_trials']:>7} {s['n_feasible']:>6} "
            f"{s['n_pareto']:>8} {s['mean_bacc']:>6.3f} "
            f"{s['mean_sparsity']:>7.3f} {s['stability_f1']:>7.3f} "
            f"{s['best_shd']:>8} {s['mean_time']:>6.1f}"
        )

    # Structural agreement between Pareto solutions
    if len(enhanced_solutions) >= 2:
        agreement = compute_structural_agreement(enhanced_solutions, args.n_vars)
        mean_agreement = np.mean(agreement[np.triu_indices(len(agreement), k=1)])
        print(f"\nMean pairwise structural agreement: {mean_agreement:.3f}")

        # Agreement within vs between processors
        pareto_procs = [sol["metrics"].get("processor_type", "") for sol in enhanced_solutions]
        within = []
        between = []
        for i in range(len(enhanced_solutions)):
            for j in range(i + 1, len(enhanced_solutions)):
                if pareto_procs[i] == pareto_procs[j]:
                    within.append(agreement[i, j])
                else:
                    between.append(agreement[i, j])

        if within:
            print(f"  Within-processor agreement:  {np.mean(within):.3f} (n={len(within)} pairs)")
        if between:
            print(f"  Between-processor agreement: {np.mean(between):.3f} (n={len(between)} pairs)")

    # Key findings
    print("\nKey Findings:")
    pareto_procs = [sol["metrics"].get("processor_type", "") for sol in enhanced_solutions]
    for pt in PROCESSOR_TYPES:
        count = pareto_procs.count(pt)
        if count > 0:
            print(f"  {pt}: {count} Pareto solutions")

    # Which processor produces sparsest Pareto solutions?
    if enhanced_solutions:
        sparsest = max(enhanced_solutions, key=lambda s: s["objectives"][1])
        densest = min(enhanced_solutions, key=lambda s: s["objectives"][1])
        print(
            f"\n  Sparsest Pareto: {sparsest['metrics']['processor_type']} "
            f"(sparsity={sparsest['objectives'][1]:.3f})"
        )
        print(
            f"  Most accurate:  {max(enhanced_solutions, key=lambda s: s['objectives'][0])['metrics']['processor_type']} "
            f"(BAcc={max(enhanced_solutions, key=lambda s: s['objectives'][0])['objectives'][0]:.3f})"
        )

    # Save results
    if args.output:
        output = {
            "args": vars(args),
            "n_edges": n_edges,
            "time": elapsed,
            "n_pareto": len(enhanced_solutions),
            "processor_stats": proc_stats,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
