#!/usr/bin/env python
"""
Experiment C.7: Noise Sensitivity Analysis.

Adds increasing levels of noise to synthetic data and measures degradation
of Pareto front quality (hypervolume, edge stability, structure recovery).
Compares Gaussian vs Laplace noise. Identifies robust processor types.

Usage:
    uv run python scripts/experiment_c7_noise_sensitivity.py --quick
    uv run python scripts/experiment_c7_noise_sensitivity.py \
        --n-trials 15 --n-vars 10 --n-samples 500 --max-iter 100
"""

import argparse
import json
import time
import sys
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jcce.data.dag_generator import DAGConfig, generate_dag, count_edges
from jcce.data.scm import SCMConfig, LinearSCM
from jcce.utils.metrics import compute_structure_metrics
from jcce.structure_learning.optuna_search import (
    suggest_hyperparams,
    create_optuna_objective,
    extract_pareto_solutions,
    constraints_func,
)
from jcce.analysis.hypervolume import compute_hypervolume_2d, extract_pareto_front_2d
from jcce.analysis.pareto_stability import (
    compute_edge_stability,
    compare_stability_to_ground_truth,
)


# ============================================================================
# Data Generation
# ============================================================================

def generate_data_at_noise(A, n_samples, noise_scale, noise_type, seed):
    """Generate classification data from given DAG at specified noise level.

    Returns (X, Y) where Y is binary (median-split of last node).
    """
    scm_config = SCMConfig(noise_scale=noise_scale, noise_type=noise_type)
    scm = LinearSCM(A, scm_config)
    key = random.PRNGKey(seed)
    X_full = scm.sample(n_samples, key)

    Y_continuous = X_full[:, -1]
    Y_binary = (Y_continuous > jnp.median(Y_continuous)).astype(jnp.float32)
    X = X_full[:, :-1]

    return jnp.array(X, dtype=jnp.float32), Y_binary


# ============================================================================
# Run One Noise Condition
# ============================================================================

def run_noise_condition(X, Y, n_vars, A_true, n_trials, max_iter, seed,
                        noise_scale, noise_type):
    """Run Optuna search at one noise level and compute metrics.

    Returns dict with HV, stability metrics, structure recovery, processor info.
    """
    sampler = TPESampler(
        multivariate=True,
        group=True,
        seed=seed,
        n_startup_trials=min(5, n_trials),
        constraints_func=constraints_func,
        constant_liar=True,
    )

    study = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=sampler,
    )

    objective = create_optuna_objective(
        X=X, Y=Y, n_vars=n_vars,
        max_iter=max_iter, use_v7=True,
        jax_key_seed=seed, verbose=False,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    # Extract Pareto solutions
    enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)

    # Hypervolume
    if enhanced_solutions:
        points = np.array([[s['objectives'][0], s['objectives'][1]]
                           for s in enhanced_solutions])
        pareto = extract_pareto_front_2d(points)
        hv = compute_hypervolume_2d(pareto, ref_point=np.array([0.0, 0.0]))
    else:
        hv = 0.0

    # Edge stability + ground truth comparison
    # Build full A_true including Y node (last row/col = 0)
    n_total = n_vars + 1
    A_true_full = np.zeros((n_total, n_total))
    A_true_full[:n_vars, :n_vars] = np.array(A_true)

    stability_result = {}
    if enhanced_solutions:
        stab = compute_edge_stability(enhanced_solutions, n_vars, threshold=0.1)
        gt_result = compare_stability_to_ground_truth(stab, A_true_full, min_frequency=0.5)
        stability_result = gt_result
        stability_result['n_stable_08'] = int(np.sum(stab >= 0.8))
    else:
        stability_result = {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'n_stable': 0, 'n_true': int(np.sum(np.abs(A_true) > 1e-6)),
            'n_stable_08': 0,
        }

    # Best solution structure recovery
    best_shd = float('inf')
    best_f1 = 0.0
    if enhanced_solutions:
        for sol in enhanced_solutions:
            A_est = sol['metrics'].get('structure_A_est')
            if A_est is not None:
                A_est_X = np.array(A_est)[:n_vars, :n_vars]
                sm = compute_structure_metrics(
                    jnp.array(A_est_X), jnp.array(A_true)
                )
                if sm['shd'] < best_shd:
                    best_shd = sm['shd']
                    best_f1 = sm['f1']

    # Processor breakdown
    proc_counts = {}
    for sol in enhanced_solutions:
        pt = sol['metrics'].get('processor_type', 'unknown')
        proc_counts[pt] = proc_counts.get(pt, 0) + 1

    # Count states
    n_feasible = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE
                      and t.user_attrs.get('h_A', 1.0) < 0.1])
    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])

    return {
        'noise_scale': noise_scale,
        'noise_type': noise_type,
        'hv': hv,
        'time': elapsed,
        'n_pareto': len(enhanced_solutions),
        'n_feasible': n_feasible,
        'n_completed': n_completed,
        'best_shd': int(best_shd) if best_shd < float('inf') else -1,
        'best_f1': best_f1,
        'stability_precision': stability_result.get('precision', 0.0),
        'stability_recall': stability_result.get('recall', 0.0),
        'stability_f1': stability_result.get('f1', 0.0),
        'n_stable_08': stability_result.get('n_stable_08', 0),
        'processor_counts': proc_counts,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='C.7: Noise Sensitivity Analysis')
    parser.add_argument('--n-trials', type=int, default=15)
    parser.add_argument('--n-vars', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--noise-scales', nargs='+', type=float,
                        default=[0.1, 0.3, 0.5, 1.0, 2.0])
    parser.add_argument('--noise-types', nargs='+',
                        default=['gaussian', 'laplace'])
    parser.add_argument('--expected-degree', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 trials, 20 iters, 5 vars, fewer noise levels')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.max_iter = 20
        args.n_vars = 5
        args.noise_scales = [0.1, 0.5, 2.0]
        args.noise_types = ['gaussian']

    print(f"NOISE SENSITIVITY ANALYSIS (C.7)")
    print(f"=" * 70)
    print(f"Data: {args.n_samples} samples, {args.n_vars} vars, ER-{args.expected_degree}")
    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")
    print(f"Noise scales: {args.noise_scales}")
    print(f"Noise types: {args.noise_types}")

    # Generate ONE DAG (shared across all noise conditions)
    dag_config = DAGConfig(
        num_nodes=args.n_vars + 1,
        graph_type='erdos_renyi',
        expected_degree=args.expected_degree,
        seed=args.seed,
    )
    A = generate_dag(dag_config)
    A_true = np.array(A)[:args.n_vars, :args.n_vars]
    n_edges = int(np.sum(np.abs(A_true) > 1e-6))
    print(f"Ground truth DAG: {n_edges} edges among {args.n_vars} vars")

    all_results = []

    for noise_type in args.noise_types:
        print(f"\n--- Noise type: {noise_type} ---")

        for noise_scale in args.noise_scales:
            print(f"\n  noise_scale={noise_scale}...", end='', flush=True)

            X, Y = generate_data_at_noise(
                A, args.n_samples, noise_scale, noise_type,
                seed=args.seed + int(noise_scale * 100),
            )

            result = run_noise_condition(
                X, Y, args.n_vars, A_true,
                args.n_trials, args.max_iter,
                seed=args.seed + int(noise_scale * 100),
                noise_scale=noise_scale,
                noise_type=noise_type,
            )
            all_results.append(result)

            print(f" HV={result['hv']:.3f} bestSHD={result['best_shd']} "
                  f"stabF1={result['stability_f1']:.3f} "
                  f"({result['time']:.1f}s)")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {args.n_vars} vars, {n_edges} true edges, "
          f"{args.n_trials} trials x {args.max_iter} iters\n")
    print(f"{'Noise Type':<10} {'Scale':>6} {'HV':>6} {'BestSHD':>8} "
          f"{'BestF1':>7} {'StabF1':>7} {'Stab>0.8':>9} {'Pareto':>7}")
    print(f"{'-' * 10} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 7}")

    for r in all_results:
        print(f"{r['noise_type']:<10} {r['noise_scale']:>6.1f} "
              f"{r['hv']:>6.3f} {r['best_shd']:>8} "
              f"{r['best_f1']:>7.3f} {r['stability_f1']:>7.3f} "
              f"{r['n_stable_08']:>9} {r['n_pareto']:>7}")

    # Degradation analysis
    print(f"\nDegradation Analysis:")
    for noise_type in args.noise_types:
        type_results = [r for r in all_results if r['noise_type'] == noise_type]
        if len(type_results) >= 2:
            hvs = [r['hv'] for r in type_results]
            f1s = [r['best_f1'] for r in type_results]
            scales = [r['noise_scale'] for r in type_results]
            hv_drop = (hvs[0] - hvs[-1]) / hvs[0] * 100 if hvs[0] > 0 else 0
            f1_drop = (f1s[0] - f1s[-1]) / f1s[0] * 100 if f1s[0] > 0 else 0
            print(f"  {noise_type}: HV drops {hv_drop:.1f}% from "
                  f"noise={scales[0]} to noise={scales[-1]}")
            print(f"  {noise_type}: F1 drops {f1_drop:.1f}% from "
                  f"noise={scales[0]} to noise={scales[-1]}")

    # Save results
    if args.output:
        output = {
            'args': vars(args),
            'n_edges': n_edges,
            'results': all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
