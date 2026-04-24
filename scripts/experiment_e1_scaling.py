#!/usr/bin/env python
"""
Experiment E.1: Scaling Experiments.

Measures JCCE performance as dimensionality increases:
- d ∈ {10, 20, 50, 100} variables
- Graph types: ER-2, ER-4, SF-2, SF-4
- SCM types: linear, nonlinear (MLP)

Metrics: wall-clock, SHD, F1, balanced accuracy, PEHE, hypervolume.

Usage:
    uv run python scripts/experiment_e1_scaling.py --quick
    uv run python scripts/experiment_e1_scaling.py \
        --dims 10 20 50 --n-trials 15 --n-samples 500 --max-iter 100
"""

import argparse
import json
import time
import sys
import os

import gc

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import SCMConfig, LinearSCM, NonlinearMLPSCM
from jcce.utils.metrics import compute_structure_metrics, compute_varsortability
from jcce.structure_learning.optuna_search import (
    create_optuna_objective,
    extract_pareto_solutions,
    constraints_func,
)
from jcce.analysis.hypervolume import compute_hypervolume_2d, extract_pareto_front_2d


# ============================================================================
# Data Generation
# ============================================================================

def generate_scaling_data(n_vars, n_samples, graph_type, expected_degree,
                          scm_type, noise_scale, seed,
                          min_y_parents=1, max_attempts=100):
    """Generate synthetic data at given dimensionality.

    Rejects DAGs where Y (last node) has fewer than min_y_parents parents.
    Returns (X, Y, A_true_X, n_edges, varsortability).
    """
    Y_idx = n_vars

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
                print(f" [DGP rejected {attempt} seed(s), using seed={current_seed}]",
                      end='', flush=True)
            break
    else:
        raise RuntimeError(
            f"Could not find DAG with Y having >= {min_y_parents} parents "
            f"after {max_attempts} attempts (seeds {seed}-{seed + max_attempts - 1})"
        )

    scm_config = SCMConfig(noise_scale=noise_scale)

    key = random.PRNGKey(current_seed + 1000)
    if scm_type == 'linear':
        scm = LinearSCM(A, scm_config)
        X_full = scm.sample(n_samples, key)
    elif scm_type == 'nonlinear':
        scm = NonlinearMLPSCM(A, scm_config, key=random.PRNGKey(current_seed + 2000))
        X_full = scm.sample(n_samples, key)
    else:
        raise ValueError(f"Unknown scm_type: {scm_type}")

    Y_continuous = X_full[:, -1]
    Y_binary = (Y_continuous > jnp.median(Y_continuous)).astype(jnp.float32)
    X = X_full[:, :n_vars]
    A_true = np.array(A)[:n_vars, :n_vars]
    n_edges = int(np.sum(np.abs(A_true) > 1e-6))

    vs = compute_varsortability(np.array(X_full), np.array(A))

    return (jnp.array(X, dtype=jnp.float32), Y_binary, A_true,
            n_edges, vs)


# ============================================================================
# Run Single Configuration
# ============================================================================

def run_single_config(X, Y, n_vars, A_true, n_trials, max_iter, seed):
    """Run Optuna search and return metrics."""
    sampler = TPESampler(
        multivariate=True, group=True, seed=seed,
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

    enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)

    # Metrics
    n_feasible = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE
                      and t.user_attrs.get('h_A', 1.0) < 0.1])
    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])

    # Hypervolume
    if enhanced_solutions:
        points = np.array([[s['objectives'][0], s['objectives'][1]]
                           for s in enhanced_solutions])
        pareto = extract_pareto_front_2d(points)
        hv = compute_hypervolume_2d(pareto, ref_point=np.array([0.0, 0.0]))
    else:
        hv = 0.0

    # Best metrics
    best_bacc = 0.0
    best_shd = float('inf')
    best_f1 = 0.0
    for sol in enhanced_solutions:
        bacc = sol['objectives'][0]
        if bacc > best_bacc:
            best_bacc = bacc
        A_est = sol['metrics'].get('structure_A_est')
        if A_est is not None:
            sm = compute_structure_metrics(
                jnp.array(np.array(A_est)[:n_vars, :n_vars]),
                jnp.array(A_true),
            )
            if sm['shd'] < best_shd:
                best_shd = sm['shd']
                best_f1 = sm['f1']

    # Time per trial
    time_per_trial = elapsed / max(n_completed, 1)

    return {
        'hv': hv,
        'best_bacc': best_bacc,
        'best_shd': int(best_shd) if best_shd < float('inf') else -1,
        'best_f1': best_f1,
        'n_pareto': len(enhanced_solutions),
        'n_feasible': n_feasible,
        'n_completed': n_completed,
        'total_time': elapsed,
        'time_per_trial': time_per_trial,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='E.1: Scaling Experiments')
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 20, 50, 100])
    parser.add_argument('--graph-types', type=str, nargs='+',
                        default=['erdos_renyi', 'scale_free'])
    parser.add_argument('--degrees', type=float, nargs='+', default=[2.0, 4.0])
    parser.add_argument('--scm-types', type=str, nargs='+',
                        default=['linear', 'nonlinear'])
    parser.add_argument('--n-trials', type=int, default=15)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--noise-scale', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: d={5,10}, ER-2 only, 5 trials, 20 iters')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.dims = [5, 10]
        args.graph_types = ['erdos_renyi']
        args.degrees = [2.0]
        args.scm_types = ['linear']
        args.n_trials = 5
        args.max_iter = 20

    # Build condition list
    conditions = []
    for gt in args.graph_types:
        for deg in args.degrees:
            for scm in args.scm_types:
                label = f"{gt[:2].upper()}-{int(deg)}" if gt != 'scale_free' else f"SF-{int(deg)}"
                conditions.append({
                    'graph_type': gt,
                    'degree': deg,
                    'scm_type': scm,
                    'label': f"{label}/{scm[:3]}",
                })

    print(f"SCALING EXPERIMENTS (E.1)")
    print(f"=" * 70)
    print(f"Dimensions: {args.dims}")
    print(f"Conditions: {len(conditions)} ({', '.join(c['label'] for c in conditions)})")
    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")
    print(f"Samples: {args.n_samples}, noise: {args.noise_scale}")

    all_results = []

    for cond in conditions:
        print(f"\n{'=' * 70}")
        print(f"Condition: {cond['label']} "
              f"({cond['graph_type']}, degree={cond['degree']}, {cond['scm_type']})")

        for d in args.dims:
            # Clear JAX caches and Python garbage between dimensions to prevent
            # GPU memory fragmentation (caused E1 OOM at d>=50 on server)
            jax.clear_caches()
            gc.collect()

            print(f"\n  d={d}...", end='', flush=True)

            try:
                X, Y, A_true, n_edges, vs = generate_scaling_data(
                    d, args.n_samples, cond['graph_type'], cond['degree'],
                    cond['scm_type'], args.noise_scale, args.seed,
                )
            except Exception as e:
                print(f" DATA GENERATION FAILED: {e}")
                all_results.append({
                    'd': d, 'condition': cond['label'],
                    'graph_type': cond['graph_type'], 'degree': cond['degree'],
                    'scm_type': cond['scm_type'], 'error': str(e),
                    'hv': 0.0, 'best_bacc': 0.0, 'best_shd': -1, 'best_f1': 0.0,
                    'n_pareto': 0, 'n_feasible': 0, 'n_completed': 0,
                    'total_time': 0.0, 'time_per_trial': 0.0,
                    'n_edges': -1, 'varsortability': 0.0,
                })
                continue

            print(f" {n_edges} edges, varsort={vs:.3f}", end='', flush=True)

            try:
                result = run_single_config(
                    X, Y, d, A_true, args.n_trials, args.max_iter, args.seed,
                )
            except Exception as e:
                print(f" OPTIMIZATION FAILED: {e}")
                all_results.append({
                    'd': d, 'condition': cond['label'],
                    'graph_type': cond['graph_type'], 'degree': cond['degree'],
                    'scm_type': cond['scm_type'], 'error': str(e),
                    'hv': 0.0, 'best_bacc': 0.0, 'best_shd': -1, 'best_f1': 0.0,
                    'n_pareto': 0, 'n_feasible': 0, 'n_completed': 0,
                    'total_time': 0.0, 'time_per_trial': 0.0,
                    'n_edges': n_edges, 'varsortability': vs,
                })
                continue

            result.update({
                'd': d,
                'condition': cond['label'],
                'graph_type': cond['graph_type'],
                'degree': cond['degree'],
                'scm_type': cond['scm_type'],
                'n_edges': n_edges,
                'varsortability': vs,
            })
            all_results.append(result)

            print(f" | HV={result['hv']:.3f} BAcc={result['best_bacc']:.3f} "
                  f"SHD={result['best_shd']} F1={result['best_f1']:.3f} "
                  f"t={result['total_time']:.1f}s ({result['time_per_trial']:.1f}s/trial)")

    # Summary tables
    print(f"\n{'=' * 70}")
    print(f"SCALING SUMMARY\n")

    # Group by condition
    for cond in conditions:
        cond_results = [r for r in all_results if r['condition'] == cond['label']]
        if not cond_results:
            continue

        print(f"\n{cond['label']}:")
        print(f"  {'d':>5} {'Edges':>6} {'HV':>7} {'BAcc':>6} {'SHD':>5} "
              f"{'F1':>6} {'Feas':>5} {'Time':>7} {'t/trial':>8} {'Varsort':>8}")
        print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*5} "
              f"{'-'*6} {'-'*5} {'-'*7} {'-'*8} {'-'*8}")

        for r in sorted(cond_results, key=lambda x: x['d']):
            print(f"  {r['d']:>5} {r['n_edges']:>6} {r['hv']:>7.3f} "
                  f"{r['best_bacc']:>6.3f} {r['best_shd']:>5} "
                  f"{r['best_f1']:>6.3f} {r['n_feasible']:>5} "
                  f"{r['total_time']:>7.1f} {r['time_per_trial']:>8.1f} "
                  f"{r['varsortability']:>8.3f}")

    # Scaling analysis
    print(f"\nScaling Analysis:")
    for cond in conditions:
        cond_results = sorted(
            [r for r in all_results if r['condition'] == cond['label']],
            key=lambda x: x['d']
        )
        if len(cond_results) >= 2:
            d_small = cond_results[0]
            d_large = cond_results[-1]
            time_ratio = d_large['total_time'] / max(d_small['total_time'], 1e-6)
            d_ratio = d_large['d'] / d_small['d']
            # Estimate scaling exponent: time ∝ d^α → α = log(t2/t1) / log(d2/d1)
            alpha = np.log(time_ratio) / np.log(d_ratio) if d_ratio > 1 else 0
            print(f"  {cond['label']}: time scales as d^{alpha:.2f} "
                  f"(d={d_small['d']}→{d_large['d']}: {d_small['total_time']:.1f}s → "
                  f"{d_large['total_time']:.1f}s)")

            f1_drop = d_large['best_f1'] - d_small['best_f1']
            print(f"    F1 change: {f1_drop:+.3f} "
                  f"(d={d_small['d']}: {d_small['best_f1']:.3f} → "
                  f"d={d_large['d']}: {d_large['best_f1']:.3f})")

    # Save results
    if args.output:
        output = {
            'args': vars(args),
            'results': all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
