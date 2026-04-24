#!/usr/bin/env python
"""
Experiment C.6: Ablation of JCCE Components.

Removes each component of the unified framework and measures impact:
1. Full JCCE (baseline)
2. No structure learning (freeze_A=True — random DAG, only train processor)
3. No classification loss (lambda_class=0 — structure-only)
4. No effect estimation (use_amortized_effects=False — no DragonNet)
5. No bow-free constraint (lambda_bow=0 — allow both directed + bidirected)

Validates that each component contributes to the unified framework.

Usage:
    uv run python scripts/experiment_c6_component_ablation.py --quick
    uv run python scripts/experiment_c6_component_ablation.py \
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

from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import SCMConfig, LinearSCM
from jcce.utils.metrics import compute_structure_metrics
from jcce.structure_learning.optuna_search import (
    suggest_hyperparams,
    create_optuna_objective,
    extract_pareto_solutions,
    constraints_func,
)
from jcce.analysis.hypervolume import compute_hypervolume_2d, extract_pareto_front_2d


# ============================================================================
# Data Generation
# ============================================================================

def generate_classification_data(n_vars, n_samples, expected_degree, noise_scale, seed,
                                  min_y_parents=1, max_attempts=100):
    """Generate synthetic classification data from a linear SEM.

    Uses rejection sampling to ensure Y (last node) has at least
    min_y_parents parents, avoiding degenerate DGPs where Y is pure noise.
    """
    Y_idx = n_vars  # Last node is Y

    for attempt in range(max_attempts):
        current_seed = seed + attempt
        dag_config = DAGConfig(
            num_nodes=n_vars + 1,
            graph_type='erdos_renyi',
            expected_degree=expected_degree,
            seed=current_seed,
        )
        A = generate_dag(dag_config)
        A_np = np.array(A)

        # Count parents of Y: A[Y_idx, j] != 0 means j -> Y
        n_parents_y = int(np.sum(np.abs(A_np[Y_idx, :]) > 1e-6))

        if n_parents_y >= min_y_parents:
            if attempt > 0:
                print(f"  [DGP] Rejected {attempt} seed(s); "
                      f"seed={current_seed} gives Y {n_parents_y} parent(s)")
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
# Ablation Conditions
# ============================================================================

ABLATION_CONDITIONS = [
    {
        'name': 'full_jcce',
        'description': 'Full JCCE (baseline)',
        'golem_overrides': {},
        'use_v7': True,
    },
    {
        'name': 'no_structure',
        'description': 'No structure learning (freeze A)',
        'golem_overrides': {'freeze_A': True},
        'use_v7': True,
    },
    {
        'name': 'no_classification',
        'description': 'No classification loss (lambda_class=0)',
        'golem_overrides': {'lambda_class_override': 0.0},
        'use_v7': True,
        # We override lambda_class via suggest_fn
    },
    {
        'name': 'no_effects',
        'description': 'No effect estimation',
        'golem_overrides': {'use_amortized_effects': False},
        'use_v7': False,  # Disable v7 entirely
    },
    {
        'name': 'no_bow',
        'description': 'No bow-free constraint (lambda_bow=0)',
        'golem_overrides': {'lambda_bow': 0.0},
        'use_v7': True,
    },
]


def make_suggest_fn_with_class_override(lambda_class_val):
    """Create a suggest_fn that overrides lambda_class."""
    def suggest_fn(trial, use_v7=True):
        config = suggest_hyperparams(trial, use_v7=use_v7)
        config['lambda_class'] = lambda_class_val
        return config
    return suggest_fn


# ============================================================================
# Run One Ablation Condition
# ============================================================================

def run_ablation_condition(cond, X, Y, n_vars, A_true, n_trials, max_iter, seed):
    """Run Optuna search with one ablation condition."""
    print(f"\n  [{cond['name']}] {cond['description']}")

    sampler = TPESampler(
        multivariate=True, group=True,
        seed=seed,
        n_startup_trials=min(5, n_trials),
        constraints_func=constraints_func,
        constant_liar=True,
    )
    study = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=sampler,
    )

    # Determine suggest_fn override
    suggest_fn = None
    if cond.get('golem_overrides', {}).get('lambda_class_override') is not None:
        suggest_fn = make_suggest_fn_with_class_override(
            cond['golem_overrides']['lambda_class_override']
        )

    # Build golem_overrides (remove our internal keys)
    golem_overrides = {k: v for k, v in cond.get('golem_overrides', {}).items()
                       if k != 'lambda_class_override'}

    objective = create_optuna_objective(
        X=X, Y=Y, n_vars=n_vars,
        max_iter=max_iter,
        use_v7=cond['use_v7'],
        golem_overrides=golem_overrides,
        jax_key_seed=seed,
        verbose=False,
        suggest_fn=suggest_fn,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    enhanced_solutions, _ = extract_pareto_solutions(
        study, use_v7=cond['use_v7'])

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

    # Best BAcc and structure recovery
    best_bacc = 0.0
    best_shd = float('inf')
    best_struct_f1 = 0.0
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
                best_struct_f1 = sm['f1']

    # Mean BAcc across all completed feasible trials
    feasible_baccs = [t.values[0] for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE
                      and t.user_attrs.get('h_A', 1.0) < 0.1
                      and t.values is not None]
    mean_bacc = float(np.mean(feasible_baccs)) if feasible_baccs else 0.0

    result = {
        'name': cond['name'],
        'description': cond['description'],
        'hv': hv,
        'time': elapsed,
        'n_pareto': len(enhanced_solutions),
        'n_feasible': n_feasible,
        'n_completed': n_completed,
        'best_bacc': best_bacc,
        'mean_bacc': mean_bacc,
        'best_shd': int(best_shd) if best_shd < float('inf') else -1,
        'best_struct_f1': best_struct_f1,
    }

    print(f"    HV={hv:.3f} BAcc={best_bacc:.3f}(best)/{mean_bacc:.3f}(mean) "
          f"SHD={result['best_shd']} F1={best_struct_f1:.3f} "
          f"Feasible={n_feasible}/{n_trials} ({elapsed:.1f}s)")

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='C.6: Component Ablation')
    parser.add_argument('--n-trials', type=int, default=15)
    parser.add_argument('--n-vars', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--expected-degree', type=float, default=2.0)
    parser.add_argument('--noise-scale', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 trials, 20 iters, 5 vars')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.max_iter = 20
        args.n_vars = 5

    print(f"COMPONENT ABLATION (C.6)")
    print(f"=" * 70)
    print(f"Data: {args.n_samples} samples, {args.n_vars} vars, "
          f"ER-{args.expected_degree}, noise={args.noise_scale}")
    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")

    # Generate shared data
    X, Y, A_true = generate_classification_data(
        args.n_vars, args.n_samples, args.expected_degree,
        args.noise_scale, args.seed,
    )
    n_edges = int(np.sum(np.abs(A_true) > 1e-6))
    print(f"Ground truth: {n_edges} edges")

    # Run all conditions
    all_results = []
    for cond in ABLATION_CONDITIONS:
        result = run_ablation_condition(
            cond, X, Y, args.n_vars, A_true,
            args.n_trials, args.max_iter, args.seed,
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"ABLATION SUMMARY\n")
    print(f"{'Condition':<22} {'HV':>6} {'BestBAcc':>9} {'MeanBAcc':>9} "
          f"{'BestSHD':>8} {'F1':>6} {'Feas':>5}")
    print(f"{'-' * 22} {'-' * 6} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 6} {'-' * 5}")

    baseline = all_results[0]
    for r in all_results:
        print(f"{r['name']:<22} {r['hv']:>6.3f} {r['best_bacc']:>9.3f} "
              f"{r['mean_bacc']:>9.3f} {r['best_shd']:>8} "
              f"{r['best_struct_f1']:>6.3f} {r['n_feasible']:>5}")

    # Impact analysis
    print(f"\nImpact (delta from full_jcce baseline):")
    for r in all_results[1:]:
        delta_hv = r['hv'] - baseline['hv']
        delta_bacc = r['best_bacc'] - baseline['best_bacc']
        delta_f1 = r['best_struct_f1'] - baseline['best_struct_f1']
        print(f"  {r['name']:<22} deltaHV={delta_hv:+.3f} "
              f"deltaBAcc={delta_bacc:+.3f} deltaF1={delta_f1:+.3f}")

    # Save results
    if args.output:
        output = {
            'args': vars(args),
            'n_edges': n_edges,
            'results': all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
