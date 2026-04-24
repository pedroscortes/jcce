#!/usr/bin/env python
"""
Experiment C.9: Multi-Fidelity Ablation (2x2 design).

{pruning ON/OFF} x {conditional/flat search space}
Validates each outer-loop design choice independently.

Usage:
    uv run python scripts/experiment_c9_multi_fidelity_ablation.py --quick
    uv run python scripts/experiment_c9_multi_fidelity_ablation.py \
        --n-trials 20 --n-vars 10 --n-samples 500 --max-iter 100
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
from jcce.structure_learning.optuna_search import (
    PROCESSOR_TYPES,
    PROCESSOR_SEARCH_SPACE,
    FIXED_PROCESSOR_PARAMS,
    V7_EFFECT_HIDDEN_DIMS,
    V7_EFFECT_EMBED_DIMS,
    V7_LAMBDA_EFFECTS,
    V7_EFFECT_WARMUP_ITERS,
    V7_LAMBDA_CONFOUND_SPARSE,
    V7_LAMBDA_BOW,
    V7_EFFECT_REFINEMENT_ITERS,
    PruningTracker,
    suggest_hyperparams,
    create_optuna_objective,
    constraints_func,
    extract_pareto_solutions,
)
from jcce.analysis.hypervolume import compute_hypervolume_2d, extract_pareto_front_2d
from jcce.analysis.pareto_stability import compute_edge_stability
from jcce.utils.metrics import compute_structure_metrics, compute_sid


# ============================================================================
# Flat Search Space (suggests ALL processor params for every trial)
# ============================================================================

def suggest_hyperparams_flat(trial: optuna.Trial, use_v7: bool = True) -> dict:
    """Suggest hyperparams with flat (non-conditional) search space.

    Suggests ALL processor-specific params for every trial (inflated space).
    Only the selected processor's params are used in processor_config.
    """
    processor_type = trial.suggest_categorical('processor_type', PROCESSOR_TYPES)

    lambda_1 = trial.suggest_float('lambda_1', 0.005, 0.5, log=True)
    lambda_2 = trial.suggest_float('lambda_2', 0.001, 1.0, log=True)
    lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
    lambda_class = trial.suggest_categorical('lambda_class', [0.1, 0.5, 1.0, 2.0, 5.0])

    # Flat: suggest params for ALL processors (inflated space)
    all_params = {}
    for pt in PROCESSOR_TYPES:
        space = PROCESSOR_SEARCH_SPACE[pt]
        for param_name, values in space.items():
            key = f'{pt}_{param_name}'
            all_params[key] = trial.suggest_categorical(key, values)

    # But processor_config only uses the selected processor's params
    processor_config = {}
    space = PROCESSOR_SEARCH_SPACE[processor_type]
    for param_name in space:
        key = f'{processor_type}_{param_name}'
        processor_config[param_name] = all_params[key]

    if processor_type in FIXED_PROCESSOR_PARAMS:
        processor_config.update(FIXED_PROCESSOR_PARAMS[processor_type])

    config = {
        'processor_type': processor_type,
        'processor_config': processor_config,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'lambda_class': lambda_class,
        'lr': lr,
    }

    if use_v7:
        config['effect_hidden_dim'] = trial.suggest_categorical(
            'effect_hidden_dim', V7_EFFECT_HIDDEN_DIMS)
        config['effect_embed_dim'] = trial.suggest_categorical(
            'effect_embed_dim', V7_EFFECT_EMBED_DIMS)
        config['lambda_effect'] = trial.suggest_categorical(
            'lambda_effect', V7_LAMBDA_EFFECTS)
        config['effect_warmup_iter'] = trial.suggest_categorical(
            'effect_warmup_iter', V7_EFFECT_WARMUP_ITERS)
        config['lambda_confound_sparse'] = trial.suggest_categorical(
            'lambda_confound_sparse', V7_LAMBDA_CONFOUND_SPARSE)
        config['lambda_bow_v7'] = trial.suggest_categorical(
            'lambda_bow_v7', V7_LAMBDA_BOW)
        config['effect_refinement_iters'] = trial.suggest_categorical(
            'effect_refinement_iters', V7_EFFECT_REFINEMENT_ITERS)

    return config


# ============================================================================
# Data Generation (same as C.3)
# ============================================================================

def generate_classification_data(n_vars, n_samples, expected_degree, noise_scale,
                                  seed, min_y_parents=1, max_attempts=100):
    """Generate synthetic classification data from a linear SEM.

    Uses rejection sampling to ensure Y (last node) has at least
    ``min_y_parents`` parents, avoiding degenerate DGPs where Y is
    pure noise.

    Returns:
        (X, Y, A_true): Features, binary labels, and ground truth DAG
    """
    Y_idx = n_vars  # last node

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

    return jnp.array(X, dtype=jnp.float32), Y_binary, np.array(A)


# ============================================================================
# Condition Runner
# ============================================================================

def run_condition(name, X, Y, n_vars, n_trials, max_iter, seed,
                  use_pruning, use_conditional, A_true=None):
    """Run one ablation condition.

    Returns dict with HV, time, n_feasible, n_pruned, stability, SHD/F1 metrics.
    """
    print(f"\n  [{name}] pruning={'ON' if use_pruning else 'OFF'}, "
          f"space={'conditional' if use_conditional else 'flat'}")

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

    pruning_tracker = PruningTracker(
        n_startup_trials=min(3, n_trials),
        percentile=25.0,
    ) if use_pruning else None

    suggest_fn = suggest_hyperparams if use_conditional else suggest_hyperparams_flat

    objective = create_optuna_objective(
        X=X,
        Y=Y,
        n_vars=n_vars,
        max_iter=max_iter,
        use_v7=True,
        pruning_tracker=pruning_tracker,
        jax_key_seed=seed,
        verbose=False,
        suggest_fn=suggest_fn,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    # Count states
    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    n_feasible = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE
                      and t.user_attrs.get('h_A', 1.0) < 0.1])

    # Hypervolume
    enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)
    if enhanced_solutions:
        points = np.array([[s['objectives'][0], s['objectives'][1]]
                           for s in enhanced_solutions])
        pareto = extract_pareto_front_2d(points)
        hv = compute_hypervolume_2d(pareto, ref_point=np.array([0.0, 0.0]))
    else:
        hv = 0.0

    # Edge stability
    if enhanced_solutions:
        stability_matrix = compute_edge_stability(enhanced_solutions, n_vars)
        n_stable = int(np.sum(stability_matrix > 0.8))
    else:
        n_stable = 0

    # Structure metrics (SHD/F1/SID) vs ground truth
    best_shd, best_f1, best_sid = None, None, None
    if A_true is not None and enhanced_solutions:
        import jax.numpy as jnp
        A_true_sub = A_true[:n_vars, :n_vars]
        for sol in enhanced_solutions:
            A_est = sol.get('metrics', {}).get('structure_A_est')
            if A_est is not None:
                A_est = np.array(A_est)[:n_vars, :n_vars]
                sm = compute_structure_metrics(
                    jnp.array(A_est), jnp.array(A_true_sub))
                if best_f1 is None or sm['f1'] > best_f1:
                    best_f1 = sm['f1']
                    best_shd = sm['shd']
                    best_sid = compute_sid(A_est, A_true_sub)

    result = {
        'name': name,
        'hv': hv,
        'time': elapsed,
        'n_feasible': n_feasible,
        'n_completed': n_completed,
        'n_pruned': n_pruned,
        'n_stable_edges': n_stable,
        'n_pareto': len(enhanced_solutions),
        'best_f1': best_f1,
        'best_shd': best_shd,
        'best_sid': best_sid,
    }

    shd_str = f" F1={best_f1:.3f} SHD={best_shd} SID={best_sid}" if best_f1 is not None else ""
    print(f"    HV={hv:.3f} Time={elapsed:.1f}s "
          f"Feasible={n_feasible}/{n_trials} Pruned={n_pruned} "
          f"Stable(>0.8)={n_stable}{shd_str}")

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='C.9: Multi-Fidelity Ablation')
    parser.add_argument('--n-trials', type=int, default=20)
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

    print(f"MULTI-FIDELITY ABLATION (C.9)")
    print(f"=" * 70)
    print(f"Data: {args.n_samples} samples, {args.n_vars} vars, ER-{args.expected_degree}")
    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")

    # Generate shared data
    X, Y, A_true = generate_classification_data(
        args.n_vars, args.n_samples, args.expected_degree, args.noise_scale, args.seed
    )

    # Four conditions
    conditions = [
        ('no_prune + flat',        False, False),
        ('prune + flat',           True,  False),
        ('no_prune + conditional', False, True),
        ('prune + conditional',    True,  True),
    ]

    condition_results = []
    for name, use_pruning, use_conditional in conditions:
        result = run_condition(
            name, X, Y, args.n_vars, args.n_trials, args.max_iter,
            args.seed, use_pruning, use_conditional, A_true=A_true,
        )
        condition_results.append(result)

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Condition':<25} {'HV':>6} {'Time(s)':>8} {'Feasible':>9} "
          f"{'Pruned':>7} {'Stable':>7} {'F1':>6} {'SHD':>5}")
    print(f"{'-' * 25} {'-' * 6} {'-' * 8} {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 5}")

    for r in condition_results:
        f1_str = f"{r['best_f1']:.3f}" if r['best_f1'] is not None else "N/A"
        shd_str = f"{r['best_shd']}" if r['best_shd'] is not None else "N/A"
        print(f"{r['name']:<25} {r['hv']:>6.3f} {r['time']:>8.1f} "
              f"{r['n_feasible']:>4}/{args.n_trials:<4} {r['n_pruned']:>7} "
              f"{r['n_stable_edges']:>7} {f1_str:>6} {shd_str:>5}")

    # Factor analysis
    print(f"\nFactor Analysis:")

    # Pruning effect: average of (prune - no_prune) across both spaces
    hv_no_prune = np.mean([r['hv'] for r in condition_results if 'no_prune' in r['name']])
    hv_prune = np.mean([r['hv'] for r in condition_results if r['name'].startswith('prune')])
    time_no_prune = np.mean([r['time'] for r in condition_results if 'no_prune' in r['name']])
    time_prune = np.mean([r['time'] for r in condition_results if r['name'].startswith('prune')])
    delta_hv_prune = hv_prune - hv_no_prune
    delta_time_prune = (time_prune - time_no_prune) / time_no_prune * 100 if time_no_prune > 0 else 0

    # Conditional effect: average of (conditional - flat) across both pruning settings
    hv_flat = np.mean([r['hv'] for r in condition_results if 'flat' in r['name'] and 'conditional' not in r['name']])
    hv_cond = np.mean([r['hv'] for r in condition_results if 'conditional' in r['name']])
    time_flat = np.mean([r['time'] for r in condition_results if 'flat' in r['name'] and 'conditional' not in r['name']])
    time_cond = np.mean([r['time'] for r in condition_results if 'conditional' in r['name']])
    delta_hv_cond = hv_cond - hv_flat
    delta_time_cond = (time_cond - time_flat) / time_flat * 100 if time_flat > 0 else 0

    print(f"  Pruning:     deltaHV = {delta_hv_prune:+.3f}, deltaTime = {delta_time_prune:+.1f}%")
    print(f"  Conditional: deltaHV = {delta_hv_cond:+.3f}, deltaTime = {delta_time_cond:+.1f}%")

    # Save results
    if args.output:
        output = {
            'args': vars(args),
            'conditions': condition_results,
            'factor_analysis': {
                'pruning': {
                    'delta_hv': delta_hv_prune,
                    'delta_time_pct': delta_time_prune,
                },
                'conditional': {
                    'delta_hv': delta_hv_cond,
                    'delta_time_pct': delta_time_cond,
                },
            },
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
