#!/usr/bin/env python
"""
Experiment C.8: Dataset Shift Robustness.

Train on distribution A (e.g., low noise), evaluate on distribution B
(same DAG, different noise/SEM parameters). Tests whether Pareto-optimal
hyperparameters transfer and whether stable edges remain stable under
distribution shift.

Usage:
    uv run python scripts/experiment_c8_dataset_shift.py --quick
    uv run python scripts/experiment_c8_dataset_shift.py \
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
from jcce.structure_learning.jcce_learner import (
    create_processor,
    learn_structure,
)
from jcce.structure_learning.optuna_search import (
    suggest_hyperparams,
    create_optuna_objective,
    extract_pareto_solutions,
    constraints_func,
    get_trial_artifacts,
)
from jcce.analysis.hypervolume import compute_hypervolume_2d, extract_pareto_front_2d
from jcce.analysis.pareto_stability import (
    compute_edge_stability,
    compare_stability_to_ground_truth,
)


# ============================================================================
# Distribution Generation
# ============================================================================

def generate_distribution(A, n_samples, noise_scale, noise_type, seed):
    """Generate (X, Y) from given DAG with specified noise parameters."""
    scm_config = SCMConfig(noise_scale=noise_scale, noise_type=noise_type)
    scm = LinearSCM(A, scm_config)
    key = random.PRNGKey(seed)
    X_full = scm.sample(n_samples, key)

    Y_continuous = X_full[:, -1]
    Y_binary = (Y_continuous > jnp.median(Y_continuous)).astype(jnp.float32)
    X = X_full[:, :-1]

    return jnp.array(X, dtype=jnp.float32), Y_binary


# ============================================================================
# Evaluate Pareto solutions on shifted data
# ============================================================================

def evaluate_solutions_on_data(enhanced_solutions, X, Y, n_vars, A_true, max_iter, seed):
    """Re-evaluate Pareto configs on new data.

    For each Pareto solution, re-train GOLEM with the SAME hyperparameters
    but on the new (shifted) data. Returns per-solution metrics.
    """
    results = []

    for idx, sol in enumerate(enhanced_solutions):
        config = {
            'processor_type': sol['metrics']['processor_type'],
            'processor_config': sol['metrics']['processor_config'],
            'lambda_1': sol['metrics']['lambda_1'],
            'lambda_2': sol['metrics']['lambda_2'],
            'lambda_class': sol['metrics']['lambda_class'],
            'lr': sol['metrics']['lr'],
        }

        # v7 params
        for k in ['effect_hidden_dim', 'effect_embed_dim', 'lambda_effect',
                   'effect_warmup_iter', 'lambda_confound_sparse', 'lambda_bow']:
            if k in sol['metrics']:
                config[k] = sol['metrics'][k]

        key = random.PRNGKey(seed + idx * 100)
        key, proc_key, train_key = random.split(key, 3)

        processor = create_processor(
            config['processor_type'],
            key=proc_key,
            n_features=n_vars,
            **config['processor_config'],
        )

        Y_for_v7 = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        try:
            A_est, _, _, metrics = learn_structure(
                data=X, Y=Y_for_v7, Y_idx=n_vars,
                processor=processor, key=train_key,
                processor_type=config['processor_type'],
                lambda_1=config['lambda_1'],
                lambda_2_init=config['lambda_2'],
                lambda_class=config['lambda_class'],
                lr=config['lr'],
                max_iter=max_iter,
                patience=25,
                verbose=0,
                task='classification',
                use_adaptive_curriculum=True,
                effect_hidden_dim=config.get('effect_hidden_dim', 64),
                effect_embed_dim=config.get('effect_embed_dim', 16),
                lambda_effect=config.get('lambda_effect', 10.0),
                effect_warmup_iter=config.get('effect_warmup_iter', 20),
                lambda_confound_sparse=config.get('lambda_confound_sparse', 0.05),
                lambda_bow=config.get('lambda_bow', 0.3),
                effect_refinement_iters=config.get('effect_refinement_iters', 50),
            )

            bacc = float(metrics.get('balanced_accuracy', 0.0))
            h_A = float(metrics.get('final_h_A', 1.0))

            A_est_X = np.array(A_est)[:n_vars, :n_vars]
            sm = compute_structure_metrics(
                jnp.array(A_est_X), jnp.array(A_true)
            )

            results.append({
                'processor_type': config['processor_type'],
                'bacc': bacc,
                'h_A': h_A,
                'shd': sm['shd'],
                'f1': sm['f1'],
                'A_est': np.array(A_est),
                'feasible': h_A < 0.1,
            })
        except Exception as e:
            results.append({
                'processor_type': config['processor_type'],
                'bacc': 0.0, 'h_A': 1.0, 'shd': -1, 'f1': 0.0,
                'A_est': None, 'feasible': False, 'error': str(e),
            })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='C.8: Dataset Shift Robustness')
    parser.add_argument('--n-trials', type=int, default=15)
    parser.add_argument('--n-vars', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--expected-degree', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5 trials, 20 iters, 5 vars')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.max_iter = 20
        args.n_vars = 5

    # Define distribution pairs (train_noise, test_noise)
    shift_conditions = [
        {'name': 'same (control)',
         'train_noise': 0.5, 'train_type': 'gaussian',
         'test_noise': 0.5, 'test_type': 'gaussian'},
        {'name': 'noise increase',
         'train_noise': 0.3, 'train_type': 'gaussian',
         'test_noise': 1.0, 'test_type': 'gaussian'},
        {'name': 'noise decrease',
         'train_noise': 1.0, 'train_type': 'gaussian',
         'test_noise': 0.3, 'test_type': 'gaussian'},
        {'name': 'noise type shift',
         'train_noise': 0.5, 'train_type': 'gaussian',
         'test_noise': 0.5, 'test_type': 'laplace'},
    ]

    if args.quick:
        shift_conditions = shift_conditions[:2]  # Just control + one shift

    print(f"DATASET SHIFT ROBUSTNESS (C.8)")
    print(f"=" * 70)
    print(f"Data: {args.n_samples} samples, {args.n_vars} vars, "
          f"ER-{args.expected_degree}")
    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")
    print(f"Conditions: {len(shift_conditions)}")

    # Generate ONE shared DAG
    dag_config = DAGConfig(
        num_nodes=args.n_vars + 1,
        graph_type='erdos_renyi',
        expected_degree=args.expected_degree,
        seed=args.seed,
    )
    A = generate_dag(dag_config)
    A_true = np.array(A)[:args.n_vars, :args.n_vars]
    n_edges = int(np.sum(np.abs(A_true) > 1e-6))
    print(f"Ground truth DAG: {n_edges} edges (shared across all conditions)")

    all_results = []

    for cond in shift_conditions:
        print(f"\n{'=' * 70}")
        print(f"Condition: {cond['name']}")
        print(f"  Train: noise={cond['train_noise']}, type={cond['train_type']}")
        print(f"  Test:  noise={cond['test_noise']}, type={cond['test_type']}")

        # Generate train + test data
        X_train, Y_train = generate_distribution(
            A, args.n_samples, cond['train_noise'], cond['train_type'],
            seed=args.seed,
        )
        X_test, Y_test = generate_distribution(
            A, args.n_samples, cond['test_noise'], cond['test_type'],
            seed=args.seed + 5000,
        )

        # Train: Run Optuna search on train data
        print(f"  Training {args.n_trials} trials on train distribution...")
        sampler = TPESampler(
            multivariate=True, group=True,
            seed=args.seed,
            n_startup_trials=min(5, args.n_trials),
            constraints_func=constraints_func,
            constant_liar=True,
        )
        study = optuna.create_study(
            directions=['maximize', 'maximize'],
            sampler=sampler,
        )
        objective = create_optuna_objective(
            X=X_train, Y=Y_train, n_vars=args.n_vars,
            max_iter=args.max_iter, use_v7=True,
            jax_key_seed=args.seed, verbose=False,
        )

        t0 = time.time()
        study.optimize(objective, n_trials=args.n_trials)
        train_time = time.time() - t0

        enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)
        n_pareto = len(enhanced_solutions)

        # Train metrics
        train_baccs = [s['objectives'][0] for s in enhanced_solutions]
        mean_train_bacc = float(np.mean(train_baccs)) if train_baccs else 0.0

        print(f"  Train: {n_pareto} Pareto solutions, "
              f"mean BAcc={mean_train_bacc:.3f} ({train_time:.1f}s)")

        # Edge stability on train
        n_total = args.n_vars + 1
        A_true_full = np.zeros((n_total, n_total))
        A_true_full[:args.n_vars, :args.n_vars] = A_true

        if enhanced_solutions:
            train_stab = compute_edge_stability(enhanced_solutions, args.n_vars,
                                                 threshold=0.1)
            train_gt = compare_stability_to_ground_truth(
                train_stab, A_true_full, min_frequency=0.5)
        else:
            train_gt = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        # Test: Re-evaluate Pareto configs on test data
        if enhanced_solutions:
            print(f"  Evaluating {n_pareto} Pareto configs on test distribution...")
            t0 = time.time()
            test_results = evaluate_solutions_on_data(
                enhanced_solutions, X_test, Y_test, args.n_vars, A_true,
                args.max_iter, seed=args.seed + 10000,
            )
            test_time = time.time() - t0

            test_baccs = [r['bacc'] for r in test_results if r['feasible']]
            test_f1s = [r['f1'] for r in test_results if r['feasible']]
            test_feasible = sum(1 for r in test_results if r['feasible'])

            mean_test_bacc = float(np.mean(test_baccs)) if test_baccs else 0.0
            mean_test_f1 = float(np.mean(test_f1s)) if test_f1s else 0.0

            # Edge stability on test solutions
            test_solutions_for_stab = []
            for r in test_results:
                if r['A_est'] is not None and r['feasible']:
                    test_solutions_for_stab.append({
                        'metrics': {'structure_A_est': r['A_est']},
                    })
            if test_solutions_for_stab:
                test_stab = compute_edge_stability(
                    test_solutions_for_stab, args.n_vars, threshold=0.1)
                test_gt = compare_stability_to_ground_truth(
                    test_stab, A_true_full, min_frequency=0.5)
            else:
                test_gt = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

            print(f"  Test:  {test_feasible}/{n_pareto} feasible, "
                  f"mean BAcc={mean_test_bacc:.3f} ({test_time:.1f}s)")
        else:
            mean_test_bacc = 0.0
            mean_test_f1 = 0.0
            test_feasible = 0
            test_gt = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        bacc_drop = mean_train_bacc - mean_test_bacc
        stab_drop = train_gt['f1'] - test_gt['f1']

        result = {
            'condition': cond['name'],
            'train_noise': cond['train_noise'],
            'train_type': cond['train_type'],
            'test_noise': cond['test_noise'],
            'test_type': cond['test_type'],
            'n_pareto': n_pareto,
            'train_bacc': mean_train_bacc,
            'test_bacc': mean_test_bacc,
            'bacc_drop': bacc_drop,
            'train_stab_f1': train_gt['f1'],
            'test_stab_f1': test_gt['f1'],
            'stab_f1_drop': stab_drop,
            'test_feasible': test_feasible,
        }
        all_results.append(result)

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY\n")
    print(f"{'Condition':<20} {'TrainBAcc':>10} {'TestBAcc':>9} {'Drop':>7} "
          f"{'TrainStab':>10} {'TestStab':>9} {'Drop':>7}")
    print(f"{'-' * 20} {'-' * 10} {'-' * 9} {'-' * 7} "
          f"{'-' * 10} {'-' * 9} {'-' * 7}")

    for r in all_results:
        print(f"{r['condition']:<20} {r['train_bacc']:>10.3f} "
              f"{r['test_bacc']:>9.3f} {r['bacc_drop']:>+7.3f} "
              f"{r['train_stab_f1']:>10.3f} {r['test_stab_f1']:>9.3f} "
              f"{r['stab_f1_drop']:>+7.3f}")

    # Interpretation
    print(f"\nInterpretation:")
    control = next((r for r in all_results if 'control' in r['condition']), None)
    for r in all_results:
        if 'control' in r['condition']:
            continue
        if control and r['bacc_drop'] > control['bacc_drop'] + 0.05:
            print(f"  {r['condition']}: BAcc degrades significantly "
                  f"(drop={r['bacc_drop']:+.3f} vs control={control['bacc_drop']:+.3f})")
        else:
            print(f"  {r['condition']}: BAcc robust "
                  f"(drop={r['bacc_drop']:+.3f})")

    # Save results
    if args.output:
        output = {'args': vars(args), 'n_edges': n_edges, 'results': all_results}
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
