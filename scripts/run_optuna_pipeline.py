#!/usr/bin/env python
"""
CLI entry point for Optuna-based JCCE optimization.

Usage:
    # Quick local test (LUCAS, 20 trials)
    uv run python scripts/run_optuna_pipeline.py --dataset lucas --n-trials 20 --max-iter 30

    # Full run with CV
    uv run python scripts/run_optuna_pipeline.py --dataset lucas --n-trials 100 --max-iter 300 --run-cv

    # With PC warm-start
    uv run python scripts/run_optuna_pipeline.py --dataset lucas --n-trials 100 --use-pc-warmstart
"""

import argparse
import pickle
import time
import sys
import os
import json
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_dataset(name: str, task: str = 'classification'):
    """Load a dataset by name. Returns (X, Y, true_mb, true_graph, n_vars, feature_names)."""
    if name == 'synthetic':
        rng = np.random.RandomState(42)
        n_samples, n_vars = 200, 5
        X = rng.randn(n_samples, n_vars).astype(np.float32)
        Y = (rng.randn(n_samples) > 0).astype(np.float32)
        return X, Y, None, None, n_vars, [f'X{i}' for i in range(n_vars)]

    else:
        from jcce.data.benchmark_loader import load_dataset as bm_load
        X, Y, config = bm_load(name)
        true_mb = config.get('true_mb')
        true_dag = config.get('true_dag')
        n_vars = X.shape[1]
        feature_names = config.get('feature_names', [f'X{i}' for i in range(n_vars)])
        return X, Y, true_mb, true_dag, n_vars, feature_names


RESULTS_DIR = Path('results/optuna_v16')


def _convert_optuna_to_ablation_format(result, dataset_name, args):
    """
    Convert Optuna pipeline result dict to the same structure as
    run_ablation_study.py's run_ablation() output, so analysis code
    can load both interchangeably.
    """
    enhanced_sols = result.get('enhanced_solutions', [])

    # Build pareto_solutions list (same schema as ablation)
    pareto_solutions_data = []
    raw_sol_metrics = []
    for sol_data in enhanced_sols:
        m = sol_data.get('metrics', {})
        if m is None or 'structure_A_est' not in m:
            continue
        raw_sol_metrics.append(m)
        sol_entry = {
            'A_est': np.array(m['structure_A_est']),
            'A_weights': np.array(m['A_weights']) if m.get('A_weights') is not None else None,
            'A_confound': np.array(m['A_confound']) if m.get('A_confound') is not None else None,
            'A_confound_weights': np.array(m['A_confound_weights']) if m.get('A_confound_weights') is not None else None,
            'processor_type': m.get('processor_type', '?'),
            'processor_config': m.get('processor_config', {}),
            'mb_indices': m.get('mb_indices', []),
            'markov_blanket': m.get('markov_blanket', []),
            'causal_effects': m.get('causal_effects', {}),
            'n_confound_edges': m.get('n_confound_edges', 0),
            'n_edges': m.get('structure_n_edges', 0),
            'h_A': m.get('structure_h_A', 0),
            'accuracy': m.get('classification_accuracy', 0),
            'balanced_accuracy': m.get('classification_balanced_accuracy', 0),
            'f1': m.get('classification_f1', 0),
            'precision': m.get('classification_precision', 0),
            'recall': m.get('classification_recall', 0),
            'roc_auc': m.get('classification_roc_auc', 0),
            # Structure recovery metrics (available when ground truth exists)
            'structure_shd': m.get('structure_shd'),
            'structure_sid': m.get('structure_sid'),
            'structure_edge_f1': m.get('structure_edge_f1'),
            'structure_edge_precision': m.get('structure_edge_precision'),
            'structure_edge_recall': m.get('structure_edge_recall'),
            'hyperparams': {
                'lambda_1': m.get('lambda_1'),
                'lambda_2': m.get('lambda_2'),
                'lambda_class': m.get('lambda_class'),
                'lr': m.get('lr'),
            },
        }
        # Processor params (needed for CF rerun)
        if '_processor_params' in m:
            from jcce.validation.counterfactual_runner import convert_params_to_numpy
            sol_entry['processor_params'] = convert_params_to_numpy(m['_processor_params'])
        # Diagnostics (gradient cosines, bow-free, residual normality)
        for diag_key in ['gradient_diagnostics', 'bow_free_violations',
                         'residuals_non_gaussian', 'residual_normality_pvals']:
            if diag_key in m:
                sol_entry[diag_key] = m[diag_key]
        pareto_solutions_data.append(sol_entry)

    # Find best solution by balanced accuracy
    best_pareto_idx = -1
    best_bacc = -1
    for i, m in enumerate(raw_sol_metrics):
        bacc = m.get('classification_balanced_accuracy', 0.0)
        if bacc > best_bacc:
            best_bacc = bacc
            best_pareto_idx = i
    best_sol_metrics = raw_sol_metrics[best_pareto_idx] if best_pareto_idx >= 0 else None

    # Attach CV results per-solution (Optuna stores them in a separate list)
    cv_results = result.get('cv_results', [])
    for cv in cv_results:
        idx = cv.get('solution_idx')
        if idx is not None and idx < len(pareto_solutions_data):
            pareto_solutions_data[idx]['cv'] = {
                'accuracy_mean': cv.get('cv_accuracy_mean', 0),
                'accuracy_std': cv.get('cv_accuracy_std', 0),
                'precision_mean': cv.get('cv_precision_mean', 0),
                'precision_std': cv.get('cv_precision_std', 0),
                'recall_mean': cv.get('cv_recall_mean', 0),
                'recall_std': cv.get('cv_recall_std', 0),
                'f1_mean': cv.get('cv_f1_mean', 0),
                'f1_std': cv.get('cv_f1_std', 0),
                'balanced_acc_mean': cv.get('cv_bacc_mean', 0),
                'balanced_acc_std': cv.get('cv_bacc_std', 0),
                'roc_auc_mean': cv.get('cv_roc_auc_mean', 0),
                'roc_auc_std': cv.get('cv_roc_auc_std', 0),
                'mb_jaccard_mean': cv.get('cv_mb_jaccard', 0),
            }
            if cv.get('cv_mb_f1_mean') is not None:
                pareto_solutions_data[idx]['cv']['mb_f1_mean'] = cv['cv_mb_f1_mean']

    # Re-select best solution by CV BAcc (not training BAcc) when CV available
    cv_best_idx = -1
    cv_best_bacc = -1
    for i, ps in enumerate(pareto_solutions_data):
        cv_data = ps.get('cv', {})
        cv_bacc = cv_data.get('balanced_acc_mean', -1)
        if cv_bacc > cv_best_bacc:
            cv_best_bacc = cv_bacc
            cv_best_idx = i
    if cv_best_idx >= 0 and cv_best_bacc > 0:
        best_pareto_idx = cv_best_idx
        best_sol_metrics = raw_sol_metrics[best_pareto_idx]

    # Attach DML results per-solution
    dml_results = result.get('dml_results', [])
    for dml in dml_results:
        idx = dml.get('solution_idx')
        if idx is not None and idx < len(pareto_solutions_data):
            pareto_solutions_data[idx]['dml'] = dml.get('dml_result', {})
            pareto_solutions_data[idx]['dml_n_parents'] = dml.get('n_parents', 0)
            pareto_solutions_data[idx]['dml_n_significant'] = dml.get('n_significant', 0)

    # Attach all-edges DML results per-solution
    all_edges_dml_results = result.get('all_edges_dml_results', [])
    for ae in all_edges_dml_results:
        idx = ae.get('solution_idx')
        if idx is not None and idx < len(pareto_solutions_data):
            pareto_solutions_data[idx]['all_edges_dml'] = ae.get('all_edges_dml', {})
            pareto_solutions_data[idx]['dml_causal_effects'] = ae.get('dml_causal_effects', {})

    # Attach CF results per-solution
    cf_results = result.get('cf_results', [])
    for cf in cf_results:
        idx = cf.get('solution_idx')
        if idx is not None and idx < len(pareto_solutions_data):
            pareto_solutions_data[idx]['cf'] = cf.get('cf', {})

    # Build top-level result dict (same schema as run_ablation)
    out = {
        'ablation_id': 'optuna',
        'ablation_name': 'Optuna TPE',
        'ablation_description': f'Optuna TPE optimization ({args.n_trials} trials, {args.max_iter} iter)',
        'dataset': dataset_name,
        'dataset_config': {},  # Filled below
        'is_pipeline_baseline': False,
        'quick_mode': False,
        'optuna_time': result.get('pipeline_time', 0),
        'nsga2_time': 0,  # Not applicable, but keeps schema consistent
        'total_time': result.get('pipeline_time', 0),
        'n_pareto_solutions': len(pareto_solutions_data),
        'pareto_front': result.get('pareto_front', []),
        'golem_overrides': _parse_golem_overrides(args.golem_override),

        # Optuna-specific metadata
        'n_trials_completed': result.get('n_trials_completed', 0),
        'n_trials_pruned': result.get('n_trials_pruned', 0),
        'n_trials_failed': result.get('n_trials_failed', 0),
        'seed': args.seed,

        # Best solution training metrics
        'best_training_balanced_acc': best_bacc,
        'best_pareto_idx': best_pareto_idx,
        'best_processor': best_sol_metrics.get('processor_type', '?') if best_sol_metrics else '?',
        'best_mb_size': best_sol_metrics.get('mb_size', 0) if best_sol_metrics else 0,
        'best_n_edges': best_sol_metrics.get('structure_n_edges', 0) if best_sol_metrics else 0,
        'best_h_A': best_sol_metrics.get('structure_h_A', 0) if best_sol_metrics else 0,
        'diverse_indices': list(range(min(5, len(pareto_solutions_data)))),

        # Structural validation
        'structural_validation': result.get('structural_validation', {}),
    }

    # Best solution model (backwards compat)
    if best_sol_metrics:
        model_dict = {
            'A_est': np.array(best_sol_metrics['structure_A_est']),
            'A_weights': np.array(best_sol_metrics['A_weights']) if best_sol_metrics.get('A_weights') is not None else None,
            'A_confound': np.array(best_sol_metrics['A_confound']) if best_sol_metrics.get('A_confound') is not None else None,
            'A_confound_weights': np.array(best_sol_metrics['A_confound_weights']) if best_sol_metrics.get('A_confound_weights') is not None else None,
            'processor_type': best_sol_metrics.get('processor_type', 'mlp'),
            'processor_config': best_sol_metrics.get('processor_config', {}),
            'mb_indices': best_sol_metrics.get('mb_indices', []),
            'markov_blanket': best_sol_metrics.get('markov_blanket', []),
            'causal_effects': best_sol_metrics.get('causal_effects', {}),
            'n_confound_edges': best_sol_metrics.get('n_confound_edges', 0),
            'hyperparams': {
                'lambda_1': best_sol_metrics.get('lambda_1'),
                'lambda_2': best_sol_metrics.get('lambda_2'),
                'lambda_class': best_sol_metrics.get('lambda_class'),
                'lr': best_sol_metrics.get('lr'),
            },
        }
        if '_processor_params' in best_sol_metrics:
            from jcce.validation.counterfactual_runner import convert_params_to_numpy
            model_dict['processor_params'] = convert_params_to_numpy(best_sol_metrics['_processor_params'])
        out['model'] = model_dict

    # Store all Pareto solutions (with per-solution DML/CV attached)
    out['pareto_solutions'] = pareto_solutions_data

    # Backwards-compatible top-level CV/DML keys from best solution
    if best_pareto_idx >= 0 and best_pareto_idx < len(pareto_solutions_data):
        best_ps = pareto_solutions_data[best_pareto_idx]

        if 'cv' in best_ps:
            cv = best_ps['cv']
            out['cv_accuracy_mean'] = cv['accuracy_mean']
            out['cv_accuracy_std'] = cv['accuracy_std']
            out['cv_f1_mean'] = cv['f1_mean']
            out['cv_f1_std'] = cv['f1_std']
            out['cv_balanced_acc_mean'] = cv['balanced_acc_mean']
            out['cv_balanced_acc_std'] = cv['balanced_acc_std']
            out['cv_roc_auc_mean'] = cv['roc_auc_mean']
            out['cv_roc_auc_std'] = cv['roc_auc_std']
            out['cv_mb_jaccard'] = cv.get('mb_jaccard_mean')
            if 'mb_f1_mean' in cv:
                out['cv_mb_f1_mean'] = cv['mb_f1_mean']

        if 'dml' in best_ps:
            out['dml_n_parents'] = best_ps.get('dml_n_parents', 0)
            out['dml_n_significant'] = best_ps.get('dml_n_significant', 0)
            out['dml_parent_effects'] = best_ps['dml']

        if 'cf' in best_ps:
            cf = best_ps['cf']
            out['cf_n_instances'] = cf.get('n_instances', 0)
            out['cf_n_valid'] = cf.get('n_valid', 0)
            out['cf_validity_rate'] = cf.get('validity_rate', 0)
            out['cf_avg_sparsity'] = cf.get('avg_sparsity', 0)
            out['cf_avg_distance'] = cf.get('avg_distance', 0)
            out['cf_avg_plausibility_ratio'] = cf.get('avg_plausibility_ratio', 0)
            out['cf_n_plausible'] = cf.get('n_plausible', 0)
            out['cf_avg_causal_validity'] = cf.get('avg_causal_validity', 0)
            out['cf_details'] = cf

    return out


def _run_rerun_modes(args):
    """Handle --rerun-* flags: load existing pkl, rerun specific post-hoc phases, update."""
    from sklearn.model_selection import train_test_split

    result_path = RESULTS_DIR / args.dataset / f'optuna_seed{args.seed}.pkl'
    if not result_path.exists():
        print(f"No result file: {result_path}")
        return

    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    if 'pareto_solutions' not in result:
        print(f"No pareto_solutions in result (old format?)")
        return

    # Load full dataset
    X_full, Y_full, true_mb, true_graph, n_vars, feature_names = load_dataset(
        args.dataset, args.task
    )

    # Recreate the same 70/30 split used during training
    X_struct, X_effect, Y_struct, Y_effect = train_test_split(
        X_full, Y_full, test_size=0.3, stratify=Y_full, random_state=42
    )

    best_idx = result.get('best_pareto_idx', 0)
    n_solutions = len(result['pareto_solutions'])
    if args.sol == 'all':
        sol_indices = list(range(n_solutions))
    elif args.sol is not None:
        sol_indices = args.sol
    else:
        sol_indices = [best_idx]

    # Build ds_config for CF
    ds_config = {'feature_names': feature_names}
    if args.dataset != 'synthetic':
        try:
            from jcce.data.benchmark_loader import load_dataset as bm_load
            _, _, _cfg = bm_load(args.dataset)
            ds_config.update({k: v for k, v in _cfg.items() if k != 'true_dag'})
        except Exception:
            pass

    # --rerun-dml
    if getattr(args, 'rerun_dml', False):
        from jcce.validation.multi_parent_dml import run_multi_parent_dml
        print(f"\n  Re-running DML: {args.dataset}, solutions: {sol_indices}")

        for sol_idx in sol_indices:
            if sol_idx >= n_solutions:
                print(f"  Sol {sol_idx}: index out of range (max {n_solutions-1})")
                continue

            ps = result['pareto_solutions'][sol_idx]
            A_est = np.array(ps['A_est'])
            Y_idx = A_est.shape[0] - 1

            try:
                dml_result = run_multi_parent_dml(
                    X=X_effect, Y=Y_effect,
                    A_est=A_est, Y_idx=Y_idx,
                    feature_names=feature_names,
                    parent_threshold=0.01,
                    n_dml_folds=5 if X_effect.shape[0] >= 500 else 3,
                    run_refutation=True,
                    verbose=True,
                )
                result['pareto_solutions'][sol_idx]['dml'] = dml_result.to_dict()
                result['pareto_solutions'][sol_idx]['dml_n_parents'] = dml_result.n_parents_discovered
                result['pareto_solutions'][sol_idx]['dml_n_significant'] = dml_result.n_parents_significant

                if sol_idx == best_idx:
                    result['dml_parent_effects'] = dml_result.to_dict().get('parent_effects', {})

                sig = dml_result.n_parents_significant
                total = dml_result.n_parents_discovered
                print(f"    Sol {sol_idx}: DML {sig}/{total} significant")
            except Exception as e:
                print(f"    Sol {sol_idx}: DML failed: {e}")
                import traceback
                traceback.print_exc()

        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Updated results: {result_path}")

    # --rerun-all-edges-dml
    if getattr(args, 'rerun_all_edges_dml', False):
        from jcce.validation.all_edges_dml import run_all_edges_dml
        print(f"\n  Re-running all-edges DML: {args.dataset}, solutions: {sol_indices}")

        for sol_idx in sol_indices:
            if sol_idx >= n_solutions:
                print(f"  Sol {sol_idx}: index out of range (max {n_solutions-1})")
                continue

            ps = result['pareto_solutions'][sol_idx]
            A_est = np.array(ps['A_est'])
            Y_idx = A_est.shape[0] - 1

            try:
                ae_result = run_all_edges_dml(
                    X=X_effect, Y=Y_effect,
                    A_est=A_est, Y_idx=Y_idx,
                    feature_names=feature_names,
                    edge_threshold=0.01,
                    n_dml_folds=5 if X_effect.shape[0] >= 500 else 3,
                    run_refutation=True,
                    verbose=True,
                )
                if ae_result is not None:
                    result['pareto_solutions'][sol_idx]['all_edges_dml'] = ae_result.to_storage_dict()
                    result['pareto_solutions'][sol_idx]['dml_causal_effects'] = ae_result.to_causal_effects_dict()
                    sig = ae_result.n_significant_fdr
                    total = ae_result.n_edges
                    print(f"    Sol {sol_idx}: All-edges DML {sig}/{total} significant (FDR)")
            except Exception as e:
                print(f"    Sol {sol_idx}: All-edges DML failed: {e}")
                import traceback
                traceback.print_exc()

        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Updated results: {result_path}")

    # --rerun-cv
    if getattr(args, 'rerun_cv', False):
        from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv
        n_folds = 3 if X_effect.shape[0] < 500 else 5
        print(f"\n  Re-running CV: {args.dataset}, solutions: {sol_indices}")

        for sol_idx in sol_indices:
            if sol_idx >= n_solutions:
                print(f"  Sol {sol_idx}: index out of range (max {n_solutions-1})")
                continue

            ps = result['pareto_solutions'][sol_idx]
            hyperparams = dict(ps.get('hyperparams', {}))
            hyperparams['processor_config'] = ps.get('processor_config', {})
            hyperparams.setdefault('lambda_1', 0.02)
            hyperparams.setdefault('lambda_2', 0.01)
            hyperparams.setdefault('lambda_class', 1.0)
            hyperparams.setdefault('lr', 0.001)

            A_init = ps.get('A_est')
            if A_init is None:
                print(f"  Sol {sol_idx}: no A_est saved")
                continue

            try:
                cv_result = evaluate_pareto_solution_cv(
                    X=X_effect, Y=Y_effect,
                    hyperparams=hyperparams,
                    processor_type=ps.get('processor_type', 'elm'),
                    A_init=np.array(A_init),
                    n_folds=n_folds,
                    true_mb=true_mb,
                    golem_max_iter=args.max_iter // 3,
                    cold_start=True,
                    task=args.task,
                    verbose=True,
                    use_v7=True,
                )
                cv_dict = {
                    'accuracy_mean': cv_result.accuracy_mean,
                    'accuracy_std': cv_result.accuracy_std,
                    'precision_mean': cv_result.precision_mean,
                    'precision_std': cv_result.precision_std,
                    'recall_mean': cv_result.recall_mean,
                    'recall_std': cv_result.recall_std,
                    'f1_mean': cv_result.f1_mean,
                    'f1_std': cv_result.f1_std,
                    'balanced_acc_mean': cv_result.balanced_acc_mean,
                    'balanced_acc_std': cv_result.balanced_acc_std,
                    'roc_auc_mean': cv_result.roc_auc_mean,
                    'roc_auc_std': cv_result.roc_auc_std,
                    'mb_jaccard_mean': cv_result.mb_jaccard_mean,
                }
                if cv_result.mb_f1_per_fold:
                    cv_dict['mb_f1_mean'] = float(np.mean(cv_result.mb_f1_per_fold))

                result['pareto_solutions'][sol_idx]['cv'] = cv_dict

                # Update top-level keys if best solution
                if sol_idx == best_idx:
                    result['cv_accuracy_mean'] = cv_dict['accuracy_mean']
                    result['cv_accuracy_std'] = cv_dict['accuracy_std']
                    result['cv_f1_mean'] = cv_dict['f1_mean']
                    result['cv_f1_std'] = cv_dict['f1_std']
                    result['cv_balanced_acc_mean'] = cv_dict['balanced_acc_mean']
                    result['cv_balanced_acc_std'] = cv_dict['balanced_acc_std']
                    result['cv_roc_auc_mean'] = cv_dict['roc_auc_mean']
                    result['cv_roc_auc_std'] = cv_dict['roc_auc_std']
                    result['cv_mb_jaccard'] = cv_dict.get('mb_jaccard_mean')
                    if 'mb_f1_mean' in cv_dict:
                        result['cv_mb_f1_mean'] = cv_dict['mb_f1_mean']

                print(f"    Sol {sol_idx}: CV BAcc={cv_result.balanced_acc_mean:.4f}"
                      f"+-{cv_result.balanced_acc_std:.4f}")
            except Exception as e:
                print(f"    Sol {sol_idx}: CV failed: {e}")
                import traceback
                traceback.print_exc()

        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Updated results: {result_path}")

    # --rerun-cf
    if getattr(args, 'rerun_cf', False):
        from jcce.validation.counterfactual_runner import (
            run_counterfactual_evaluation,
            reconstruct_processor,
        )
        print(f"\n  Re-running CF: {args.dataset}, solutions: {sol_indices}")

        for sol_idx in sol_indices:
            if sol_idx >= n_solutions:
                print(f"  Sol {sol_idx}: index out of range (max {n_solutions-1})")
                continue

            ps = result['pareto_solutions'][sol_idx]
            if 'processor_params' not in ps and '_processor_params' not in ps:
                print(f"  Sol {sol_idx}: no processor_params saved, cannot rerun CF")
                continue

            proc_params = ps.get('processor_params') or ps.get('_processor_params')
            proc_type = ps.get('processor_type', 'elm')
            proc_config = ps.get('processor_config', {})

            try:
                processor, params_jax = reconstruct_processor(
                    proc_type, proc_config, proc_params,
                )

                cf_result = run_counterfactual_evaluation(
                    X=np.array(X_struct), Y=np.array(Y_struct),
                    A_est=np.array(ps['A_est']),
                    processor_trained=processor,
                    processor_params=params_jax,
                    ds_config=ds_config,
                    A_weights=np.array(ps['A_weights']) if ps.get('A_weights') is not None else None,
                    A_confound=np.array(ps['A_confound_weights']) if ps.get('A_confound_weights') is not None else None,
                    n_instances=30,
                    cf_pop_size=50,
                    cf_n_gen=50,
                    verbose=True,
                )

                if cf_result:
                    result['pareto_solutions'][sol_idx]['cf'] = cf_result

                    # Update top-level keys if best solution
                    if sol_idx == best_idx:
                        result['cf_n_instances'] = cf_result['n_instances']
                        result['cf_n_valid'] = cf_result['n_valid']
                        result['cf_validity_rate'] = cf_result['validity_rate']
                        result['cf_avg_sparsity'] = cf_result['avg_sparsity']
                        result['cf_avg_distance'] = cf_result['avg_distance']
                        result['cf_avg_plausibility_ratio'] = cf_result['avg_plausibility_ratio']
                        result['cf_n_plausible'] = cf_result['n_plausible']
                        result['cf_avg_causal_validity'] = cf_result['avg_causal_validity']
                        result['cf_details'] = cf_result

                    print(f"    Sol {sol_idx}: CF {cf_result['n_valid']}/{cf_result['n_instances']} valid")
            except Exception as e:
                print(f"    Sol {sol_idx}: CF failed: {e}")
                import traceback
                traceback.print_exc()

        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Updated results: {result_path}")


def _parse_golem_overrides(override_list):
    """Parse --golem-override key=value pairs into a dict."""
    overrides = {}
    for item in (override_list or []):
        if '=' not in item:
            continue
        key, val = item.split('=', 1)
        # Parse booleans and numbers
        if val.lower() == 'true':
            val = True
        elif val.lower() == 'false':
            val = False
        else:
            try:
                val = float(val)
                if val == int(val):
                    val = int(val)
            except ValueError:
                pass
        overrides[key] = val
    return overrides if overrides else None


def main():
    parser = argparse.ArgumentParser(description='JCCE Optuna Optimization')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['lucas', 'heart_disease', 'breast_cancer',
                                 'diabetes', 'sachs', 'asia', 'child',
                                 'neuropathic_pain', 'alarm', 'insurance',
                                 'synthetic'],
                        help='Dataset to optimize on')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of Optuna trials')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max GOLEM iterations per trial')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'])
    parser.add_argument('--use-pc-warmstart', action='store_true',
                        help='Run PC algorithm for warm-start')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='CV folds')
    parser.add_argument('--cv-max-solutions', type=int, default=0,
                        help='Max Pareto solutions for CV (0=all)')
    parser.add_argument('--dml-max-solutions', type=int, default=0,
                        help='Max Pareto solutions for DML (0=all)')
    parser.add_argument('--no-cv', action='store_true',
                        help='Skip post-hoc CV (on by default)')
    parser.add_argument('--no-dml', action='store_true',
                        help='Skip post-hoc DML (on by default)')
    parser.add_argument('--no-cf', action='store_true',
                        help='Skip post-hoc counterfactual evaluation (on by default)')
    parser.add_argument('--no-all-edges-dml', action='store_true',
                        help='Skip all-edges DML (on by default)')
    parser.add_argument('--no-structural-validation', action='store_true',
                        help='Skip structural validation (on by default)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g. sqlite:///study.db)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Optuna study name')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override results directory (default: results/optuna_v16)')
    parser.add_argument('--golem-override', action='append', default=[],
                        help='Override golem param: key=value (e.g. use_structural_dml=True)')
    # Rerun flags (load existing pkl, rerun specific post-hoc phases)
    parser.add_argument('--rerun-posthoc', action='store_true',
                        help='Re-run all post-hoc phases from saved result pkl')
    parser.add_argument('--rerun-dml', action='store_true',
                        help='Re-run only DML from saved result pkl')
    parser.add_argument('--rerun-all-edges-dml', action='store_true',
                        help='Re-run all-edges DML from saved result pkl')
    parser.add_argument('--rerun-cv', action='store_true',
                        help='Re-run only CV from saved result pkl')
    parser.add_argument('--rerun-cf', action='store_true',
                        help='Re-run only CF from saved result pkl')
    parser.add_argument('--sol', type=str, nargs='+', default=None,
                        help='Solution index(es) or "all" for rerun (e.g., --sol 0 2 or --sol all)')
    args = parser.parse_args()

    # Override results directory if specified
    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    # Expand --rerun-posthoc into individual flags
    if args.rerun_posthoc:
        args.rerun_dml = True
        args.rerun_all_edges_dml = True
        args.rerun_cv = True
        args.rerun_cf = True

    # Parse --sol
    if args.sol is not None:
        if 'all' in args.sol:
            args.sol = 'all'
        else:
            args.sol = [int(s) for s in args.sol]

    # Check if any rerun mode is active
    _any_rerun = any([
        getattr(args, 'rerun_dml', False),
        getattr(args, 'rerun_all_edges_dml', False),
        getattr(args, 'rerun_cv', False),
        getattr(args, 'rerun_cf', False),
    ])

    if _any_rerun:
        _run_rerun_modes(args)
        return

    print(f"JCCE Optuna Optimization")
    print(f"  Dataset: {args.dataset}")
    print(f"  Trials: {args.n_trials}, Max iter: {args.max_iter}")
    print(f"  Seed: {args.seed}")

    # Load data
    X, Y, true_mb, true_graph, n_vars, feature_names = load_dataset(
        args.dataset, args.task
    )
    print(f"  Data: {X.shape[0]} samples, {n_vars} features")
    if true_mb:
        print(f"  True MB: {true_mb}")

    # Run pipeline
    from jcce.structure_learning.experiment_runner import run_full_optuna_pipeline

    # Build ds_config for CF
    ds_config = {'feature_names': feature_names}
    if args.dataset != 'synthetic':
        try:
            from jcce.data.benchmark_loader import load_dataset as bm_load
            _, _, _cfg = bm_load(args.dataset)
            ds_config.update({k: v for k, v in _cfg.items() if k != 'true_dag'})
        except Exception:
            pass

    result = run_full_optuna_pipeline(
        X=X,
        Y=Y,
        n_vars=n_vars,
        n_trials=args.n_trials,
        max_iter=args.max_iter,
        use_v7=True,
        task=args.task,
        jax_key_seed=args.seed,
        verbose=not args.quiet,
        use_pc_warmstart=args.use_pc_warmstart,
        run_cv=not args.no_cv,
        n_folds=args.n_folds,
        cv_max_solutions=args.cv_max_solutions or 999,  # 0 = all
        cv_golem_max_iter=max(args.max_iter // 3, 100),
        run_dml=not args.no_dml,
        dml_max_solutions=args.dml_max_solutions or 999,  # 0 = all
        run_all_edges_dml=not args.no_all_edges_dml,
        run_cf=not args.no_cf,
        run_structural_validation=not args.no_structural_validation,
        feature_names=feature_names,
        ds_config=ds_config,
        true_mb=true_mb,
        true_graph=true_graph,
        storage=args.storage,
        study_name=args.study_name or f'jcce_{args.dataset}_{args.seed}',
        golem_overrides=_parse_golem_overrides(args.golem_override),
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {result['n_trials_completed']}")
    print(f"Pruned: {result['n_trials_pruned']}")
    print(f"Pareto solutions: {len(result['enhanced_solutions'])}")
    print(f"Total time: {result['pipeline_time']:.1f}s")

    for i, sol in enumerate(result['enhanced_solutions']):
        m = sol['metrics']
        print(f"  [{i}] {m['processor_type']}: "
              f"BAcc={m['classification_balanced_accuracy']:.3f} "
              f"Sparsity={m['mb_sparsity']:.3f} "
              f"h_A={m['structure_h_A']:.4f} "
              f"MB={m.get('markov_blanket', [])}")

    if result.get('cv_results'):
        print(f"\nCV Results (fixed-structure):")
        for cv in result['cv_results']:
            print(f"  {cv['processor_type']}: "
                  f"CV BAcc={cv['cv_bacc_mean']:.3f}+-{cv['cv_bacc_std']:.3f} "
                  f"({cv['cv_time']:.1f}s)")

    if result.get('dml_results'):
        print(f"\nDML Results (sample-split AIPW):")
        for dml in result['dml_results']:
            print(f"  {dml['processor_type']}: "
                  f"{dml['n_significant']}/{dml['n_parents']} significant effects")

    if result.get('all_edges_dml_results'):
        print(f"\nAll-Edges DML Results:")
        for ae in result['all_edges_dml_results']:
            print(f"  {ae['processor_type']}: "
                  f"{ae['n_significant_fdr']}/{ae['n_edges']} significant (FDR)")

    if result.get('cf_results'):
        print(f"\nCounterfactual Results:")
        for cf in result['cf_results']:
            cfr = cf['cf']
            print(f"  {cf['processor_type']}: "
                  f"{cfr['n_valid']}/{cfr['n_instances']} valid "
                  f"(sparsity={cfr['avg_sparsity']:.1f})")

    if result.get('structural_validation'):
        sv = result['structural_validation']
        print(f"\nStructural Validation:")
        if 'bootstrap' in sv and 'error' not in sv['bootstrap']:
            bs = sv['bootstrap']
            print(f"  Bootstrap: {bs.get('n_stable_edges_80', '?')} stable edges (B={bs.get('B', '?')})")
        if 'lovo' in sv and 'error' not in sv['lovo']:
            lv = sv['lovo']
            print(f"  LOVO: mean F1={lv['mean_sub_f1']:.3f} "
                  f"({lv['n_recovered']}/{lv['n_variables']} above 0.8)")
        if 'cinelli_sensitivity' in sv and 'error' not in sv['cinelli_sensitivity']:
            cs = sv['cinelli_sensitivity']
            print(f"  Cinelli RV: {cs.get('rv', '?'):.3f}")

    # ===================================================================
    # Save results in ablation-compatible format
    # ===================================================================
    ablation_result = _convert_optuna_to_ablation_format(
        result, args.dataset, args
    )

    # Fill dataset_config (exclude unpicklable true_dag if large)
    if args.dataset != 'synthetic':
        try:
            from jcce.data.benchmark_loader import load_dataset as bm_load
            _, _, ds_config = bm_load(args.dataset)
            ablation_result['dataset_config'] = {
                k: v for k, v in ds_config.items() if k != 'true_dag'
            }
        except Exception:
            pass

    out_dir = RESULTS_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'optuna_seed{args.seed}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(ablation_result, f)
    print(f"\nResults saved to: {out_path}")
    print(f"  (Compatible with run_ablation_study.py pickle format)")


if __name__ == '__main__':
    main()
