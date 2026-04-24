"""
Optuna study runner with multi-GPU support and post-hoc pipeline integration.

Phase A.2+A.3: High-level orchestrator that wraps optuna_search.py with:
- PC algorithm warm-start (seeded via enqueue_trial)
- Multi-GPU launcher (separate processes per GPU, shared SQLite)
- Post-hoc validation pipeline integration (K-fold CV, DML)
"""

import os
import sys
import time
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import optuna

from jcce.structure_learning.optuna_search import (
    PROCESSOR_TYPES,
    PROCESSOR_SEARCH_SPACE,
    FIXED_PROCESSOR_PARAMS,
    run_optuna_search,
    extract_pareto_solutions,
    get_trial_artifacts,
)
from jcce.analysis.topsis_ranking import topsis_rank


def _topsis_select(enhanced_solutions, max_n):
    """Select top-N solutions by TOPSIS ranking on BAcc + sparsity."""
    if not enhanced_solutions or max_n <= 0:
        return []
    ranked_indices, _ = topsis_rank(enhanced_solutions)
    return ranked_indices[:min(max_n, len(ranked_indices))]


# ============================================================================
# PC Warm-Start → Enqueue Seed Trials
# ============================================================================

def create_pc_seed_trials(
    X: jnp.ndarray,
    alpha: float = 0.05,
    max_cond_size: int = 2,
    n_seeds: int = 5,
    verbose: bool = False,
) -> Tuple[Optional[jnp.ndarray], List[Dict[str, Any]]]:
    """
    Run PC algorithm and generate seed trial configs for Optuna.

    PC provides a structural prior. We create seed trials that combine the PC
    structure (as A_init) with diverse hyperparameter settings, giving Optuna
    a warm start.

    Args:
        X: Feature matrix (n_samples, n_features)
        alpha: CI test significance level
        max_cond_size: Max conditioning set size
        n_seeds: Number of seed configs to generate (one per processor type)
        verbose: Print progress

    Returns:
        (pc_A_init, seed_configs):
            - pc_A_init: PC adjacency matrix (n_vars, n_vars) or None
            - seed_configs: List of dicts with params for study.enqueue_trial()
    """
    try:
        from jcce.structure_learning.jcce_learner import get_pc_warmstart

        if verbose:
            print("Running PC algorithm for warm-start...")

        pc_A = get_pc_warmstart(X, alpha=alpha, max_cond_size=max_cond_size, verbose=verbose)

        # Expand to (n_vars+1, n_vars+1) for v7 (Y is last variable)
        n_vars = pc_A.shape[0]
        n_total = n_vars + 1
        pc_A_init = jnp.zeros((n_total, n_total))
        pc_A_init = pc_A_init.at[:n_vars, :n_vars].set(pc_A)

        if verbose:
            n_edges = int(jnp.sum(jnp.abs(pc_A) > 0.01))
            print(f"PC warm-start: {n_edges} edges (expanded to {n_total}x{n_total})")

    except Exception as e:
        if verbose:
            print(f"PC warm-start failed: {e}, proceeding without")
        return None, []

    # Generate seed configs: one per processor type with reasonable defaults
    seed_configs = []
    default_hp = {
        'lambda_1': 0.02,
        'lambda_2': 0.01,
        'lr': 0.001,
        'lambda_class': 1.0,
    }

    for proc_type in PROCESSOR_TYPES[:n_seeds]:
        params = {
            'processor_type': proc_type,
            **default_hp,
        }
        # Add processor-specific params
        space = PROCESSOR_SEARCH_SPACE[proc_type]
        for param_name, values in space.items():
            params[f'{proc_type}_{param_name}'] = values[0]  # first (default) value

        seed_configs.append(params)

    return pc_A_init, seed_configs


def enqueue_seed_trials(
    study: optuna.Study,
    seed_configs: List[Dict[str, Any]],
    verbose: bool = False,
) -> int:
    """
    Enqueue seed trial configurations into an Optuna study.

    Args:
        study: Optuna study
        seed_configs: List of param dicts from create_pc_seed_trials()
        verbose: Print progress

    Returns:
        Number of trials enqueued
    """
    n_enqueued = 0
    for config in seed_configs:
        try:
            study.enqueue_trial(config)
            n_enqueued += 1
        except Exception as e:
            if verbose:
                print(f"  Failed to enqueue seed: {e}")

    if verbose and n_enqueued > 0:
        print(f"  Enqueued {n_enqueued} PC-seeded trials")

    return n_enqueued


# ============================================================================
# Post-Hoc Pipeline Integration
# ============================================================================

def run_posthoc_dml(
    enhanced_solutions: List[Dict[str, Any]],
    X_effect: np.ndarray,
    Y_effect: np.ndarray,
    feature_names: Optional[List[str]] = None,
    known_treatment_idx: Optional[int] = None,
    max_solutions: int = 10,
    n_dml_folds: int = 5,
    run_refutation: bool = True,
    n_refutation_sims: int = 100,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run DML effect estimation on Pareto solutions using held-out data.

    Uses X_effect (held-out 30%) to avoid the double-dipping problem
    (Gradu et al. JASA 2024): the same data used for structure discovery
    must NOT be reused for effect estimation.

    Args:
        enhanced_solutions: From run_optuna_search()['enhanced_solutions']
        X_effect: Held-out feature matrix for DML (NOT the search data)
        Y_effect: Held-out target vector for DML
        feature_names: Optional feature names
        known_treatment_idx: Known treatment variable for comparison
        max_solutions: Max solutions to evaluate
        n_dml_folds: Number of DML cross-fitting folds
        run_refutation: Run refutation tests
        n_refutation_sims: MC simulations per refutation test
        verbose: Print progress

    Returns:
        List of dicts with DML results per solution
    """
    from jcce.validation.multi_parent_dml import run_multi_parent_dml

    if not enhanced_solutions:
        if verbose:
            print("No solutions for DML.")
        return []

    selected = _topsis_select(enhanced_solutions, max_solutions)
    if verbose:
        print(f"\nPost-hoc DML: evaluating {len(selected)}/{len(enhanced_solutions)} "
              f"solutions on held-out data (n={X_effect.shape[0]})")

    dml_results = []
    for rank in selected:
        sol = enhanced_solutions[rank]
        m = sol['metrics']
        # Use continuous A_weights for DML instead of binary A_est.
        # Binary thresholding (0.05*max_weight) kills X->Y edges because they are
        # naturally weaker than X->X weights. DML applies its own 0.01 threshold,
        # so continuous weights let it discover X->Y parents correctly.
        A_est = m.get('A_weights', m.get('structure_A_est'))
        if A_est is None:
            continue

        A_est_np = np.array(A_est)
        Y_idx = A_est_np.shape[0] - 1
        processor_type = m.get('processor_type', 'elm')

        try:
            dml_result = run_multi_parent_dml(
                X=X_effect, Y=Y_effect,
                A_est=A_est_np, Y_idx=Y_idx,
                feature_names=feature_names,
                known_treatment_idx=known_treatment_idx,
                parent_threshold=0.01,
                n_dml_folds=n_dml_folds if X_effect.shape[0] >= 500 else 3,
                run_refutation=run_refutation,
                n_refutation_sims=n_refutation_sims,
                verbose=(verbose and rank == 0),
            )

            result_dict = {
                'solution_idx': rank,
                'processor_type': processor_type,
                'dml_result': dml_result.to_dict() if dml_result else None,
                'n_parents': dml_result.n_parents_discovered if dml_result else 0,
                'n_significant': dml_result.n_parents_significant if dml_result else 0,
            }
            dml_results.append(result_dict)

            if verbose:
                sig = dml_result.n_parents_significant if dml_result else 0
                total = dml_result.n_parents_discovered if dml_result else 0
                print(f"  [{rank}] {processor_type}: DML {sig}/{total} significant")

        except Exception as e:
            if verbose:
                print(f"  [{rank}] {processor_type}: DML failed: {e}")

    return dml_results


def run_posthoc_all_edges_dml(
    enhanced_solutions: List[Dict[str, Any]],
    X_effect: np.ndarray,
    Y_effect: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_solutions: int = 10,
    n_dml_folds: int = 5,
    run_refutation: bool = True,
    n_refutation_sims: int = 100,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run all-edges DML effect estimation on Pareto solutions using held-out data.

    Unlike multi-parent DML (which only estimates Y's parents), this estimates
    causal effects for ALL edges in the learned DAG with FDR correction.

    Args:
        enhanced_solutions: From run_optuna_search()['enhanced_solutions']
        X_effect: Held-out feature matrix for DML (NOT the search data)
        Y_effect: Held-out target vector for DML
        feature_names: Optional feature names
        max_solutions: Max solutions to evaluate
        n_dml_folds: Number of DML cross-fitting folds
        run_refutation: Run refutation tests
        n_refutation_sims: MC simulations per refutation test
        verbose: Print progress

    Returns:
        List of dicts with all-edges DML results per solution
    """
    from jcce.validation.all_edges_dml import run_all_edges_dml

    if not enhanced_solutions:
        if verbose:
            print("No solutions for all-edges DML.")
        return []

    n_eval = min(max_solutions, len(enhanced_solutions))
    if verbose:
        print(f"\nPost-hoc all-edges DML: evaluating {n_eval}/{len(enhanced_solutions)} "
              f"solutions on held-out data (n={X_effect.shape[0]})")

    all_edges_results = []
    for rank in range(n_eval):
        sol = enhanced_solutions[rank]
        m = sol['metrics']
        # Use continuous A_weights (same rationale as per-solution DML above)
        A_est = m.get('A_weights', m.get('structure_A_est'))
        if A_est is None:
            continue

        A_est_np = np.array(A_est)
        Y_idx = A_est_np.shape[0] - 1
        processor_type = m.get('processor_type', 'elm')

        try:
            ae_result = run_all_edges_dml(
                X=X_effect, Y=Y_effect,
                A_est=A_est_np, Y_idx=Y_idx,
                feature_names=feature_names,
                edge_threshold=0.01,
                n_dml_folds=n_dml_folds if X_effect.shape[0] >= 500 else 3,
                run_refutation=run_refutation,
                n_refutation_sims=n_refutation_sims,
                verbose=(verbose and rank == 0),
            )

            if ae_result is not None:
                result_dict = {
                    'solution_idx': rank,
                    'processor_type': processor_type,
                    'all_edges_dml': ae_result.to_storage_dict(),
                    'dml_causal_effects': ae_result.to_causal_effects_dict(),
                    'n_edges': ae_result.n_edges,
                    'n_significant_fdr': ae_result.n_significant_fdr,
                }
                all_edges_results.append(result_dict)

                if verbose:
                    sig = ae_result.n_significant_fdr
                    total = ae_result.n_edges
                    print(f"  [{rank}] {processor_type}: All-edges DML "
                          f"{sig}/{total} significant (FDR)")

        except Exception as e:
            if verbose:
                print(f"  [{rank}] {processor_type}: All-edges DML failed: {e}")

    return all_edges_results


def run_posthoc_cf(
    enhanced_solutions: List[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    ds_config: dict,
    max_solutions: int = 5,
    n_instances: int = 30,
    cf_pop_size: int = 50,
    cf_n_gen: int = 50,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run counterfactual evaluation on Pareto solutions.

    Requires _processor_params in the solution metrics (added by
    extract_pareto_solutions when proc_params are in artifacts).

    Args:
        enhanced_solutions: From run_optuna_search()['enhanced_solutions']
        X: Feature matrix (structure training data)
        Y: Target vector
        ds_config: Dataset config dict (for feature names, immutable features, etc.)
        max_solutions: Max solutions to evaluate
        n_instances: Number of instances per solution
        cf_pop_size: NSGA-II population size for CF search
        cf_n_gen: NSGA-II generations for CF search
        verbose: Print progress

    Returns:
        List of dicts with CF results per solution
    """
    from jcce.validation.counterfactual_runner import (
        run_counterfactual_evaluation,
        reconstruct_processor,
        convert_params_to_jax,
    )

    if not enhanced_solutions:
        if verbose:
            print("No solutions for CF evaluation.")
        return []

    n_eval = min(max_solutions, len(enhanced_solutions))
    if verbose:
        print(f"\nPost-hoc CF: evaluating {n_eval}/{len(enhanced_solutions)} solutions")

    cf_results = []
    for rank in range(n_eval):
        sol = enhanced_solutions[rank]
        m = sol['metrics']

        # Need processor params + A_est for CF
        if (m.get('structure_A_est') is None
                or m.get('_processor_params') is None):
            if verbose:
                print(f"  [{rank}] Skipping CF: missing processor_params or A_est")
            continue

        processor_type = m.get('processor_type', 'elm')
        processor_config = m.get('processor_config', {})
        proc_params_numpy = m['_processor_params']

        try:
            # Reconstruct processor from saved type/config/params
            processor, params_jax = reconstruct_processor(
                processor_type, processor_config, proc_params_numpy,
            )

            cf_result = run_counterfactual_evaluation(
                X=np.array(X), Y=np.array(Y),
                A_est=np.array(m['structure_A_est']),
                processor_trained=processor,
                processor_params=params_jax,
                ds_config=ds_config,
                A_weights=np.array(m['A_weights']) if m.get('A_weights') is not None else None,
                A_confound=np.array(m['A_confound_weights']) if m.get('A_confound_weights') is not None else None,
                n_instances=n_instances,
                cf_pop_size=cf_pop_size,
                cf_n_gen=cf_n_gen,
                verbose=(verbose and rank == 0),
            )

            if cf_result is not None:
                result_dict = {
                    'solution_idx': rank,
                    'processor_type': processor_type,
                    'cf': cf_result,
                }
                cf_results.append(result_dict)

                if verbose:
                    print(f"  [{rank}] {processor_type}: CF "
                          f"{cf_result['n_valid']}/{cf_result['n_instances']} valid "
                          f"(sparsity={cf_result['avg_sparsity']:.1f})")

        except Exception as e:
            if verbose:
                print(f"  [{rank}] {processor_type}: CF failed: {e}")

    return cf_results


def run_posthoc_cv(
    enhanced_solutions: List[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    max_solutions: int = 5,
    golem_max_iter: int = 50,
    cold_start: bool = False,
    task: str = 'classification',
    true_mb: Optional[List[int]] = None,
    true_dag: Optional[np.ndarray] = None,
    verbose: bool = False,
    parallel_folds: bool = False,
    freeze_structure: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run K-fold CV on top Pareto solutions from Optuna study.

    By default uses fixed-structure CV (freeze_structure=True): the adjacency
    matrix A is frozen across folds and only the processor is retrained.
    This isolates predictive quality from structural instability (6/6 LLM consensus).

    Args:
        enhanced_solutions: From run_optuna_search()['enhanced_solutions']
        X: Full dataset features (numpy)
        Y: Full dataset labels (numpy)
        n_folds: Number of CV folds
        max_solutions: Max solutions to evaluate (expensive)
        golem_max_iter: GOLEM iters per fold
        cold_start: Retrain from random init (unbiased but slower)
        task: 'classification' or 'regression'
        true_mb: Ground truth Markov blanket
        true_dag: Ground truth DAG
        verbose: Print progress
        parallel_folds: Use multi-GPU for fold parallelism
        freeze_structure: If True (default), freeze A during CV folds.
            Only processor weights are retrained per fold.

    Returns:
        List of dicts with CV results per solution
    """
    from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

    if not enhanced_solutions:
        if verbose:
            print("No solutions to evaluate.")
        return []

    # Select top solutions by TOPSIS ranking
    indices = _topsis_select(enhanced_solutions, max_solutions)

    if verbose:
        print(f"\nPost-hoc CV: evaluating {len(indices)}/{len(enhanced_solutions)} "
              f"Pareto solutions ({n_folds}-fold)")

    cv_results = []
    for rank, idx in enumerate(indices):
        sol = enhanced_solutions[idx]
        m = sol['metrics']

        # Build hyperparams dict
        hyperparams = {
            'lambda_1': m.get('lambda_1', 0.02),
            'lambda_2': m.get('lambda_2', 0.01),
            'lambda_class': m.get('lambda_class', 1.0),
            'lr': m.get('lr', 0.001),
            'processor_config': m.get('processor_config', {}),
        }
        for key in ['effect_hidden_dim', 'effect_embed_dim', 'lambda_effect',
                     'effect_warmup_iter', 'lambda_confound_sparse', 'lambda_bow']:
            if key in m:
                hyperparams[key] = m[key]

        processor_type = m.get('processor_type', 'elm')
        A_init = m.get('structure_A_est')
        if A_init is None:
            if verbose:
                print(f"  [{rank}] SKIP: no adjacency matrix")
            continue

        # Extract trained processor params for warm-starting CV folds
        proc_params = m.get('_processor_params', None)

        # Continuous A_weights for prediction (matches training-time weighting)
        A_weights_cont = m.get('A_weights')
        if A_weights_cont is not None:
            A_weights_cont = np.array(A_weights_cont)
        A_conf_weights = m.get('A_confound_weights')
        if A_conf_weights is not None:
            A_conf_weights = np.array(A_conf_weights)

        try:
            cv_start = time.time()
            cv_result = evaluate_pareto_solution_cv(
                X=X,
                Y=Y,
                hyperparams=hyperparams,
                processor_type=processor_type,
                A_init=np.array(A_init),
                processor_params_init=proc_params,
                n_folds=n_folds,
                true_mb=true_mb,
                true_dag=true_dag,
                golem_max_iter=golem_max_iter,
                cold_start=cold_start,
                task=task,
                verbose=False,
                parallel_folds=parallel_folds,
                freeze_structure=freeze_structure,
                A_weights_continuous=A_weights_cont,
                A_confound_weights=A_conf_weights,
            )
            cv_time = time.time() - cv_start

            result_dict = {
                'solution_idx': idx,
                'processor_type': processor_type,
                'train_bacc': m.get('classification_balanced_accuracy', 0.0),
                'cv_accuracy_mean': cv_result.accuracy_mean,
                'cv_accuracy_std': cv_result.accuracy_std,
                'cv_precision_mean': cv_result.precision_mean,
                'cv_precision_std': cv_result.precision_std,
                'cv_recall_mean': cv_result.recall_mean,
                'cv_recall_std': cv_result.recall_std,
                'cv_bacc_mean': cv_result.balanced_acc_mean,
                'cv_bacc_std': cv_result.balanced_acc_std,
                'cv_f1_mean': cv_result.f1_mean,
                'cv_f1_std': cv_result.f1_std,
                'cv_roc_auc_mean': cv_result.roc_auc_mean,
                'cv_roc_auc_std': cv_result.roc_auc_std,
                'cv_mb_jaccard': cv_result.mb_jaccard_mean,
                'cv_time': cv_time,
                'cv_result': cv_result,
            }
            if cv_result.mb_f1_per_fold:
                result_dict['cv_mb_f1_mean'] = float(np.mean(cv_result.mb_f1_per_fold))
            if cv_result.edge_f1_per_fold:
                result_dict['cv_edge_f1_mean'] = float(np.mean(cv_result.edge_f1_per_fold))
            if cv_result.edge_precision_per_fold:
                result_dict['cv_edge_precision_mean'] = float(np.mean(cv_result.edge_precision_per_fold))
            if cv_result.edge_recall_per_fold:
                result_dict['cv_edge_recall_mean'] = float(np.mean(cv_result.edge_recall_per_fold))

            cv_results.append(result_dict)

            if verbose:
                print(f"  [{rank}] {processor_type}: "
                      f"CV BAcc={cv_result.balanced_acc_mean:.3f}"
                      f"+-{cv_result.balanced_acc_std:.3f} "
                      f"({cv_time:.1f}s)")

        except Exception as e:
            if verbose:
                print(f"  [{rank}] {processor_type}: CV failed: {e}")

    return cv_results


# ============================================================================
# Multi-GPU Launcher
# ============================================================================

def launch_multi_gpu_workers(
    n_gpus: int,
    n_trials_per_gpu: int,
    study_name: str,
    storage: str,
    script_args: List[str],
    verbose: bool = True,
) -> List[subprocess.Popen]:
    """
    Launch separate worker processes per GPU.

    Each worker runs an independent Optuna optimization loop sharing
    the same SQLite study. TPESampler with constant_liar=True prevents
    duplicate sampling across workers.

    Args:
        n_gpus: Number of GPU workers to launch
        n_trials_per_gpu: Trials per worker
        study_name: Shared study name
        storage: SQLite storage URL
        script_args: Additional CLI args for the worker script
        verbose: Print launch info

    Returns:
        List of Popen processes
    """
    worker_script = os.path.join(
        os.path.dirname(__file__), '..', '..', 'scripts', 'jcce_hpo_worker.py'
    )
    worker_script = os.path.abspath(worker_script)

    if not os.path.exists(worker_script):
        raise FileNotFoundError(
            f"Worker script not found: {worker_script}. "
            f"Create scripts/jcce_hpo_worker.py first."
        )

    processes = []
    for gpu_id in range(n_gpus):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        cmd = [
            sys.executable, worker_script,
            '--study-name', study_name,
            '--storage', storage,
            '--n-trials', str(n_trials_per_gpu),
            '--gpu-id', str(gpu_id),
        ] + script_args

        if verbose:
            print(f"  Launching GPU {gpu_id}: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
        )
        processes.append(proc)

    if verbose:
        print(f"  {n_gpus} workers launched")

    return processes


# ============================================================================
# Post-hoc Structural Validation
# ============================================================================

def _run_structural_validation(
    enhanced_solutions: List[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    n_vars: int,
    max_iter: int,
    dml_results: List[Dict[str, Any]],
    bootstrap_B: int = 50,
    lovo_variables: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run post-hoc structural validation on the best Pareto solution.

    Runs bootstrap stability, LOVO CV, and (if DML results available)
    Cinelli sensitivity analysis.

    Returns dict with results keyed by validation type.
    """
    # Pick best feasible solution by TOPSIS ranking (BAcc + sparsity)
    best_indices = _topsis_select(enhanced_solutions, 1)
    best = enhanced_solutions[best_indices[0]]
    m = best['metrics']
    A_est = np.array(m.get('structure_A_est'))
    processor_type = m.get('processor_type', 'elm')

    if verbose:
        print(f"\nStructural validation on best solution ({processor_type}):")

    # Build hyperparams for bootstrap (which uses GOLEM directly)
    hyperparams = {
        'lambda_1': m.get('lambda_1', 0.02),
        'lambda_2': m.get('lambda_2', 0.01),
        'lambda_class': m.get('lambda_class', 1.0),
        'lr': m.get('lr', 0.001),
        'processor_config': m.get('processor_config', {}),
    }
    for key in ['effect_hidden_dim', 'effect_embed_dim', 'lambda_effect',
                'effect_warmup_iter', 'lambda_confound_sparse', 'lambda_bow']:
        if key in m:
            hyperparams[key] = m[key]

    # Iterations for validation relearning — needs the full budget to be fair.
    # LOVO/self-compat do cold-start relearning; giving them fewer iterations
    # than the original training biases toward non-recovery.
    learn_max_iter = max_iter

    # Generic learner for LOVO/self-compat (takes full data matrix, returns A)
    def _generic_learner(data_matrix):
        """Relearn DAG from (n, d) data matrix. Returns (d, d) adjacency."""
        from jcce.structure_learning.jcce_learner import learn_structure
        from jcce.structure_learning.jcce_learner import create_processor
        import jax
        d = data_matrix.shape[1]
        X_learn = jnp.array(data_matrix[:, :d-1], dtype=jnp.float32)
        Y_learn = jnp.array(data_matrix[:, d-1].reshape(-1, 1), dtype=jnp.float32)
        n_feat = d - 1

        proc_config = hyperparams.get('processor_config', {})
        key = jax.random.PRNGKey(42)
        key, proc_key = jax.random.split(key)
        processor = create_processor(
            processor_type, key=proc_key, n_features=n_feat, **proc_config
        )

        # Pass all v7 hyperparams from Optuna trial for consistent re-training.
        A_new, _, _, _ = learn_structure(
            data=X_learn,
            Y=Y_learn,
            Y_idx=n_feat,
            processor=processor,
            key=key,
            processor_type=processor_type,
            lambda_1=hyperparams.get('lambda_1', 0.02),
            lambda_2_init=hyperparams.get('lambda_2', 0.01),
            lambda_class=hyperparams.get('lambda_class', 1.0),
            lr=hyperparams.get('lr', 0.001),
            max_iter=learn_max_iter,
            patience=25,
            verbose=0,
            n_latent_confounders=hyperparams.get('n_latent_confounders', 5),
            effect_hidden_dim=hyperparams.get('effect_hidden_dim', 64),
            effect_embed_dim=hyperparams.get('effect_embed_dim', 16),
            lambda_effect=hyperparams.get('lambda_effect', 0.5),
            effect_warmup_iter=hyperparams.get('effect_warmup_iter', 20),
            lambda_confound_sparse=hyperparams.get('lambda_confound_sparse', 0.02),
            lambda_bow=hyperparams.get('lambda_bow', 0.3),
        )
        return np.array(A_new)

    results = {}
    data_full = np.column_stack([np.array(X), np.array(Y)])

    # ---- Bootstrap DAG stability ----
    try:
        from jcce.validation.bootstrap_stability import bootstrap_dag_stability
        if verbose:
            print(f"  Bootstrap stability (B={bootstrap_B})...")
        bootstrap_result = bootstrap_dag_stability(
            X=np.array(X),
            Y=np.array(Y),
            hyperparams=hyperparams,
            processor_type=processor_type,
            B=bootstrap_B,
            max_iter=learn_max_iter,
            A_init=A_est,
            seed=42,
            verbose=False,
        )
        results['bootstrap'] = bootstrap_result.to_dict()
        if verbose:
            stable = bootstrap_result.get_stable_edges(min_frequency=0.8)
            print(f"    {len(stable)} stable edges (freq >= 0.8)")
    except Exception as e:
        if verbose:
            print(f"    Bootstrap failed: {e}")
        results['bootstrap'] = {'error': str(e)}

    # ---- LOVO CV ----
    try:
        from jcce.validation.lovo_cv import lovo_cv
        # Default: test all X variables but NOT Y (last column).
        # Removing Y and treating X[-1] as target is semantically wrong.
        if lovo_variables is None:
            lovo_variables = list(range(n_vars))  # X variables only (0..n_vars-1)
        if verbose:
            print(f"  LOVO CV ({len(lovo_variables)} variables)...")
        lovo_result = lovo_cv(
            data_full, A_est, _generic_learner,
            variables=lovo_variables,
            verbose=False,
        )
        results['lovo'] = lovo_result.to_dict()
        if verbose:
            print(f"    Mean sub-DAG F1: {lovo_result.mean_sub_f1:.3f} "
                  f"({lovo_result.n_recovered}/{lovo_result.n_variables} above 0.8 threshold)")
    except Exception as e:
        if verbose:
            print(f"    LOVO failed: {e}")
        results['lovo'] = {'error': str(e)}

    # ---- Cinelli sensitivity (only if DML results available) ----
    if dml_results:
        try:
            from jcce.validation.cinelli_sensitivity import cinelli_sensitivity
            best_dml = dml_results[0]
            treatment_idx = best_dml.get('treatment_idx', 0)
            if verbose:
                print(f"  Cinelli sensitivity analysis...")
            sensitivity_result = cinelli_sensitivity(
                Y=np.array(Y),
                T=np.array(X)[:, treatment_idx],
                X=np.delete(np.array(X), treatment_idx, axis=1),
                treatment_name=best_dml.get('treatment_name', f'X{treatment_idx}'),
            )
            results['cinelli_sensitivity'] = sensitivity_result.to_dict()
            if verbose:
                print(f"    RV = {sensitivity_result.rv:.3f}")
        except Exception as e:
            if verbose:
                print(f"    Cinelli failed: {e}")
            results['cinelli_sensitivity'] = {'error': str(e)}

    return results


# ============================================================================
# Full Pipeline: Search + CV
# ============================================================================

def run_full_optuna_pipeline(
    X: np.ndarray,
    Y: np.ndarray,
    n_vars: int,
    n_trials: int = 100,
    max_iter: int = 300,
    use_v7: bool = True,
    task: str = 'classification',
    golem_overrides: Optional[Dict[str, Any]] = None,
    jax_key_seed: int = 0,
    verbose: bool = True,
    # PC warm-start
    use_pc_warmstart: bool = False,
    # Post-hoc CV (mandatory by default — fixed-structure)
    run_cv: bool = True,
    n_folds: int = 5,
    cv_max_solutions: int = 5,
    cv_golem_max_iter: int = 100,
    cv_cold_start: bool = False,
    # Post-hoc DML (mandatory by default — sample-split AIPW)
    run_dml: bool = True,
    dml_max_solutions: int = 5,
    dml_run_refutation: bool = True,
    # Post-hoc all-edges DML (mandatory by default)
    run_all_edges_dml: bool = True,
    all_edges_dml_max_solutions: int = 5,
    # Post-hoc counterfactual evaluation (mandatory by default)
    run_cf: bool = True,
    cf_max_solutions: int = 5,
    cf_n_instances: int = 30,
    cf_pop_size: int = 50,
    cf_n_gen: int = 50,
    # Post-hoc structural validation (mandatory by default — runs on best solution)
    run_structural_validation: bool = True,
    bootstrap_B: int = 50,
    lovo_variables: Optional[List[int]] = None,
    # Ground truth
    true_graph: Optional[np.ndarray] = None,
    true_mb: Optional[List[int]] = None,
    # Dataset metadata
    feature_names: Optional[List[str]] = None,
    known_treatment_idx: Optional[int] = None,
    ds_config: Optional[Dict[str, Any]] = None,
    # Storage
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    # Multi-GPU
    parallel_folds: bool = False,
) -> Dict[str, Any]:
    """
    Run complete Optuna pipeline: PC warm-start → search → post-hoc DML + CV.

    This is the top-level entry point that replaces the NSGA-II flow in
    test_lucas_v15_quick.py and run_v16_ablation.py.

    When run_dml=True, applies a 70/30 sample split: search uses X_struct (70%),
    DML uses X_effect (30%) to avoid double-dipping (Gradu et al. JASA 2024).

    When run_structural_validation=True, runs post-hoc diagnostics on the best
    Pareto solution: bootstrap DAG stability, LOVO CV, self-compatibility check,
    and Cinelli sensitivity analysis (if DML results available).

    Args:
        X: Feature matrix (numpy array, n_samples × n_features)
        Y: Target variable (numpy array, n_samples)
        n_vars: Number of features (= X.shape[1])
        n_trials: Number of Optuna trials
        max_iter: Max GOLEM iterations per trial
        use_v7: Use v7 with effect estimation
        task: 'classification' or 'regression'
        golem_overrides: Override dict for GOLEM
        jax_key_seed: Random seed
        verbose: Print progress
        use_pc_warmstart: Run PC algorithm for seed trials
        run_cv: Run post-hoc K-fold CV on Pareto solutions
        n_folds: CV folds
        cv_max_solutions: Max solutions for CV
        cv_golem_max_iter: GOLEM iters per CV fold
        cv_cold_start: Cold-start CV (unbiased but slower)
        run_dml: Run post-hoc DML effect estimation on held-out data
        dml_max_solutions: Max solutions for DML
        dml_run_refutation: Run refutation tests in DML
        true_graph: Ground truth DAG
        true_mb: Ground truth Markov blanket
        feature_names: Feature names for DML reporting
        known_treatment_idx: Known treatment variable index
        storage: Optuna storage URL
        study_name: Study name
        parallel_folds: Multi-GPU fold parallelism

    Returns:
        Dict with:
            - All keys from run_optuna_search()
            - 'pc_A_init': PC adjacency matrix or None
            - 'cv_results': List of CV result dicts (if run_cv=True)
            - 'pipeline_time': Total wall-clock time
    """
    pipeline_start = time.time()

    X_np = np.array(X) if isinstance(X, jnp.ndarray) else X
    Y_np = np.array(Y) if isinstance(Y, jnp.ndarray) else Y

    # ---- Sample splitting for DML (Gradu et al. JASA 2024) ----
    X_effect_np = None
    Y_effect_np = None
    if run_dml:
        from sklearn.model_selection import train_test_split
        X_struct_np, X_effect_np, Y_struct_np, Y_effect_np = train_test_split(
            X_np, Y_np, test_size=0.3, stratify=Y_np, random_state=42
        )
        # Search uses X_struct (70%), DML uses X_effect (30%)
        X_search = jnp.array(X_struct_np)
        Y_search = jnp.array(Y_struct_np)
        if verbose:
            print(f"Sample split: {X_struct_np.shape[0]} structure (search) / "
                  f"{X_effect_np.shape[0]} effect (DML+CV)")
    else:
        # No DML: search uses full data (backward compatible)
        X_search = jnp.array(X_np)
        Y_search = jnp.array(Y_np)

    true_graph_jax = jnp.array(true_graph) if true_graph is not None else None

    # ---- Step 1: PC warm-start ----
    pc_A_init = None
    if use_pc_warmstart:
        pc_A_init, seed_configs = create_pc_seed_trials(
            X_search, verbose=verbose,
        )
    else:
        seed_configs = []

    # ---- Step 2: Optuna search ----
    search_result = run_optuna_search(
        X=X_search,
        Y=Y_search,
        n_vars=n_vars,
        n_trials=n_trials,
        max_iter=max_iter,
        use_v7=use_v7,
        task=task,
        golem_overrides=golem_overrides,
        jax_key_seed=jax_key_seed,
        verbose=verbose,
        true_graph=true_graph_jax,
        true_mb=true_mb,
        enable_warm_start=True,
        storage=storage,
        study_name=study_name,
    )

    # Enqueue seed trials (for future runs with load_if_exists)
    if seed_configs and search_result.get('study'):
        enqueue_seed_trials(search_result['study'], seed_configs, verbose=verbose)

    # ---- Step 3: Post-hoc DML on held-out data ----
    dml_results = []
    if run_dml and search_result['enhanced_solutions'] and X_effect_np is not None:
        dml_results = run_posthoc_dml(
            enhanced_solutions=search_result['enhanced_solutions'],
            X_effect=X_effect_np,
            Y_effect=Y_effect_np,
            feature_names=feature_names,
            known_treatment_idx=known_treatment_idx,
            max_solutions=dml_max_solutions,
            run_refutation=dml_run_refutation,
            verbose=verbose,
        )

    # ---- Step 3b: Post-hoc all-edges DML on held-out data ----
    all_edges_dml_results = []
    if run_all_edges_dml and search_result['enhanced_solutions'] and X_effect_np is not None:
        all_edges_dml_results = run_posthoc_all_edges_dml(
            enhanced_solutions=search_result['enhanced_solutions'],
            X_effect=X_effect_np,
            Y_effect=Y_effect_np,
            feature_names=feature_names,
            max_solutions=all_edges_dml_max_solutions,
            run_refutation=dml_run_refutation,
            verbose=verbose,
        )

    # ---- Step 4: Post-hoc CV (uses held-out data if DML split exists) ----
    cv_results = []
    if run_cv and search_result['enhanced_solutions']:
        # When DML split exists, CV uses held-out data too (consistent validation)
        cv_X = X_effect_np if X_effect_np is not None else X_np
        cv_Y = Y_effect_np if Y_effect_np is not None else Y_np
        # Adaptive n_folds: reduce to 3 for small held-out sets to avoid
        # tiny folds (e.g. 90 samples / 5 folds = 18 per fold)
        effective_n_folds = min(n_folds, 3) if cv_X.shape[0] < 500 else n_folds
        if effective_n_folds != n_folds and verbose:
            print(f"  [CV] Reduced n_folds {n_folds}→{effective_n_folds} "
                  f"(n_effect={cv_X.shape[0]} < 500)")
        cv_results = run_posthoc_cv(
            enhanced_solutions=search_result['enhanced_solutions'],
            X=cv_X,
            Y=cv_Y,
            n_folds=effective_n_folds,
            max_solutions=cv_max_solutions,
            golem_max_iter=cv_golem_max_iter,
            cold_start=cv_cold_start,
            task=task,
            true_mb=true_mb,
            true_dag=true_graph,
            verbose=verbose,
            parallel_folds=parallel_folds,
        )

    # ---- Step 4b: Post-hoc counterfactual evaluation ----
    cf_results = []
    if run_cf and search_result['enhanced_solutions']:
        # CF uses structure training data (X_struct) — needs the trained model's data distribution
        cf_X = X_struct_np if run_dml else X_np
        cf_Y = Y_struct_np if run_dml else Y_np
        # Build ds_config from available info if not provided
        _ds_config = ds_config or {}
        if feature_names and 'feature_names' not in _ds_config:
            _ds_config = dict(_ds_config)
            _ds_config['feature_names'] = feature_names
        cf_results = run_posthoc_cf(
            enhanced_solutions=search_result['enhanced_solutions'],
            X=cf_X,
            Y=cf_Y,
            ds_config=_ds_config,
            max_solutions=cf_max_solutions,
            n_instances=cf_n_instances,
            cf_pop_size=cf_pop_size,
            cf_n_gen=cf_n_gen,
            verbose=verbose,
        )

    # ---- Step 5: Structural validation (on best Pareto solution) ----
    structural_validation = {}
    if run_structural_validation and search_result['enhanced_solutions']:
        structural_validation = _run_structural_validation(
            enhanced_solutions=search_result['enhanced_solutions'],
            X=X_np,
            Y=Y_np,
            n_vars=n_vars,
            max_iter=max_iter,
            dml_results=dml_results,
            bootstrap_B=bootstrap_B,
            lovo_variables=lovo_variables,
            verbose=verbose,
        )

    pipeline_time = time.time() - pipeline_start

    if verbose:
        print(f"\nPipeline complete in {pipeline_time:.1f}s")
        if cv_results:
            best_cv = max(cv_results, key=lambda r: r['cv_bacc_mean'])
            print(f"  Best CV BAcc: {best_cv['cv_bacc_mean']:.3f}"
                  f"+-{best_cv['cv_bacc_std']:.3f} ({best_cv['processor_type']})")
        if dml_results:
            best_dml = max(dml_results, key=lambda r: r.get('n_significant', 0))
            print(f"  Best DML: {best_dml['n_significant']}/{best_dml['n_parents']} "
                  f"significant ({best_dml['processor_type']})")

    return {
        **search_result,
        'pc_A_init': pc_A_init,
        'cv_results': cv_results,
        'dml_results': dml_results,
        'all_edges_dml_results': all_edges_dml_results,
        'cf_results': cf_results,
        'structural_validation': structural_validation,
        'pipeline_time': pipeline_time,
    }
