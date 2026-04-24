#!/usr/bin/env python3
"""
JCCE v16 Comprehensive Ablation Study

Runs all ablations and structural comparisons:

Core Ablations (A1-A7): Remove one component at a time
  A1: No curriculum        (disable adaptive curriculum phases)
  A2: No bow-free          (disable bow-free constraint on A_confound)
  A3: No L_effect          (disable DragonNet effect loss)
  A4: No kappa constraint  (disable condition number constraint)
  A5: No D-optimality      (disable identifiability regularizer)
  A6: No outcome sink      (disable A[Y,:]=0 enforcement)
  A7: Freeze A             (fix A, only train processor params)

Structural Comparisons (B1-B6): Alternative architectures/methods
  B1: MLP only             (single processor, no architecture search)
  B2: Random search        (replace NSGA-II with random sampling)
  B3: PC + LogReg          (pipeline: PC algorithm → LogisticRegression)
  B4: NOTEARS + LogReg     (pipeline: NOTEARS → LogisticRegression)
  B5: GES + LogReg         (pipeline: GES → LogisticRegression)
  B6: Full JCCE (baseline) (the complete v16 framework)

Usage:
    # Run specific ablation on specific dataset
    uv run python scripts/run_ablation_study.py --ablation A1 --dataset lucas

    # Run all ablations on LUCAS
    uv run python scripts/run_ablation_study.py --all --dataset lucas

    # Run all ablations on all datasets
    uv run python scripts/run_ablation_study.py --all --all-datasets

    # Quick test mode (tiny config, for validation)
    uv run python scripts/run_ablation_study.py --ablation A1 --dataset lucas --quick

    # List available ablations
    uv run python scripts/run_ablation_study.py --list

Server usage:
    nohup uv run python scripts/run_ablation_study.py --all --all-datasets > ablation_v16.log 2>&1 &
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

import argparse
import pickle
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import jcce.structure_learning.nsga2_search as nsga2_module
from jcce.data.benchmark_loader import load_dataset
from jcce.structure_learning.nsga2_search import run_nsga2
from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

# =============================================================================
# Output directory
# =============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results" / "ablation_v16"


# =============================================================================
# Ablation Configurations
# =============================================================================


def get_ablation_config(ablation_id: str) -> dict:
    """
    Returns NSGA-II kwargs overrides for each ablation.

    The baseline (B6) uses all defaults. Each ablation disables one component.
    """
    configs = {
        # =============================================
        # Core Ablations: Remove one component
        # =============================================
        "A1": {
            "name": "No Curriculum",
            "description": "Disable adaptive curriculum (all phases active from start)",
            "golem_overrides": {"use_adaptive_curriculum": False},
        },
        "A1b": {
            "name": "Fixed Curriculum (33/33/33)",
            "description": "Three phases with equal fixed iteration splits instead of convergence triggers",
            "golem_overrides": {
                "use_adaptive_curriculum": False,
                "curriculum_phase_splits": (0.333, 0.667),
            },
        },
        "A1c": {
            "name": "Two-Phase Only",
            "description": "Structure then classification only, no effect estimation phase",
            "golem_overrides": {
                "lambda_effect": 0.0,
                "use_amortized_effects": False,
                "use_adaptive_curriculum": True,
            },
        },
        "A8": {
            "name": "PCGrad Gradient Surgery",
            "description": "Use PCGrad to project conflicting per-task gradients on A_direct",
            "golem_overrides": {"use_pcgrad": True},
        },
        "A9": {
            "name": "Structural DML Effects",
            "description": "Replace shared DragonNet with independent linear models on X[Z]",
            "golem_overrides": {"use_structural_dml": True, "use_dragonnet": False},
        },
        "A2": {
            "name": "No Bow-Free",
            "description": "Disable bow-free constraint (allow directed + bidirected on same edge)",
            "golem_overrides": {"lambda_bow": 0.0},
        },
        "A3": {
            "name": "No Effect Loss",
            "description": "Disable DragonNet effect estimation loss",
            "golem_overrides": {"lambda_effect": 0.0, "use_amortized_effects": False},
        },
        "A4": {
            "name": "No Kappa Constraint",
            "description": "Disable condition number constraint in NSGA-II",
            "nsga2_overrides": {"use_condition_constraint": False},
        },
        "A5": {
            "name": "No D-Optimality",
            "description": "Disable identifiability regularizer in GOLEM loss",
            "golem_overrides": {"lambda_ident": 0.0},
        },
        "A6": {
            "name": "No Outcome Sink",
            "description": "Do NOT enforce A[Y,:]=0 (allow Y to have outgoing edges)",
            "golem_overrides": {"enforce_outcome_sink": False},
        },
        "A7": {
            "name": "Freeze A",
            "description": "Fix adjacency matrix (no structure learning), only train processor",
            "golem_overrides": {"freeze_A": True},
        },
        # =============================================
        # Structural Comparisons
        # =============================================
        "B1": {
            "name": "MLP Only",
            "description": "Single processor type (MLP), no architecture search",
            "processor_override": ["mlp"],
        },
        "B2": {
            "name": "Random Search",
            "description": "Replace NSGA-II with random sampling (no evolutionary optimization)",
            "random_search": True,
        },
        "B3": {
            "name": "PC + LogReg",
            "description": "Pipeline baseline: PC algorithm → feature selection → LogisticRegression",
            "pipeline_baseline": "pc",
        },
        "B4": {
            "name": "NOTEARS + LogReg",
            "description": "Pipeline baseline: NOTEARS → feature selection → LogisticRegression",
            "pipeline_baseline": "notears",
        },
        "B5": {
            "name": "GES + LogReg",
            "description": "Pipeline baseline: GES → feature selection → LogisticRegression",
            "pipeline_baseline": "ges",
        },
        "B6": {
            "name": "Full JCCE (Baseline)",
            "description": "Complete v16 framework with all components enabled",
            # No overrides — this is the baseline
        },
    }

    if ablation_id not in configs:
        raise ValueError(f"Unknown ablation: {ablation_id}. Available: {list(configs.keys())}")

    return configs[ablation_id]


# =============================================================================
# Pipeline Baselines (B3-B5)
# =============================================================================


def run_pipeline_baseline(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    feature_names: list,
    n_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run a traditional pipeline: structure learning → feature selection → classifier.

    Steps:
    1. Learn DAG structure (PC/NOTEARS/GES)
    2. Extract Markov Blanket from DAG
    3. Train LogisticRegression on MB features
    4. Evaluate with K-fold CV
    """
    from sklearn.preprocessing import StandardScaler

    if verbose:
        print(f"\n  Pipeline baseline: {method.upper()} + LogisticRegression")

    # Step 1: Learn DAG structure
    A_learned = None
    structure_time_start = time.time()

    if method == "pc":
        try:
            from causallearn.search.ConstraintBased.PC import pc

            result = pc(X, alpha=0.05, indep_test="fisherz")
            A_learned = result.G.graph  # Adjacency matrix
            # Convert CPDAG to DAG-like (take absolute values)
            A_learned = np.abs(A_learned).astype(float)
        except ImportError:
            warnings.warn("causal-learn not installed. Using correlation-based fallback.")
            A_learned = _correlation_dag(X)

    elif method == "notears":
        try:
            from notears.linear import notears_linear

            A_learned = notears_linear(X, lambda1=0.1, loss_type="l2")
            A_learned = (np.abs(A_learned) > 0.3).astype(float)
        except ImportError:
            try:
                # Try jcce_learner's built-in NOTEARS-like
                import jax.numpy as jnp
                from jax import random

                from jcce.structure_learning.jcce_learner import (
                    _learn_structure_legacy,
                    create_processor,
                )

                # Simple GOLEM run with high sparsity (NOTEARS-like)
                n_vars = X.shape[1]
                Y_aug = np.zeros(X.shape[0])  # Dummy Y
                data = np.column_stack([X, Y_aug]).astype(np.float32)
                processor = create_processor("elm", key=random.PRNGKey(42), n_features=n_vars)
                A_est, _, _, _ = _learn_structure_legacy(
                    data=jnp.array(data),
                    Y=jnp.array(Y_aug),
                    Y_idx=n_vars,
                    processor=processor,
                    key=random.PRNGKey(42),
                    lambda_1=0.1,
                    lambda_class=0.0,  # No classification loss
                    max_iter=200,
                    verbose=False,
                )
                A_learned = np.array(A_est[:n_vars, :n_vars])
            except Exception:
                warnings.warn("NOTEARS not available. Using correlation-based fallback.")
                A_learned = _correlation_dag(X)

    elif method == "ges":
        try:
            from causallearn.search.ScoreBased.GES import ges

            result = ges(X, score_func="local_score_BIC")
            A_learned = result["G"].graph
            A_learned = np.abs(A_learned).astype(float)
        except ImportError:
            warnings.warn("causal-learn not installed. Using correlation-based fallback.")
            A_learned = _correlation_dag(X)

    structure_time = time.time() - structure_time_start

    # Step 2: Extract MB (parents + children of last variable = proxy for Y)
    n_vars = X.shape[1]
    # Use features connected to any variable as MB proxy
    # Since we don't have Y in the DAG, use top correlated features
    if A_learned is not None and A_learned.shape[0] >= n_vars:
        # Sum of incoming + outgoing edges per variable
        edge_importance = np.sum(np.abs(A_learned[:n_vars, :n_vars]), axis=0) + np.sum(
            np.abs(A_learned[:n_vars, :n_vars]), axis=1
        )
        # Select features with any edge
        mb_indices = np.where(edge_importance > 0)[0].tolist()
        if len(mb_indices) == 0:
            # Fallback: use all features
            mb_indices = list(range(n_vars))
    else:
        mb_indices = list(range(n_vars))

    if verbose:
        print(f"    Structure learning: {structure_time:.1f}s")
        print(f"    MB: {len(mb_indices)} features selected")

    # Step 3+4: K-fold CV with LogisticRegression on MB features
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    acc_folds, f1_folds, bacc_folds, auc_folds = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, Y)):
        X_train, X_test = X[train_idx][:, mb_indices], X[test_idx][:, mb_indices]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Standardize per fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)
        Y_prob = (
            clf.predict_proba(X_test)[:, 1] if len(clf.classes_) == 2 else clf.predict_proba(X_test)
        )

        acc_folds.append(accuracy_score(Y_test, Y_pred))
        f1_folds.append(f1_score(Y_test, Y_pred, zero_division=0))
        bacc_folds.append(balanced_accuracy_score(Y_test, Y_pred))
        try:
            auc_folds.append(roc_auc_score(Y_test, Y_prob))
        except ValueError:
            auc_folds.append(0.5)

    result = {
        "method": f"{method}+logreg",
        "accuracy_mean": float(np.mean(acc_folds)),
        "accuracy_std": float(np.std(acc_folds)),
        "f1_mean": float(np.mean(f1_folds)),
        "f1_std": float(np.std(f1_folds)),
        "balanced_acc_mean": float(np.mean(bacc_folds)),
        "balanced_acc_std": float(np.std(bacc_folds)),
        "roc_auc_mean": float(np.mean(auc_folds)),
        "roc_auc_std": float(np.std(auc_folds)),
        "mb_size": len(mb_indices),
        "mb_indices": mb_indices,
        "structure_time": structure_time,
        "n_edges": int(np.sum(A_learned > 0)) if A_learned is not None else 0,
        "n_folds": n_folds,
    }

    if verbose:
        print(f"    Results ({n_folds}-fold CV):")
        print(f"      Accuracy:     {result['accuracy_mean']:.4f} +/- {result['accuracy_std']:.4f}")
        print(f"      F1:           {result['f1_mean']:.4f} +/- {result['f1_std']:.4f}")
        print(
            f"      Balanced Acc: {result['balanced_acc_mean']:.4f} +/- {result['balanced_acc_std']:.4f}"
        )
        print(f"      ROC AUC:      {result['roc_auc_mean']:.4f} +/- {result['roc_auc_std']:.4f}")

    return result


def _correlation_dag(X: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Fallback: create pseudo-DAG from correlation matrix."""
    corr = np.corrcoef(X.T)
    A = (np.abs(corr) > threshold).astype(float)
    np.fill_diagonal(A, 0)
    # Make it a DAG (upper triangular)
    A = np.triu(A, k=1)
    return A


import warnings

# =============================================================================
# CF Checkpoint Save/Load (for --rerun-cf without re-running NSGA-II)
# Shared implementations in jcce/validation/counterfactual_runner.py
# =============================================================================
from jcce.validation.counterfactual_runner import (
    convert_params_to_numpy as _convert_params_to_numpy,
)
from jcce.validation.counterfactual_runner import (
    load_cf_checkpoint as _load_cf_checkpoint,
)
from jcce.validation.counterfactual_runner import (
    reconstruct_processor as _reconstruct_processor_impl,
)
from jcce.validation.counterfactual_runner import (
    run_counterfactual_evaluation as _run_counterfactual_evaluation,
)
from jcce.validation.counterfactual_runner import (
    save_cf_checkpoint as _save_cf_checkpoint,
)


def _select_diverse_posthoc(pareto_solutions: list, n_max: int = 5) -> list:
    """Select diverse Pareto solutions for expensive post-hoc validation (CV, CF).

    Returns list of indices into pareto_solutions, choosing:
    1. Best balanced accuracy
    2. Sparsest with reasonable acc (>=60%)
    3. Most edges with reasonable acc
    4+. Fill with diverse edge counts
    """
    if not pareto_solutions:
        return []
    if len(pareto_solutions) <= n_max:
        return list(range(len(pareto_solutions)))

    selected = []
    used = set()

    # 1. Best accuracy
    best_idx = max(
        range(len(pareto_solutions)), key=lambda i: pareto_solutions[i].get("balanced_accuracy", 0)
    )
    selected.append(best_idx)
    used.add(best_idx)

    # 2. Sparsest with reasonable accuracy
    candidates = [
        (i, s)
        for i, s in enumerate(pareto_solutions)
        if i not in used and s.get("balanced_accuracy", 0) >= 0.60
    ]
    if candidates:
        sparse = min(candidates, key=lambda x: x[1].get("n_edges", 999))
        selected.append(sparse[0])
        used.add(sparse[0])

    # 3. Most edges (most complex structure)
    candidates = [
        (i, s)
        for i, s in enumerate(pareto_solutions)
        if i not in used and s.get("balanced_accuracy", 0) >= 0.60
    ]
    if candidates:
        dense = max(candidates, key=lambda x: x[1].get("n_edges", 0))
        selected.append(dense[0])
        used.add(dense[0])

    # 4+. Fill with diverse edge counts
    edge_counts = {pareto_solutions[i].get("n_edges", 0) for i in selected}
    remaining = sorted(
        [(i, s) for i, s in enumerate(pareto_solutions) if i not in used],
        key=lambda x: x[1].get("balanced_accuracy", 0),
        reverse=True,
    )
    for idx, sol in remaining:
        if len(selected) >= n_max:
            break
        n_edges = sol.get("n_edges", 0)
        if n_edges not in edge_counts:
            selected.append(idx)
            used.add(idx)
            edge_counts.add(n_edges)

    # If still not enough, add remaining by accuracy
    for idx, sol in remaining:
        if len(selected) >= n_max:
            break
        if idx not in used:
            selected.append(idx)

    return selected


def _reconstruct_processor(processor_type, processor_config, processor_params_numpy):
    """Backwards-compatible wrapper around shared reconstruct_processor."""
    return _reconstruct_processor_impl(processor_type, processor_config, processor_params_numpy)


# =============================================================================
# Main Ablation Runner
# =============================================================================


def run_ablation(
    ablation_id: str,
    dataset_name: str,
    quick: bool = False,
    verbose: bool = True,
    n_gen_override: int = None,
    pop_size_override: int = None,
    extra_golem_overrides: dict = None,
) -> dict:
    """
    Run a single ablation experiment.

    Returns dict with all results.
    """
    abl_config = get_ablation_config(ablation_id)

    if verbose:
        print("=" * 80)
        print(f"ABLATION {ablation_id}: {abl_config['name']}")
        print(f"Dataset: {dataset_name}")
        print(f"Description: {abl_config['description']}")
        print("=" * 80)

    # Load dataset
    X, Y, ds_config = load_dataset(dataset_name)

    if verbose:
        print(f"\nData: X{X.shape}, Y{Y.shape}, balance: {Y.mean():.3f}")

    # Pipeline baselines (B3-B5) bypass NSGA-II entirely
    if "pipeline_baseline" in abl_config:
        method = abl_config["pipeline_baseline"]
        n_folds = 3 if X.shape[0] < 500 else 5
        start = time.time()
        result = run_pipeline_baseline(
            X,
            Y,
            method=method,
            feature_names=ds_config["feature_names"],
            n_folds=n_folds,
            verbose=verbose,
        )
        result["ablation_id"] = ablation_id
        result["ablation_name"] = abl_config["name"]
        result["dataset"] = dataset_name
        result["total_time"] = time.time() - start
        result["is_pipeline_baseline"] = True
        return result

    # v16 sample splitting for DML
    X_struct, X_effect, Y_struct, Y_effect = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=42
    )

    if verbose:
        print(
            f"Sample split: {X_struct.shape[0]} structure (NSGA-II) / "
            f"{X_effect.shape[0]} effect (DML+CV validation)"
        )

    # Determine NSGA-II config
    if quick:
        pop_size = ds_config.get("quick_pop_size", 6)
        n_gen = ds_config.get("quick_n_gen", 3)
        max_iter = ds_config.get("quick_max_iter", 30)
        processors = ["elm", "mlp"]
    else:
        pop_size = ds_config.get("pop_size", 20)
        n_gen = ds_config.get("n_gen", 30)
        max_iter = ds_config.get("max_iter", 100)
        processors = None  # Use all

    # Apply CLI overrides (useful for server tuning)
    if n_gen_override is not None:
        n_gen = n_gen_override
    if pop_size_override is not None:
        pop_size = pop_size_override

    # Apply processor override (B1)
    if "processor_override" in abl_config:
        original_processors = nsga2_module.PROCESSOR_TYPES
        nsga2_module.PROCESSOR_TYPES = abl_config["processor_override"]

    # Random search (B2): use large pop, 1 generation (no evolution)
    if abl_config.get("random_search", False):
        pop_size = pop_size * n_gen  # Total budget as single generation
        n_gen = 1
        if verbose:
            print(f"  Random search: pop={pop_size}, gen=1 (no evolution)")

    # Build NSGA-II kwargs
    nsga2_kwargs = {
        "X": jnp.array(X_struct),
        "Y": jnp.array(Y_struct),
        "n_vars": X.shape[1],
        "true_mb": ds_config.get("true_mb"),
        "pop_size": pop_size,
        "n_generations": n_gen,
        "stage1_max_iter": max_iter,
        "jax_key_seed": 42,
        "verbose": verbose,
        "use_v7": True,
        "use_condition_constraint": True,
        "condition_threshold": 100.0,
        "task": ds_config.get("task", "classification"),
        # Phase 3: Algorithmic optimizations
        "enable_warm_start": True,
        "warm_start_prob": 0.3,
    }

    # Apply NSGA-II overrides (A4)
    if "nsga2_overrides" in abl_config:
        nsga2_kwargs.update(abl_config["nsga2_overrides"])

    # Apply processor list
    if processors and "processor_override" not in abl_config:
        original_processors = nsga2_module.PROCESSOR_TYPES
        nsga2_module.PROCESSOR_TYPES = processors

    golem_overrides = abl_config.get("golem_overrides", {})

    # Merge extra overrides (from CLI --golem-override or programmatic call)
    if extra_golem_overrides:
        golem_overrides.update(extra_golem_overrides)

    # Pass GOLEM overrides to NSGA-II (threaded through to evaluate_genome → GOLEM)
    nsga2_kwargs["golem_overrides"] = golem_overrides

    # Run NSGA-II
    start_time = time.time()

    try:
        results = run_nsga2(**nsga2_kwargs)
    finally:
        # Restore processors
        if "processor_override" in abl_config or processors:
            nsga2_module.PROCESSOR_TYPES = ["elm", "mlp", "gnn", "transformer", "mamba"]

    nsga2_time = time.time() - start_time
    pareto = results["pareto_front"]
    enhanced_sols = results.get("enhanced_solutions", [])

    if verbose:
        print(f"\n  NSGA-II completed in {nsga2_time:.1f}s")
        print(f"  Pareto front: {len(pareto)} solutions")

    # ===================================================================
    # Build Pareto solutions list + keep raw metrics for post-hoc
    # ===================================================================
    pareto_solutions_data = []
    raw_sol_metrics = []  # parallel list — raw enhanced_sol metrics (with _processor etc.)
    for sol_data in enhanced_sols:
        m = sol_data.get("metrics", {}) if isinstance(sol_data, dict) else sol_data
        if m is None or "structure_A_est" not in m:
            continue
        raw_sol_metrics.append(m)
        sol_entry = {
            "A_est": np.array(m["structure_A_est"]),
            "A_weights": np.array(m["A_weights"]) if m.get("A_weights") is not None else None,
            "A_confound": np.array(m["A_confound"]) if m.get("A_confound") is not None else None,
            "A_confound_weights": np.array(m["A_confound_weights"])
            if m.get("A_confound_weights") is not None
            else None,
            "processor_type": m.get("processor_type", "?"),
            "processor_config": m.get("processor_config", {}),
            "mb_indices": m.get("mb_indices", []),
            "markov_blanket": m.get("markov_blanket", []),
            "causal_effects": m.get("causal_effects", {}),
            "n_confound_edges": m.get("n_confound_edges", 0),
            "n_edges": m.get("structure_n_edges", 0),
            "h_A": m.get("structure_h_A", 0),
            "accuracy": m.get("classification_accuracy", 0),
            "balanced_accuracy": m.get("classification_balanced_accuracy", 0),
            "f1": m.get("classification_f1", 0),
            "precision": m.get("classification_precision", 0),
            "recall": m.get("classification_recall", 0),
            "roc_auc": m.get("classification_roc_auc", 0),
            # Structure recovery metrics (available when ground truth exists, e.g. LUCAS)
            "structure_shd": m.get("structure_shd"),
            "structure_sid": m.get("structure_sid"),
            "structure_edge_f1": m.get("structure_edge_f1"),
            "structure_edge_precision": m.get("structure_edge_precision"),
            "structure_edge_recall": m.get("structure_edge_recall"),
            "hyperparams": {
                "lambda_1": m.get("lambda_1"),
                "lambda_2": m.get("lambda_2"),
                "lambda_class": m.get("lambda_class"),
                "lr": m.get("lr"),
            },
        }
        if m.get("_processor_params") is not None:
            try:
                sol_entry["processor_params"] = _convert_params_to_numpy(m["_processor_params"])
            except Exception:
                pass
        pareto_solutions_data.append(sol_entry)

    # Find best solution by balanced accuracy
    best_pareto_idx = -1
    best_bacc = -1
    for i, m in enumerate(raw_sol_metrics):
        bacc = m.get("classification_balanced_accuracy", m.get("balanced_accuracy", 0.0))
        if bacc > best_bacc:
            best_bacc = bacc
            best_pareto_idx = i
    best_sol_metrics = raw_sol_metrics[best_pareto_idx] if best_pareto_idx >= 0 else None

    # Select diverse solutions for expensive post-hoc (CV, CF)
    diverse_indices = _select_diverse_posthoc(pareto_solutions_data, n_max=5)
    if best_pareto_idx >= 0 and best_pareto_idx not in diverse_indices:
        diverse_indices.insert(0, best_pareto_idx)

    if verbose:
        print("\n  Post-hoc pipeline:")
        print(f"    DML: all {len(raw_sol_metrics)} Pareto solutions")
        print(f"    CV+CF: {len(diverse_indices)} diverse solutions (indices: {diverse_indices})")
        if best_pareto_idx >= 0:
            print(f"    Best solution: index {best_pareto_idx} (BAcc={best_bacc:.4f})")

    # ===================================================================
    # P3: Multi-Parent DML on ALL Pareto solutions (cheap — no retraining)
    # ===================================================================
    if verbose:
        print(f"\n  P3: Running DML on {len(raw_sol_metrics)} solutions...")
    try:
        from jcce.validation.multi_parent_dml import run_multi_parent_dml
    except ImportError:
        run_multi_parent_dml = None

    if run_multi_parent_dml is not None:
        for i, m in enumerate(raw_sol_metrics):
            if m.get("structure_A_est") is None:
                continue
            try:
                A_est_dml = np.array(m["structure_A_est"])
                Y_idx_dml = A_est_dml.shape[0] - 1
                dml_result = run_multi_parent_dml(
                    X=X_effect,
                    Y=Y_effect,
                    A_est=A_est_dml,
                    Y_idx=Y_idx_dml,
                    feature_names=ds_config.get("feature_names"),
                    known_treatment_idx=ds_config.get("treatment_idx"),
                    parent_threshold=0.01,
                    n_dml_folds=5 if X_effect.shape[0] >= 500 else 3,
                    run_refutation=not quick,
                    n_refutation_sims=50 if quick else 100,
                    verbose=(verbose and i == best_pareto_idx),
                )
                pareto_solutions_data[i]["dml"] = dml_result.to_dict()
                pareto_solutions_data[i]["dml_n_parents"] = dml_result.n_parents_discovered
                pareto_solutions_data[i]["dml_n_significant"] = dml_result.n_parents_significant
                if verbose:
                    sig = dml_result.n_parents_significant
                    total = dml_result.n_parents_discovered
                    star = " *" if i == best_pareto_idx else ""
                    print(f"    Sol {i}: DML {sig}/{total} significant{star}")
            except Exception as e:
                if verbose:
                    print(f"    Sol {i}: DML failed: {e}")

    # ===================================================================
    # P3b: All-Edges DML on ALL Pareto solutions
    # ===================================================================
    if verbose:
        print(f"\n  P3b: Running all-edges DML on {len(raw_sol_metrics)} solutions...")
    try:
        from jcce.validation.all_edges_dml import run_all_edges_dml as run_all_edges_dml_fn
    except ImportError:
        run_all_edges_dml_fn = None

    if run_all_edges_dml_fn is not None:
        for i, m in enumerate(raw_sol_metrics):
            if m.get("structure_A_est") is None:
                continue
            try:
                A_est_ae = np.array(m["structure_A_est"])
                Y_idx_ae = A_est_ae.shape[0] - 1
                ae_result = run_all_edges_dml_fn(
                    X=X_effect,
                    Y=Y_effect,
                    A_est=A_est_ae,
                    Y_idx=Y_idx_ae,
                    feature_names=ds_config.get("feature_names"),
                    edge_threshold=0.01,
                    n_dml_folds=5 if X_effect.shape[0] >= 500 else 3,
                    run_refutation=not quick,
                    n_refutation_sims=50 if quick else 100,
                    verbose=(verbose and i == best_pareto_idx),
                )
                if ae_result is not None:
                    pareto_solutions_data[i]["all_edges_dml"] = ae_result.to_storage_dict()
                    pareto_solutions_data[i]["dml_causal_effects"] = (
                        ae_result.to_causal_effects_dict()
                    )
                    if verbose:
                        sig = ae_result.n_significant_fdr
                        total = ae_result.n_edges
                        star = " *" if i == best_pareto_idx else ""
                        print(f"    Sol {i}: All-edges DML {sig}/{total} significant (FDR){star}")
            except Exception as e:
                if verbose:
                    print(f"    Sol {i}: All-edges DML failed: {e}")

    # ===================================================================
    # P2: K-fold CV on diverse solutions (expensive — retrains GOLEM)
    # ===================================================================
    n_folds = 3 if X_effect.shape[0] < 500 else 5
    if verbose:
        print(f"\n  P2: Running {n_folds}-fold CV on {len(diverse_indices)} diverse solutions...")

    for i in diverse_indices:
        m = raw_sol_metrics[i]
        hyperparams = {
            "lambda_1": m.get("lambda_1", 0.02),
            "lambda_2": m.get("lambda_2", 0.01),
            "lambda_class": m.get("lambda_class", 1.0),
            "lr": m.get("lr", 0.001),
            "processor_config": m.get("processor_config", {}),
        }
        for key in [
            "effect_hidden_dim",
            "effect_embed_dim",
            "lambda_effect",
            "effect_warmup_iter",
            "lambda_confound_sparse",
            "lambda_bow",
        ]:
            if key in m:
                hyperparams[key] = m[key]

        A_init = m.get("structure_A_est", None)
        if A_init is None:
            continue
        try:
            cv_result = evaluate_pareto_solution_cv(
                X=X_effect,
                Y=Y_effect,
                hyperparams=hyperparams,
                processor_type=m.get("processor_type", "elm"),
                A_init=np.array(A_init),
                processor_params_init=m.get("_processor_params"),
                n_folds=n_folds,
                true_mb=ds_config.get("true_mb"),
                golem_max_iter=100 if not quick else 20,
                cold_start=False,
                task=ds_config.get("task", "classification"),
                verbose=(verbose and i == best_pareto_idx),
                use_v7=True,
                freeze_structure=True,
            )
            pareto_solutions_data[i]["cv"] = {
                "accuracy_mean": cv_result.accuracy_mean,
                "accuracy_std": cv_result.accuracy_std,
                "precision_mean": cv_result.precision_mean,
                "precision_std": cv_result.precision_std,
                "recall_mean": cv_result.recall_mean,
                "recall_std": cv_result.recall_std,
                "f1_mean": cv_result.f1_mean,
                "f1_std": cv_result.f1_std,
                "balanced_acc_mean": cv_result.balanced_acc_mean,
                "balanced_acc_std": cv_result.balanced_acc_std,
                "roc_auc_mean": cv_result.roc_auc_mean,
                "roc_auc_std": cv_result.roc_auc_std,
                "mb_jaccard_mean": cv_result.mb_jaccard_mean,
            }
            if cv_result.mb_f1_per_fold:
                pareto_solutions_data[i]["cv"]["mb_f1_mean"] = float(
                    np.mean(cv_result.mb_f1_per_fold)
                )
            if verbose:
                star = " *" if i == best_pareto_idx else ""
                print(
                    f"    Sol {i}: CV BAcc={cv_result.balanced_acc_mean:.4f}"
                    f"+-{cv_result.balanced_acc_std:.4f}{star}"
                )
        except Exception as e:
            if verbose:
                print(f"    Sol {i}: CV failed: {e}")

    # Re-select best solution by CV BAcc (not training BAcc) when CV available
    cv_best_idx = -1
    cv_best_bacc = -1
    for i, ps in enumerate(pareto_solutions_data):
        cv_data = ps.get("cv", {})
        cv_bacc = cv_data.get("balanced_acc_mean", -1)
        if cv_bacc > cv_best_bacc:
            cv_best_bacc = cv_bacc
            cv_best_idx = i
    if cv_best_idx >= 0 and cv_best_bacc > 0:
        if verbose and cv_best_idx != best_pareto_idx:
            print(
                f"    Best by CV BAcc: idx={cv_best_idx} (CV={cv_best_bacc:.4f}) "
                f"replaces idx={best_pareto_idx} (train={best_bacc:.4f})"
            )
        best_pareto_idx = cv_best_idx
        best_sol_metrics = raw_sol_metrics[best_pareto_idx]

    # ===================================================================
    # P4: Counterfactual evaluation on diverse solutions
    # ===================================================================
    ckpt_dir = RESULTS_DIR / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""

    if verbose:
        print(f"\n  P4: Running CF on {len(diverse_indices)} diverse solutions...")

    for i in diverse_indices:
        m = raw_sol_metrics[i]
        if (
            m.get("structure_A_est") is None
            or m.get("_processor") is None
            or m.get("_processor_params") is None
        ):
            continue

        # Save CF checkpoint (best solution uses default name for --rerun-cf compat)
        ckpt_suffix = "" if i == best_pareto_idx else f"_sol{i}"
        ckpt_path = ckpt_dir / f"{ablation_id}{suffix}_cf_checkpoint{ckpt_suffix}.pkl"
        try:
            _save_cf_checkpoint(
                checkpoint_path=ckpt_path,
                X=X_struct,
                Y=Y_struct,
                A_est=np.array(m["structure_A_est"]),
                A_weights=np.array(m["A_weights"]) if m.get("A_weights") is not None else None,
                A_confound_weights=np.array(m["A_confound_weights"])
                if m.get("A_confound_weights") is not None
                else None,
                processor_type=m.get("processor_type", "mlp"),
                processor_config=m.get("processor_config", {}),
                processor_params=m["_processor_params"],
                ds_config=ds_config,
            )
        except Exception as e:
            if verbose:
                print(f"    Sol {i}: CF checkpoint save failed: {e}")

        try:
            cf_result = _run_counterfactual_evaluation(
                X=X_struct,
                Y=Y_struct,
                A_est=np.array(m["structure_A_est"]),
                processor_trained=m["_processor"],
                processor_params=m["_processor_params"],
                ds_config=ds_config,
                A_weights=np.array(m["A_weights"]) if m.get("A_weights") is not None else None,
                A_confound=np.array(m["A_confound_weights"])
                if m.get("A_confound_weights") is not None
                else None,
                n_instances=30 if not quick else 10,
                cf_pop_size=50 if not quick else 20,
                cf_n_gen=50 if not quick else 15,
                verbose=(verbose and i == best_pareto_idx),
            )
            if cf_result:
                pareto_solutions_data[i]["cf"] = cf_result
                if verbose:
                    star = " *" if i == best_pareto_idx else ""
                    print(
                        f"    Sol {i}: CF {cf_result['n_valid']}/{cf_result['n_instances']} valid "
                        f"(sparsity={cf_result['avg_sparsity']:.1f}){star}"
                    )
        except Exception as e:
            if verbose:
                print(f"    Sol {i}: CF failed: {e}")
                import traceback

                traceback.print_exc()

    # ===================================================================
    # P5: Structural validation (on best Pareto solution)
    # ===================================================================
    structural_validation = {}
    if best_pareto_idx >= 0 and best_sol_metrics is not None:
        if verbose:
            print("\n  P5: Running structural validation on best solution...")
        try:
            from jcce.structure_learning.experiment_runner import _run_structural_validation

            # Build enhanced_solutions list (just best) for the API
            _enhanced_sols = [
                {
                    "metrics": best_sol_metrics,
                }
            ]

            # Build dml_results list for Cinelli
            _dml_results = []
            best_ps = (
                pareto_solutions_data[best_pareto_idx]
                if best_pareto_idx < len(pareto_solutions_data)
                else {}
            )
            if "dml" in best_ps:
                _dml_results = [best_ps["dml"]]

            structural_validation = _run_structural_validation(
                enhanced_solutions=_enhanced_sols,
                X=np.array(X_struct),
                Y=np.array(Y_struct),
                n_vars=X_struct.shape[1],
                max_iter=max_iter,
                dml_results=_dml_results,
                bootstrap_B=20 if quick else 50,
                verbose=verbose,
            )
            if verbose and structural_validation:
                if (
                    "bootstrap" in structural_validation
                    and "error" not in structural_validation["bootstrap"]
                ):
                    bs = structural_validation["bootstrap"]
                    print(f"    Bootstrap: {bs.get('n_stable_edges_80', '?')} stable edges")
                if "lovo" in structural_validation and "error" not in structural_validation["lovo"]:
                    lv = structural_validation["lovo"]
                    print(f"    LOVO: {lv['stability_score']:.1%} stability")
        except Exception as e:
            if verbose:
                print(f"    Structural validation failed: {e}")
                import traceback

                traceback.print_exc()

    # ===================================================================
    # Build result dict
    # ===================================================================
    result = {
        "ablation_id": ablation_id,
        "ablation_name": abl_config["name"],
        "ablation_description": abl_config["description"],
        "dataset": dataset_name,
        "dataset_config": {k: v for k, v in ds_config.items() if k not in ("true_dag",)},
        "is_pipeline_baseline": False,
        "quick_mode": quick,
        "nsga2_time": nsga2_time,
        "total_time": time.time() - start_time,
        "n_pareto_solutions": len(pareto),
        "pareto_front": pareto,
        "golem_overrides": golem_overrides,
        # Best solution training metrics
        "best_training_balanced_acc": best_bacc,
        "best_pareto_idx": best_pareto_idx,
        "best_processor": best_sol_metrics.get("processor_type", "?") if best_sol_metrics else "?",
        "best_mb_size": best_sol_metrics.get("mb_size", 0) if best_sol_metrics else 0,
        "best_n_edges": best_sol_metrics.get("structure_n_edges", 0) if best_sol_metrics else 0,
        "best_h_A": best_sol_metrics.get("structure_h_A", 0) if best_sol_metrics else 0,
        "diverse_indices": diverse_indices,
        # Structural validation
        "structural_validation": structural_validation,
    }

    # Save full model artifacts for best solution (backwards compat)
    if best_sol_metrics:
        result["model"] = {
            "A_est": np.array(best_sol_metrics["structure_A_est"]),
            "A_weights": np.array(best_sol_metrics["A_weights"])
            if best_sol_metrics.get("A_weights") is not None
            else None,
            "A_confound": np.array(best_sol_metrics["A_confound"])
            if best_sol_metrics.get("A_confound") is not None
            else None,
            "A_confound_weights": np.array(best_sol_metrics["A_confound_weights"])
            if best_sol_metrics.get("A_confound_weights") is not None
            else None,
            "processor_type": best_sol_metrics.get("processor_type", "mlp"),
            "processor_config": best_sol_metrics.get("processor_config", {}),
            "processor_params": _convert_params_to_numpy(best_sol_metrics["_processor_params"]),
            "mb_indices": best_sol_metrics.get("mb_indices", []),
            "markov_blanket": best_sol_metrics.get("markov_blanket", []),
            "causal_effects": best_sol_metrics.get("causal_effects", {}),
            "n_confound_edges": best_sol_metrics.get("n_confound_edges", 0),
            "hyperparams": {
                "lambda_1": best_sol_metrics.get("lambda_1"),
                "lambda_2": best_sol_metrics.get("lambda_2"),
                "lambda_class": best_sol_metrics.get("lambda_class"),
                "lr": best_sol_metrics.get("lr"),
            },
        }

    # Store all Pareto solutions (with per-solution DML/CV/CF results attached)
    result["pareto_solutions"] = pareto_solutions_data

    # Backwards-compatible top-level CV/DML/CF keys from best solution
    if best_pareto_idx >= 0 and best_pareto_idx < len(pareto_solutions_data):
        best_ps = pareto_solutions_data[best_pareto_idx]

        if "cv" in best_ps:
            cv = best_ps["cv"]
            result["cv_accuracy_mean"] = cv["accuracy_mean"]
            result["cv_accuracy_std"] = cv["accuracy_std"]
            result["cv_f1_mean"] = cv["f1_mean"]
            result["cv_f1_std"] = cv["f1_std"]
            result["cv_balanced_acc_mean"] = cv["balanced_acc_mean"]
            result["cv_balanced_acc_std"] = cv["balanced_acc_std"]
            result["cv_roc_auc_mean"] = cv["roc_auc_mean"]
            result["cv_roc_auc_std"] = cv["roc_auc_std"]
            result["cv_mb_jaccard"] = cv.get("mb_jaccard_mean")
            if "mb_f1_mean" in cv:
                result["cv_mb_f1_mean"] = cv["mb_f1_mean"]

        if "dml" in best_ps:
            result["dml_n_parents"] = best_ps["dml_n_parents"]
            result["dml_n_significant"] = best_ps["dml_n_significant"]
            result["dml_parent_effects"] = best_ps["dml"]

        if "cf" in best_ps:
            cf = best_ps["cf"]
            result["cf_n_instances"] = cf["n_instances"]
            result["cf_n_valid"] = cf["n_valid"]
            result["cf_validity_rate"] = cf["validity_rate"]
            result["cf_avg_sparsity"] = cf["avg_sparsity"]
            result["cf_avg_distance"] = cf["avg_distance"]
            result["cf_avg_plausibility_ratio"] = cf["avg_plausibility_ratio"]
            result["cf_n_plausible"] = cf["n_plausible"]
            result["cf_avg_causal_validity"] = cf["avg_causal_validity"]
            result["cf_details"] = cf

    if verbose:
        print(f"\n  {'=' * 60}")
        print(f"  ABLATION {ablation_id} RESULTS: {abl_config['name']}")
        print(f"  {'=' * 60}")
        print(f"  Pareto solutions: {len(pareto_solutions_data)}")
        print(f"  Best solution (idx {best_pareto_idx}): BAcc={best_bacc:.4f}")
        # Per-solution summary
        for i, ps in enumerate(pareto_solutions_data):
            star = " *" if i == best_pareto_idx else ""
            parts = [f"Sol {i}{star}: BAcc={ps['balanced_accuracy']:.3f}"]
            parts.append(f"edges={ps['n_edges']}")
            parts.append(f"|MB|={len(ps['mb_indices'])}")
            parts.append(f"proc={ps['processor_type']}")
            if "dml_n_significant" in ps:
                parts.append(f"DML={ps['dml_n_significant']}/{ps['dml_n_parents']}")
            if "cv" in ps:
                parts.append(f"CV={ps['cv']['balanced_acc_mean']:.3f}")
            if "cf" in ps:
                parts.append(f"CF={ps['cf']['n_valid']}/{ps['cf']['n_instances']}")
            print(f"    {', '.join(parts)}")
        print(f"  Total time: {result['total_time']:.1f}s")

    return result


# =============================================================================
# Summary Table
# =============================================================================


def print_summary_table(all_results: list):
    """Print a comparison table of all ablation results."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY SUMMARY")
    print("=" * 100)

    # Header
    print(
        f"{'ID':<4} {'Name':<25} {'Dataset':<12} {'CV BAcc':<16} {'CV AUC':<16} "
        f"{'MB':<5} {'DML':<7} {'CF':<10} {'Time':<8}"
    )
    print("-" * 120)

    for r in sorted(all_results, key=lambda x: x["ablation_id"]):
        ablation_id = r["ablation_id"]
        name = r.get("ablation_name", r.get("method", "?"))[:24]
        dataset = r["dataset"][:11]

        if r.get("is_pipeline_baseline", False):
            bacc = f"{r['balanced_acc_mean']:.4f}+/-{r['balanced_acc_std']:.4f}"
            auc = f"{r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f}"
            mb = str(r.get("mb_size", "?"))
        elif "cv_balanced_acc_mean" in r:
            bacc = f"{r['cv_balanced_acc_mean']:.4f}+/-{r['cv_balanced_acc_std']:.4f}"
            auc = f"{r['cv_roc_auc_mean']:.4f}+/-{r['cv_roc_auc_std']:.4f}"
            mb = str(r.get("best_mb_size", "?"))
        else:
            bacc = f"{r.get('best_training_balanced_acc', 0):.4f} (train)"
            auc = "N/A"
            mb = str(r.get("best_mb_size", "?"))

        # DML column
        if "dml_n_parents" in r:
            dml = f"{r['dml_n_significant']}/{r['dml_n_parents']}"
        else:
            dml = "-"

        # CF column: valid/total (sparsity)
        if "cf_n_valid" in r:
            cf = f"{r['cf_n_valid']}/{r['cf_n_instances']}"
        else:
            cf = "-"

        t = f"{r.get('total_time', 0):.0f}s"
        print(
            f"{ablation_id:<4} {name:<25} {dataset:<12} {bacc:<16} {auc:<16} "
            f"{mb:<5} {dml:<7} {cf:<10} {t:<8}"
        )

    print("=" * 120)
    print("DML: significant_parents / total_parents | CF: valid_CFs / total_instances")


# =============================================================================
# Main
# =============================================================================

ALL_ABLATIONS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "B1", "B2", "B3", "B4", "B5", "B6"]

ALL_DATASETS = [
    "lucas",
    "diabetes",
    "heart_disease",
    "breast_cancer",
    "sachs",
    "asia",
    "child",
    "neuropathic_pain",
    "alarm",
    "insurance",
]


def main():
    parser = argparse.ArgumentParser(description="JCCE v16 Ablation Study")
    parser.add_argument("--ablation", type=str, nargs="+", help="Ablation ID(s) (A1-A7, B1-B6)")
    parser.add_argument("--dataset", type=str, nargs="+", default=["lucas"], help="Dataset name(s)")
    parser.add_argument("--all", action="store_true", help="Run all ablations")
    parser.add_argument("--all-datasets", action="store_true", help="Run on all datasets")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (tiny config)")
    parser.add_argument("--list", action="store_true", help="List available ablations")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--n-gen", type=int, default=None, help="Override n_generations (e.g., 20)")
    parser.add_argument(
        "--pop-size", type=int, default=None, help="Override population size (e.g., 15)"
    )
    parser.add_argument(
        "--rerun-cf",
        action="store_true",
        help="Re-run only counterfactual evaluation from saved checkpoint (no NSGA-II)",
    )
    parser.add_argument(
        "--rerun-cv",
        action="store_true",
        help="Re-run only K-fold CV from saved result pkl (no NSGA-II)",
    )
    parser.add_argument(
        "--rerun-dml",
        action="store_true",
        help="Re-run only DML validation from saved result pkl (no NSGA-II)",
    )
    parser.add_argument(
        "--rerun-all-edges-dml",
        action="store_true",
        help="Re-run all-edges DML from saved result pkl (no NSGA-II)",
    )
    parser.add_argument(
        "--rerun-posthoc",
        action="store_true",
        help="Re-run all post-hoc phases (DML + CV + CF) from saved result pkl",
    )
    parser.add_argument(
        "--sol",
        type=str,
        nargs="+",
        default=None,
        help='Solution index(es) or "all" (e.g., --sol 0 2 5 or --sol all)',
    )
    parser.add_argument(
        "--golem-override",
        action="append",
        default=[],
        help="Extra golem override: key=value (merged with ablation config)",
    )
    args = parser.parse_args()

    # Expand --rerun-posthoc into individual flags
    if args.rerun_posthoc:
        args.rerun_dml = True
        args.rerun_all_edges_dml = True
        args.rerun_cv = True
        args.rerun_cf = True

    # Parse --sol: convert "all" or integer strings to a list (or None)
    if args.sol is not None:
        if "all" in args.sol:
            args.sol = "all"  # Special sentinel
        else:
            args.sol = [int(s) for s in args.sol]

    # --rerun-dml mode: load result pkl, rerun DML, update results
    # Order matches main pipeline: DML (P3) → CV (P2) → CF (P4)
    if args.rerun_dml:
        from jcce.validation.multi_parent_dml import run_multi_parent_dml

        ablations = args.ablation or ["B6"]
        datasets = ALL_DATASETS if args.all_datasets else args.dataset
        for dataset_name in datasets:
            X_full, Y_full, ds_config = load_dataset(dataset_name)
            # Recreate the same 70/30 split used during training
            X_struct, X_effect, Y_struct, Y_effect = train_test_split(
                X_full, Y_full, test_size=0.3, stratify=Y_full, random_state=42
            )
            for ablation_id in ablations:
                suffix = "_quick" if args.quick else ""
                result_path = RESULTS_DIR / dataset_name / f"{ablation_id}{suffix}.pkl"

                if not result_path.exists():
                    print(f"  No result file: {result_path}")
                    continue

                with open(result_path, "rb") as f:
                    result = pickle.load(f)

                if "pareto_solutions" not in result:
                    print("  No pareto_solutions in result (old format?)")
                    continue

                best_idx = result.get("best_pareto_idx", 0)
                n_solutions = len(result["pareto_solutions"])
                if args.sol == "all":
                    sol_indices = list(range(n_solutions))
                elif args.sol is not None:
                    sol_indices = args.sol
                else:
                    sol_indices = [best_idx]

                print(f"\n  DML: {ablation_id} / {dataset_name}, solutions: {sol_indices}")

                for sol_idx in sol_indices:
                    if sol_idx >= n_solutions:
                        print(f"  Sol {sol_idx}: index out of range (max {n_solutions - 1})")
                        continue

                    ps = result["pareto_solutions"][sol_idx]
                    A_est_dml = np.array(ps["A_est"])
                    Y_idx_dml = A_est_dml.shape[0] - 1

                    try:
                        dml_result = run_multi_parent_dml(
                            X=X_effect,
                            Y=Y_effect,
                            A_est=A_est_dml,
                            Y_idx=Y_idx_dml,
                            parent_threshold=0.01,
                            n_dml_folds=5 if X_effect.shape[0] >= 500 else 3,
                            run_refutation=not args.quick,
                            n_refutation_sims=50 if args.quick else 100,
                            verbose=True,
                        )
                        result["pareto_solutions"][sol_idx]["dml"] = dml_result.to_dict()
                        result["pareto_solutions"][sol_idx]["dml_n_parents"] = (
                            dml_result.n_parents_discovered
                        )
                        result["pareto_solutions"][sol_idx]["dml_n_significant"] = (
                            dml_result.n_parents_significant
                        )

                        if sol_idx == best_idx:
                            result["dml_parent_effects"] = dml_result.to_dict().get(
                                "parent_effects", {}
                            )

                        sig = dml_result.n_parents_significant
                        total = dml_result.n_parents_discovered
                        print(f"    Sol {sol_idx}: DML {sig}/{total} significant")
                    except Exception as e:
                        print(f"    Sol {sol_idx}: DML failed: {e}")
                        import traceback

                        traceback.print_exc()

                with open(result_path, "wb") as f:
                    pickle.dump(result, f)
                print(f"  Updated results: {result_path}")

        if not (args.rerun_all_edges_dml or args.rerun_cv or args.rerun_cf):
            return

    # --rerun-all-edges-dml mode: load result pkl, rerun all-edges DML, update results
    if getattr(args, "rerun_all_edges_dml", False):
        from jcce.validation.all_edges_dml import run_all_edges_dml

        ablations = args.ablation or ["B6"]
        datasets = ALL_DATASETS if args.all_datasets else args.dataset
        for dataset_name in datasets:
            X_full, Y_full, ds_config = load_dataset(dataset_name)
            X_struct, X_effect, Y_struct, Y_effect = train_test_split(
                X_full, Y_full, test_size=0.3, stratify=Y_full, random_state=42
            )
            for ablation_id in ablations:
                suffix = "_quick" if args.quick else ""
                result_path = RESULTS_DIR / dataset_name / f"{ablation_id}{suffix}.pkl"

                if not result_path.exists():
                    print(f"  No result file: {result_path}")
                    continue

                with open(result_path, "rb") as f:
                    result = pickle.load(f)

                if "pareto_solutions" not in result:
                    print("  No pareto_solutions in result (old format?)")
                    continue

                best_idx = result.get("best_pareto_idx", 0)
                n_solutions = len(result["pareto_solutions"])
                if args.sol == "all":
                    sol_indices = list(range(n_solutions))
                elif args.sol is not None:
                    sol_indices = args.sol
                else:
                    sol_indices = [best_idx]

                print(
                    f"\n  All-edges DML: {ablation_id} / {dataset_name}, solutions: {sol_indices}"
                )

                for sol_idx in sol_indices:
                    if sol_idx >= n_solutions:
                        print(f"  Sol {sol_idx}: index out of range (max {n_solutions - 1})")
                        continue

                    ps = result["pareto_solutions"][sol_idx]
                    A_est_dml = np.array(ps["A_est"])
                    Y_idx_dml = A_est_dml.shape[0] - 1

                    try:
                        all_edges_result = run_all_edges_dml(
                            X=X_effect,
                            Y=Y_effect,
                            A_est=A_est_dml,
                            Y_idx=Y_idx_dml,
                            feature_names=ds_config.get("feature_names"),
                            edge_threshold=0.01,
                            n_dml_folds=5 if X_effect.shape[0] >= 500 else 3,
                            run_refutation=not args.quick,
                            n_refutation_sims=50 if args.quick else 100,
                            verbose=True,
                        )
                        if all_edges_result is not None:
                            result["pareto_solutions"][sol_idx]["all_edges_dml"] = (
                                all_edges_result.to_storage_dict()
                            )
                            result["pareto_solutions"][sol_idx]["dml_causal_effects"] = (
                                all_edges_result.to_causal_effects_dict()
                            )
                            sig = all_edges_result.n_significant_fdr
                            total = all_edges_result.n_edges
                            print(
                                f"    Sol {sol_idx}: All-edges DML {sig}/{total} significant (FDR)"
                            )
                    except Exception as e:
                        print(f"    Sol {sol_idx}: All-edges DML failed: {e}")
                        import traceback

                        traceback.print_exc()

                with open(result_path, "wb") as f:
                    pickle.dump(result, f)
                print(f"  Updated results: {result_path}")

        if not (args.rerun_cv or args.rerun_cf):
            return

    # --rerun-cv mode: load result pkl, reconstruct solution, run CV, update results
    if args.rerun_cv:
        from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

        ablations = args.ablation or ["B6"]
        datasets = ALL_DATASETS if args.all_datasets else args.dataset
        for dataset_name in datasets:
            X_full, Y_full, ds_config = load_dataset(dataset_name)
            X_struct, X_effect, Y_struct, Y_effect = train_test_split(
                X_full, Y_full, test_size=0.3, stratify=Y_full, random_state=42
            )
            for ablation_id in ablations:
                suffix = "_quick" if args.quick else ""
                result_path = RESULTS_DIR / dataset_name / f"{ablation_id}{suffix}.pkl"

                if not result_path.exists():
                    print(f"  No result file: {result_path}")
                    continue

                with open(result_path, "rb") as f:
                    result = pickle.load(f)

                if "pareto_solutions" not in result:
                    print("  No pareto_solutions in result (old format?)")
                    continue

                best_idx = result.get("best_pareto_idx", 0)
                n_solutions = len(result["pareto_solutions"])
                if args.sol == "all":
                    sol_indices = list(range(n_solutions))
                elif args.sol is not None:
                    sol_indices = args.sol
                else:
                    sol_indices = [best_idx]
                n_folds = 3 if X_effect.shape[0] < 500 else 5

                print(f"  Solutions to process: {sol_indices}")

                for sol_idx in sol_indices:
                    if sol_idx >= n_solutions:
                        print(f"  Sol {sol_idx}: index out of range (max {n_solutions - 1})")
                        continue

                    ps = result["pareto_solutions"][sol_idx]
                    print(f"\n{'=' * 60}")
                    print(f"RE-RUNNING CV: {ablation_id} / {dataset_name} / sol {sol_idx}")
                    print(
                        f"  Processor: {ps['processor_type']}, edges={ps['n_edges']}, "
                        f"|MB|={len(ps['mb_indices'])}"
                    )
                    print(f"{'=' * 60}")

                    hyperparams = dict(ps.get("hyperparams", {}))
                    hyperparams["processor_config"] = ps.get("processor_config", {})
                    # Set defaults for missing keys
                    hyperparams.setdefault("lambda_1", 0.02)
                    hyperparams.setdefault("lambda_2", 0.01)
                    hyperparams.setdefault("lambda_class", 1.0)
                    hyperparams.setdefault("lr", 0.001)

                    A_init = ps.get("A_est")
                    if A_init is None:
                        print(f"  Sol {sol_idx}: no A_est saved")
                        continue

                    try:
                        # Determine max_iter from ablation config
                        abl_cfg = get_ablation_config(ablation_id)
                        golem_ov = abl_cfg.get("golem_overrides", {})
                        max_iter = golem_ov.get("max_iter", 4000)

                        cv_result = evaluate_pareto_solution_cv(
                            X=X_effect,
                            Y=Y_effect,
                            hyperparams=hyperparams,
                            processor_type=ps.get("processor_type", "elm"),
                            A_init=np.array(A_init),
                            processor_params_init=ps.get("processor_params"),
                            n_folds=n_folds,
                            true_mb=ds_config.get("true_mb"),
                            golem_max_iter=100 if not args.quick else 20,
                            cold_start=False,
                            task=ds_config.get("task", "classification"),
                            verbose=True,
                            use_v7=True,
                            freeze_structure=True,
                        )
                        cv_dict = {
                            "accuracy_mean": cv_result.accuracy_mean,
                            "accuracy_std": cv_result.accuracy_std,
                            "precision_mean": cv_result.precision_mean,
                            "precision_std": cv_result.precision_std,
                            "recall_mean": cv_result.recall_mean,
                            "recall_std": cv_result.recall_std,
                            "f1_mean": cv_result.f1_mean,
                            "f1_std": cv_result.f1_std,
                            "balanced_acc_mean": cv_result.balanced_acc_mean,
                            "balanced_acc_std": cv_result.balanced_acc_std,
                            "roc_auc_mean": cv_result.roc_auc_mean,
                            "roc_auc_std": cv_result.roc_auc_std,
                            "mb_jaccard_mean": cv_result.mb_jaccard_mean,
                        }
                        if cv_result.mb_f1_per_fold:
                            cv_dict["mb_f1_mean"] = float(np.mean(cv_result.mb_f1_per_fold))

                        # Update per-solution CV
                        result["pareto_solutions"][sol_idx]["cv"] = cv_dict

                        # Update top-level keys if this is the best solution
                        if sol_idx == best_idx:
                            result["cv_accuracy_mean"] = cv_dict["accuracy_mean"]
                            result["cv_accuracy_std"] = cv_dict["accuracy_std"]
                            result["cv_f1_mean"] = cv_dict["f1_mean"]
                            result["cv_f1_std"] = cv_dict["f1_std"]
                            result["cv_balanced_acc_mean"] = cv_dict["balanced_acc_mean"]
                            result["cv_balanced_acc_std"] = cv_dict["balanced_acc_std"]
                            result["cv_roc_auc_mean"] = cv_dict["roc_auc_mean"]
                            result["cv_roc_auc_std"] = cv_dict["roc_auc_std"]
                            result["cv_mb_jaccard"] = cv_dict.get("mb_jaccard_mean")
                            if "mb_f1_mean" in cv_dict:
                                result["cv_mb_f1_mean"] = cv_dict["mb_f1_mean"]

                        with open(result_path, "wb") as f:
                            pickle.dump(result, f)
                        print(
                            f"  CV BAcc={cv_result.balanced_acc_mean:.4f}"
                            f"+-{cv_result.balanced_acc_std:.4f}"
                        )
                        print(f"  Updated results: {result_path}")
                    except Exception as e:
                        print(f"  Sol {sol_idx}: CV failed: {e}")
                        import traceback

                        traceback.print_exc()

        if not args.rerun_cf:
            return

    # --rerun-cf mode: load checkpoint or reconstruct from pkl, run CF, update results
    if args.rerun_cf:
        ablations = args.ablation or ["B6"]
        datasets = ALL_DATASETS if args.all_datasets else args.dataset
        for dataset_name in datasets:
            for ablation_id in ablations:
                suffix = "_quick" if args.quick else ""
                result_path = RESULTS_DIR / dataset_name / f"{ablation_id}{suffix}.pkl"

                # Load result pkl to determine solution indices
                result = None
                if result_path.exists():
                    with open(result_path, "rb") as f:
                        result = pickle.load(f)

                best_idx = result.get("best_pareto_idx", 0) if result else 0
                n_solutions = len(result.get("pareto_solutions", [])) if result else 0
                if args.sol == "all":
                    sol_indices = list(range(n_solutions))
                elif args.sol is not None:
                    sol_indices = args.sol
                else:
                    sol_indices = [best_idx]

                print(f"\n  CF: {ablation_id} / {dataset_name}, solutions: {sol_indices}")

                for sol_idx in sol_indices:
                    print(f"\n{'=' * 60}")
                    print(f"RE-RUNNING CF: {ablation_id} / {dataset_name} / sol {sol_idx}")
                    print(f"{'=' * 60}")

                    # Try loading checkpoint first
                    ckpt_suffix = "" if sol_idx == best_idx else f"_sol{sol_idx}"
                    ckpt_path = (
                        RESULTS_DIR
                        / dataset_name
                        / f"{ablation_id}{suffix}_cf_checkpoint{ckpt_suffix}.pkl"
                    )

                    if ckpt_path.exists():
                        print(f"  Loading checkpoint: {ckpt_path}")
                        X, Y, A_est, A_weights, A_confound, processor, params, ds_config = (
                            _load_cf_checkpoint(ckpt_path)
                        )
                    elif (
                        result
                        and "pareto_solutions" in result
                        and sol_idx < len(result["pareto_solutions"])
                    ):
                        # Reconstruct from pkl's pareto_solutions entry
                        ps = result["pareto_solutions"][sol_idx]
                        if "processor_params" not in ps:
                            print(f"  Sol {sol_idx}: no processor_params saved, cannot rerun CF")
                            continue
                        print(f"  Reconstructing from result pkl (sol {sol_idx})")
                        X, Y, ds_config = load_dataset(dataset_name)
                        A_est = np.array(ps["A_est"])
                        A_weights = (
                            np.array(ps["A_weights"]) if ps.get("A_weights") is not None else None
                        )
                        A_confound = (
                            np.array(ps["A_confound_weights"])
                            if ps.get("A_confound_weights") is not None
                            else None
                        )
                        processor, params = _reconstruct_processor(
                            ps["processor_type"],
                            ps.get("processor_config", {}),
                            ps["processor_params"],
                        )
                    else:
                        print(f"  No checkpoint or pkl data found for sol {sol_idx}")
                        continue

                    cf_result = _run_counterfactual_evaluation(
                        X=X,
                        Y=Y,
                        A_est=A_est,
                        processor_trained=processor,
                        processor_params=params,
                        ds_config=ds_config,
                        A_weights=A_weights,
                        A_confound=A_confound,
                        n_instances=30 if not args.quick else 10,
                        cf_pop_size=50 if not args.quick else 20,
                        cf_n_gen=50 if not args.quick else 15,
                        verbose=True,
                    )

                    # Update saved result pickle
                    if cf_result and result is not None:
                        # Update per-solution CF
                        if "pareto_solutions" in result and sol_idx < len(
                            result["pareto_solutions"]
                        ):
                            result["pareto_solutions"][sol_idx]["cf"] = cf_result
                        # Update top-level keys if this is the best solution
                        if sol_idx == best_idx:
                            result["cf_n_instances"] = cf_result["n_instances"]
                            result["cf_n_valid"] = cf_result["n_valid"]
                            result["cf_validity_rate"] = cf_result["validity_rate"]
                            result["cf_avg_sparsity"] = cf_result["avg_sparsity"]
                            result["cf_avg_distance"] = cf_result["avg_distance"]
                            result["cf_avg_plausibility_ratio"] = cf_result[
                                "avg_plausibility_ratio"
                            ]
                            result["cf_n_plausible"] = cf_result["n_plausible"]
                            result["cf_avg_causal_validity"] = cf_result["avg_causal_validity"]
                            result["cf_details"] = cf_result
                        with open(result_path, "wb") as f:
                            pickle.dump(result, f)
                        print(f"  Updated results: {result_path}")
                    elif cf_result:
                        print("  CF completed but no result file to update")

        return

    if args.list:
        print("\nAvailable ablations:")
        for aid in ALL_ABLATIONS:
            cfg = get_ablation_config(aid)
            print(f"  {aid}: {cfg['name']} — {cfg['description']}")
        print(f"\nAvailable datasets: {', '.join(ALL_DATASETS)}")
        return

    # Determine what to run
    ablations = ALL_ABLATIONS if args.all else args.ablation
    datasets = ALL_DATASETS if args.all_datasets else args.dataset

    if not args.all and not args.ablation:
        parser.error("Specify --ablation ID or --all")

    # Run
    all_results = []
    total_start = time.time()

    for dataset_name in datasets:
        for ablation_id in ablations:
            try:
                # Parse CLI --golem-override into dict
                cli_overrides = {}
                if hasattr(args, "golem_override") and args.golem_override:
                    for item in args.golem_override:
                        if "=" not in item:
                            continue
                        key, val = item.split("=", 1)
                        if val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                        else:
                            try:
                                val = float(val)
                                if val == int(val):
                                    val = int(val)
                            except ValueError:
                                pass
                        cli_overrides[key] = val

                result = run_ablation(
                    ablation_id=ablation_id,
                    dataset_name=dataset_name,
                    quick=args.quick,
                    verbose=not args.quiet,
                    n_gen_override=args.n_gen,
                    pop_size_override=args.pop_size,
                    extra_golem_overrides=cli_overrides if cli_overrides else None,
                )
                all_results.append(result)

                # Save individual result
                out_dir = RESULTS_DIR / dataset_name
                out_dir.mkdir(parents=True, exist_ok=True)
                suffix = "_quick" if args.quick else ""
                out_path = out_dir / f"{ablation_id}{suffix}.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump(result, f)

            except Exception as e:
                print(f"\n  ERROR in {ablation_id}/{dataset_name}: {e}")
                import traceback

                traceback.print_exc()
                all_results.append(
                    {
                        "ablation_id": ablation_id,
                        "dataset": dataset_name,
                        "error": str(e),
                    }
                )

    # Summary
    total_time = time.time() - total_start
    print_summary_table([r for r in all_results if "error" not in r])
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"summary{'_quick' if args.quick else ''}.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
