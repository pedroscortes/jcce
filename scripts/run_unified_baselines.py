#!/usr/bin/env python
"""
Unified Baseline Runner for JCCE.

Runs all baselines on all datasets with the same evaluation protocol:
  1. SHAP feature selection + 5 processors
  2. Structure learning (PC/GES/FCI/DAGMA/NOTEARS/GOLEM/DirectLiNGAM) → MB → 5 processors
  3. CASTLE (direct competitor)
  4. All features + 5 processors (upper bound)

Uses JCCE's own processor implementations (JAX) for fair comparison.

Usage:
    uv run python scripts/run_unified_baselines.py --dataset lucas
    uv run python scripts/run_unified_baselines.py --all
    uv run python scripts/run_unified_baselines.py --all --algorithms pc ges dagma
    uv run python scripts/run_unified_baselines.py --all --quick  # fewer epochs
"""

import argparse
import json
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np

from jcce.baselines.castle import train_castle
from jcce.baselines.classifier import train_and_evaluate_processor
from jcce.data.benchmark_loader import list_datasets, load_dataset
from jcce.structure_learning.markov_blanket import (
    validate_markov_blanket,
)
from jcce.structure_learning.postprocess import learn_and_postprocess

# ============================================================================
# Configuration
# ============================================================================

PROCESSOR_TYPES = ["mlp", "transformer", "mamba", "elm", "gnn"]

STRUCTURE_ALGORITHMS = [
    "pc",
    "ges",
    "fci",
    "dagma",
    "notears",
    "golem",
    "directlingam",
]

# Default hyperparameters per structure algorithm
ALGO_DEFAULTS = {
    "pc": {"alpha": 0.05, "max_cond_size": 3},
    "ges": {},
    "fci": {"alpha": 0.05, "max_cond_size": 3},
    "dagma": {"n_iterations": 300, "alpha_sparse": 0.02, "learning_rate": 3e-4},
    "notears": {"n_outer": 20, "n_inner": 200, "alpha_sparse": 0.02, "learning_rate": 3e-4},
    "golem": {"n_epochs": 500, "alpha_sparse": 0.02, "learning_rate": 3e-4},
    "directlingam": {"prune_threshold": 0.85},
}


def compute_structure_metrics(A_est: np.ndarray, A_true: np.ndarray) -> dict:
    """Compute SHD, F1, precision, recall between estimated and true DAG."""
    A_est_bin = (np.abs(A_est) > 0.01).astype(int)
    A_true_bin = (np.abs(A_true) > 0.01).astype(int)

    # Align shapes
    d = min(A_est_bin.shape[0], A_true_bin.shape[0])
    A_e = A_est_bin[:d, :d]
    A_t = A_true_bin[:d, :d]

    tp = int(np.sum((A_e == 1) & (A_t == 1)))
    fp = int(np.sum((A_e == 1) & (A_t == 0)))
    fn = int(np.sum((A_e == 0) & (A_t == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    shd = fp + fn  # structural Hamming distance (ignoring edge direction for simplicity)

    return {
        "shd": shd,
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_mb_metrics(mb_est: list, mb_true: list) -> dict:
    """Compare estimated MB against ground truth MB."""
    est_set = set(mb_est)
    true_set = set(mb_true)

    tp = len(est_set & true_set)
    fp = len(est_set - true_set)
    fn = len(true_set - est_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mb_f1": f1,
        "mb_precision": precision,
        "mb_recall": recall,
        "mb_size_est": len(est_set),
        "mb_size_true": len(true_set),
        "mb_tp": tp,
        "mb_fp": fp,
        "mb_fn": fn,
    }


def run_shap_baseline(X, Y, feature_names, seed=42) -> dict:
    """SHAP feature selection via XGBoost TreeExplainer."""
    try:
        import shap
        from xgboost import XGBClassifier
    except ImportError:
        return {"error": "shap or xgboost not installed"}

    model = XGBClassifier(n_estimators=100, verbosity=0, use_label_encoder=False, random_state=seed)
    model.fit(X, Y)

    explainer = shap.TreeExplainer(model)
    n_explain = min(1000, len(X))
    shap_values = explainer.shap_values(X[:n_explain])

    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    ranking = np.argsort(mean_shap)[::-1]

    # Adaptive k: features explaining 80% of total importance
    total = mean_shap.sum()
    cumsum = np.cumsum(mean_shap[ranking])
    k = int(np.searchsorted(cumsum, 0.80 * total)) + 1
    k = max(3, min(k, len(feature_names) // 2))

    selected_idx = ranking[:k].tolist()
    selected_names = [feature_names[i] for i in selected_idx]

    return {
        "selected_idx": selected_idx,
        "selected_names": selected_names,
        "k": k,
        "importance": {feature_names[i]: float(mean_shap[i]) for i in ranking},
    }


# ============================================================================
# Main per-dataset analysis
# ============================================================================


def run_dataset(
    dataset_name: str,
    algorithms: list = None,
    processors: list = None,
    seed: int = 42,
    quick: bool = False,
    verbose: int = 1,
) -> dict:
    """Run all baselines for one dataset."""
    if algorithms is None:
        algorithms = STRUCTURE_ALGORITHMS
    if processors is None:
        processors = PROCESSOR_TYPES

    print(f"\n{'=' * 70}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'=' * 70}")

    # Load data
    X, Y, config = load_dataset(dataset_name)
    Y_int = np.asarray(Y).astype(int)
    feature_names = config.get("feature_names", [f"X{i}" for i in range(X.shape[1])])
    true_dag = config.get("true_dag")
    true_mb = config.get("true_mb")
    has_true_dag = config.get("has_true_dag", False)

    n, d = X.shape
    print(f"  Shape: ({n}, {d}), Target: {config.get('target_name', '?')}")
    if has_true_dag:
        print(
            f"  Ground truth available: DAG={'yes' if true_dag is not None else 'no'}, MB={true_mb}"
        )

    key = random.PRNGKey(seed)
    n_epochs_cls = 100 if quick else 200
    n_splits = 5

    # ================================================================
    # SAMPLE SPLITTING: 70/30, matching GOLEM-GA's protocol exactly.
    # Structure learning and feature selection use X_struct (70%).
    # Classification CV uses X_effect (30%).
    # This ensures fair comparison: baselines and GOLEM-GA see
    # the same data partitions (same seed, same split ratio).
    # ================================================================
    from sklearn.model_selection import train_test_split

    X_struct, X_effect, Y_struct, Y_effect = train_test_split(
        X, Y_int, test_size=0.3, stratify=Y_int, random_state=42
    )
    print(
        f"  Sample split: X_struct={X_struct.shape[0]} (70%), "
        f"X_effect={X_effect.shape[0]} (30%), seed=42"
    )

    # Standardize using X_struct statistics (avoid data leakage from X_effect)
    mu, sigma = X_struct.mean(axis=0), X_struct.std(axis=0) + 1e-8
    X_struct_std = (X_struct - mu) / sigma
    X_effect_std = (X_effect - mu) / sigma

    # For structure learning: augment X_struct with Y_struct as last column
    Y_struct_int = np.asarray(Y_struct).astype(int)
    Y_effect_int = np.asarray(Y_effect).astype(int)

    result = {
        "dataset": dataset_name,
        "n_samples": n,
        "n_features": d,
        "n_struct": X_struct.shape[0],
        "n_effect": X_effect.shape[0],
        "feature_names": feature_names,
        "has_true_dag": has_true_dag,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "baselines": {},
    }

    # ------------------------------------------------------------------
    # Baseline 0: All features + processors (upper bound)
    # Evaluated on X_effect (30%) to match GOLEM-GA protocol
    # ------------------------------------------------------------------
    print("\n  --- B0: All Features + Processors ---")
    for proc in processors:
        label = f"all_features_{proc}"
        t0 = time.time()
        try:
            res = train_and_evaluate_processor(
                X_effect_std,
                Y_effect_int,
                processor_type=proc,
                n_splits=n_splits,
                seed=seed,
                n_epochs=n_epochs_cls,
            )
            res["time"] = time.time() - t0
            res["feature_set"] = "all"
            res["n_features_used"] = d
            result["baselines"][label] = res
            print(
                f"    {proc:14s}: BAcc={res['balanced_acc_mean']:.3f}+-{res['balanced_acc_std']:.3f} "
                f"F1={res['f1_mean']:.3f} ({res['time']:.1f}s)"
            )
        except Exception as e:
            print(f"    {proc:14s}: FAILED — {e}")
            result["baselines"][label] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Baseline 1: SHAP feature selection + processors
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Baseline 1: SHAP feature selection + processors
    # SHAP runs on X_struct (70%), classification CV on X_effect (30%)
    # ------------------------------------------------------------------
    print("\n  --- B1: SHAP Feature Selection + Processors ---")
    shap_result = run_shap_baseline(X_struct, Y_struct_int, feature_names, seed)
    result["shap"] = shap_result

    if "error" not in shap_result:
        shap_idx = shap_result["selected_idx"]
        X_shap_effect = X_effect_std[:, shap_idx]
        print(f"    Selected {len(shap_idx)} features: {shap_result['selected_names']}")

        if true_mb:
            mb_metrics = compute_mb_metrics(shap_idx, true_mb)
            shap_result["mb_metrics"] = mb_metrics
            print(f"    MB overlap: F1={mb_metrics['mb_f1']:.3f}")

        for proc in processors:
            label = f"shap_{proc}"
            t0 = time.time()
            try:
                res = train_and_evaluate_processor(
                    X_shap_effect,
                    Y_effect_int,
                    processor_type=proc,
                    n_splits=n_splits,
                    seed=seed,
                    n_epochs=n_epochs_cls,
                )
                res["time"] = time.time() - t0
                res["feature_set"] = "shap"
                res["n_features_used"] = len(shap_idx)
                res["selected_features"] = shap_result["selected_names"]
                result["baselines"][label] = res
                print(
                    f"    {proc:14s}: BAcc={res['balanced_acc_mean']:.3f}+-{res['balanced_acc_std']:.3f} "
                    f"F1={res['f1_mean']:.3f} ({res['time']:.1f}s)"
                )
            except Exception as e:
                print(f"    {proc:14s}: FAILED — {e}")
                result["baselines"][label] = {"error": str(e)}
    else:
        print(f"    SHAP failed: {shap_result['error']}")

    # ------------------------------------------------------------------
    # Baseline 2: Structure Learning → MB → Processors
    # Structure learning on X_struct (70%), classification on X_effect (30%)
    # ------------------------------------------------------------------
    # Y_idx: position of target in the FULL variable set
    target_name = config.get("target_name", None)
    Y_idx = d  # Y appended as last column
    X_with_Y_struct = np.concatenate(
        [X_struct_std, Y_struct_int[:, None].astype(np.float32)], axis=1
    )
    X_with_Y_jax = jnp.array(X_with_Y_struct)

    for algo in algorithms:
        print(f"\n  --- B2: {algo.upper()} → MB → Processors ---")
        t0 = time.time()

        try:
            key, algo_key = random.split(key)
            algo_kwargs = {**ALGO_DEFAULTS.get(algo, {})}

            # Quick mode: reduce iterations
            if quick:
                for k in ["n_iterations", "n_inner", "n_epochs", "n_outer"]:
                    if k in algo_kwargs:
                        algo_kwargs[k] = max(50, algo_kwargs[k] // 3)

            A_dag = learn_and_postprocess(
                X_with_Y_jax,
                algo_key,
                algorithm=algo,
                verbose=verbose >= 2,
                protected_idx=Y_idx,
                **algo_kwargs,
            )
            A_np = np.array(A_dag)
            algo_time = time.time() - t0

            # Structure metrics
            struct_metrics = {}
            if has_true_dag and true_dag is not None:
                struct_metrics = compute_structure_metrics(A_np, np.array(true_dag))
                print(
                    f"    Structure: SHD={struct_metrics['shd']}, "
                    f"F1={struct_metrics['edge_f1']:.3f}"
                )

            # Extract MB
            mb_info = validate_markov_blanket(jnp.array(A_np), Y_idx)
            mb_indices_raw = [int(i) for i in mb_info["mb_all"] if int(i) != Y_idx]
            # Filter to feature indices only (< d)
            mb_indices = [i for i in mb_indices_raw if i < d]

            print(f"    {algo.upper()}: {len(mb_indices)} MB features (of {d}) in {algo_time:.1f}s")

            if true_mb:
                mb_met = compute_mb_metrics(mb_indices, true_mb)
                print(
                    f"    MB overlap: F1={mb_met['mb_f1']:.3f} "
                    f"(P={mb_met['mb_precision']:.3f}, R={mb_met['mb_recall']:.3f})"
                )
            else:
                mb_met = {}

            # Classify on MB features
            if len(mb_indices) == 0:
                print("    Empty MB — skipping classification")
                result["baselines"][f"{algo}_summary"] = {
                    "algo_time": algo_time,
                    "A_est": A_np.tolist(),
                    "n_edges": int(np.sum(np.abs(A_np) > 0.1)),
                    "structure_metrics": struct_metrics,
                    "mb_metrics": mb_met,
                    "mb_indices": mb_indices,
                    "error": "empty_mb",
                }
                continue

            X_mb_effect = X_effect_std[:, mb_indices]

            for proc in processors:
                label = f"{algo}_{proc}"
                t1 = time.time()
                try:
                    res = train_and_evaluate_processor(
                        X_mb_effect,
                        Y_effect_int,
                        processor_type=proc,
                        n_splits=n_splits,
                        seed=seed,
                        n_epochs=n_epochs_cls,
                    )
                    res["time"] = time.time() - t1
                    res["algo_time"] = algo_time
                    res["feature_set"] = f"{algo}_mb"
                    res["n_features_used"] = len(mb_indices)
                    res["mb_indices"] = mb_indices
                    res["mb_names"] = [
                        feature_names[i] for i in mb_indices if i < len(feature_names)
                    ]
                    res["structure_metrics"] = struct_metrics
                    res["mb_metrics"] = mb_met
                    result["baselines"][label] = res
                    print(
                        f"    {proc:14s}: BAcc={res['balanced_acc_mean']:.3f}+-{res['balanced_acc_std']:.3f} "
                        f"F1={res['f1_mean']:.3f} ({res['time']:.1f}s)"
                    )
                except Exception as e:
                    print(f"    {proc:14s}: FAILED — {e}")
                    result["baselines"][label] = {"error": str(e)}

            # Save algo-level summary
            mb_names = [feature_names[i] for i in mb_indices if i < len(feature_names)]
            result["baselines"][f"{algo}_summary"] = {
                "algo_time": algo_time,
                "A_est": A_np.tolist(),
                "n_edges": int(np.sum(np.abs(A_np) > 0.01)),
                "structure_metrics": struct_metrics,
                "mb_metrics": mb_met,
                "mb_indices": mb_indices,
                "mb_names": mb_names,
            }

        except Exception as e:
            print(f"    {algo.upper()} FAILED: {e}")
            if verbose >= 2:
                traceback.print_exc()
            result["baselines"][f"{algo}_summary"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Baseline 3: CASTLE (direct competitor)
    # ------------------------------------------------------------------
    print("\n  --- B3: CASTLE ---")
    t0 = time.time()
    try:
        key, castle_key = random.split(key)
        castle_epochs = 150 if quick else 300
        # Run both regression (original paper) and classification (BCE) modes
        # Original paper uses MSE even for classification, evaluates AUROC on raw output
        castle_result = None
        for castle_task in ["regression", "classification"]:
            key, ck = random.split(key)
            res = train_castle(
                X_struct_std,
                Y_struct_int.astype(np.float32),
                key=ck,
                task=castle_task,
                n_epochs=castle_epochs,
                verbose=verbose,
            )
            bacc = res["metrics"].get("balanced_accuracy", 0)
            if castle_result is None or bacc > castle_result["metrics"].get("balanced_accuracy", 0):
                castle_result = res
                castle_result["metrics"]["castle_task"] = castle_task
        if verbose >= 1:
            print(f"    Best mode: {castle_result['metrics'].get('castle_task', '?')}")
        castle_time = time.time() - t0

        castle_entry = {
            "balanced_acc_mean": castle_result["metrics"].get("balanced_accuracy", 0),
            "f1_mean": castle_result["metrics"].get("f1", 0),
            "roc_auc_mean": castle_result["metrics"].get("roc_auc", 0),
            "time": castle_time,
            "h_A": castle_result["metrics"]["h_A"],
            "mb_indices": castle_result["metrics"].get("mb_indices", []),
            "mb_size": castle_result["metrics"].get("mb_size", 0),
            "n_edges": int(np.sum(np.abs(castle_result["A_est"]) > 0.1)),
            "A_est": castle_result["A_est"].tolist(),
            "feature_set": "castle_joint",
            "castle_task": castle_result["metrics"].get("castle_task", "unknown"),
        }

        # Structure metrics
        if has_true_dag and true_dag is not None:
            s_met = compute_structure_metrics(castle_result["A_est"], np.array(true_dag))
            castle_entry["structure_metrics"] = s_met
            print(f"    Structure: SHD={s_met['shd']}, F1={s_met['edge_f1']:.3f}")

        # MB metrics
        if true_mb and castle_result["metrics"].get("mb_indices"):
            mb_met = compute_mb_metrics(castle_result["metrics"]["mb_indices"], true_mb)
            castle_entry["mb_metrics"] = mb_met
            print(f"    MB: F1={mb_met['mb_f1']:.3f}")

        result["baselines"]["castle"] = castle_entry
        print(
            f"    CASTLE: BAcc={castle_entry['balanced_acc_mean']:.3f} "
            f"h(A)={castle_entry['h_A']:.4f} ({castle_time:.1f}s)"
        )

        # CASTLE MB → 5 processors on X_effect (30%)
        castle_mb = castle_result["metrics"].get("mb_indices", [])
        if castle_mb:
            X_castle_mb_effect = X_effect_std[:, castle_mb]
            for proc in processors:
                label = f"castle_mb_{proc}"
                t1 = time.time()
                try:
                    res = train_and_evaluate_processor(
                        X_castle_mb_effect,
                        Y_effect_int,
                        processor_type=proc,
                        n_splits=n_splits,
                        seed=seed,
                        n_epochs=n_epochs_cls,
                    )
                    res["time"] = time.time() - t1
                    res["feature_set"] = "castle_mb"
                    res["n_features_used"] = len(castle_mb)
                    result["baselines"][label] = res
                    print(
                        f"    {proc:14s} (MB): BAcc={res['balanced_acc_mean']:.3f} "
                        f"F1={res['f1_mean']:.3f}"
                    )
                except Exception as e:
                    result["baselines"][label] = {"error": str(e)}

    except Exception as e:
        print(f"    CASTLE FAILED: {e}")
        if verbose >= 2:
            traceback.print_exc()
        result["baselines"]["castle"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Baseline 4: Modern baselines (SDCD, DiffAN)
    # Structure learning on X_struct (70%), classification on X_effect (30%)
    # ------------------------------------------------------------------
    print("\n  --- B4: Modern Baselines (SDCD, DiffAN) ---")

    try:
        from jcce.baselines.modern_baselines import run_sdcd

        print("    Running SDCD...")
        t0 = time.time()
        A_sdcd, sdcd_info = run_sdcd(
            X_struct_std, feature_names=feature_names, verbose=verbose >= 2
        )
        sdcd_time = time.time() - t0

        # Extract MB from SDCD DAG
        A_sdcd_np = np.array(A_sdcd)
        mb_sdcd = [
            i for i in range(d) if abs(A_sdcd_np[i, :]).sum() > 0 or abs(A_sdcd_np[:, i]).sum() > 0
        ]
        # Filter to parents/children of Y-like variable (last col or highest connectivity)
        sdcd_entry = {
            "algo_time": sdcd_time,
            "A_est": A_sdcd_np.tolist(),
            "n_edges": int(np.sum(A_sdcd_np != 0)),
            "mb_indices": mb_sdcd[: d // 2],  # Cap at d/2
        }
        if has_true_dag and true_dag is not None:
            s_met = compute_structure_metrics(A_sdcd_np, np.array(true_dag))
            sdcd_entry["structure_metrics"] = s_met
            print(
                f"    SDCD: {sdcd_entry['n_edges']} edges, SHD={s_met['shd']}, F1={s_met['edge_f1']:.3f} ({sdcd_time:.1f}s)"
            )
        else:
            print(f"    SDCD: {sdcd_entry['n_edges']} edges ({sdcd_time:.1f}s)")
        result["baselines"]["sdcd_summary"] = sdcd_entry

        # SDCD MB → 5 processors on X_effect
        if mb_sdcd:
            X_sdcd_mb = X_effect_std[:, mb_sdcd[: d // 2]]
            for proc in processors:
                label = f"sdcd_{proc}"
                try:
                    res = train_and_evaluate_processor(
                        X_sdcd_mb,
                        Y_effect_int,
                        processor_type=proc,
                        n_splits=n_splits,
                        seed=seed,
                        n_epochs=n_epochs_cls,
                    )
                    res["time"] = time.time() - t0
                    res["feature_set"] = "sdcd_mb"
                    result["baselines"][label] = res
                    print(
                        f"    {proc:14s}: BAcc={res['balanced_acc_mean']:.3f} F1={res['f1_mean']:.3f}"
                    )
                except Exception as e:
                    result["baselines"][label] = {"error": str(e)}
    except Exception as e:
        print(f"    SDCD FAILED: {e}")
        result["baselines"]["sdcd_summary"] = {"error": str(e)}

    try:
        from jcce.baselines.modern_baselines import run_diffan

        print("    Running DiffAN...")
        t0 = time.time()
        A_diffan, diffan_info = run_diffan(X_struct_std, verbose=verbose >= 2)
        diffan_time = time.time() - t0

        A_diffan_np = np.array(A_diffan)
        mb_diffan = [
            i
            for i in range(d)
            if abs(A_diffan_np[i, :]).sum() > 0 or abs(A_diffan_np[:, i]).sum() > 0
        ]
        diffan_entry = {
            "algo_time": diffan_time,
            "A_est": A_diffan_np.tolist(),
            "n_edges": int(np.sum(A_diffan_np != 0)),
            "mb_indices": mb_diffan[: d // 2],
            "topological_order": diffan_info.get("topological_order", []),
        }
        if has_true_dag and true_dag is not None:
            s_met = compute_structure_metrics(A_diffan_np, np.array(true_dag))
            diffan_entry["structure_metrics"] = s_met
            print(
                f"    DiffAN: {diffan_entry['n_edges']} edges, SHD={s_met['shd']}, F1={s_met['edge_f1']:.3f} ({diffan_time:.1f}s)"
            )
        else:
            print(f"    DiffAN: {diffan_entry['n_edges']} edges ({diffan_time:.1f}s)")
        result["baselines"]["diffan_summary"] = diffan_entry

        # DiffAN MB → 5 processors on X_effect
        if mb_diffan:
            X_diffan_mb = X_effect_std[:, mb_diffan[: d // 2]]
            for proc in processors:
                label = f"diffan_{proc}"
                try:
                    res = train_and_evaluate_processor(
                        X_diffan_mb,
                        Y_effect_int,
                        processor_type=proc,
                        n_splits=n_splits,
                        seed=seed,
                        n_epochs=n_epochs_cls,
                    )
                    res["time"] = time.time() - t0
                    res["feature_set"] = "diffan_mb"
                    result["baselines"][label] = res
                    print(
                        f"    {proc:14s}: BAcc={res['balanced_acc_mean']:.3f} F1={res['f1_mean']:.3f}"
                    )
                except Exception as e:
                    result["baselines"][label] = {"error": str(e)}
    except Exception as e:
        print(f"    DiffAN FAILED: {e}")
        result["baselines"]["diffan_summary"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Post-hoc Validation: DML + Cinelli + Refutation on all baselines
    # Runs on X_effect (30%) using each baseline's learned DAG.
    # Matches GOLEM-GA's 7-step validation protocol exactly.
    # ------------------------------------------------------------------
    print("\n  --- Validation Protocol (on X_effect, 30%) ---")

    from jcce.validation.multi_parent_dml import run_multi_parent_dml

    # Identify methods with stored A_est (structure-learning methods + castle)
    methods_to_validate = []
    for algo in algorithms:
        key = f"{algo}_summary"
        data = result["baselines"].get(key, {})
        if isinstance(data, dict) and "A_est" in data:
            methods_to_validate.append((algo, key, np.array(data["A_est"])))
    # Also validate CASTLE
    castle_data = result["baselines"].get("castle", {})
    if isinstance(castle_data, dict) and "A_est" in castle_data:
        methods_to_validate.append(("castle", "castle", np.array(castle_data["A_est"])))

    for method_name, method_key, A_baseline in methods_to_validate:
        print(f"\n    Validating: {method_name.upper()}")

        # A_baseline may be (d+1)x(d+1) augmented or (d)x(d) non-augmented.
        # run_multi_parent_dml expects augmented (n_features+1, n_features+1).
        n_f = X_effect_std.shape[1]
        if A_baseline.shape[0] == n_f:
            # Need to augment: add Y row/col. Put Y as last variable.
            A_aug = np.zeros((n_f + 1, n_f + 1), dtype=A_baseline.dtype)
            A_aug[:n_f, :n_f] = A_baseline
            # Copy MB edges to Y column (parents of Y)
            mb = result["baselines"][method_key].get("mb_indices", [])
            for idx in mb:
                if idx < n_f:
                    A_aug[idx, n_f] = (
                        np.max(np.abs(A_baseline[idx, :])) + 0.1
                    )  # synthetic edge weight
        else:
            A_aug = A_baseline

        try:
            dml_result = run_multi_parent_dml(
                X=X_effect_std,
                Y=Y_effect_int.astype(np.float32),
                A_est=A_aug,
                Y_idx=n_f,
                feature_names=feature_names,
                parent_threshold=0.01,
                n_dml_folds=5,
                run_refutation=True,
                n_refutation_sims=50,
                verbose=verbose >= 2,
            )
            if dml_result is not None:
                result["baselines"][method_key]["validation"] = {
                    "dml_n_parents": dml_result.n_parents_discovered,
                    "dml_n_significant": dml_result.n_parents_significant,
                    "dml_parent_results": [
                        {
                            "feature": e.feature_name,
                            "ate": float(e.dml_result.ate_mean),
                            "significant": e.is_significant,
                        }
                        for e in (dml_result.parent_results or [])
                    ],
                }
                print(
                    f"      DML: {dml_result.n_parents_significant}/{dml_result.n_parents_discovered} "
                    f"parents significant"
                )
            else:
                print("      DML: no parents found")
                result["baselines"][method_key]["validation"] = {
                    "dml_n_parents": 0,
                    "dml_n_significant": 0,
                }

            # Cinelli sensitivity
            try:
                from jcce.validation.cinelli_sensitivity import cinelli_sensitivity

                parents_of_Y = np.where(np.abs(A_aug[:n_f, n_f]) > 0.01)[0]
                if len(parents_of_Y) > 0:
                    strongest = int(parents_of_Y[np.argmax(np.abs(A_aug[parents_of_Y, n_f]))])
                    # cinelli_sensitivity expects (Y, T, X) as separate arrays
                    T_cinelli = X_effect_std[:, strongest]
                    X_cinelli = np.delete(X_effect_std, strongest, axis=1)
                    t_name = (
                        feature_names[strongest]
                        if strongest < len(feature_names)
                        else f"X{strongest}"
                    )
                    cinelli_r = cinelli_sensitivity(
                        Y=Y_effect_int.astype(np.float32),
                        T=T_cinelli,
                        X=X_cinelli,
                        treatment_name=t_name,
                    )
                    rv = cinelli_r.rv if hasattr(cinelli_r, "rv") else cinelli_r.get("rv", None)
                    result["baselines"][method_key]["validation"]["cinelli_rv"] = (
                        float(rv) if rv is not None else None
                    )
                    print(
                        f"      Cinelli RV: {rv:.3f}"
                        if isinstance(rv, (int, float))
                        else f"      Cinelli: {rv}"
                    )
            except Exception as ce:
                print(f"      Cinelli: failed ({ce})")

        except Exception as val_e:
            print(f"      Validation failed: {val_e}")
            if verbose >= 2:
                traceback.print_exc()

    return result


# ============================================================================
# Summary Table
# ============================================================================


def print_summary(all_results: dict):
    """Print a consolidated comparison table."""
    print(f"\n\n{'=' * 90}")
    print("SUMMARY: Best BAcc per Method x Dataset")
    print(f"{'=' * 90}")

    datasets = sorted(all_results.keys())

    # Collect all method families
    method_families = ["all_features", "shap", "castle"]
    for algo in STRUCTURE_ALGORITHMS:
        method_families.append(algo)

    # Header
    header = f"{'Method':<20s}"
    for ds in datasets:
        header += f" {ds[:10]:>10s}"
    print(header)
    print("-" * len(header))

    for method in method_families:
        row = f"{method:<20s}"
        for ds in datasets:
            baselines = all_results[ds].get("baselines", {})
            # Find best processor for this method
            best_bacc = 0
            for key, val in baselines.items():
                if key.startswith(method + "_") or key == method:
                    bacc = val.get("balanced_acc_mean", 0)
                    if isinstance(bacc, (int, float)) and bacc > best_bacc:
                        best_bacc = bacc
            if best_bacc > 0:
                row += f" {best_bacc:>10.3f}"
            else:
                row += f" {'---':>10s}"
        print(row)


# ============================================================================
# Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Unified JCCE Baselines")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help=f"Structure algorithms (default: all). Options: {STRUCTURE_ALGORITHMS}",
    )
    parser.add_argument(
        "--processors",
        nargs="+",
        default=None,
        help=f"Processor types (default: all). Options: {PROCESSOR_TYPES}",
    )
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results/baselines_unified")
    args = parser.parse_args()

    if args.all:
        datasets = list_datasets()
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ["lucas"]  # default

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets: {datasets}")
    print(f"Algorithms: {args.algorithms or STRUCTURE_ALGORITHMS}")
    print(f"Processors: {args.processors or PROCESSOR_TYPES}")
    print(f"Quick mode: {args.quick}")

    all_results = {}

    for ds in datasets:
        try:
            result = run_dataset(
                ds,
                algorithms=args.algorithms,
                processors=args.processors,
                seed=args.seed,
                quick=args.quick,
                verbose=args.verbose,
            )
            all_results[ds] = result

            # Save per-dataset
            with open(output_dir / f"{ds}_baselines.pkl", "wb") as f:
                pickle.dump(result, f)

        except Exception as e:
            print(f"\nFATAL ERROR on {ds}: {e}")
            traceback.print_exc()

    # Summary
    print_summary(all_results)

    # Save all
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    pkl_path = output_dir / f"all_baselines_{timestamp}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)

    json_path = output_dir / f"all_baselines_{timestamp}.json"
    try:
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=_json_safe)
    except (ValueError, TypeError) as e:
        print(f"\n  JSON save failed ({e}), pkl saved successfully")

    print(f"\nResults saved to {output_dir}/")
    print(f"  JSON: {json_path.name}")
    print(f"  PKL:  {pkl_path.name}")


if __name__ == "__main__":
    main()
