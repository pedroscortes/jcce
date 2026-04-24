#!/usr/bin/env python
"""
Experiment C.2: Unified vs Pipeline Ablation.

Compares three approaches on the same synthetic data:
1. JCCE Unified — GOLEM v7 (structure + classification + effects in one pass)
2. Pipeline A — PC → MB feature selection → logistic regression → OLS ATE
3. Pipeline B — GOLEM structure-only (lambda_class=0) → MB → classifier → ATE

Same total data, same ground truth. Metrics: SHD, balanced accuracy, MB F1,
ATE quality (PEHE, ATE error).

Usage:
    uv run python scripts/experiment_c2_unified_vs_pipeline.py --quick
    uv run python scripts/experiment_c2_unified_vs_pipeline.py \
        --n-trials 15 --n-vars 10 --n-samples 500 --max-iter 100
"""

import argparse
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np
import optuna
from jax import random
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import LinearSCM, SCMConfig
from jcce.evaluation.effect_metrics import compute_pehe
from jcce.structure_learning.optuna_search import (
    constraints_func,
    create_optuna_objective,
    extract_pareto_solutions,
)
from jcce.structure_learning.pc import learn_with_pc
from jcce.utils.metrics import compute_structure_metrics

# ============================================================================
# Data Generation
# ============================================================================


def generate_classification_data(
    n_vars, n_samples, expected_degree, noise_scale, seed, min_y_parents=1, max_attempts=100
):
    """Generate synthetic data with ground truth DAG and effects.

    Rejects DAGs where Y (last node) has fewer than min_y_parents parents,
    since such DGPs make classification and effect comparison vacuous.
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

    return jnp.array(X, dtype=jnp.float32), Y_binary, A_true, A


def extract_mb_from_adjacency(A, target_idx, n_vars, threshold=0.3):
    """Extract Markov Blanket of target from adjacency matrix.

    MB(Y) = parents(Y) ∪ children(Y) ∪ spouses(Y).
    """
    A_np = np.array(A)
    d = A_np.shape[0]
    A_bin = (np.abs(A_np) > threshold).astype(int)

    parents = set()
    children = set()
    spouses = set()

    # Parents: A[target, j] != 0 means j -> target
    for j in range(d):
        if j != target_idx and A_bin[target_idx, j]:
            parents.add(j)

    # Children: A[j, target] != 0 means target -> j
    for j in range(d):
        if j != target_idx and A_bin[j, target_idx]:
            children.add(j)

    # Spouses: other parents of children of target
    for child in children:
        for j in range(d):
            if j != target_idx and j != child and A_bin[child, j]:
                spouses.add(j)

    mb = parents | children | spouses
    # Filter to feature variables only (exclude Y itself)
    mb = sorted([v for v in mb if v < n_vars])
    return mb


def compute_ground_truth_ates(A, Y_idx, n_vars):
    """Compute ground truth total causal effects from linear SCM."""
    scm = LinearSCM(A, SCMConfig())
    B = scm.compute_total_effects()
    return {j: float(B[Y_idx, j]) for j in range(n_vars)}


def compute_ground_truth_mb(A, Y_idx, n_vars):
    """Extract ground truth MB from the true DAG."""
    return extract_mb_from_adjacency(A, Y_idx, n_vars, threshold=1e-6)


def mb_f1(pred_mb, true_mb, n_vars):
    """Compute F1 for Markov Blanket recovery."""
    pred_set = set(pred_mb)
    true_set = set(true_mb)
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(true_set) if true_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


# ============================================================================
# Pipeline A: PC → Feature Selection → Classifier → ATE
# ============================================================================


def ols_ate(X, Y, treatment_idx, confounders):
    """Estimate ATE of X[:, treatment_idx] on Y using OLS adjustment.

    Fits Y = β_t * T + β_c * X_conf + intercept via least squares.
    Returns β_t as the ATE estimate.
    """
    T = X[:, treatment_idx : treatment_idx + 1]  # (n, 1)

    if confounders:
        conf_indices = [c for c in confounders if c != treatment_idx]
        if conf_indices:
            X_conf = X[:, np.array(conf_indices)]
            design = np.hstack(
                [
                    np.ones((len(X), 1)),
                    np.array(T),
                    np.array(X_conf),
                ]
            )
        else:
            design = np.hstack([np.ones((len(X), 1)), np.array(T)])
    else:
        design = np.hstack([np.ones((len(X), 1)), np.array(T)])

    Y_np = np.array(Y).reshape(-1)
    # OLS: β = (X^T X)^{-1} X^T Y
    beta = np.linalg.lstsq(design, Y_np, rcond=None)[0]
    return float(beta[1])  # Coefficient of T


def logistic_regression_bacc(X_train, Y_train, X_test, Y_test):
    """Simple logistic regression via gradient descent. Returns balanced accuracy."""
    n_features = X_train.shape[1]
    if n_features == 0:
        return 0.5

    X_tr = np.array(X_train)
    Y_tr = np.array(Y_train).reshape(-1)
    X_te = np.array(X_test)
    Y_te = np.array(Y_test).reshape(-1)

    # Add intercept
    X_tr = np.hstack([np.ones((len(X_tr), 1)), X_tr])
    X_te = np.hstack([np.ones((len(X_te), 1)), X_te])

    # Initialize weights
    w = np.zeros(X_tr.shape[1])

    # Gradient descent
    lr = 0.01
    for _ in range(200):
        logits = X_tr @ w
        probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        grad = X_tr.T @ (probs - Y_tr) / len(Y_tr)
        w -= lr * grad

    # Predict on test
    logits = X_te @ w
    preds = (logits > 0).astype(float)

    # Balanced accuracy
    pos = Y_te == 1
    neg = Y_te == 0
    tpr = np.mean(preds[pos] == Y_te[pos]) if np.any(pos) else 0.5
    tnr = np.mean(preds[neg] == Y_te[neg]) if np.any(neg) else 0.5
    return float(0.5 * (tpr + tnr))


def run_pipeline_a(X, Y, n_vars, A_true_full, true_ates, true_mb, seed):
    """Pipeline A: PC → MB selection → logistic regression → OLS ATE."""
    print("\n  [Pipeline A] PC → Select → Classify → ATE")
    t0 = time.time()

    # Step 1: PC algorithm on full data (X + Y column)
    Y_col = np.array(Y).reshape(-1, 1)
    data_full = jnp.hstack([X, jnp.array(Y_col, dtype=jnp.float32)])
    key = random.PRNGKey(seed)
    A_pc = learn_with_pc(data_full, key, alpha=0.05, max_cond_size=2, verbose=False)
    A_pc_np = np.array(A_pc)

    # Step 2: Extract MB
    Y_idx = n_vars
    mb_pc = extract_mb_from_adjacency(A_pc_np, Y_idx, n_vars, threshold=0.5)

    # Structure metrics (PC on X-only subgraph)
    A_pc_X = A_pc_np[:n_vars, :n_vars]
    A_true_X = np.array(A_true_full)[:n_vars, :n_vars]
    struct_metrics = compute_structure_metrics(jnp.array(A_pc_X), jnp.array(A_true_X))

    # Step 3: Train-test split (80/20)
    n = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    if mb_pc:
        X_mb_train = np.array(X[train_idx][:, np.array(mb_pc)])
        X_mb_test = np.array(X[test_idx][:, np.array(mb_pc)])
    else:
        print("    WARNING: PC found empty MB — falling back to all features")
        X_mb_train = np.array(X[train_idx])
        X_mb_test = np.array(X[test_idx])
        mb_pc = list(range(n_vars))

    bacc = logistic_regression_bacc(X_mb_train, Y[train_idx], X_mb_test, Y[test_idx])

    # Step 4: Estimate ATEs via OLS
    pred_ates = {}
    for t_idx in mb_pc:
        confounders = [c for c in mb_pc if c != t_idx]
        ate = ols_ate(np.array(X), np.array(Y), t_idx, confounders)
        pred_ates[t_idx] = ate

    # ATE quality
    common_vars = set(pred_ates.keys()) & set(true_ates.keys())
    if common_vars:
        tau_pred = jnp.array([pred_ates[v] for v in sorted(common_vars)])
        tau_true = jnp.array([true_ates[v] for v in sorted(common_vars)])
        pehe = compute_pehe(tau_pred, tau_true)
        mean_ate_err = float(np.mean([abs(pred_ates[v] - true_ates[v]) for v in common_vars]))
    else:
        pehe = float("inf")
        mean_ate_err = float("inf")

    # MB F1
    mb_f1_val, mb_prec, mb_rec = mb_f1(mb_pc, true_mb, n_vars)

    elapsed = time.time() - t0

    result = {
        "method": "Pipeline A (PC)",
        "shd": struct_metrics["shd"],
        "struct_f1": struct_metrics["f1"],
        "bacc": bacc,
        "mb_size": len(mb_pc),
        "mb_f1": mb_f1_val,
        "pehe": pehe,
        "mean_ate_error": mean_ate_err,
        "n_ate_matched": len(common_vars),
        "time": elapsed,
    }
    print(
        f"    SHD={result['shd']} BAcc={bacc:.3f} MB_F1={mb_f1_val:.3f} "
        f"PEHE={pehe:.4f} ({elapsed:.1f}s)"
    )
    return result


# ============================================================================
# Pipeline B: GOLEM structure-only → MB → Classify → ATE
# ============================================================================


def run_pipeline_b(X, Y, n_vars, A_true_full, true_ates, true_mb, n_trials, max_iter, seed):
    """Pipeline B: GOLEM (no classification) → MB → classifier → OLS ATE."""
    print("\n  [Pipeline B] GOLEM(λ_class=0) → Select → Classify → ATE")
    t0 = time.time()

    # Step 1: Run GOLEM with lambda_class=0 via suggest_fn override
    def suggest_fn_no_class(trial, use_v7=True):
        from jcce.structure_learning.optuna_search import suggest_hyperparams

        config = suggest_hyperparams(trial, use_v7=use_v7)
        config["lambda_class"] = 0.0  # Structure-only
        return config

    sampler = TPESampler(
        multivariate=True,
        group=True,
        seed=seed,
        n_startup_trials=min(5, n_trials),
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
        n_vars=n_vars,
        max_iter=max_iter,
        use_v7=True,
        jax_key_seed=seed,
        verbose=False,
        suggest_fn=suggest_fn_no_class,
    )
    study.optimize(objective, n_trials=n_trials)

    enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)

    # Pick best feasible solution by structure quality
    best_sol = None
    best_shd = float("inf")
    for sol in enhanced_solutions:
        A_est = sol["metrics"].get("structure_A_est")
        if A_est is not None:
            sm = compute_structure_metrics(
                jnp.array(np.array(A_est)[:n_vars, :n_vars]),
                jnp.array(np.array(A_true_full)[:n_vars, :n_vars]),
            )
            if sm["shd"] < best_shd:
                best_shd = sm["shd"]
                best_sol = sol

    if best_sol is None and enhanced_solutions:
        best_sol = enhanced_solutions[0]

    if best_sol is None:
        elapsed = time.time() - t0
        result = {
            "method": "Pipeline B (GOLEM→Classify)",
            "shd": -1,
            "struct_f1": 0.0,
            "bacc": 0.5,
            "mb_size": 0,
            "mb_f1": 0.0,
            "pehe": float("inf"),
            "mean_ate_error": float("inf"),
            "n_ate_matched": 0,
            "time": elapsed,
        }
        print(f"    No feasible solutions ({elapsed:.1f}s)")
        return result

    # Step 2: Extract MB from learned structure
    A_est = np.array(best_sol["metrics"].get("structure_A_est", np.zeros((n_vars + 1, n_vars + 1))))
    mb_b = extract_mb_from_adjacency(A_est, n_vars, n_vars, threshold=0.3)

    A_est_X = A_est[:n_vars, :n_vars]
    A_true_X = np.array(A_true_full)[:n_vars, :n_vars]
    struct_metrics = compute_structure_metrics(jnp.array(A_est_X), jnp.array(A_true_X))

    # Step 3: Train classifier on MB features
    n = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    if mb_b:
        X_mb_train = np.array(X[train_idx][:, np.array(mb_b)])
        X_mb_test = np.array(X[test_idx][:, np.array(mb_b)])
    else:
        print("    WARNING: GOLEM found empty MB — falling back to all features")
        X_mb_train = np.array(X[train_idx])
        X_mb_test = np.array(X[test_idx])
        mb_b = list(range(n_vars))

    bacc = logistic_regression_bacc(X_mb_train, Y[train_idx], X_mb_test, Y[test_idx])

    # Step 4: Estimate ATEs via OLS
    pred_ates = {}
    for t_idx in mb_b:
        confounders = [c for c in mb_b if c != t_idx]
        ate = ols_ate(np.array(X), np.array(Y), t_idx, confounders)
        pred_ates[t_idx] = ate

    # ATE quality
    common_vars = set(pred_ates.keys()) & set(true_ates.keys())
    if common_vars:
        tau_pred = jnp.array([pred_ates[v] for v in sorted(common_vars)])
        tau_true = jnp.array([true_ates[v] for v in sorted(common_vars)])
        pehe = compute_pehe(tau_pred, tau_true)
        mean_ate_err = float(np.mean([abs(pred_ates[v] - true_ates[v]) for v in common_vars]))
    else:
        pehe = float("inf")
        mean_ate_err = float("inf")

    mb_f1_val, _, _ = mb_f1(mb_b, true_mb, n_vars)

    elapsed = time.time() - t0
    result = {
        "method": "Pipeline B (GOLEM→Classify)",
        "shd": struct_metrics["shd"],
        "struct_f1": struct_metrics["f1"],
        "bacc": bacc,
        "mb_size": len(mb_b),
        "mb_f1": mb_f1_val,
        "pehe": pehe,
        "mean_ate_error": mean_ate_err,
        "n_ate_matched": len(common_vars),
        "time": elapsed,
    }
    print(
        f"    SHD={result['shd']} BAcc={bacc:.3f} MB_F1={mb_f1_val:.3f} "
        f"PEHE={pehe:.4f} ({elapsed:.1f}s)"
    )
    return result


# ============================================================================
# JCCE Unified
# ============================================================================


def run_jcce_unified(X, Y, n_vars, A_true_full, true_ates, true_mb, n_trials, max_iter, seed):
    """JCCE Unified: GOLEM v7 (structure + classification + effects in one pass)."""
    print("\n  [JCCE Unified] GOLEM v7 (all-in-one)")
    t0 = time.time()

    sampler = TPESampler(
        multivariate=True,
        group=True,
        seed=seed,
        n_startup_trials=min(5, n_trials),
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
        n_vars=n_vars,
        max_iter=max_iter,
        use_v7=True,
        jax_key_seed=seed,
        verbose=False,
    )
    study.optimize(objective, n_trials=n_trials)

    enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)

    # Pick best solution by BAcc
    best_sol = None
    best_bacc = 0.0
    for sol in enhanced_solutions:
        if sol["objectives"][0] > best_bacc:
            best_bacc = sol["objectives"][0]
            best_sol = sol

    if best_sol is None:
        elapsed = time.time() - t0
        result = {
            "method": "JCCE Unified",
            "shd": -1,
            "struct_f1": 0.0,
            "bacc": 0.5,
            "mb_size": 0,
            "mb_f1": 0.0,
            "pehe": float("inf"),
            "mean_ate_error": float("inf"),
            "n_ate_matched": 0,
            "time": elapsed,
        }
        print(f"    No feasible solutions ({elapsed:.1f}s)")
        return result

    # Structure metrics
    A_est = np.array(best_sol["metrics"].get("structure_A_est", np.zeros((n_vars + 1, n_vars + 1))))
    A_est_X = A_est[:n_vars, :n_vars]
    A_true_X = np.array(A_true_full)[:n_vars, :n_vars]
    struct_metrics = compute_structure_metrics(jnp.array(A_est_X), jnp.array(A_true_X))

    bacc = best_sol["objectives"][0]

    # MB recovery
    mb_jcce = extract_mb_from_adjacency(A_est, n_vars, n_vars, threshold=0.3)
    mb_f1_val, _, _ = mb_f1(mb_jcce, true_mb, n_vars)

    # ATE quality from v7 causal_effects
    effects = best_sol["metrics"].get("causal_effects", {})
    pred_ates = {}
    for key, val in effects.items():
        if "->Y" in key or "→Y" in key:
            var_str = key.split("->")[0] if "->" in key else key.split("→")[0]
            var_idx = int(var_str.replace("X", ""))
            pred_ates[var_idx] = float(val)

    common_vars = set(pred_ates.keys()) & set(true_ates.keys())
    if common_vars:
        tau_pred = jnp.array([pred_ates[v] for v in sorted(common_vars)])
        tau_true = jnp.array([true_ates[v] for v in sorted(common_vars)])
        pehe = compute_pehe(tau_pred, tau_true)
        mean_ate_err = float(np.mean([abs(pred_ates[v] - true_ates[v]) for v in common_vars]))
    else:
        pehe = float("inf")
        mean_ate_err = float("inf")

    elapsed = time.time() - t0
    result = {
        "method": "JCCE Unified",
        "shd": struct_metrics["shd"],
        "struct_f1": struct_metrics["f1"],
        "bacc": bacc,
        "mb_size": len(mb_jcce),
        "mb_f1": mb_f1_val,
        "pehe": pehe,
        "mean_ate_error": mean_ate_err,
        "n_ate_matched": len(common_vars),
        "time": elapsed,
    }
    print(
        f"    SHD={result['shd']} BAcc={bacc:.3f} MB_F1={mb_f1_val:.3f} "
        f"PEHE={pehe:.4f} ({elapsed:.1f}s)"
    )
    return result


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="C.2: Unified vs Pipeline")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-vars", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--expected-degree", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 5 trials, 20 iters, 5 vars"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Real dataset name (lucas, sachs, etc.)"
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.max_iter = 20
        args.n_vars = 5

    print("UNIFIED VS PIPELINE COMPARISON (C.2)")
    print("=" * 70)

    if args.dataset:
        from jcce.data.benchmark_loader import load_dataset as load_benchmark

        X_np, Y_np, config = load_benchmark(args.dataset)
        X = jnp.array(X_np, dtype=jnp.float32)
        Y = jnp.array(Y_np, dtype=jnp.float32)
        args.n_vars = X.shape[1]
        A_full = config.get("true_dag")
        has_ground_truth = config.get("has_true_dag", False) and A_full is not None
        print(f"Dataset: {args.dataset} ({X.shape[0]} samples, {args.n_vars} vars)")
    else:
        has_ground_truth = True
        X, Y, A_true, A_full = generate_classification_data(
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

    if has_ground_truth:
        if args.dataset:
            A_true = np.array(A_full)[: args.n_vars, : args.n_vars]
        n_edges = int(np.sum(np.abs(A_true) > 1e-6))
        Y_idx = args.n_vars
        true_ates = compute_ground_truth_ates(A_full, Y_idx, args.n_vars)
        true_mb = compute_ground_truth_mb(A_full, Y_idx, args.n_vars)
    else:
        n_edges = -1
        Y_idx = args.n_vars
        true_ates = {}
        true_mb = []
        A_full = np.zeros((args.n_vars + 1, args.n_vars + 1))
    print(f"Ground truth: {n_edges} edges, MB(Y) = {true_mb} ({len(true_mb)} members)")

    all_results = []

    # Run all three methods
    r_a = run_pipeline_a(X, Y, args.n_vars, A_full, true_ates, true_mb, args.seed)
    all_results.append(r_a)

    r_b = run_pipeline_b(
        X, Y, args.n_vars, A_full, true_ates, true_mb, args.n_trials, args.max_iter, args.seed
    )
    all_results.append(r_b)

    r_u = run_jcce_unified(
        X, Y, args.n_vars, A_full, true_ates, true_mb, args.n_trials, args.max_iter, args.seed
    )
    all_results.append(r_u)

    # Summary table
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY\n")
    print(
        f"{'Method':<30} {'SHD':>5} {'F1':>6} {'BAcc':>6} "
        f"{'MB_F1':>6} {'PEHE':>8} {'ATEErr':>8} {'Time':>7}"
    )
    print(f"{'-' * 30} {'-' * 5} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 7}")

    for r in all_results:
        pehe_str = f"{r['pehe']:>8.4f}" if r["pehe"] < float("inf") else "     inf"
        ate_str = (
            f"{r['mean_ate_error']:>8.4f}" if r["mean_ate_error"] < float("inf") else "     inf"
        )
        print(
            f"{r['method']:<30} {r['shd']:>5} {r['struct_f1']:>6.3f} "
            f"{r['bacc']:>6.3f} {r['mb_f1']:>6.3f} "
            f"{pehe_str} {ate_str} {r['time']:>7.1f}"
        )

    # Analysis
    print("\nAnalysis:")
    unified = next((r for r in all_results if "Unified" in r["method"]), None)
    if unified:
        for r in all_results:
            if "Unified" not in r["method"]:
                delta_bacc = unified["bacc"] - r["bacc"]
                delta_f1 = unified["struct_f1"] - r["struct_f1"]
                print(f"  Unified vs {r['method']}: ΔBAcc={delta_bacc:+.3f} ΔF1={delta_f1:+.3f}")
                if r["pehe"] < float("inf") and unified["pehe"] < float("inf"):
                    delta_pehe = unified["pehe"] - r["pehe"]
                    print(
                        f"    ΔPEHE={delta_pehe:+.4f} ({'better' if delta_pehe < 0 else 'worse'})"
                    )

    # Save results
    if args.output:
        output = {
            "args": vars(args),
            "n_edges": n_edges,
            "true_mb": true_mb,
            "true_ates": {str(k): v for k, v in true_ates.items()},
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
