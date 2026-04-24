#!/usr/bin/env python
"""
Experiment C.10: ATE Quality Evaluation.

Evaluates treatment effect estimation quality by comparing JCCE's predicted
ATEs against ground truth from synthetic linear SCMs.

For each Pareto solution from an Optuna search:
1. Extract predicted causal_effects (Xi->Y ATEs from DragonNet)
2. Compare against ground truth effects from (I-A)^{-1}
3. Report PEHE, ATE error, ATE bias across the Pareto front

Usage:
    uv run python scripts/experiment_c10_ate_quality.py --quick
    uv run python scripts/experiment_c10_ate_quality.py \
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
from jcce.evaluation.effect_metrics import (
    compute_ate_bias,
    compute_pehe,
)
from jcce.structure_learning.optuna_search import (
    constraints_func,
    create_optuna_objective,
    extract_pareto_solutions,
)

# ============================================================================
# Ground Truth Effects
# ============================================================================


def compute_ground_truth_ates(A, Y_idx):
    """Compute ground truth ATE for each variable -> Y.

    For linear SCM X = (I-A)^{-1}ε, the total causal effect of X_j on X_i
    is B[i,j] where B = (I-A)^{-1}.

    Returns:
        dict mapping variable index -> true ATE (B[Y_idx, var_idx])
    """
    scm = LinearSCM(A, SCMConfig())
    B = scm.compute_total_effects()
    true_ates = {}
    for j in range(A.shape[0]):
        if j != Y_idx:
            true_ates[j] = float(B[Y_idx, j])
    return true_ates


# ============================================================================
# ATE Quality Evaluation
# ============================================================================


def evaluate_ate_quality(enhanced_solutions, true_ates, n_vars):
    """Evaluate ATE quality for each Pareto solution.

    Returns list of per-solution ATE metrics.
    """
    results = []
    Y_idx = n_vars  # Y is the last node

    for sol in enhanced_solutions:
        effects = sol["metrics"].get("causal_effects", {})
        if not effects:
            results.append(
                {
                    "has_effects": False,
                    "ate_errors": {},
                    "mean_ate_error": float("inf"),
                    "pehe": float("inf"),
                }
            )
            continue

        # Extract predicted ATEs for Xi->Y pairs
        pred_ates = {}
        for key, val in effects.items():
            if "->Y" in key or "→Y" in key:
                # Parse 'X3->Y' or 'X3→Y'
                var_str = key.split("->")[0] if "->" in key else key.split("→")[0]
                var_idx = int(var_str.replace("X", ""))
                pred_ates[var_idx] = float(val)

        # Compare predicted vs true ATEs
        common_vars = set(pred_ates.keys()) & set(true_ates.keys())

        if not common_vars:
            results.append(
                {
                    "has_effects": True,
                    "n_effects": len(pred_ates),
                    "n_matched": 0,
                    "ate_errors": {},
                    "mean_ate_error": float("inf"),
                    "pehe": float("inf"),
                }
            )
            continue

        ate_errors = {}
        tau_pred_list = []
        tau_true_list = []
        for v in sorted(common_vars):
            pred = pred_ates[v]
            true = true_ates[v]
            ate_errors[v] = abs(pred - true)
            tau_pred_list.append(pred)
            tau_true_list.append(true)

        tau_pred = jnp.array(tau_pred_list)
        tau_true = jnp.array(tau_true_list)

        pehe = compute_pehe(tau_pred, tau_true)
        mean_ate_error = float(np.mean(list(ate_errors.values())))
        ate_bias = compute_ate_bias(tau_pred, float(jnp.mean(tau_true)))

        results.append(
            {
                "has_effects": True,
                "n_effects": len(pred_ates),
                "n_matched": len(common_vars),
                "ate_errors": ate_errors,
                "mean_ate_error": mean_ate_error,
                "pehe": pehe,
                "ate_bias": ate_bias,
                "pred_ates": pred_ates,
            }
        )

    return results


# ============================================================================
# Data Generation
# ============================================================================


def generate_classification_data(
    n_vars, n_samples, expected_degree, noise_scale, seed, min_y_parents=1, max_attempts=100
):
    """Generate synthetic classification data with known ground truth effects.

    Rejects DAGs where Y (last node) has fewer than min_y_parents parents,
    since such DGPs make ATE quality comparison vacuous.
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


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="C.10: ATE Quality Evaluation")
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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.max_iter = 20
        args.n_vars = 5

    print("ATE QUALITY EVALUATION (C.10)")
    print("=" * 70)
    print(
        f"Data: {args.n_samples} samples, {args.n_vars} vars, "
        f"ER-{args.expected_degree}, noise={args.noise_scale}"
    )
    print(f"Budget: {args.n_trials} trials x {args.max_iter} iters")

    # Generate data
    X, Y, A_true, A_full = generate_classification_data(
        args.n_vars,
        args.n_samples,
        args.expected_degree,
        args.noise_scale,
        args.seed,
    )
    n_edges = int(np.sum(np.abs(A_true) > 1e-6))
    Y_idx = args.n_vars  # Y is the last node

    # Compute ground truth effects
    true_ates = compute_ground_truth_ates(A_full, Y_idx)
    print(f"Ground truth: {n_edges} edges among {args.n_vars} vars")
    print("Ground truth ATEs (Xi->Y):")
    for v in sorted(true_ates.keys()):
        if v < args.n_vars:
            print(f"  X{v}->Y: {true_ates[v]:+.4f}")

    # Run Optuna search
    print(f"\nRunning {args.n_trials} trials...")
    sampler = TPESampler(
        multivariate=True,
        group=True,
        seed=args.seed,
        n_startup_trials=min(5, args.n_trials),
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
        n_vars=args.n_vars,
        max_iter=args.max_iter,
        use_v7=True,
        jax_key_seed=args.seed,
        verbose=False,
    )

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    elapsed = time.time() - t0

    enhanced_solutions, _ = extract_pareto_solutions(study, use_v7=True)
    print(f"Completed in {elapsed:.1f}s: {len(enhanced_solutions)} Pareto solutions")

    # Evaluate ATE quality
    ate_results = evaluate_ate_quality(enhanced_solutions, true_ates, args.n_vars)

    # Summary
    print(f"\n{'=' * 70}")
    print("ATE QUALITY SUMMARY\n")

    valid_results = [r for r in ate_results if r.get("n_matched", 0) > 0]

    if valid_results:
        print(f"{'Sol#':>5} {'BAcc':>6} {'PEHE':>8} {'MeanErr':>8} {'ATEBias':>8} {'#Match':>7}")
        print(f"{'-' * 5} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")

        for i, (sol, ate_r) in enumerate(zip(enhanced_solutions, ate_results)):
            if ate_r.get("n_matched", 0) > 0:
                bacc = sol["objectives"][0]
                print(
                    f"{i:>5} {bacc:>6.3f} {ate_r['pehe']:>8.4f} "
                    f"{ate_r['mean_ate_error']:>8.4f} "
                    f"{ate_r.get('ate_bias', 0):>+8.4f} "
                    f"{ate_r['n_matched']:>7}"
                )

        # Aggregate metrics
        pehes = [r["pehe"] for r in valid_results]
        ate_errors = [r["mean_ate_error"] for r in valid_results]
        print(f"\nAggregate (across {len(valid_results)} solutions with effects):")
        print(f"  Mean PEHE:       {np.mean(pehes):.4f} (std={np.std(pehes):.4f})")
        print(f"  Best PEHE:       {min(pehes):.4f}")
        print(f"  Mean ATE Error:  {np.mean(ate_errors):.4f}")
        print(f"  Best ATE Error:  {min(ate_errors):.4f}")

        # Per-variable analysis
        print("\nPer-Variable ATE Analysis (best solution by PEHE):")
        best_idx = np.argmin(pehes)
        best_r = valid_results[best_idx]
        for v in sorted(true_ates.keys()):
            if v < args.n_vars and v in best_r.get("pred_ates", {}):
                pred = best_r["pred_ates"][v]
                true = true_ates[v]
                err = abs(pred - true)
                print(f"  X{v}->Y: true={true:+.4f} pred={pred:+.4f} err={err:.4f}")
    else:
        print("  No Pareto solutions with effect estimates found.")

    # Save results
    if args.output:
        output = {
            "args": vars(args),
            "n_edges": n_edges,
            "true_ates": {str(k): v for k, v in true_ates.items()},
            "n_pareto": len(enhanced_solutions),
            "ate_results": [
                {k: v for k, v in r.items() if k not in ("ate_errors", "pred_ates")}
                for r in ate_results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
