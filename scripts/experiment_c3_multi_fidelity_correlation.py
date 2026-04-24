#!/usr/bin/env python
"""
Experiment C.3: Multi-Fidelity Rank Correlation Analysis.

For N configs, run GOLEM to full budget recording metrics at every 10 iters.
Extract at rungs [30, 90, 270, max_iter]. Compute Spearman rho(metric@rung,
metric@final). Validates Hyperband-style pruning for causal discovery.

Usage:
    uv run python scripts/experiment_c3_multi_fidelity_correlation.py --quick
    uv run python scripts/experiment_c3_multi_fidelity_correlation.py \
        --n-trials 30 --n-vars 10 --n-samples 500 --max-iter 300
"""

import argparse
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np
from jax import random
from scipy import stats

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import optuna

from jcce.data.dag_generator import DAGConfig, generate_dag
from jcce.data.scm import LinearSCM, SCMConfig
from jcce.structure_learning.jcce_learner import (
    create_processor,
    learn_structure,
)
from jcce.structure_learning.optuna_search import suggest_hyperparams
from jcce.utils.metrics import compute_structure_metrics

# ============================================================================
# Recording Callback
# ============================================================================


class RecordingCallback:
    """Stores metrics at every 10-iter checkpoint during GOLEM training."""

    def __init__(self):
        self.records = {}  # iter_num -> {balanced_accuracy, h_A, loss}

    def __call__(self, iter_num: int, metrics: dict) -> None:
        self.records[iter_num] = {
            "balanced_accuracy": metrics["balanced_accuracy"],
            "h_A": metrics["h_A"],
            "loss": metrics["loss"],
        }

    def get_at_rung(self, rung: int):
        """Return metrics at the exact rung, or nearest-below recorded iteration."""
        if rung in self.records:
            return self.records[rung]
        # Find nearest-below
        available = sorted(k for k in self.records if k <= rung)
        if not available:
            return None
        return self.records[available[-1]]


# ============================================================================
# Data Generation
# ============================================================================


def generate_classification_data(
    n_vars, n_samples, expected_degree, noise_scale, seed, min_y_parents=1, max_attempts=100
):
    """Generate synthetic classification data from a linear SEM.

    Creates an ER DAG with n_vars+1 nodes, samples via LinearSCM, then
    binarizes the last node at median to create Y.

    Uses rejection sampling to ensure Y (last node) has at least
    min_y_parents parents, avoiding degenerate DGPs where Y is pure noise.

    Returns:
        (X_jnp, Y_jnp, A_true_np) where A_true is (n_vars, n_vars).
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

    # Last node becomes binary Y (threshold at median)
    Y_continuous = X_full[:, -1]
    Y_binary = (Y_continuous > jnp.median(Y_continuous)).astype(jnp.float32)

    # Remove last col from X
    X = X_full[:, :n_vars]
    A_true = np.array(A)[:n_vars, :n_vars]

    return jnp.array(X, dtype=jnp.float32), Y_binary, A_true


# ============================================================================
# Single Trial Runner
# ============================================================================


def run_single_trial(trial_idx, config, X, Y, n_vars, max_iter, seed):
    """Run a single GOLEM trial to completion with recording callback.

    Returns dict with config, records, final_metrics, A_est.
    """
    key = random.PRNGKey(seed + trial_idx * 100)
    key, proc_key, train_key = random.split(key, 3)

    processor = create_processor(
        config["processor_type"],
        key=proc_key,
        n_features=n_vars,
        **config["processor_config"],
    )

    callback = RecordingCallback()
    Y_for_v7 = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    Y_idx = n_vars

    try:
        A_est, _, _, metrics = learn_structure(
            data=X,
            Y=Y_for_v7,
            Y_idx=Y_idx,
            processor=processor,
            key=train_key,
            processor_type=config["processor_type"],
            lambda_1=config["lambda_1"],
            lambda_2_init=config["lambda_2"],
            lambda_class=config["lambda_class"],
            lr=config["lr"],
            max_iter=max_iter,
            patience=max_iter + 1,  # No early stopping — run to completion
            verbose=0,
            task="classification",
            use_adaptive_curriculum=True,
            lambda_ident=0.01,
            use_amortized_effects=True,
            enforce_outcome_sink=True,
            iteration_callback=callback,
            effect_hidden_dim=config.get("effect_hidden_dim", 64),
            effect_embed_dim=config.get("effect_embed_dim", 16),
            lambda_effect=config.get("lambda_effect", 10.0),
            effect_warmup_iter=config.get("effect_warmup_iter", 20),
            lambda_confound_sparse=config.get("lambda_confound_sparse", 0.05),
            lambda_bow=config.get("lambda_bow_v7", 0.3),
            effect_refinement_iters=config.get("effect_refinement_iters", 50),
        )
    except Exception as e:
        print(f"  Trial {trial_idx} failed: {e}")
        return None

    # Compute structure metrics on X-only subgraph
    A_est_X = np.array(A_est)[:n_vars, :n_vars]
    struct_metrics = compute_structure_metrics(
        jnp.array(A_est_X), jnp.array(np.zeros((n_vars, n_vars)))
    )

    return {
        "config": config,
        "records": callback.records,
        "callback": callback,
        "final_metrics": {
            "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
            "h_A": float(metrics.get("final_h_A", 1.0)),
            "loss": float(callback.records[max(callback.records.keys())]["loss"])
            if callback.records
            else 0.0,
            "f1": struct_metrics["f1"],
            "shd": struct_metrics["shd"],
        },
        "A_est": np.array(A_est),
    }


# ============================================================================
# Rank Correlation Computation
# ============================================================================


def compute_rank_correlations(trial_results, rungs):
    """Compute Spearman rank correlations between rung metrics and final metrics.

    Args:
        trial_results: List of dicts from run_single_trial (None entries filtered).
        rungs: List of rung iteration numbers.

    Returns:
        Dict mapping (rung_metric, rung, final_metric) -> (rho, pvalue).
    """
    results = [r for r in trial_results if r is not None]
    if len(results) < 3:
        return {}

    rung_metrics = ["balanced_accuracy", "h_A", "loss"]
    final_metrics = ["balanced_accuracy", "h_A", "loss", "f1", "shd"]

    correlations = {}

    for rung in rungs:
        for rm in rung_metrics:
            rung_values = []
            for r in results:
                at_rung = r["callback"].get_at_rung(rung)
                if at_rung is not None:
                    rung_values.append(at_rung[rm])
                else:
                    rung_values.append(None)

            for fm in final_metrics:
                final_values = [r["final_metrics"][fm] for r in results]

                # Filter pairs where rung value is available
                pairs = [(rv, fv) for rv, fv in zip(rung_values, final_values) if rv is not None]

                if len(pairs) < 3:
                    continue

                rv_arr = np.array([p[0] for p in pairs])
                fv_arr = np.array([p[1] for p in pairs])

                # Skip if constant
                if np.std(rv_arr) < 1e-10 or np.std(fv_arr) < 1e-10:
                    correlations[(rm, rung, fm)] = (0.0, 1.0)
                    continue

                rho, pval = stats.spearmanr(rv_arr, fv_arr)
                correlations[(rm, rung, fm)] = (float(rho), float(pval))

    return correlations


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="C.3: Multi-Fidelity Rank Correlations")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-vars", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--expected-degree", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 5 trials, 30 iters, 5 vars"
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.max_iter = 30
        args.n_vars = 5

    rungs = sorted(set([30, 90, 270, args.max_iter]) & set(range(0, args.max_iter + 1)))
    # Ensure final rung is always included
    if args.max_iter not in rungs:
        rungs.append(args.max_iter)
    # For short runs, adjust rungs to what's achievable
    rungs = [r for r in rungs if r <= args.max_iter]

    print("MULTI-FIDELITY RANK CORRELATIONS (C.3)")
    print("=" * 60)

    # Generate data
    print(
        f"\nGenerating data: {args.n_samples} samples, {args.n_vars} vars, "
        f"ER-{args.expected_degree} linear SEM"
    )
    X, Y, A_true = generate_classification_data(
        args.n_vars, args.n_samples, args.expected_degree, args.noise_scale, args.seed
    )
    n_edges = int(np.sum(np.abs(A_true) > 1e-6))
    print(f"Generated DAG: {n_edges} edges")
    print(f"Rungs: {rungs}")

    # Generate configs via Optuna's suggest_hyperparams
    print(f"\nRunning {args.n_trials} trials to max_iter={args.max_iter}...")
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.RandomSampler(seed=args.seed),
    )

    configs = []
    for i in range(args.n_trials):
        trial = study.ask()
        config = suggest_hyperparams(trial, use_v7=True)
        configs.append(config)
        study.tell(trial, [0.0, 0.0])  # Dummy values

    # Run trials
    trial_results = []
    t0 = time.time()
    for i, config in enumerate(configs):
        t_trial = time.time()
        print(
            f"  Trial {i + 1}/{args.n_trials} ({config['processor_type']})...", end="", flush=True
        )
        result = run_single_trial(i, config, X, Y, args.n_vars, args.max_iter, args.seed)
        trial_results.append(result)
        if result is not None:
            print(
                f" BAcc={result['final_metrics']['balanced_accuracy']:.3f} "
                f"h_A={result['final_metrics']['h_A']:.4f} "
                f"({time.time() - t_trial:.1f}s)"
            )
        else:
            print(f" FAILED ({time.time() - t_trial:.1f}s)")

    total_time = time.time() - t0
    n_success = sum(1 for r in trial_results if r is not None)
    print(f"\nCompleted: {n_success}/{args.n_trials} in {total_time:.1f}s")

    # Compute rank correlations
    correlations = compute_rank_correlations(trial_results, rungs)

    # Display results
    print(f"\n{'=' * 60}")
    print(
        f"Data: {args.n_samples} samples, {args.n_vars} vars, "
        f"ER-{args.expected_degree} linear SEM, {n_edges} edges\n"
    )

    final_targets = ["balanced_accuracy", "shd"]
    rung_metrics = ["balanced_accuracy", "h_A", "loss"]

    for ft in final_targets:
        ft_label = "BAcc" if ft == "balanced_accuracy" else ft.upper()
        print(f"rho(metric@rung -> {ft_label}@final):")
        for rung in rungs:
            parts = []
            for rm in rung_metrics:
                key = (rm, rung, ft)
                if key in correlations:
                    rho, pval = correlations[key]
                    rm_label = {"balanced_accuracy": "BAcc", "h_A": "h_A", "loss": "loss"}[rm]
                    p_str = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
                    parts.append(f"{rm_label}@{rung}: {rho:+.2f} ({p_str})")
                else:
                    rm_label = {"balanced_accuracy": "BAcc", "h_A": "h_A", "loss": "loss"}[rm]
                    parts.append(f"{rm_label}@{rung}: N/A")
            print(f"  {'    '.join(parts)}")
        print()

    # Conclusion
    best_rho = 0.0
    best_rung = None
    for (rm, rung, fm), (rho, pval) in correlations.items():
        if fm == "balanced_accuracy" and abs(rho) > best_rho:
            best_rho = abs(rho)
            best_rung = rung
    if best_rung is not None:
        budget_pct = best_rung / args.max_iter * 100
        verdict = "supports" if best_rho >= 0.7 else "weak support for"
        print(
            f"Conclusion: best rho={best_rho:.2f} at rung {best_rung} "
            f"({budget_pct:.0f}% budget) -> {verdict} pruning at that rung."
        )
    else:
        print("Conclusion: insufficient data for rank correlation analysis.")

    # Save results
    if args.output:
        output = {
            "args": vars(args),
            "n_edges": n_edges,
            "n_success": n_success,
            "total_time": total_time,
            "rungs": rungs,
            "correlations": {
                f"{rm}@{rung}->{fm}": {"rho": rho, "pvalue": pval}
                for (rm, rung, fm), (rho, pval) in correlations.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
