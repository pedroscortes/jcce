#!/usr/bin/env python
"""
Server orchestration script for Phase C experiments.

Launches all C.1-C.11 experiments with full parameters on the server.
Supports both synthetic and real datasets (via --dataset).
Saves all outputs to a timestamped results directory.

Usage:
    # Full run (synthetic data, server parameters)
    uv run python scripts/run_phase_c_server.py

    # With real dataset for C.1, C.2, C.5
    uv run python scripts/run_phase_c_server.py --dataset lucas

    # Quick local validation
    uv run python scripts/run_phase_c_server.py --quick

    # Select specific experiments
    uv run python scripts/run_phase_c_server.py --experiments c3 c5 c9

    # With cached NSGA-II results for C.1
    uv run python scripts/run_phase_c_server.py --nsga2-results path/to/results.json
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Experiment definitions: (id, script, description, supports_dataset)
EXPERIMENTS = [
    ("c1", "compare_nsga2_vs_optuna.py", "NSGA-II vs Optuna", True),
    ("c2", "experiment_c2_unified_vs_pipeline.py", "Unified vs Pipeline", True),
    ("c3", "experiment_c3_multi_fidelity_correlation.py", "Multi-Fidelity Correlation", False),
    ("c4", "experiment_c4_pareto_stability.py", "Pareto Edge Stability", False),
    ("c5", "experiment_c5_processor_analysis.py", "Processor Analysis", True),
    ("c6", "experiment_c6_component_ablation.py", "Component Ablation", False),
    ("c7", "experiment_c7_noise_sensitivity.py", "Noise Sensitivity", False),
    ("c8", "experiment_c8_dataset_shift.py", "Dataset Shift Robustness", False),
    ("c9", "experiment_c9_multi_fidelity_ablation.py", "Multi-Fidelity Ablation", False),
    ("c10", "experiment_c10_ate_quality.py", "ATE Quality", False),
    ("c11", "experiment_c11_varsortability.py", "Varsortability Check", False),
]

# Full server parameters per experiment
FULL_PARAMS = {
    "c1": {"n_evals": 50, "max_iter": 150},
    "c2": {"n_trials": 30, "n_vars": 10, "n_samples": 1000, "max_iter": 150},
    "c3": {"n_trials": 30, "n_vars": 10, "n_samples": 1000, "max_iter": 300},
    "c4": {
        "n_trials": 30,
        "n_vars": 10,
        "n_samples": 1000,
        "max_iter": 150,
        "graph_types": ["erdos_renyi", "scale_free"],
    },
    "c5": {"n_trials": 50, "n_vars": 10, "n_samples": 1000, "max_iter": 150},
    "c6": {"n_trials": 20, "n_vars": 10, "n_samples": 1000, "max_iter": 150},
    "c7": {"n_trials": 20, "n_vars": 10, "n_samples": 1000, "max_iter": 150},
    "c8": {"n_trials": 20, "n_vars": 10, "n_samples": 1000, "max_iter": 150},
    "c9": {"n_trials": 30, "n_vars": 10, "n_samples": 1000, "max_iter": 150},
    "c10": {"n_trials": 30, "n_vars": 10, "n_samples": 1000, "max_iter": 150},
    "c11": {
        "n_vars": 10,
        "n_samples": 2000,
        "graph_types": ["erdos_renyi", "chain", "scale_free"],
        "noise_scales": [0.1, 0.3, 0.5, 1.0, 2.0],
    },
}

# Quick mode overrides
QUICK_PARAMS = {
    "c1": {"n_evals": 5, "max_iter": 20},
    "c2": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c3": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 30},
    "c4": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c5": {"n_trials": 10, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c6": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c7": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c8": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c9": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c10": {"n_trials": 5, "n_vars": 5, "n_samples": 200, "max_iter": 20},
    "c11": {"n_vars": 5, "n_samples": 200},
}


def build_command(
    exp_id, script_name, params, output_path, dataset=None, nsga2_results=None, seed=42
):
    """Build command list for a single experiment."""
    cmd = ["uv", "run", "python", str(SCRIPT_DIR / script_name)]

    for key, val in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in val)
        else:
            cmd.extend([flag, str(val)])

    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--output", str(output_path)])

    # Dataset mode for supported experiments
    if dataset and exp_id in ("c2", "c5"):
        cmd.extend(["--dataset", dataset])
    elif dataset and exp_id == "c1":
        cmd.extend(["--dataset", dataset])

    # Cached NSGA-II results for C.1
    if nsga2_results and exp_id == "c1":
        cmd.extend(["--nsga2-results", nsga2_results])

    return cmd


def run_experiment(
    exp_id, script_name, description, params, output_dir, dataset=None, nsga2_results=None, seed=42
):
    """Run a single experiment and return result dict."""
    output_path = output_dir / f"{exp_id}_results.json"
    cmd = build_command(
        exp_id,
        script_name,
        params,
        output_path,
        dataset=dataset,
        nsga2_results=nsga2_results,
        seed=seed,
    )

    print(f"\n{'=' * 70}")
    print(f"[{exp_id.upper()}] {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output:  {output_path}")
    print(f"{'=' * 70}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    success = result.returncode == 0
    status = "PASS" if success else f"FAIL (exit {result.returncode})"
    print(f"\n  [{exp_id.upper()}] {status} ({elapsed:.1f}s)")

    return {
        "exp_id": exp_id,
        "description": description,
        "success": success,
        "exit_code": result.returncode,
        "time": elapsed,
        "output_file": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase C Server Orchestration")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: reduced parameters for local testing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Real dataset for C.1, C.2, C.5 (e.g., lucas, sachs)",
    )
    parser.add_argument(
        "--nsga2-results", type=str, default=None, help="Cached NSGA-II results JSON for C.1"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (e.g., c3 c5 c9)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/phase_c_<timestamp>)",
    )
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "quick" if args.quick else "full"
        ds = f"_{args.dataset}" if args.dataset else ""
        output_dir = PROJECT_ROOT / "results" / f"phase_c_{mode}{ds}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter experiments
    if args.experiments:
        selected = set(args.experiments)
        experiments = [(eid, s, d, ds) for eid, s, d, ds in EXPERIMENTS if eid in selected]
    else:
        experiments = EXPERIMENTS

    # Choose params
    params_map = QUICK_PARAMS if args.quick else FULL_PARAMS

    print("PHASE C SERVER ORCHESTRATION")
    print("=" * 70)
    print(f"Mode: {'QUICK (local)' if args.quick else 'FULL (server)'}")
    print(f"Dataset: {args.dataset or 'synthetic'}")
    print(f"Experiments: {len(experiments)} ({', '.join(e[0] for e in experiments)})")
    print(f"Output: {output_dir}")
    if args.nsga2_results:
        print(f"Cached NSGA-II: {args.nsga2_results}")

    # Run experiments sequentially
    all_results = []
    total_t0 = time.time()

    for exp_id, script_name, description, supports_dataset in experiments:
        params = params_map.get(exp_id, {})
        dataset = args.dataset if supports_dataset else None

        result = run_experiment(
            exp_id,
            script_name,
            description,
            params,
            output_dir,
            dataset=dataset,
            nsga2_results=args.nsga2_results,
            seed=args.seed,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n{'=' * 70}")
    print("PHASE C SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Exp':<6} {'Description':<35} {'Status':<10} {'Time':>8}")
    print(f"{'-' * 6} {'-' * 35} {'-' * 10} {'-' * 8}")

    n_pass = 0
    n_fail = 0
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        if r["success"]:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{r['exp_id']:<6} {r['description']:<35} {status:<10} {r['time']:>7.1f}s")

    print(f"\nTotal: {n_pass}/{n_pass + n_fail} passed, {total_elapsed:.1f}s total")
    print(f"Results: {output_dir}")

    # Save summary
    summary = {
        "mode": "quick" if args.quick else "full",
        "dataset": args.dataset,
        "seed": args.seed,
        "total_time": total_elapsed,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "results": all_results,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
