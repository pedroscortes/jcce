#!/usr/bin/env python
"""
Master orchestration script for full JCCE experiment suite.

Schedules jobs across 2 GPUs with priority-ordered queue.
All experiments use fixed validation code (AIPW SE, fixed-structure CV,
structural validation, DML sample splitting).

Usage:
    # Run everything
    uv run python scripts/run_full_server.py

    # Quick test mode (5 trials, 20 iters)
    uv run python scripts/run_full_server.py --quick

    # Only Groups 1+2
    uv run python scripts/run_full_server.py --groups 1 2

    # Subset of datasets
    uv run python scripts/run_full_server.py --groups 1 --datasets lucas heart_disease

    # Dry run (print commands without executing)
    uv run python scripts/run_full_server.py --dry-run
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Job definition
# ============================================================================


@dataclass
class Job:
    name: str
    group: int
    command: List[str]
    gpu: Optional[int] = None  # None = auto-assign
    extra_env: Dict[str, str] = field(default_factory=dict)
    # Filled at runtime
    returncode: Optional[int] = None
    wall_time: float = 0.0
    log_path: str = ""


# ============================================================================
# GPU Scheduler
# ============================================================================


class GPUScheduler:
    """Simple 2-GPU round-robin scheduler with job queue."""

    def __init__(self, n_gpus: int = 2):
        self.n_gpus = n_gpus
        # gpu_id -> (Job, Popen, start_time) or None
        self.slots: Dict[int, Optional[Tuple[Job, subprocess.Popen, float]]] = {
            i: None for i in range(n_gpus)
        }

    def find_free_gpu(self) -> Optional[int]:
        """Return first free GPU id, or None."""
        for gpu_id, slot in self.slots.items():
            if slot is None:
                return gpu_id
        return None

    def n_running(self) -> int:
        return sum(1 for s in self.slots.values() if s is not None)

    def submit(self, job: Job, log_dir: str) -> int:
        """Submit job to a free GPU. Returns GPU id."""
        gpu_id = job.gpu if job.gpu is not None else self.find_free_gpu()
        if gpu_id is None:
            raise RuntimeError("No free GPU slot")

        job.log_path = os.path.join(log_dir, f"{job.name}.log")
        log_file = open(job.log_path, "w")

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), **job.extra_env}
        proc = subprocess.Popen(
            job.command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )

        self.slots[gpu_id] = (job, proc, time.time())
        return gpu_id

    def wait_any(self, poll_interval: float = 5.0) -> Tuple[int, Job, int]:
        """
        Block until any running job finishes.
        Returns (gpu_id, job, returncode).
        """
        while True:
            for gpu_id, slot in self.slots.items():
                if slot is None:
                    continue
                job, proc, start_time = slot
                ret = proc.poll()
                if ret is not None:
                    job.returncode = ret
                    job.wall_time = time.time() - start_time
                    self.slots[gpu_id] = None
                    return gpu_id, job, ret
            time.sleep(poll_interval)

    def wait_all(self) -> List[Tuple[int, Job, int]]:
        """Wait for all running jobs to finish."""
        results = []
        while self.n_running() > 0:
            results.append(self.wait_any())
        return results

    def status_line(self, queue_size: int) -> str:
        """One-line status string."""
        parts = []
        for gpu_id, slot in self.slots.items():
            if slot is None:
                parts.append(f"[GPU {gpu_id}] idle")
            else:
                job, _, start_time = slot
                elapsed = time.time() - start_time
                mins = int(elapsed // 60)
                parts.append(f"[GPU {gpu_id}] {job.name} ({mins}m)")
        parts.append(f"Queue: {queue_size} jobs")
        return " | ".join(parts)


# ============================================================================
# Job definitions
# ============================================================================

ALL_DATASETS = [
    "lucas",
    "heart_disease",
    "breast_cancer",
    "diabetes",
    "sachs",
    "asia",
    "child",
    "neuropathic_pain",
    "alarm",
    "insurance",
]

OPTUNA_CONFIGS = {
    "lucas": {"n_trials": 100, "max_iter": 300},
    "heart_disease": {"n_trials": 100, "max_iter": 150},
    "breast_cancer": {"n_trials": 100, "max_iter": 150},
    "diabetes": {"n_trials": 100, "max_iter": 100},
    "sachs": {"n_trials": 100, "max_iter": 150},
    "asia": {"n_trials": 100, "max_iter": 100},
    "child": {"n_trials": 100, "max_iter": 150},
    "neuropathic_pain": {"n_trials": 100, "max_iter": 150},
    "alarm": {"n_trials": 100, "max_iter": 150},
    "insurance": {"n_trials": 100, "max_iter": 150},
}

QUICK_OPTUNA = {"n_trials": 5, "max_iter": 20}


def _build_group_jobs(
    group: int,
    datasets: List[str],
    quick: bool = False,
    seed: int = 42,
) -> List[Job]:
    """Build jobs for a single group."""
    jobs = []

    # Group 1: Optuna full pipeline (4 datasets)
    if group == 1:
        for ds in datasets:
            if ds not in OPTUNA_CONFIGS:
                continue
            cfg = QUICK_OPTUNA if quick else OPTUNA_CONFIGS[ds]
            jobs.append(
                Job(
                    name=f"optuna_{ds}",
                    group=1,
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_optuna_pipeline.py",
                        "--dataset",
                        ds,
                        "--n-trials",
                        str(cfg["n_trials"]),
                        "--max-iter",
                        str(cfg["max_iter"]),
                        "--seed",
                        str(seed),
                    ],
                )
            )

    # Group 2: NSGA-II baseline (4 datasets)
    if group == 2:
        for ds in datasets:
            if ds not in OPTUNA_CONFIGS:
                continue
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/run_ablation_study.py",
                "--ablation",
                "B6",
                "--dataset",
                ds,
            ]
            if quick:
                cmd.append("--quick")
            jobs.append(
                Job(
                    name=f"nsga2_{ds}",
                    group=2,
                    command=cmd,
                )
            )

    # Group 3: Phase C synthetic (sequential, 1 GPU)
    if group == 3:
        cmd = ["uv", "run", "python", "scripts/run_phase_c_server.py"]
        if quick:
            cmd.append("--quick")
        jobs.append(
            Job(
                name="phase_c_synthetic",
                group=3,
                command=cmd,
            )
        )

    # Group 4: Phase C real (C1/C2/C5 per dataset)
    if group == 4:
        for ds in datasets:
            if ds not in OPTUNA_CONFIGS:
                continue
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/run_phase_c_server.py",
                "--dataset",
                ds,
                "--experiments",
                "c1",
                "c2",
                "c5",
            ]
            if quick:
                cmd.append("--quick")
            jobs.append(
                Job(
                    name=f"phase_c_real_{ds}",
                    group=4,
                    command=cmd,
                )
            )

    # Group 6: Definitive Optuna (structural DML + PCGrad)
    # Replaces Group 1 results with the improved effect estimation
    if group == 6:
        for ds in datasets:
            if ds not in OPTUNA_CONFIGS:
                continue
            cfg = QUICK_OPTUNA if quick else OPTUNA_CONFIGS[ds]
            jobs.append(
                Job(
                    name=f"optuna_dml_{ds}",
                    group=6,
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_optuna_pipeline.py",
                        "--dataset",
                        ds,
                        "--n-trials",
                        str(cfg["n_trials"]),
                        "--max-iter",
                        str(cfg["max_iter"]),
                        "--seed",
                        str(seed),
                        "--golem-override",
                        "use_structural_dml=True",
                        "--golem-override",
                        "use_dragonnet=False",
                        "--golem-override",
                        "use_pcgrad=True",
                    ],
                )
            )

    # Group 7: Definitive NSGA-II (structural DML + PCGrad + fixed sparsity)
    if group == 7:
        for ds in datasets:
            if ds not in OPTUNA_CONFIGS:
                continue
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/run_ablation_study.py",
                "--ablation",
                "B6",
                "--dataset",
                ds,
                "--golem-override",
                "use_structural_dml=True",
                "--golem-override",
                "use_dragonnet=False",
                "--golem-override",
                "use_pcgrad=True",
            ]
            if quick:
                cmd.append("--quick")
            jobs.append(
                Job(
                    name=f"nsga2_dml_{ds}",
                    group=7,
                    command=cmd,
                )
            )

    # Group 8: Real-data ablations via Optuna (faster than NSGA-II)
    # Tests component impact on 3 datasets with 30 Optuna trials each
    if group == 8:
        ablation_datasets = [ds for ds in ["lucas", "sachs", "diabetes"] if ds in datasets]
        abl_trials = 10 if quick else 30
        abl_iter = 50 if quick else 150

        # Each ablation config: (name, golem_overrides)
        ablation_configs = [
            ("B6_baseline", {}),  # Full system (reference)
            ("A1_no_curriculum", {"use_adaptive_curriculum": False}),
            (
                "A1b_fixed_thirds",
                {"use_adaptive_curriculum": False, "curriculum_phase_splits": "(0.333,0.667)"},
            ),
            ("A2_no_bow", {"lambda_bow": 0.0}),
            ("A3_no_effects", {"lambda_effect": 0.0, "use_amortized_effects": False}),
            ("A8_pcgrad", {"use_pcgrad": True}),
            ("A9_struct_dml", {"use_structural_dml": True, "use_dragonnet": False}),
            (
                "A9A8_dml_pcgrad",
                {"use_structural_dml": True, "use_dragonnet": False, "use_pcgrad": True},
            ),
        ]

        for ds in ablation_datasets:
            cfg = OPTUNA_CONFIGS.get(ds, {"n_trials": abl_trials, "max_iter": abl_iter})
            for abl_name, overrides in ablation_configs:
                cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/run_optuna_pipeline.py",
                    "--dataset",
                    ds,
                    "--n-trials",
                    str(abl_trials),
                    "--max-iter",
                    str(min(abl_iter, cfg["max_iter"])),
                    "--seed",
                    "42",
                    "--results-dir",
                    f"results/ablation_v16/{abl_name}",
                ]
                for k, v in overrides.items():
                    cmd.extend(["--golem-override", f"{k}={v}"])
                jobs.append(
                    Job(
                        name=f"abl_{abl_name}_{ds}",
                        group=8,
                        command=cmd,
                    )
                )

    # Group 9: Baselines with fair 70/30 split + validation
    if group == 9:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_unified_baselines.py",
            "--all",
        ]
        if quick:
            cmd.append("--quick")
        jobs.append(
            Job(
                name="baselines_all",
                group=9,
                command=cmd,
            )
        )

    # Group 10: Semiparametric efficiency coverage experiment
    if group == 10:
        for config_id in [1, 2, 5, 6]:
            for n in [500, 1000, 2000]:
                n_reps = 10 if quick else 50
                mi = 50 if quick else 150
                jobs.append(
                    Job(
                        name=f"coverage_c{config_id}_n{n}",
                        group=10,
                        command=[
                            "uv",
                            "run",
                            "python",
                            "scripts/theory/coverage_experiment.py",
                            "--config",
                            str(config_id),
                            "--n_samples",
                            str(n),
                            "--n_reps",
                            str(n_reps),
                            "--max_iter",
                            str(mi),
                            "--processor",
                            "elm",
                        ],
                    )
                )

    # Group 11: Post-article experiments (processor comparison + scalability validation)
    # Revised 2026-04-15: slimmed down to minimum experiments that inform decisions
    if group == 11:
        # 11a: New processor GO/NO-GO gate
        # Only 4 processors (DAG-Attention vs Transformer, CausalMamba vs Mamba)
        # Only 3 datasets (LUCAS + Sachs have ground truth, Heart is real clinical)
        # Existing MLP/ELM/GNN results from Groups 8+7 serve as baselines
        for ds in ["lucas", "sachs", "heart_disease"]:
            if ds not in [d for d in datasets if d in OPTUNA_CONFIGS]:
                continue
            mi = 50 if quick else 150
            ns = 1 if quick else 3
            jobs.append(
                Job(
                    name=f"proc_gate_{ds}",
                    group=11,
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/theory/test_new_processors_server.py",
                        "--dataset",
                        ds,
                        "--max-iter",
                        str(mi),
                        "--n-seeds",
                        str(ns),
                    ],
                )
            )

        # 11b: Two-stage optimization — single dataset is sufficient for go/no-go
        # LUCAS only (d=11, ground truth, fast). If it works here, test more later.
        mi = 50 if quick else 150
        ns = 1 if quick else 3
        jobs.append(
            Job(
                name="two_stage_lucas",
                group=11,
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/theory/test_two_stage_server.py",
                    "--dataset",
                    "lucas",
                    "--max-iter",
                    str(mi),
                    "--n-seeds",
                    str(ns),
                ]
                + (["--quick"] if quick else []),
            )
        )

        # 11c: Phase-aware Hyperband — single dataset is sufficient
        # LUCAS only. We already know from local tests that standard Hyperband
        # fails (Spearman 0.22). This tests if REAL GOLEM-GA phase transitions
        # produce better rank correlation than the simulation.
        nt = 8 if quick else 20
        mi = 50 if quick else 150
        jobs.append(
            Job(
                name="hyperband_lucas",
                group=11,
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/theory/test_hyperband_server.py",
                    "--dataset",
                    "lucas",
                    "--n-trials",
                    str(nt),
                    "--max-iter",
                    str(mi),
                ]
                + (["--quick"] if quick else []),
            )
        )

    # Group 5: E.1 scaling
    # Disable XLA command buffers to prevent GPU memory fragmentation at d>=50
    if group == 5:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/experiment_e1_scaling.py",
            "--dims",
            "10",
            "20",
            "50",
            "100",
            "--n-trials",
            "15",
            "--n-samples",
            "500",
            "--max-iter",
            "100",
        ]
        if quick:
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/experiment_e1_scaling.py",
                "--quick",
            ]
        jobs.append(
            Job(
                name="e1_scaling",
                group=5,
                command=cmd,
                extra_env={"XLA_FLAGS": "--xla_gpu_enable_command_buffer="},
            )
        )

    return jobs


def build_jobs(
    groups: List[int],
    datasets: List[str],
    quick: bool = False,
    seed: int = 42,
) -> List[Job]:
    """Build the full job queue respecting user-specified group order."""
    jobs = []
    for group in groups:
        jobs.extend(_build_group_jobs(group, datasets, quick, seed))
    return jobs


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Master orchestration for full JCCE experiment suite"
    )
    parser.add_argument(
        "--groups", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Job groups to run (1-5)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to include",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test mode (5 trials, 20 iters)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--n-gpus", type=int, default=2, help="Number of GPUs available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--poll-interval", type=float, default=10.0, help="Seconds between status checks"
    )
    args = parser.parse_args()

    # Build job queue
    jobs = build_jobs(
        groups=args.groups,
        datasets=args.datasets,
        quick=args.quick,
        seed=args.seed,
    )

    if not jobs:
        print("No jobs to run.")
        return

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if args.quick else "full"
    log_dir = os.path.join("results", f"full_run_{mode}_{timestamp}")

    print("JCCE Full Experiment Suite")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"  Groups: {args.groups} (in this order)")
    print(f"  Datasets: {args.datasets}")
    print(f"  GPUs: {args.n_gpus}")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Log dir: {log_dir}")
    print()

    # Dry run: just print commands
    if args.dry_run:
        for job in jobs:
            gpu_str = f"GPU {job.gpu}" if job.gpu is not None else "GPU auto"
            print(f"[Group {job.group}] {job.name} ({gpu_str})")
            print(f"  {' '.join(job.command)}")
            print()
        return

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Run with GPU scheduler
    scheduler = GPUScheduler(n_gpus=args.n_gpus)
    queue = list(jobs)
    completed: List[Job] = []
    failed: List[Job] = []
    start_time = time.time()

    print(f"{'=' * 70}")
    print(f"Starting execution at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 70}")

    while queue or scheduler.n_running() > 0:
        # Submit jobs to free GPUs
        while queue and scheduler.find_free_gpu() is not None:
            job = queue.pop(0)
            gpu_id = scheduler.submit(job, log_dir)
            print(f"\n[START] {job.name} on GPU {gpu_id}")
            print(f"  cmd: {' '.join(job.command)}")
            print(f"  log: {job.log_path}")

        # Print status (full line for nohup compatibility)
        print(f"  {scheduler.status_line(len(queue))}", flush=True)

        if scheduler.n_running() == 0:
            break

        # Wait for any job to finish
        gpu_id, job, returncode = scheduler.wait_any(poll_interval=args.poll_interval)

        mins = int(job.wall_time // 60)
        secs = int(job.wall_time % 60)
        if returncode == 0:
            completed.append(job)
            print(f"\n[DONE] {job.name} ({mins}m{secs}s) on GPU {gpu_id}")
        else:
            failed.append(job)
            print(f"\n[FAIL] {job.name} (rc={returncode}, {mins}m{secs}s) on GPU {gpu_id}")
            print(f"  Check log: {job.log_path}")

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'=' * 70}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total time: {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m")
    print(f"Completed: {len(completed)}/{len(jobs)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for job in failed:
            print(f"  - {job.name} (rc={job.returncode})")

    print("\nPer-job times:")
    for job in completed + failed:
        status = "OK" if job.returncode == 0 else f"FAIL(rc={job.returncode})"
        mins = int(job.wall_time // 60)
        print(f"  {job.name:<30} {mins:>4}m  {status}")

    # Write summary JSON
    summary = {
        "mode": "quick" if args.quick else "full",
        "groups": args.groups,
        "datasets": args.datasets,
        "n_gpus": args.n_gpus,
        "seed": args.seed,
        "total_time_s": total_time,
        "n_completed": len(completed),
        "n_failed": len(failed),
        "jobs": [
            {
                "name": job.name,
                "group": job.group,
                "command": " ".join(job.command),
                "returncode": job.returncode,
                "wall_time_s": job.wall_time,
                "log_path": job.log_path,
            }
            for job in completed + failed
        ],
    }
    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
