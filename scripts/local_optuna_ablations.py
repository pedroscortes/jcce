#!/usr/bin/env python3
"""Local Optuna ablations on real data: quick mode (20 trials) on CPU.

Tests structural DML, PCGrad, curriculum variants on LUCAS, Diabetes, Sachs.

Usage:
    uv run python scripts/local_optuna_ablations.py
    uv run python scripts/local_optuna_ablations.py --datasets lucas
    uv run python scripts/local_optuna_ablations.py --n-trials 10
"""

import argparse
import numpy as np
import time
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_one_config(dataset_name, config_name, golem_overrides, n_trials=20, max_iter=100):
    """Run a single Optuna config and return summary."""
    from jcce.data.benchmark_loader import load_dataset
    from jcce.structure_learning.experiment_runner import run_full_optuna_pipeline

    X, Y, config = load_dataset(dataset_name)
    n_vars = X.shape[1]
    feature_names = config.get('feature_names', [])
    true_mb = config.get('true_mb')
    true_dag = config.get('true_dag')

    print(f"\n    {config_name} ({dataset_name}, {n_trials} trials, {max_iter} iter)...")
    t0 = time.time()

    try:
        result = run_full_optuna_pipeline(
            X=X, Y=Y, n_vars=n_vars,
            n_trials=n_trials, max_iter=max_iter,
            use_v7=True, task='classification',
            jax_key_seed=42, verbose=False,
            run_cv=True, n_folds=5,
            cv_max_solutions=3, cv_golem_max_iter=max_iter // 3,
            run_dml=True, dml_max_solutions=3,
            run_all_edges_dml=False,
            run_cf=False,
            run_structural_validation=False,
            feature_names=feature_names,
            true_mb=true_mb, true_graph=true_dag,
            golem_overrides=golem_overrides,
        )

        elapsed = time.time() - t0

        # Extract key metrics
        best_bacc = result.get('best_training_balanced_acc', 0)
        cv_bacc = result.get('cv_balanced_acc_mean', 0)
        cv_std = result.get('cv_balanced_acc_std', 0)
        dml_n_sig = result.get('dml_n_significant', 0)
        dml_n_parents = result.get('dml_n_parents', 0)
        n_pareto = result.get('n_pareto_solutions', 0)

        # Get ATEs from best solution
        ps = result.get('pareto_solutions', [])
        ates = {}
        if ps:
            best = ps[result.get('best_pareto_idx', 0)]
            ce = best.get('causal_effects', {})
            for k, v in ce.items():
                if '->Y' in k:
                    ate = v.get('ate', v) if isinstance(v, dict) else v
                    if isinstance(ate, (int, float)) and abs(float(ate)) > 0.001:
                        ates[k] = float(ate)

        summary = {
            'config': config_name,
            'dataset': dataset_name,
            'time': elapsed,
            'best_bacc': best_bacc,
            'cv_bacc': cv_bacc,
            'cv_std': cv_std,
            'dml_sig': f"{dml_n_sig}/{dml_n_parents}",
            'n_pareto': n_pareto,
            'top_ates': dict(sorted(ates.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
        }

        print(f"      BAcc={best_bacc:.3f}, CV={cv_bacc:.3f}±{cv_std:.3f}, "
              f"DML={dml_n_sig}/{dml_n_parents} sig, Pareto={n_pareto}, {elapsed:.0f}s")
        if ates:
            print(f"      Top ATEs: {', '.join(f'{k}={v:.3f}' for k,v in list(ates.items())[:3])}")

        return summary

    except Exception as e:
        elapsed = time.time() - t0
        print(f"      FAILED: {e} ({elapsed:.0f}s)")
        import traceback; traceback.print_exc()
        return {'config': config_name, 'dataset': dataset_name, 'error': str(e), 'time': elapsed}


def main():
    parser = argparse.ArgumentParser(description='Local Optuna ablations')
    parser.add_argument('--datasets', nargs='+', default=['lucas', 'diabetes', 'sachs'])
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--max-iter', type=int, default=100)
    args = parser.parse_args()

    configs = [
        ('B6: DragonNet (baseline)', {}),
        ('A9: Structural DML', {'use_structural_dml': True, 'use_dragonnet': False}),
        ('A8: PCGrad + DragonNet', {'use_pcgrad': True}),
        ('A9+A8: StructDML + PCGrad', {'use_structural_dml': True, 'use_dragonnet': False, 'use_pcgrad': True}),
        ('A1: No Curriculum', {'use_adaptive_curriculum': False}),
        ('A3: No Effects', {'lambda_effect': 0.0, 'use_amortized_effects': False}),
    ]

    print(f"LOCAL OPTUNA ABLATIONS")
    print(f"{'='*70}")
    print(f"Datasets: {args.datasets}")
    print(f"Configs: {len(configs)}")
    print(f"Trials: {args.n_trials}, Max iter: {args.max_iter}")
    print(f"Device: CPU")
    print(f"Started: {datetime.now().isoformat()}")

    all_results = []

    for ds in args.datasets:
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'='*70}")

        for config_name, overrides in configs:
            r = run_one_config(ds, config_name, overrides,
                               n_trials=args.n_trials, max_iter=args.max_iter)
            all_results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<30s} {'Dataset':<12s} {'BAcc':>6s} {'CV BAcc':>10s} {'DML sig':>8s} {'Time':>6s}")
    print("-" * 75)
    for r in all_results:
        if 'error' in r:
            print(f"{r['config']:<30s} {r['dataset']:<12s} FAILED")
            continue
        print(f"{r['config']:<30s} {r['dataset']:<12s} {r['best_bacc']:>6.3f} "
              f"{r['cv_bacc']:>5.3f}±{r['cv_std']:.3f} {r['dml_sig']:>8s} {r['time']:>5.0f}s")

    # Save results
    out_path = Path('results/local_ablations.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
