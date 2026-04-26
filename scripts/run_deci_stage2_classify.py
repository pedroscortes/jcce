#!/usr/bin/env python3
"""DECI Stage 2: load DAGs from Stage 1, train classifiers on MB features.

Reads results/deci_stage1/<dataset>/A_bin.npy and X/Y splits, then runs
the same 5-processor evaluation as run_unified_baselines.py and patches
the results into the existing baselines PKL.

Run from main JAX venv (uv run).
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from jcce.baselines.classifier import train_and_evaluate_processor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_ROOT / 'results' / 'deci_stage1'


def get_markov_blanket(A: np.ndarray, target_idx: int) -> list:
    parents = set(np.where(A[:, target_idx] > 0)[0]) - {target_idx}
    children = set(np.where(A[target_idx, :] > 0)[0]) - {target_idx}
    spouses = set()
    for c in children:
        spouses |= set(np.where(A[:, c] > 0)[0]) - {target_idx, c}
    return sorted([int(i) for i in (parents | children | spouses) - {target_idx}])


def process_dataset(name: str):
    base = STAGE1_DIR / name
    if not base.exists():
        return {'error': f'stage1 missing: {base}'}

    A_bin = np.load(base / 'A_bin.npy')
    X_struct = np.load(base / 'X_struct_std.npy')
    X_effect = np.load(base / 'X_effect_std.npy')
    Y_struct = np.load(base / 'Y_struct.npy')
    Y_effect = np.load(base / 'Y_effect.npy')
    with open(base / 'info.json') as f:
        info = json.load(f)

    d = info['n_features']
    target_idx = d  # Y is last col
    mb = get_markov_blanket(A_bin, target_idx)
    mb_features = [i for i in mb if i < d]

    print(f"\n=== {name.upper()} ===  edges={info['n_edges']} MB={len(mb_features)}", flush=True)

    if not mb_features:
        info['per_processor'] = {}
        info['error'] = 'empty_mb'
        return info

    Y_effect_int = Y_effect.astype(int)
    X_mb = X_effect[:, mb_features]

    per_proc = {}
    for proc in ['mlp', 'transformer', 'mamba', 'elm', 'gnn']:
        try:
            t0 = time.time()
            res = train_and_evaluate_processor(
                X_mb, Y_effect_int, processor_type=proc,
                n_splits=5, seed=42, n_epochs=200,
            )
            res['time'] = time.time() - t0
            res['feature_set'] = 'deci_mb'
            per_proc[proc] = res
            print(f"  {proc:14s}: BAcc={res['balanced_acc_mean']:.3f} "
                  f"F1={res['f1_mean']:.3f} ({res['time']:.1f}s)", flush=True)
        except Exception as e:
            print(f"  {proc:14s}: FAILED — {e}", flush=True)
            per_proc[proc] = {'error': str(e)}

    info['per_processor'] = per_proc
    info['mb_indices'] = mb_features
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--output', type=str,
                   default=str(PROJECT_ROOT / 'official_results' / 'deci' / 'deci_results.pkl'))
    args = p.parse_args()

    all_ds = ['lucas', 'sachs', 'asia', 'diabetes', 'heart_disease',
              'breast_cancer', 'child', 'alarm', 'insurance', 'neuropathic_pain']
    datasets = args.datasets or all_ds

    results = {}
    for ds in datasets:
        try:
            results[ds] = process_dataset(ds)
        except Exception as e:
            print(f"  ERROR on {ds}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[ds] = {'error': str(e)}

    # Save standalone PKL
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out}")

    # Summary table
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Dataset':<20s} {'BAcc':>8s} {'F1':>8s} {'AUC':>8s} {'BestProc':>10s}")
    for ds in datasets:
        r = results.get(ds, {})
        per = r.get('per_processor', {})
        best_b, best_p = -1, None
        for p, pr in per.items():
            if isinstance(pr, dict) and 'balanced_acc_mean' in pr:
                if pr['balanced_acc_mean'] > best_b:
                    best_b = pr['balanced_acc_mean']; best_p = p
        if best_p:
            br = per[best_p]
            print(f"  {ds:<18s} {br['balanced_acc_mean']:>8.3f} {br['f1_mean']:>8.3f} "
                  f"{br['roc_auc_mean']:>8.3f} {best_p:>10s}")
        else:
            print(f"  {ds:<18s} {'--':>8s} {'--':>8s} {'--':>8s}")


if __name__ == '__main__':
    main()
