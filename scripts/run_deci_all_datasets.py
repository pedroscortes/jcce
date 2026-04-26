#!/usr/bin/env python3
"""Run DECI standalone on all 10 datasets.

Output format mirrors run_unified_baselines.py so results can be patched
directly into the main baselines PKL.

Workflow per dataset:
  1. Same 70/30 split as JCCE/baselines (random_state=42)
  2. DECI on X_struct (70%) -> learned DAG (Y appended as last column)
  3. Extract Markov blanket (parents/children/spouses of Y)
  4. Train 5 processors on (X_effect MB-features, Y_effect) with 5-fold CV
  5. Compute structure metrics if ground truth available

Usage (must be from .venv-deci):
    .venv-deci/bin/python scripts/run_deci_all_datasets.py
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Force CPU before any torch import (CUDA broken on host)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

# JCCE-side imports (sklearn etc) — ok to use, no JAX
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import train_test_split, StratifiedKFold

from causica.datasets.causica_dataset_format import Variable
from causica.datasets.variable_types import VariableTypeEnum
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule


# ============================================================
# DECI invocation (returns DAG)
# ============================================================
def run_deci_on_data(X: np.ndarray, max_epochs: int = 500,
                     batch_size: int = 256, embedding_size: int = 16,
                     n_graph_samples: int = 50, edge_threshold: float = 0.5):
    """Train DECI on data and return mean adjacency matrix."""
    n, d = X.shape
    cols = [f'X{i}' for i in range(d)]
    df = pd.DataFrame(X.astype(np.float32), columns=cols)
    variables = [Variable(name=c, group_name=c, type=VariableTypeEnum.CONTINUOUS)
                 for c in cols]

    dm = BasicDECIDataModule(
        dataframe=df, variables=variables,
        batch_size=min(batch_size, max(n // 4, 32)),
        normalize=True,
    )

    model = DECIModule(
        embedding_size=embedding_size,
        out_dim_g=embedding_size,
        num_layers_g=2,
        num_layers_zeta=2,
        prior_sparsity_lambda=0.05,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    train_time = time.time() - t0

    # Sample graphs and average
    graphs = []
    sem_dist = model.sem_module
    with torch.no_grad():
        for _ in range(n_graph_samples):
            samples = sem_dist().sample()
            if isinstance(samples, list):
                for s in samples:
                    graphs.append(s.graph.cpu().numpy())
            else:
                graphs.append(samples.graph.cpu().numpy())
    A_prob = np.mean(graphs, axis=0)
    A_bin = (A_prob > edge_threshold).astype(np.float32)
    np.fill_diagonal(A_bin, 0.0)
    np.fill_diagonal(A_prob, 0.0)
    return A_bin, A_prob, train_time


# ============================================================
# MB extraction
# ============================================================
def get_markov_blanket(A: np.ndarray, target_idx: int) -> list:
    """MB = parents + children + spouses (parents of children)."""
    parents = set(np.where(A[:, target_idx] > 0)[0]) - {target_idx}
    children = set(np.where(A[target_idx, :] > 0)[0]) - {target_idx}
    spouses = set()
    for c in children:
        spouses |= set(np.where(A[:, c] > 0)[0]) - {target_idx, c}
    mb = sorted((parents | children | spouses) - {target_idx})
    return [int(i) for i in mb]


# ============================================================
# Per-dataset run
# ============================================================
def process_dataset(dataset_name: str, max_epochs: int = 500):
    print(f"\n{'='*60}\nDataset: {dataset_name.upper()}\n{'='*60}", flush=True)

    # Load (need to import here, after JAX-incompatible env)
    from jcce.data.benchmark_loader import load_dataset
    from jcce.baselines.classifier import train_and_evaluate_processor
    X, Y, cfg = load_dataset(dataset_name)
    n, d = X.shape
    print(f"  Shape: {X.shape}, target: {cfg.get('target_name','Y')}", flush=True)

    # Same split as everyone else
    X_struct, X_effect, Y_struct, Y_effect = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=42
    )

    # Standardize using struct-set statistics (consistent with run_unified_baselines)
    mu, sigma = X_struct.mean(0), X_struct.std(0) + 1e-8
    X_struct_std = (X_struct - mu) / sigma
    X_effect_std = (X_effect - mu) / sigma
    Y_struct_int = Y_struct.astype(int)
    Y_effect_int = Y_effect.astype(int)

    # Append Y as last column for joint DAG learning
    XY_struct = np.concatenate([X_struct_std, Y_struct_int.reshape(-1, 1).astype(np.float32)], axis=1)
    target_idx = d  # Y is last col

    # Run DECI
    print(f"  Running DECI ({max_epochs} epochs)...", flush=True)
    A_bin, A_prob, train_time = run_deci_on_data(
        XY_struct, max_epochs=max_epochs, embedding_size=16,
    )
    n_edges = int(A_bin.sum())
    print(f"  DECI: {n_edges} edges, {train_time:.1f}s", flush=True)

    # Structure metrics if GT available
    struct_metrics = {}
    true_dag = cfg.get('true_dag')
    if true_dag is not None:
        A_true = np.array(true_dag)
        # Compare features-only submatrix
        A_feat = A_bin[:d, :d]
        A_bin_int = (A_feat > 0).astype(int)
        A_true_int = (np.array(A_true) != 0).astype(int)
        np.fill_diagonal(A_bin_int, 0)
        np.fill_diagonal(A_true_int, 0)
        tp = int(np.sum((A_bin_int == 1) & (A_true_int == 1)))
        fp = int(np.sum((A_bin_int == 1) & (A_true_int == 0)))
        fn = int(np.sum((A_bin_int == 0) & (A_true_int == 1)))
        shd = fp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        struct_metrics = {'shd': shd, 'edge_f1': f1, 'edge_precision': prec,
                          'edge_recall': rec, 'tp': tp, 'fp': fp, 'fn': fn}
        print(f"  Structure: SHD={shd}, F1={f1:.3f}", flush=True)

    # Markov blanket
    mb = get_markov_blanket(A_bin, target_idx)
    mb = [i for i in mb if i < d]  # feature-space only
    print(f"  MB: {len(mb)} features ({mb})", flush=True)

    result = {
        'dataset': dataset_name,
        'n_samples': n,
        'n_features': d,
        'algo_time': train_time,
        'A_estimated': A_bin.tolist(),
        'A_probability': A_prob.tolist(),
        'n_edges': n_edges,
        'mb_indices': mb,
        'structure_metrics': struct_metrics,
    }

    # Classification with 5 processors on MB features
    if mb:
        X_mb = X_effect_std[:, mb]
        result['per_processor'] = {}
        for proc in ['mlp', 'transformer', 'mamba', 'elm', 'gnn']:
            try:
                t0 = time.time()
                res = train_and_evaluate_processor(
                    X_mb, Y_effect_int, processor_type=proc,
                    n_splits=5, seed=42, n_epochs=200,
                )
                res['time'] = time.time() - t0
                res['feature_set'] = 'deci_mb'
                result['per_processor'][proc] = res
                print(f"    {proc:14s}: BAcc={res['balanced_acc_mean']:.3f} "
                      f"F1={res['f1_mean']:.3f} ({res['time']:.1f}s)", flush=True)
            except Exception as e:
                print(f"    {proc:14s}: FAILED — {e}", flush=True)
                result['per_processor'][proc] = {'error': str(e)}
    else:
        print("  Empty MB — skipping classification", flush=True)
        result['error'] = 'empty_mb'

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Subset of datasets (default: all 10)')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--output', type=str,
                        default='results/baselines_unified/deci_all_datasets.pkl')
    args = parser.parse_args()

    all_datasets = ['lucas', 'sachs', 'asia', 'diabetes', 'heart_disease',
                    'breast_cancer', 'child', 'alarm', 'insurance',
                    'neuropathic_pain']
    datasets = args.datasets or all_datasets

    results = {}
    for ds in datasets:
        try:
            results[ds] = process_dataset(ds, max_epochs=args.max_epochs)
        except Exception as e:
            print(f"  ERROR on {ds}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[ds] = {'error': str(e)}

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Dataset':<20s} {'BAcc':>8s} {'F1':>8s} {'AUC':>8s} {'SHD':>6s} {'EdgeF1':>8s}")
    for ds in datasets:
        r = results.get(ds, {})
        if 'error' in r and 'per_processor' not in r:
            print(f"  {ds:<18s} ERROR: {r.get('error','?')}")
            continue
        per = r.get('per_processor', {})
        # best processor
        best_b, best_proc = -1, None
        for p, pr in per.items():
            if isinstance(pr, dict) and 'balanced_acc_mean' in pr:
                if pr['balanced_acc_mean'] > best_b:
                    best_b = pr['balanced_acc_mean']; best_proc = p
        if best_proc:
            br = per[best_proc]
            sm = r.get('structure_metrics', {})
            shd = sm.get('shd', '--')
            ef1 = sm.get('edge_f1', None)
            print(f"  {ds:<18s} {br['balanced_acc_mean']:>8.3f} {br['f1_mean']:>8.3f} "
                  f"{br['roc_auc_mean']:>8.3f} {str(shd):>6s} {(f'{ef1:.3f}' if ef1 else '--'):>8s}")


if __name__ == '__main__':
    main()
