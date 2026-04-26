#!/usr/bin/env python3
"""DECI Stage 1: train DAGs on all datasets, save DAGs to disk.

Run from .venv-deci. Stage 2 (in main JAX venv) loads the DAGs and runs
classifiers on the resulting Markov blankets.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Honor CUDA_VISIBLE_DEVICES if set (server runs); default to CPU only on
# local box where CUDA is broken. Pass --accelerator gpu to enable explicitly.
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causica.datasets.causica_dataset_format import Variable
from causica.datasets.variable_types import VariableTypeEnum
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dataset_minimal(name: str):
    """Read pre-exported numpy arrays from deci_input/."""
    base = PROJECT_ROOT / 'results' / 'deci_input' / name
    X = np.load(base / 'X.npy')
    Y = np.load(base / 'Y.npy')
    cfg = {}
    info_path = base / 'info.json'
    if info_path.exists():
        with open(info_path) as f:
            cfg = json.load(f)
    gt_path = base / 'true_dag.npy'
    if gt_path.exists():
        cfg['true_dag'] = np.load(gt_path).tolist()
    return X, Y, cfg


def run_deci_on_data(X, max_epochs=500, batch_size=256, embedding_size=16,
                     n_graph_samples=50, edge_threshold=0.5, accelerator='cpu'):
    n, d = X.shape
    cols = [f'V{i}' for i in range(d)]
    df = pd.DataFrame(X.astype(np.float32), columns=cols)
    variables = [Variable(name=c, group_name=c, type=VariableTypeEnum.CONTINUOUS)
                 for c in cols]
    dm = BasicDECIDataModule(dataframe=df, variables=variables,
                             batch_size=min(batch_size, max(n // 4, 32)),
                             normalize=True)
    model = DECIModule(embedding_size=embedding_size, out_dim_g=embedding_size,
                       num_layers_g=2, num_layers_zeta=2,
                       prior_sparsity_lambda=0.05)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator,
                         enable_progress_bar=False, logger=False,
                         enable_checkpointing=False, enable_model_summary=False)
    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    t = time.time() - t0
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
    return A_bin, A_prob, t


def get_markov_blanket(A, target_idx):
    parents = set(np.where(A[:, target_idx] > 0)[0]) - {target_idx}
    children = set(np.where(A[target_idx, :] > 0)[0]) - {target_idx}
    spouses = set()
    for c in children:
        spouses |= set(np.where(A[:, c] > 0)[0]) - {target_idx, c}
    return sorted([int(i) for i in (parents | children | spouses) - {target_idx}])


def process_dataset(name, output_dir, max_epochs=500, accelerator='cpu'):
    print(f"\n{'='*60}\nDataset: {name.upper()}\n{'='*60}", flush=True)
    X, Y, cfg = load_dataset_minimal(name)
    n, d = X.shape
    print(f"  Shape: {X.shape}", flush=True)

    X_struct, X_effect, Y_struct, Y_effect = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=42)
    mu, sigma = X_struct.mean(0), X_struct.std(0) + 1e-8
    X_struct_std = (X_struct - mu) / sigma
    X_effect_std = (X_effect - mu) / sigma

    XY_struct = np.concatenate([
        X_struct_std,
        Y_struct.astype(np.float32).reshape(-1, 1)
    ], axis=1)

    print(f"  Running DECI ({max_epochs} epochs, accelerator={accelerator})...", flush=True)
    A_bin, A_prob, t = run_deci_on_data(XY_struct, max_epochs=max_epochs,
                                         accelerator=accelerator)
    n_edges = int(A_bin.sum())
    print(f"  DECI: {n_edges} edges, {t:.1f}s", flush=True)

    target_idx = d
    mb = get_markov_blanket(A_bin, target_idx)
    mb_features = [i for i in mb if i < d]

    # Save artifacts for Stage 2
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'A_bin.npy', A_bin)
    np.save(out / 'A_prob.npy', A_prob)
    np.save(out / 'X_struct_std.npy', X_struct_std)
    np.save(out / 'X_effect_std.npy', X_effect_std)
    np.save(out / 'Y_struct.npy', Y_struct)
    np.save(out / 'Y_effect.npy', Y_effect)

    # Structure metrics
    struct_metrics = {}
    if cfg.get('true_dag') is not None:
        A_true = np.array(cfg['true_dag'])
        A_feat = (A_bin[:d, :d] > 0).astype(int)
        A_t = (np.array(A_true) != 0).astype(int)
        np.fill_diagonal(A_feat, 0); np.fill_diagonal(A_t, 0)
        tp = int(np.sum((A_feat == 1) & (A_t == 1)))
        fp = int(np.sum((A_feat == 1) & (A_t == 0)))
        fn = int(np.sum((A_feat == 0) & (A_t == 1)))
        shd = fp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        struct_metrics = {'shd': shd, 'edge_f1': f1, 'edge_precision': prec,
                          'edge_recall': rec, 'tp': tp, 'fp': fp, 'fn': fn}
        print(f"  Structure: SHD={shd}, F1={f1:.3f}", flush=True)

    info = {
        'dataset': name, 'n_samples': n, 'n_features': d,
        'algo_time': t, 'n_edges': n_edges, 'mb_indices': mb_features,
        'structure_metrics': struct_metrics,
    }
    with open(out / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Saved to {out}", flush=True)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--max_epochs', type=int, default=500)
    p.add_argument('--output_dir', type=str,
                   default='results/deci_stage1')
    p.add_argument('--accelerator', type=str, default='cpu',
                   choices=['cpu', 'gpu', 'auto'])
    args = p.parse_args()

    all_ds = ['asia', 'diabetes', 'heart_disease', 'sachs', 'lucas',
              'child', 'breast_cancer', 'insurance', 'neuropathic_pain', 'alarm']
    datasets = args.datasets or all_ds

    print(f"Running DECI Stage 1 on {len(datasets)} datasets")
    print(f"Output: {args.output_dir}\n")

    summary = {}
    for ds in datasets:
        try:
            summary[ds] = process_dataset(ds, args.output_dir, args.max_epochs,
                                           accelerator=args.accelerator)
        except Exception as e:
            print(f"  ERROR on {ds}: {e}", flush=True)
            import traceback; traceback.print_exc()
            summary[ds] = {'error': str(e)}

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for ds, info in summary.items():
        if 'error' in info:
            print(f"  {ds:<20s} ERROR: {info['error']}")
        else:
            shd = info.get('structure_metrics', {}).get('shd', '--')
            print(f"  {ds:<20s} edges={info['n_edges']:>4d}  MB={len(info['mb_indices']):>3d}  "
                  f"SHD={shd}  time={info['algo_time']:.0f}s")


if __name__ == '__main__':
    main()
