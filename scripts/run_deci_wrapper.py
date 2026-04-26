#!/usr/bin/env python3
"""DECI baseline wrapper. Runs in the .venv-deci environment.

Reads (X.npy, Y.npy, target_idx.txt) from --input_dir and writes
(A_estimated.npy, info.json) to --output_dir.

Usage (called as a subprocess from main pipeline):
    .venv-deci/bin/python scripts/run_deci_wrapper.py \\
        --input_dir /tmp/deci_in --output_dir /tmp/deci_out \\
        --epochs 500
"""
import argparse
import json
import os
import time

# Force CPU before any PyTorch import (CUDA is broken on this box)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from causica.datasets.causica_dataset_format import Variable
from causica.datasets.variable_types import VariableTypeEnum
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule


def run_deci(X: np.ndarray, Y: np.ndarray, target_idx: int,
             max_epochs: int = 500, batch_size: int = 256,
             embedding_size: int = 16) -> dict:
    """Run DECI on (X, Y) data and return DAG + ATEs.

    Args:
        X: (n, d) feature matrix
        Y: (n,) outcome (integer or continuous)
        target_idx: column index of Y in the joint (X, Y) layout (typically d)
        max_epochs: training epochs (DECI uses augmented Lagrangian over epochs)
        batch_size: minibatch size for ELBO estimation
        embedding_size: capacity of variable embeddings

    Returns:
        Dict with 'A_estimated' (d+1, d+1 numpy), 'time' seconds, 'd_total'.
    """
    n, d = X.shape
    d_total = d + 1

    # Build joint dataframe
    XY = np.concatenate([X, Y.reshape(-1, 1)], axis=1).astype(np.float32)
    cols = [f'X{i}' for i in range(d)] + ['Y']
    df = pd.DataFrame(XY, columns=cols)

    variables = [
        Variable(name=c, group_name=c, type=VariableTypeEnum.CONTINUOUS)
        for c in cols
    ]

    dm = BasicDECIDataModule(
        dataframe=df,
        variables=variables,
        batch_size=min(batch_size, n // 4),
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
    elapsed = time.time() - t0

    # Sample DAGs from posterior, average to get edge probabilities
    graphs = []
    sem_dist = model.sem_module
    with torch.no_grad():
        for _ in range(50):
            samples = sem_dist().sample()
            if isinstance(samples, list):
                for s in samples:
                    graphs.append(s.graph.numpy())
            else:
                graphs.append(samples.graph.numpy())
    A_prob = np.mean(graphs, axis=0)
    A_bin = (A_prob > 0.5).astype(np.float32)

    return {
        'A_estimated': A_bin,
        'A_probability': A_prob,
        'time': elapsed,
        'd_total': d_total,
        'n_edges': int(A_bin.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=16)
    args = parser.parse_args()

    X = np.load(f'{args.input_dir}/X.npy')
    Y = np.load(f'{args.input_dir}/Y.npy')
    with open(f'{args.input_dir}/target_idx.txt') as f:
        target_idx = int(f.read().strip())

    print(f"DECI on n={X.shape[0]}, d={X.shape[1]}", flush=True)
    out = run_deci(X, Y, target_idx,
                   max_epochs=args.max_epochs,
                   batch_size=args.batch_size,
                   embedding_size=args.embedding_size)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(f'{args.output_dir}/A_estimated.npy', out['A_estimated'])
    np.save(f'{args.output_dir}/A_probability.npy', out['A_probability'])
    info = {k: v for k, v in out.items() if k not in ('A_estimated', 'A_probability')}
    with open(f'{args.output_dir}/info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Done in {out['time']:.1f}s — {out['n_edges']} edges", flush=True)


if __name__ == '__main__':
    main()
