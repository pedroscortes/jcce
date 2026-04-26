#!/usr/bin/env python3
"""Smoke test: run DECI on a tiny synthetic dataset to verify the API."""
import sys
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")

# Tiny synthetic SEM: X0 -> X1 -> X2 (d=3, n=500)
np.random.seed(42)
n = 500
X0 = np.random.randn(n)
X1 = 0.7 * X0 + 0.3 * np.random.randn(n)
X2 = 0.6 * X1 + 0.3 * np.random.randn(n)
df = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2})
print("Data:", df.shape)

from causica.datasets.causica_dataset_format import Variable
from causica.datasets.variable_types import VariableTypeEnum
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule

# Build variable spec
variables = [
    Variable(name='X0', group_name='X0', type=VariableTypeEnum.CONTINUOUS),
    Variable(name='X1', group_name='X1', type=VariableTypeEnum.CONTINUOUS),
    Variable(name='X2', group_name='X2', type=VariableTypeEnum.CONTINUOUS),
]

dm = BasicDECIDataModule(
    dataframe=df,
    variables=variables,
    batch_size=128,
    normalize=True,
)
print("DataModule OK")

# Module with low-capacity defaults to keep it fast
model = DECIModule(
    embedding_size=8,
    out_dim_g=8,
    num_layers_g=2,
    num_layers_zeta=2,
    prior_sparsity_lambda=0.05,
)
print("Model OK")

trainer = pl.Trainer(
    max_epochs=20,
    accelerator='cpu',
    enable_progress_bar=True,
    logger=False,
    enable_checkpointing=False,
)
print("Training...")
trainer.fit(model, datamodule=dm)
print("Training done")

# Extract learned DAG
print("\n=== Extracting DAG ===")
sem_dist = model.sem_module
graphs = []
with torch.no_grad():
    for _ in range(20):
        sem_sample = sem_dist().sample()  # list of SEMs (one per sample)
        if isinstance(sem_sample, list):
            for s in sem_sample:
                graphs.append(s.graph.numpy())
        else:
            graphs.append(sem_sample.graph.numpy())
A_mean = np.mean(graphs, axis=0)
print(f"Mean adjacency:\n{A_mean.round(2)}")
print(f"Shape: {A_mean.shape}")

A_bin = (A_mean > 0.5).astype(int)
print(f"\nBinary DAG (threshold=0.5):\n{A_bin}")
print(f"\nExpected (X0->X1->X2): edges (0,1), (1,2)")
