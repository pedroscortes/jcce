#!/usr/bin/env python3
"""Export all 10 benchmark datasets as numpy arrays for DECI consumption."""
import json
import numpy as np
from pathlib import Path
from jcce.data.benchmark_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / 'results' / 'deci_input'
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ['lucas', 'sachs', 'asia', 'diabetes', 'heart_disease',
            'breast_cancer', 'child', 'alarm', 'insurance', 'neuropathic_pain']

for ds in DATASETS:
    X, Y, cfg = load_dataset(ds)
    out = OUT / ds
    out.mkdir(exist_ok=True)
    np.save(out / 'X.npy', X.astype(np.float32))
    np.save(out / 'Y.npy', Y.astype(np.float32))
    info = {'n': X.shape[0], 'd': X.shape[1],
            'feature_names': cfg.get('feature_names'),
            'target_name': cfg.get('target_name', 'Y'),
            'has_true_dag': cfg.get('true_dag') is not None}
    if cfg.get('true_dag') is not None:
        np.save(out / 'true_dag.npy', np.array(cfg['true_dag']))
    with open(out / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  {ds}: n={X.shape[0]} d={X.shape[1]}  {'(GT)' if info['has_true_dag'] else ''}")

print(f"\nSaved to {OUT}")
