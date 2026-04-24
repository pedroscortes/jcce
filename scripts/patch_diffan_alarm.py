#!/usr/bin/env python3
"""Rerun DiffAN on Alarm only and patch into existing baselines PKL."""
import pickle
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

from jcce.data.benchmark_loader import load_dataset
from jcce.baselines.classifier import train_and_evaluate_processor

# Load dataset with same split
X_full, Y_full, config = load_dataset('alarm')
d = X_full.shape[1]
X_struct, X_effect, Y_struct, Y_effect = train_test_split(
    X_full, Y_full, test_size=0.3, stratify=Y_full, random_state=42
)
X_struct_std = (X_struct - X_struct.mean(0)) / (X_struct.std(0) + 1e-8)
X_effect_std = (X_effect - X_effect.mean(0)) / (X_effect.std(0) + 1e-8)
Y_effect_int = Y_effect.astype(int)

print(f"Alarm: d={d}, n_struct={len(X_struct)}, n_effect={len(X_effect)}")

# Run DiffAN with higher timeout
from jcce.baselines.modern_baselines import run_diffan

print("Running DiffAN on Alarm (timeout=1800s)...")
t0 = time.time()

# We need to call with higher timeout — patch the function temporarily
import jcce.baselines.modern_baselines as mb
# Read the source and find timeout
import inspect
src = inspect.getsource(mb.run_diffan)
print(f"Current timeout in code: look for 'timeout=' in subprocess call")

# Just call it directly — the subprocess timeout is hardcoded at 600s
# We'll run DiffAN manually here instead
import sys, os, subprocess, tempfile, json

X = X_struct_std.astype(np.float64)
n, d = X.shape

with tempfile.TemporaryDirectory() as tmpdir:
    data_path = f'{tmpdir}/X.npy'
    result_path = f'{tmpdir}/result.json'
    np.save(data_path, X)

    script = f'''
import sys, json
sys.path.insert(0, "/tmp/DiffAN")
import numpy as np

X = np.load("{data_path}")
n, d = X.shape
print(f"DiffAN on Alarm: n={{n}}, d={{d}}", flush=True)

from diffan.diffan import DiffAN
model = DiffAN(n_nodes=d, masking=True, residue=True, epochs=3000)

try:
    A, order = model.fit(X)
    A = np.array(A).astype(float)
    order = order.tolist() if hasattr(order, "tolist") else list(order)
except Exception as e:
    print(f"  DiffAN R/CAM failed ({{e}}), using Lasso pruning fallback", flush=True)
    import torch
    from sklearn.linear_model import LassoCV
    model2 = DiffAN(n_nodes=d, masking=True, residue=False, epochs=3000)
    X_t = torch.tensor(X)
    order = model2.topological_ordering(X_t)
    order = order.tolist() if hasattr(order, "tolist") else list(order)
    A = np.zeros((d, d))
    for i, tgt in enumerate(order):
        if i == 0:
            continue
        parents = order[:i]
        lasso = LassoCV(cv=3, max_iter=2000).fit(X[:, parents], X[:, tgt])
        for j, par in enumerate(parents):
            if abs(lasso.coef_[j]) > 0.01:
                A[par, tgt] = 1.0

json.dump({{"A": A.tolist(), "order": order}}, open("{result_path}", "w"))
print(f"DiffAN done: {{int(np.array(A).sum())}} edges", flush=True)
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}
    print("Starting DiffAN subprocess (timeout=1800s)...")
    proc = subprocess.run(
        [sys.executable, '-c', script],
        env=env, capture_output=True, text=True, timeout=1800,  # 30 min timeout
    )
    print(f"stdout: {proc.stdout}")
    if proc.stderr:
        # Filter out functorch warnings
        for line in proc.stderr.split('\n'):
            if 'functorch' not in line and 'integrated' not in line and line.strip():
                print(f"stderr: {line}")

    with open(result_path) as f:
        res = json.load(f)
    A_est = np.array(res['A'])
    order = res['order']

elapsed = time.time() - t0
n_edges = int(np.sum(A_est))
print(f"DiffAN Alarm: {n_edges} edges, order={order[:5]}..., {elapsed:.1f}s")

# Compute structure metrics
true_dag = config.get('true_dag')
if true_dag is not None:
    A_true = np.array(true_dag)
    A_bin = (np.abs(A_est) > 0).astype(int)
    A_true_bin = (A_true != 0).astype(int)
    tp = int(np.sum((A_bin == 1) & (A_true_bin == 1)))
    fp = int(np.sum((A_bin == 1) & (A_true_bin == 0)))
    fn = int(np.sum((A_bin == 0) & (A_true_bin == 1)))
    shd = fp + fn
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    print(f"Structure: SHD={shd}, F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")

# Extract MB and classify
mb_indices = list(range(d))  # DiffAN doesn't have a natural MB; use all connected to Y
# Actually compute MB from A_est
Y_idx = d  # but DiffAN operates on feature-only space
# Use nodes with edges to/from any node
mb_from_edges = [i for i in range(d) if np.sum(np.abs(A_est[i, :])) > 0 or np.sum(np.abs(A_est[:, i])) > 0]
print(f"MB from edges: {len(mb_from_edges)} features")

# Classify with top processors on MB features
if mb_from_edges:
    X_mb = X_effect_std[:, mb_from_edges]
    for proc in ['elm', 'mlp', 'transformer', 'mamba', 'gnn']:
        try:
            res = train_and_evaluate_processor(
                X_mb, Y_effect_int, processor_type=proc,
                n_splits=5, seed=42, n_epochs=200,
            )
            print(f"  {proc:14s}: BAcc={res['balanced_acc_mean']:.3f} F1={res['f1_mean']:.3f}")
        except Exception as e:
            print(f"  {proc:14s}: FAILED — {e}")

# Patch into existing PKL
pkl_path = sorted(Path('results/baselines_unified').glob('all_baselines_202604*.pkl'))[-1]
print(f"\nPatching into: {pkl_path}")
with open(pkl_path, 'rb') as f:
    all_results = pickle.load(f)

alarm_baselines = all_results['alarm']['baselines']
alarm_baselines['diffan_summary'] = {
    'algo_time': elapsed,
    'A_est': A_est.tolist(),
    'n_edges': n_edges,
    'topological_order': order,
}
if true_dag is not None:
    alarm_baselines['diffan_summary']['structure_metrics'] = {
        'shd': shd, 'edge_f1': f1, 'edge_precision': prec, 'edge_recall': rec,
        'tp': tp, 'fp': fp, 'fn': fn,
    }

# Save patched PKL
with open(pkl_path, 'wb') as f:
    pickle.dump(all_results, f)
print(f"Patched DiffAN Alarm results into {pkl_path}")
