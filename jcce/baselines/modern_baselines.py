"""Modern causal discovery baselines: SDCD (ICML 2024) and DiffAN (ICLR 2023).

These wrap external implementations to produce adjacency matrices in the
same format as our existing baselines (PC, GES, DAGMA, etc.).

Usage:
    from jcce.baselines.modern_baselines import run_sdcd, run_diffan

    A_sdcd = run_sdcd(X, feature_names)   # (d, d) numpy array
    A_diffan = run_diffan(X)               # (d, d) numpy array
"""

import time
from typing import List, Optional, Tuple

import numpy as np


def run_sdcd(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_epochs: int = 500,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Run SDCD (Stable Differentiable Causal Discovery, Nazaret et al. ICML 2024).

    SDCD is designed for interventional data but works on observational data
    by setting all perturbation labels to "obs".

    Args:
        X: Feature matrix (n_samples, n_features), float32/64
        feature_names: Optional feature names
        n_epochs: Training epochs
        verbose: Print progress

    Returns:
        A_est: Adjacency matrix (n_features, n_features)
        info: Dict with timing and metadata
    """
    import os
    import sys

    import pandas as pd

    # Fix NumPy 2.0 compatibility for SDCD (uses deprecated aliases)
    for attr, replacement in [
        ("float_", "float64"),
        ("int_", "int64"),
        ("complex_", "complex128"),
        ("object_", "object_"),
        ("bool_", "bool_"),
        ("str_", "str_"),
        ("long", "int64"),
        ("unicode_", "str_"),
    ]:
        if not hasattr(np, attr):
            setattr(np, attr, getattr(np, replacement))

    # Mock wandb (SDCD dependency, has compatibility issues with our env)
    os.environ["WANDB_DISABLED"] = "true"
    if "wandb" not in sys.modules:
        mock_wandb = type(sys)("wandb")
        mock_wandb.init = lambda *a, **k: None
        mock_wandb.log = lambda *a, **k: None
        mock_wandb.finish = lambda *a, **k: None
        sys.modules["wandb"] = mock_wandb

    from sdcd import SDCD
    from sdcd.utils import create_intervention_dataset

    n, d = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]

    t0 = time.time()

    # Wrap as DataFrame with observational labels
    df = pd.DataFrame(X, columns=feature_names)
    df["perturbation_label"] = "obs"

    dataset = create_intervention_dataset(df, perturbation_colname="perturbation_label")

    # Train SDCD
    model = SDCD()
    model.train(
        dataset,
        finetune=True,
        log_wandb=False,
        verbose=verbose,
    )

    # Get adjacency matrix
    A_binary = model.get_adjacency_matrix(threshold=True)
    A_continuous = model.get_adjacency_matrix(threshold=False)

    elapsed = time.time() - t0

    # Convert to numpy
    A_binary = np.array(A_binary).astype(float)
    A_continuous = np.array(A_continuous).astype(float)

    info = {
        "algo": "sdcd",
        "time": elapsed,
        "n_edges_binary": int(np.sum(A_binary)),
        "n_edges_continuous": int(np.sum(np.abs(A_continuous) > 0.01)),
        "A_continuous": A_continuous,
    }

    if verbose:
        print(
            f"SDCD: {info['n_edges_binary']} edges (binary), "
            f"{info['n_edges_continuous']} edges (continuous), {elapsed:.1f}s"
        )

    return A_binary, info


def run_diffan(
    X: np.ndarray,
    n_epochs: int = 3000,
    masking: bool = True,
    residue: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Run DiffAN (Diffusion Models for Causal Discovery, Sanchez et al. ICLR 2023).

    DiffAN uses diffusion models to learn topological ordering via
    Hessian-based leaf-node removal, then prunes with CAM.

    Args:
        X: Feature matrix (n_samples, n_features), float32/64
        n_epochs: Training epochs for diffusion model
        masking: Use masking variant (better scaling)
        residue: Use residue-based pruning
        verbose: Print progress

    Returns:
        A_est: Adjacency matrix (n_features, n_features)
        info: Dict with timing, topological order, etc.
    """
    import json as _json
    import subprocess
    import sys
    import tempfile

    n, d = X.shape
    t0 = time.time()

    # DiffAN uses functorch ops that create CPU-only tensors. If torch is
    # already imported with CUDA visible, device mismatches are unavoidable.
    # Solution: run DiffAN in a subprocess with CUDA hidden from the start.
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = f"{tmpdir}/X.npy"
        result_path = f"{tmpdir}/result.json"
        np.save(data_path, X.astype(np.float64))

        script = f'''
import sys, json
sys.path.insert(0, "/tmp/DiffAN")
import numpy as np

X = np.load("{data_path}")
n, d = X.shape

from diffan.diffan import DiffAN
model = DiffAN(n_nodes=d, masking={masking}, residue={residue}, epochs={n_epochs})

try:
    A, order = model.fit(X)
    A = np.array(A).astype(float)
    order = order.tolist() if hasattr(order, "tolist") else list(order)
except Exception as e:
    # R/CAM failed — use Lasso fallback
    print(f"  DiffAN R/CAM failed ({{e}}), using Lasso pruning fallback", flush=True)
    import torch
    from sklearn.linear_model import LassoCV
    model2 = DiffAN(n_nodes=d, masking={masking}, residue=False, epochs={n_epochs})
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
'''
        env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1"}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if verbose and proc.stderr:
            # Print only non-warning lines
            for line in proc.stderr.split("\n"):
                if "DiffAN" in line or "FAILED" in line or "fallback" in line:
                    print(line)

        try:
            with open(result_path) as f:
                res = _json.load(f)
            A_est = np.array(res["A"])
            order = res["order"]
        except (FileNotFoundError, _json.JSONDecodeError):
            raise RuntimeError(
                f"DiffAN subprocess failed.\nstdout: {proc.stdout[-500:]}\n"
                f"stderr: {proc.stderr[-500:]}"
            )

    elapsed = time.time() - t0

    A_est = np.array(A_est).astype(float)

    info = {
        "algo": "diffan",
        "time": elapsed,
        "n_edges": int(np.sum(A_est)),
        "topological_order": order if isinstance(order, list) else list(order),
    }

    if verbose:
        print(f"DiffAN: {info['n_edges']} edges, order={info['topological_order']}, {elapsed:.1f}s")

    return A_est, info


def test_baselines():
    """Quick test on small random data."""
    np.random.seed(42)
    X = np.random.randn(200, 5).astype(np.float32)

    print("Testing SDCD...")
    try:
        A_sdcd, info_sdcd = run_sdcd(X, verbose=True)
        print(f"  Shape: {A_sdcd.shape}, edges: {info_sdcd['n_edges_binary']}")
    except Exception as e:
        print(f"  SDCD failed: {e}")

    print("\nTesting DiffAN...")
    try:
        A_diffan, info_diffan = run_diffan(X, n_epochs=500, verbose=True)
        print(f"  Shape: {A_diffan.shape}, edges: {info_diffan['n_edges']}")
    except Exception as e:
        print(f"  DiffAN failed: {e}")


if __name__ == "__main__":
    test_baselines()
