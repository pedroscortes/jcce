#!/usr/bin/env python3
"""Fix CASTLE MB classifier results that failed due to f1_macro_mean KeyError.

Loads existing baseline pkl files, re-runs only the castle_mb_* classifiers
using the already-computed CASTLE MB indices, and updates the pkl in place.

Usage:
    uv run python scripts/fix_castle_mb.py --all
    uv run python scripts/fix_castle_mb.py --dataset lucas
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jcce.baselines.classifier import train_and_evaluate_processor
from jcce.data.benchmark_loader import load_dataset

PROCESSOR_TYPES = ["mlp", "transformer", "mamba", "elm", "gnn"]
RESULTS_DIR = Path("results/baselines_unified")

ALL_DATASETS = [
    "lucas",
    "heart_disease",
    "breast_cancer",
    "asia",
    "diabetes",
    "sachs",
    "child",
    "insurance",
    "neuropathic_pain",
    "alarm",
]


def fix_dataset(dataset_name: str, seed: int = 42, n_epochs: int = 200):
    pkl_path = RESULTS_DIR / f"{dataset_name}_baselines.pkl"
    if not pkl_path.exists():
        print(f"  SKIP: {pkl_path} not found")
        return

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    baselines = result["baselines"]

    # Check if CASTLE ran successfully
    castle = baselines.get("castle", {})
    if "error" in castle:
        print(f"  SKIP: CASTLE itself failed on {dataset_name}: {castle['error']}")
        return

    mb_indices = castle.get("mb_indices", [])
    if not mb_indices:
        print(f"  SKIP: CASTLE found no MB for {dataset_name}")
        return

    # Load dataset and apply same 70/30 split as GOLEM-GA
    X, Y, config = load_dataset(dataset_name)
    Y_int = np.asarray(Y).astype(int)
    feature_names = config.get("feature_names", [])

    from sklearn.model_selection import train_test_split

    X_struct, X_effect, Y_struct, Y_effect = train_test_split(
        X, Y_int, test_size=0.3, stratify=Y_int, random_state=42
    )
    # Standardize using X_struct stats (no leakage)
    mu, sigma = X_struct.mean(axis=0), X_struct.std(axis=0) + 1e-8
    X_effect_std = (X_effect - mu) / sigma
    Y_effect_int = np.asarray(Y_effect).astype(int)

    X_castle_mb = X_effect_std[:, mb_indices]
    print(f"  CASTLE MB: {len(mb_indices)} features -> retraining 5 processors")

    n_fixed = 0
    for proc in PROCESSOR_TYPES:
        label = f"castle_mb_{proc}"
        existing = baselines.get(label, {})

        # Only re-run if it errored or is missing
        if "error" not in existing and "balanced_acc_mean" in existing:
            print(f"    {proc:14s}: already OK (BAcc={existing['balanced_acc_mean']:.3f})")
            continue

        t1 = time.time()
        try:
            res = train_and_evaluate_processor(
                X_castle_mb,
                Y_effect_int,
                processor_type=proc,
                n_splits=5,
                seed=seed,
                n_epochs=n_epochs,
            )
            res["time"] = time.time() - t1
            res["feature_set"] = "castle_mb"
            res["n_features_used"] = len(mb_indices)
            baselines[label] = res
            print(
                f"    {proc:14s}: BAcc={res['balanced_acc_mean']:.3f} "
                f"F1={res['f1_mean']:.3f} ({res['time']:.1f}s)"
            )
            n_fixed += 1
        except Exception as e:
            print(f"    {proc:14s}: FAILED: {e}")
            baselines[label] = {"error": str(e)}

    # Also add CV metrics to the castle entry itself if missing std
    if "balanced_acc_std" not in castle:
        # Find best castle_mb processor
        best_proc = None
        best_bacc = 0
        for proc in PROCESSOR_TYPES:
            label = f"castle_mb_{proc}"
            entry = baselines.get(label, {})
            bacc = entry.get("balanced_acc_mean", 0)
            if bacc > best_bacc:
                best_bacc = bacc
                best_proc = label
        if best_proc:
            best = baselines[best_proc]
            castle["balanced_acc_std"] = best.get("balanced_acc_std", 0)
            castle["f1_std"] = best.get("f1_std", 0)
            castle["roc_auc_std"] = best.get("roc_auc_std", 0)
            castle["best_processor"] = best_proc.replace("castle_mb_", "")
            print(f"  Updated CASTLE entry with std from {best_proc}")

    # Save back
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved: {n_fixed} processors fixed for {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Fix CASTLE MB classifiers")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.all else [args.dataset or "lucas"]

    for ds in datasets:
        print(f"\n{'=' * 50}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 50}")
        fix_dataset(ds, seed=args.seed, n_epochs=args.epochs)


if __name__ == "__main__":
    main()
