#!/usr/bin/env python3
"""
Run All Baselines for JCCE Comparison

This script:
1. Loads each dataset
2. Computes SHAP feature importance
3. Runs classification with:
   - All features
   - SHAP-selected features
   - JCCE MB-selected features (if available)
4. Compares results and generates reports

Usage:
    uv run python scripts/baselines/run_all_baselines.py --dataset lucas
    uv run python scripts/baselines/run_all_baselines.py --all
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from classification import run_classification_cv
from config import BASE_DATA_PATH, CV_CONFIG, DATASETS, PROCESSOR_CONFIGS, RESULTS_PATH, SHAP_CONFIG
from shap_selection import compare_feature_sets, compute_shap_importance, select_features_shap


def load_dataset(dataset_name: str) -> tuple:
    """Load a dataset and return X, y, feature_names."""

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]
    data_path = BASE_DATA_PATH / config.path

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading {config.name} from {data_path}")
    df = pd.read_csv(data_path)

    # Get feature columns
    if config.feature_cols:
        feature_cols = config.feature_cols
    else:
        feature_cols = [c for c in df.columns if c != config.target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[config.target_col].values

    # Encode target if needed
    if y.dtype == object or isinstance(y[0], str):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(y)

    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Classes: {len(np.unique(y))}, distribution: {np.bincount(y.astype(int))}")

    return X, y, feature_cols, config


def run_baseline_for_dataset(
    dataset_name: str,
    jcce_mb: Optional[List[int]] = None,
    processors: List[str] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    """
    Run all baselines for a single dataset.

    Args:
        dataset_name: Name of the dataset
        jcce_mb: JCCE-selected MB indices (if available from experiment)
        processors: List of processors to run (default: all)
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Dict with all results
    """

    if processors is None:
        processors = list(PROCESSOR_CONFIGS.keys())

    # Load data
    X, y, feature_names, config = load_dataset(dataset_name)
    n_features = X.shape[1]

    results = {
        "dataset": dataset_name,
        "config": {
            "n_samples": X.shape[0],
            "n_features": n_features,
            "n_classes": len(np.unique(y)),
            "true_mb": config.true_mb,
        },
        "shap": {},
        "feature_sets": {},
        "classification": {},
        "comparison": {},
    }

    # =========================================================================
    # Step 1: SHAP Feature Selection
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Step 1: SHAP Feature Selection")
        print("=" * 60)

    shap_result = compute_shap_importance(
        X,
        y,
        feature_names,
        model_type=SHAP_CONFIG["base_model"],
        n_estimators=SHAP_CONFIG["n_estimators"],
        max_samples=SHAP_CONFIG["max_samples"],
    )

    results["shap"]["importance"] = shap_result["importance_df"].to_dict()

    if verbose:
        print("\nSHAP Feature Importance (top 10):")
        print(shap_result["importance_df"].head(10).to_string(index=False))

    # Select features with different methods
    # Method 1: Top-k (k = |true_mb| if known, else 5)
    k = len(config.true_mb) if config.true_mb else 5
    shap_topk_idx, shap_topk_names = select_features_shap(shap_result, method="top_k", k=k)

    # Method 2: Threshold
    shap_thresh_idx, shap_thresh_names = select_features_shap(
        shap_result, method="threshold", threshold=SHAP_CONFIG["threshold"]
    )

    results["feature_sets"]["all"] = list(range(n_features))
    results["feature_sets"]["shap_topk"] = shap_topk_idx
    results["feature_sets"]["shap_thresh"] = shap_thresh_idx

    if verbose:
        print(f"\nSHAP Top-{k}: {shap_topk_names}")
        print(f"SHAP Threshold: {shap_thresh_names}")

    # Add JCCE MB if provided
    if jcce_mb is not None:
        results["feature_sets"]["jcce_mb"] = jcce_mb
        jcce_mb_names = [feature_names[i] for i in jcce_mb]
        if verbose:
            print(f"JCCE MB: {jcce_mb_names}")

    # =========================================================================
    # Step 2: Classification with Different Feature Sets
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: Classification with Different Feature Sets")
        print("=" * 60)

    for feature_set_name, feature_indices in results["feature_sets"].items():
        if verbose:
            print(f"\n--- Feature Set: {feature_set_name} ({len(feature_indices)} features) ---")

        X_subset = X[:, feature_indices]
        results["classification"][feature_set_name] = {}

        for proc_name in processors:
            if verbose:
                print(f"  Training {proc_name}...", end=" ", flush=True)

            proc_config = PROCESSOR_CONFIGS[proc_name]

            try:
                metrics = run_classification_cv(
                    X_subset,
                    y,
                    proc_name,
                    proc_config,
                    n_splits=CV_CONFIG["n_splits"],
                    random_state=CV_CONFIG["random_state"],
                    device=device,
                )
                results["classification"][feature_set_name][proc_name] = metrics

                if verbose:
                    print(
                        f"Acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}, "
                        f"F1={metrics['f1_macro_mean']:.3f}±{metrics['f1_macro_std']:.3f}"
                    )
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                results["classification"][feature_set_name][proc_name] = {"error": str(e)}

    # =========================================================================
    # Step 3: Feature Set Comparison
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Step 3: Feature Set Comparison")
        print("=" * 60)

    # Compare SHAP vs JCCE MB (if available)
    if jcce_mb is not None:
        comparison = compare_feature_sets(
            shap_topk_idx,
            jcce_mb,
            true_mb=config.true_mb,
            n_total=n_features,
        )
        results["comparison"]["shap_vs_jcce"] = comparison

        if verbose:
            print("\nSHAP vs JCCE MB:")
            print(f"  Jaccard Similarity: {comparison['jaccard_similarity']:.3f}")
            print(f"  SHAP only: {[feature_names[i] for i in comparison['shap_only']]}")
            print(f"  MB only: {[feature_names[i] for i in comparison['mb_only']]}")
            print(f"  Both: {[feature_names[i] for i in comparison['both']]}")

            if config.true_mb:
                print("\n  vs Ground Truth:")
                print(
                    f"    SHAP: P={comparison['ground_truth']['shap']['precision']:.3f}, "
                    f"R={comparison['ground_truth']['shap']['recall']:.3f}, "
                    f"F1={comparison['ground_truth']['shap']['f1']:.3f}"
                )
                print(
                    f"    MB:   P={comparison['ground_truth']['mb']['precision']:.3f}, "
                    f"R={comparison['ground_truth']['mb']['recall']:.3f}, "
                    f"F1={comparison['ground_truth']['mb']['f1']:.3f}"
                )

    # Compare to ground truth (if available)
    if config.true_mb:
        shap_vs_truth = compare_feature_sets(shap_topk_idx, config.true_mb, true_mb=config.true_mb)
        results["comparison"]["shap_vs_truth"] = shap_vs_truth["ground_truth"]["shap"]

    return results


def generate_report(results: Dict, output_dir: Path) -> str:
    """Generate a markdown report from results."""

    report = []
    report.append(f"# Baseline Results: {results['dataset']}")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Dataset info
    report.append("\n## Dataset Configuration")
    report.append(f"- Samples: {results['config']['n_samples']}")
    report.append(f"- Features: {results['config']['n_features']}")
    report.append(f"- Classes: {results['config']['n_classes']}")
    if results["config"]["true_mb"]:
        report.append(f"- Ground Truth MB: {results['config']['true_mb']}")

    # SHAP importance
    report.append("\n## SHAP Feature Importance (Top 10)")
    report.append("| Rank | Feature | Mean |SHAP| |")
    report.append("|------|---------|-------------|")
    imp_df = pd.DataFrame(results["shap"]["importance"])
    for _, row in imp_df.head(10).iterrows():
        report.append(f"| {int(row['rank'])} | {row['feature']} | {row['mean_abs_shap']:.4f} |")

    # Classification results
    report.append("\n## Classification Results (5-Fold CV)")

    for feature_set in results["classification"]:
        report.append(f"\n### Feature Set: {feature_set}")
        n_feat = len(results["feature_sets"][feature_set])
        report.append(f"*{n_feat} features*\n")

        report.append("| Processor | Accuracy | Balanced Acc | F1 Macro | AUC-ROC |")
        report.append("|-----------|----------|--------------|----------|---------|")

        for proc, metrics in results["classification"][feature_set].items():
            if "error" in metrics:
                report.append(f"| {proc} | ERROR | - | - | - |")
            else:
                acc = f"{metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}"
                bacc = f"{metrics['balanced_accuracy_mean']:.3f}±{metrics['balanced_accuracy_std']:.3f}"
                f1 = f"{metrics['f1_macro_mean']:.3f}±{metrics['f1_macro_std']:.3f}"
                auc = (
                    f"{metrics.get('roc_auc_mean', np.nan):.3f}"
                    if not np.isnan(metrics.get("roc_auc_mean", np.nan))
                    else "N/A"
                )
                report.append(f"| {proc} | {acc} | {bacc} | {f1} | {auc} |")

    # Comparison
    if results["comparison"]:
        report.append("\n## Feature Set Comparison")

        if "shap_vs_jcce" in results["comparison"]:
            comp = results["comparison"]["shap_vs_jcce"]
            report.append("\n### SHAP vs JCCE MB")
            report.append(f"- Jaccard Similarity: {comp['jaccard_similarity']:.3f}")
            report.append(f"- Overlap: {comp['overlap']} features")

            if "ground_truth" in comp:
                report.append("\n| Method | Precision | Recall | F1 |")
                report.append("|--------|-----------|--------|-----|")
                shap_gt = comp["ground_truth"]["shap"]
                mb_gt = comp["ground_truth"]["mb"]
                report.append(
                    f"| SHAP | {shap_gt['precision']:.3f} | {shap_gt['recall']:.3f} | {shap_gt['f1']:.3f} |"
                )
                report.append(
                    f"| JCCE MB | {mb_gt['precision']:.3f} | {mb_gt['recall']:.3f} | {mb_gt['f1']:.3f} |"
                )

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons for JCCE")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument(
        "--jcce-results", type=str, default=None, help="Path to JCCE results pickle"
    )
    parser.add_argument("--processors", type=str, nargs="+", default=None, help="Processors to run")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 70)
    print("JCCE Baseline Comparison Framework")
    print("=" * 70)
    print(f"Device: {args.device}")

    # Determine datasets to run
    if args.all:
        datasets = list(DATASETS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load JCCE results if provided
    jcce_results = {}
    if args.jcce_results:
        with open(args.jcce_results, "rb") as f:
            jcce_results = pickle.load(f)

    # Run baselines for each dataset
    all_results = {}
    for dataset_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"Processing: {dataset_name.upper()}")
        print("=" * 70)

        # Get JCCE MB if available
        jcce_mb = None
        if dataset_name in jcce_results:
            jcce_mb = jcce_results[dataset_name].get("best_mb")

        try:
            results = run_baseline_for_dataset(
                dataset_name,
                jcce_mb=jcce_mb,
                processors=args.processors,
                device=args.device,
                verbose=True,
            )
            all_results[dataset_name] = results

            # Save individual results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"{dataset_name}_baseline_{timestamp}.pkl"
            with open(results_file, "wb") as f:
                pickle.dump(results, f)
            print(f"\nSaved: {results_file}")

            # Generate report
            report = generate_report(results, output_dir)
            report_file = output_dir / f"{dataset_name}_baseline_{timestamp}.md"
            with open(report_file, "w") as f:
                f.write(report)
            print(f"Report: {report_file}")

        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save combined results
    if len(all_results) > 1:
        combined_file = output_dir / f"all_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(combined_file, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nCombined results: {combined_file}")

    print("\n" + "=" * 70)
    print("Baseline comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
