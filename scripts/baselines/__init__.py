"""
Baseline Comparison Framework for JCCE

This package provides tools to compare JCCE's causal feature selection
(Markov Blanket) against predictive feature selection (SHAP).

Modules:
- config: Dataset and processor configurations
- shap_selection: SHAP-based feature importance and selection
- classification: Classification baselines with multiple processors
- run_all_baselines: Main orchestration script

Usage:
    # Run baselines for LUCAS
    uv run python scripts/baselines/run_all_baselines.py --dataset lucas

    # Run all datasets
    uv run python scripts/baselines/run_all_baselines.py --all

    # With JCCE results for comparison
    uv run python scripts/baselines/run_all_baselines.py --dataset lucas \\
        --jcce-results results/lucas_v11_4_server.pkl
"""

from .config import DATASETS, PROCESSOR_CONFIGS, CV_CONFIG
from .shap_selection import compute_shap_importance, select_features_shap, compare_feature_sets
from .classification import run_classification_cv

__all__ = [
    'DATASETS',
    'PROCESSOR_CONFIGS',
    'CV_CONFIG',
    'compute_shap_importance',
    'select_features_shap',
    'compare_feature_sets',
    'run_classification_cv',
]
