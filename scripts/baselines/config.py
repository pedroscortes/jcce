"""
Baseline Comparison Configuration

Defines datasets, processors, and evaluation settings for comparing
JCCE Markov Blanket selection vs SHAP feature selection.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Base paths
BASE_DATA_PATH = Path('/home/user/Documentos/Pedro/dep/data')
RESULTS_PATH = Path('/home/user/Documentos/Pedro/dep/results/baselines')

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    path: str
    target_col: str
    feature_cols: Optional[List[str]] = None  # None = all except target
    true_mb: Optional[List[int]] = None  # Ground truth MB if known
    true_mb_names: Optional[List[str]] = None
    description: str = ""
    task_type: str = "binary"  # binary, multiclass


# Dataset configurations
DATASETS = {
    'lucas': DatasetConfig(
        name='LUCAS',
        path='benchmarks/lucas/raw/lucas0_train.csv',
        target_col='Lung_cancer',
        true_mb=[0, 4, 8, 9, 10],
        true_mb_names=['Smoking', 'Genetics', 'Fatigue', 'Allergy', 'Coughing'],
        description='Lung Cancer causal benchmark with known ground truth',
        task_type='binary',
    ),
    'heart_disease': DatasetConfig(
        name='Heart Disease',
        path='medical/heart_disease/processed/heart.csv',
        target_col='target',
        description='UCI Heart Disease dataset',
        task_type='binary',
    ),
    'breast_cancer': DatasetConfig(
        name='Breast Cancer',
        path='medical/breast_cancer/processed/breast_cancer.csv',
        target_col='diagnosis',  # May need adjustment
        description='Wisconsin Breast Cancer dataset',
        task_type='binary',
    ),
    'ai4i': DatasetConfig(
        name='AI4I Predictive Maintenance',
        path='industrial/ai4i_maintenance/processed/ai4i2020.csv',
        target_col='Machine failure',
        description='AI4I 2020 Predictive Maintenance Dataset',
        task_type='binary',
    ),
    'steel_plates': DatasetConfig(
        name='Steel Plates Faults',
        path='industrial/steel_plates/processed/steel_plates.csv',
        target_col='fault_type',  # May need adjustment
        description='Steel Plates Faults dataset',
        task_type='binary',  # or multiclass
    ),
    'tep': DatasetConfig(
        name='Tennessee Eastman Process',
        path='industrial/tep_binary/processed/tep_binary.csv',
        target_col='fault',
        description='TEP Industrial Process Fault Detection',
        task_type='binary',
    ),
}

# Processor configurations for baseline classification
PROCESSOR_CONFIGS = {
    'mlp': {
        'hidden_dims': [64, 32],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.001,
    },
    'transformer': {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.1,
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.001,
    },
    'mamba': {
        'd_model': 64,
        'd_state': 16,
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.001,
    },
    'gnn': {
        'hidden_dim': 64,
        'n_layers': 2,
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.001,
    },
    'elm': {
        'hidden_dim': 100,
        'activation': 'relu',
    },
}

# SHAP configuration
SHAP_CONFIG = {
    'base_model': 'xgboost',  # Model for computing SHAP values
    'n_estimators': 100,
    'max_samples': 1000,  # For SHAP computation efficiency
    'selection_methods': ['top_k', 'threshold'],
    'threshold': 0.01,  # Minimum mean |SHAP| for selection
}

# Cross-validation settings
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42,
}

# Metrics to compute
METRICS = [
    'accuracy',
    'balanced_accuracy',
    'f1_macro',
    'f1_weighted',
    'precision_macro',
    'recall_macro',
    'roc_auc',
]

# Feature selection comparison metrics
FEATURE_METRICS = [
    'n_features',
    'jaccard_similarity',  # Between SHAP and MB
    'precision',  # vs ground truth (if available)
    'recall',     # vs ground truth (if available)
    'f1',         # vs ground truth (if available)
]
