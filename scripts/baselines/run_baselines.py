#!/usr/bin/env python3
"""
Simplified Baseline Comparison for JCCE

Runs only the necessary baselines:
1. All Features + Classifiers (MLP, Transformer, ELM)
2. SHAP-selected Features + Classifiers

JCCE metrics come from the experiment results (no retraining needed).

Usage:
    uv run python scripts/baselines/run_baselines.py --dataset lucas
    uv run python scripts/baselines/run_baselines.py --dataset diabetes
    uv run python scripts/baselines/run_baselines.py --all
    uv run python scripts/baselines/run_baselines.py --list
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
import json
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

# SHAP
import shap
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

def _find_base_path() -> Path:
    """Walk up from the script to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    # Fallback: hardcoded server path
    return Path('/home/user/Documentos/Pedro/dep')


BASE_PATH = _find_base_path()
DATA_PATH = BASE_PATH / 'data'
RESULTS_PATH = BASE_PATH / 'results' / 'baselines'

DEFAULT_DATASETS = ['diabetes', 'heart_disease', 'breast_cancer', 'lucas']


@dataclass
class DatasetConfig:
    name: str
    path: str
    target_col: Optional[str] = None
    true_mb: Optional[List[int]] = None
    true_mb_names: Optional[List[str]] = None
    zero_as_missing_cols: Optional[List[str]] = None

DATASETS = {
    'diabetes': DatasetConfig(
        name='Diabetes (Pima)',
        path='medical/diabetes.csv',
        target_col='Outcome',
        zero_as_missing_cols=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
    ),
    'lucas': DatasetConfig(
        name='LUCAS',
        path='benchmarks/lucas/raw/lucas0_train.csv',
        target_col='Lung_cancer',
        true_mb=[0, 4, 8, 9, 10],
        true_mb_names=['Smoking', 'Genetics', 'Fatigue', 'Allergy', 'Coughing'],
    ),
    'heart_disease': DatasetConfig(
        name='Heart Disease',
        path='medical/heart_disease',  # NPY format
        target_col=None,  # Not used for NPY
    ),
    'breast_cancer': DatasetConfig(
        name='Breast Cancer',
        path='medical/breast_cancer',  # NPY format
        target_col=None,
    ),
    'ai4i': DatasetConfig(
        name='AI4I Maintenance',
        path='industrial/ai4i_maintenance',  # NPY format
        target_col=None,
    ),
    'steel_plates': DatasetConfig(
        name='Steel Plates',
        path='industrial/steel_plates',  # NPY format
        target_col=None,
    ),
    'tep': DatasetConfig(
        name='TEP',
        path='industrial/tep_binary',  # NPY format
        target_col=None,
    ),
}


# =============================================================================
# Simple Classifiers (PyTorch)
# =============================================================================

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_classes=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, d_model=64, n_classes=2):
        super().__init__()
        self.embed = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = torch.nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)


class ELMClassifier:
    """Extreme Learning Machine - fast single-layer network."""
    def __init__(self, hidden_dim=100):
        self.hidden_dim = hidden_dim

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        self.W = np.random.randn(X.shape[1], self.hidden_dim) * 0.5
        self.b = np.random.randn(self.hidden_dim) * 0.5
        H = np.maximum(0, X @ self.W + self.b)  # ReLU
        Y_onehot = np.eye(n_classes)[y.astype(int)]
        self.beta = np.linalg.pinv(H) @ Y_onehot
        return self

    def predict(self, X):
        H = np.maximum(0, X @ self.W + self.b)
        return np.argmax(H @ self.beta, axis=1)

    def predict_proba(self, X):
        H = np.maximum(0, X @ self.W + self.b)
        out = H @ self.beta
        exp_out = np.exp(out - out.max(axis=1, keepdims=True))
        return exp_out / exp_out.sum(axis=1, keepdims=True)


class MambaClassifier(torch.nn.Module):
    """Simplified Mamba-style classifier using selective state space."""
    def __init__(self, input_dim, d_model=64, d_state=16, n_classes=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection
        self.in_proj = torch.nn.Linear(input_dim, d_model)

        # SSM parameters (simplified)
        self.A = torch.nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B = torch.nn.Linear(d_model, d_state)
        self.C = torch.nn.Linear(d_state, d_model)
        self.D = torch.nn.Parameter(torch.ones(d_model))

        # Output
        self.norm = torch.nn.LayerNorm(d_model)
        self.classifier = torch.nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.in_proj(x)  # (batch, d_model)

        # Simplified SSM
        b = self.B(x)  # (batch, d_state)
        h = torch.tanh(b @ self.A.T)  # (batch, d_model)
        y = self.C(b) + self.D * x

        # Output
        y = self.norm(y)
        return self.classifier(y)


class GNNClassifier(torch.nn.Module):
    """Simple GNN-style classifier treating features as graph nodes."""
    def __init__(self, input_dim, hidden_dim=64, n_classes=2):
        super().__init__()
        self.input_dim = input_dim

        # Node embeddings (one per feature)
        self.node_embed = torch.nn.Linear(1, hidden_dim)

        # Message passing layers
        self.msg1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.msg2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)

        # Aggregation and output
        self.aggregate = torch.nn.Linear(hidden_dim * input_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # Treat each feature as a node: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)

        # Node embeddings: (batch, n_features, hidden_dim)
        h = torch.relu(self.node_embed(x))

        # Simple message passing (mean aggregation from all neighbors)
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)
        h_cat = torch.cat([h, h_mean], dim=-1)
        h = torch.relu(self.msg1(h_cat))

        # Second layer
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)
        h_cat = torch.cat([h, h_mean], dim=-1)
        h = torch.relu(self.msg2(h_cat))

        # Flatten and classify
        h = h.view(batch_size, -1)
        h = torch.relu(self.aggregate(h))
        return self.classifier(h)


# =============================================================================
# Core Functions
# =============================================================================

def impute_zeros_with_median(df: pd.DataFrame, cols: list, verbose: bool = True) -> pd.DataFrame:
    """Replace zeros with median for columns where 0 is biologically impossible (Pima Diabetes)."""
    df = df.copy()
    if verbose:
        print("\nMissing Value Imputation (zeros -> median)")
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col] == 0
        n_missing = mask.sum()
        if n_missing > 0:
            median_val = df.loc[~mask, col].median()
            df.loc[mask, col] = median_val
            if verbose:
                pct = 100 * n_missing / len(df)
                print(f"  {col}: {n_missing} zeros ({pct:.1f}%) -> median={median_val:.2f}")
    return df


def load_dataset(name: str):
    """Load dataset and return X, y, feature_names."""
    config = DATASETS[name]
    data_path = DATA_PATH / config.path

    # --- CSV format ---
    csv_path = None
    if data_path.suffix == '.csv':
        csv_path = data_path
    elif not data_path.is_dir():
        # Not a directory — maybe the .csv extension is missing, or file is elsewhere
        csv_path = data_path

    # Try primary path, then fallback to BASE_PATH / filename
    if csv_path is not None:
        candidates = [csv_path]
        # Fallback: file at project root (e.g. diabetes.csv)
        candidates.append(BASE_PATH / Path(config.path).name)
        for candidate in candidates:
            if candidate.exists():
                df = pd.read_csv(candidate)
                feature_cols = [c for c in df.columns if c != config.target_col]

                # Zero-as-missing imputation (e.g. diabetes)
                if config.zero_as_missing_cols:
                    df = impute_zeros_with_median(df, config.zero_as_missing_cols)

                X = df[feature_cols].values.astype(np.float32)
                y = df[config.target_col].values
                if y.dtype == object:
                    y = LabelEncoder().fit_transform(y)
                return X, y.astype(int), feature_cols, config

    # --- NPY format (directory with X_train.npy, y_train.npy, etc.) ---
    if data_path.is_dir():
        # Load train and test, concatenate
        X_train = np.load(data_path / 'X_train.npy')
        X_test = np.load(data_path / 'X_test.npy')
        y_train = np.load(data_path / 'y_train.npy')
        y_test = np.load(data_path / 'y_test.npy')

        X = np.vstack([X_train, X_test]).astype(np.float32)
        y = np.concatenate([y_train, y_test])

        # Load feature names if available
        feat_file = data_path / 'feature_names.txt'
        if feat_file.exists():
            with open(feat_file) as f:
                feature_cols = [line.strip() for line in f if line.strip()]
        else:
            feature_cols = [f'X{i}' for i in range(X.shape[1])]

        # Ensure y is integer for classification
        if y.dtype == float:
            y = y.astype(int)
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y)

        return X, y.astype(int), feature_cols, config

    raise FileNotFoundError(f"Could not load dataset from {data_path}")


def compute_shap_features(X, y, feature_names, k=None, threshold_pct=0.80):
    """
    Compute SHAP importance and select top features.

    Args:
        k: If provided, select top-k features
        threshold_pct: If k is None, select features explaining this % of total importance

    Returns:
        top_k_idx, top_k_names, importance_df, k_used
    """
    model = XGBClassifier(n_estimators=100, verbosity=0, use_label_encoder=False)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:min(1000, len(X))])

    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    ranking = np.argsort(mean_shap)[::-1]

    # Determine k if not provided
    if k is None:
        # Select features explaining threshold_pct of total importance
        total_importance = mean_shap.sum()
        cumsum = np.cumsum(mean_shap[ranking])
        k = np.searchsorted(cumsum, threshold_pct * total_importance) + 1
        k = max(3, min(k, len(feature_names) // 2))  # At least 3, at most half

    top_k_idx = list(ranking[:k])
    top_k_names = [feature_names[i] for i in top_k_idx]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_shap,
    }).sort_values('shap_importance', ascending=False)

    return top_k_idx, top_k_names, importance_df, k


def train_and_evaluate(X, y, model_type, device='cuda', n_splits=5):
    """Train classifier with 5-fold CV and return metrics with per-fold storage."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))

    per_fold = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        y_proba = None

        if model_type == 'elm':
            model = ELMClassifier(hidden_dim=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
        else:
            # PyTorch models
            if model_type == 'mlp':
                model = MLPClassifier(X_train.shape[1], n_classes=n_classes).to(device)
            elif model_type == 'transformer':
                model = TransformerClassifier(X_train.shape[1], n_classes=n_classes).to(device)
            elif model_type == 'mamba':
                model = MambaClassifier(X_train.shape[1], n_classes=n_classes).to(device)
            elif model_type == 'gnn':
                model = GNNClassifier(X_train.shape[1], n_classes=n_classes).to(device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            X_t = torch.FloatTensor(X_train).to(device)
            y_t = torch.LongTensor(y_train).to(device)

            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_t), y_t)
                loss.backward()
                optimizer.step()

            # Predict
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_val).to(device))
                y_pred = logits.argmax(dim=1).cpu().numpy()
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()

        # Compute all metrics for this fold
        fold_acc = accuracy_score(y_val, y_pred)
        fold_bacc = balanced_accuracy_score(y_val, y_pred)
        fold_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        fold_prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        fold_rec = recall_score(y_val, y_pred, average='macro', zero_division=0)

        # ROC AUC
        fold_auc = np.nan
        if y_proba is not None:
            try:
                if n_classes == 2:
                    fold_auc = roc_auc_score(y_val, y_proba[:, 1])
                else:
                    fold_auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
            except ValueError:
                pass

        per_fold.append({
            'fold': fold_i,
            'accuracy': fold_acc,
            'balanced_acc': fold_bacc,
            'f1_macro': fold_f1,
            'precision_macro': fold_prec,
            'recall_macro': fold_rec,
            'roc_auc': fold_auc,
        })

    # Aggregate across folds
    metric_keys = ['accuracy', 'balanced_acc', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc']
    summary = {}
    for mk in metric_keys:
        vals = [f[mk] for f in per_fold if not np.isnan(f[mk])]
        if vals:
            summary[f'{mk}_mean'] = float(np.mean(vals))
            summary[f'{mk}_std'] = float(np.std(vals))
            summary[mk] = f"{np.mean(vals):.3f}\u00b1{np.std(vals):.3f}"
        else:
            summary[f'{mk}_mean'] = np.nan
            summary[f'{mk}_std'] = np.nan
            summary[mk] = 'N/A'

    summary['per_fold'] = per_fold

    return summary


def run_baselines(dataset_name: str, device='cuda'):
    """Run all baselines for a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print('='*60)

    # Load data
    X, y, feature_names, config = load_dataset(dataset_name)
    n_classes = len(np.unique(y))
    print(f"Shape: {X.shape}, Classes: {np.bincount(y)}")

    # SHAP feature selection (adaptive k based on 80% importance threshold)
    fixed_k = len(config.true_mb) if config.true_mb else None
    shap_idx, shap_names, shap_df, k = compute_shap_features(X, y, feature_names, k=fixed_k)

    print(f"\nSHAP Top-{k} (auto-selected): {shap_names}")
    if config.true_mb:
        true_names = config.true_mb_names or [feature_names[i] for i in config.true_mb]
        print(f"True MB:     {true_names}")
        overlap = set(shap_idx) & set(config.true_mb)
        print(f"Overlap:     {len(overlap)}/{k} ({[feature_names[i] for i in overlap]})")

    # Build structured result
    shap_importance = [
        {'feature': row['feature'], 'importance': float(row['shap_importance'])}
        for _, row in shap_df.iterrows()
    ]

    dataset_result = {
        'dataset': dataset_name,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_classes': int(n_classes),
        'feature_names': list(feature_names),
        'shap_features': shap_names,
        'shap_indices': [int(i) for i in shap_idx],
        'shap_k': int(k),
        'shap_importance': shap_importance,
        'true_mb': [int(i) for i in config.true_mb] if config.true_mb else None,
        'results': {},
        'timestamp': datetime.now().isoformat(),
    }

    # Run classifiers
    header = (f"{'Model':<15} {'Features':<12} {'Accuracy':<15} {'Precision':<15} "
              f"{'Recall':<15} {'F1 Macro':<15} {'Balanced':<15}")
    print(f"\n{header}")
    print('-' * len(header))

    for model_type in ['mlp', 'transformer', 'mamba', 'gnn', 'elm']:
        for feat_set, feat_name in [('all', 'All'), ('shap', f'SHAP-{k}')]:
            X_sub = X if feat_set == 'all' else X[:, shap_idx]

            metrics = train_and_evaluate(X_sub, y, model_type, device)

            key = f"{model_type}_{feat_set}"
            dataset_result['results'][key] = metrics

            print(f"{model_type.upper():<15} {feat_name:<12} {metrics['accuracy']:<15} "
                  f"{metrics['precision_macro']:<15} {metrics['recall_macro']:<15} "
                  f"{metrics['f1_macro']:<15} {metrics['balanced_acc']:<15}")

    return dataset_result, shap_df


def make_json_serializable(obj):
    """Recursively convert numpy types and other non-serializable objects for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description='Baseline Comparison for JCCE')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--all', action='store_true',
                        help=f'Run target datasets: {DEFAULT_DATASETS}')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base-path', type=str, default=None,
                        help='Override auto-detected project base path')
    args = parser.parse_args()

    # Override base path if provided
    global BASE_PATH, DATA_PATH, RESULTS_PATH
    if args.base_path:
        BASE_PATH = Path(args.base_path)
        DATA_PATH = BASE_PATH / 'data'
        RESULTS_PATH = BASE_PATH / 'results' / 'baselines'

    print(f"Base path: {BASE_PATH}")

    if args.list:
        print("\nAvailable datasets:")
        for key, cfg in DATASETS.items():
            marker = '*' if key in DEFAULT_DATASETS else ' '
            print(f"  [{marker}] {key:<20} {cfg.name}")
        print(f"\n  * = included in --all ({', '.join(DEFAULT_DATASETS)})")
        return

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.all:
        datasets = DEFAULT_DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['lucas']

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name in datasets:
        if name not in DATASETS:
            print(f"WARNING: Unknown dataset '{name}', skipping.")
            continue
        try:
            dataset_result, shap_df = run_baselines(name, device)
            all_results[name] = dataset_result

            # Save SHAP importance CSV
            shap_df.to_csv(RESULTS_PATH / f'{name}_shap_importance.csv', index=False)
        except Exception as e:
            import traceback
            print(f"ERROR on {name}: {e}")
            traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pickle (complete data)
    pkl_path = RESULTS_PATH / f'baselines_{timestamp}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nPickle saved: {pkl_path}")

    # JSON (human-readable)
    json_path = RESULTS_PATH / f'baselines_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    print(f"JSON saved:   {json_path}")

    print(f"\n{'='*60}")
    print(f"Results saved to {RESULTS_PATH}")
    print('='*60)


if __name__ == '__main__':
    main()
