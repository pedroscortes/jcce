"""
SHAP-based Feature Selection Baseline

Uses SHAP values to select predictively important features.
This serves as a baseline to compare against JCCE's causal MB selection.

SHAP measures predictive importance (correlation-based),
while JCCE MB measures causal importance (structure-based).
"""

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_shap_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_type: str = 'xgboost',
    n_estimators: int = 100,
    max_samples: int = 1000,
    random_state: int = 42,
) -> Dict:
    """
    Compute SHAP feature importance values.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: List of feature names
        model_type: 'xgboost' or 'random_forest'
        n_estimators: Number of trees
        max_samples: Max samples for SHAP computation
        random_state: Random seed

    Returns:
        Dict with SHAP values and importance rankings
    """
    # Train model
    if model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    # Split for training (use portion for SHAP to avoid overfitting)
    X_train, X_shap, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    model.fit(X_train, y_train)

    # Compute SHAP values
    if max_samples and len(X_shap) > max_samples:
        idx = np.random.choice(len(X_shap), max_samples, replace=False)
        X_shap = X_shap[idx]

    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_shap)

    # Handle binary vs multiclass
    if isinstance(shap_values, list):
        # Multiclass: take mean absolute across classes
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values = np.abs(shap_values)

    # Compute mean absolute SHAP per feature
    mean_shap = np.mean(shap_values, axis=0)

    # Rank features
    ranking = np.argsort(mean_shap)[::-1]

    return {
        'shap_values': shap_values,
        'mean_shap': mean_shap,
        'ranking': ranking,
        'feature_names': feature_names,
        'importance_df': pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_shap,
            'rank': [list(ranking).index(i) + 1 for i in range(len(feature_names))]
        }).sort_values('mean_abs_shap', ascending=False),
    }


def select_features_shap(
    shap_result: Dict,
    method: str = 'top_k',
    k: Optional[int] = None,
    threshold: float = 0.01,
) -> Tuple[List[int], List[str]]:
    """
    Select features based on SHAP importance.

    Args:
        shap_result: Output from compute_shap_importance
        method: 'top_k' or 'threshold'
        k: Number of top features (for top_k)
        threshold: Minimum mean |SHAP| (for threshold)

    Returns:
        Tuple of (selected_indices, selected_names)
    """
    mean_shap = shap_result['mean_shap']
    feature_names = shap_result['feature_names']
    ranking = shap_result['ranking']

    if method == 'top_k':
        if k is None:
            k = max(3, len(feature_names) // 3)  # Default: top third
        selected_idx = list(ranking[:k])

    elif method == 'threshold':
        # Normalize SHAP values
        max_shap = np.max(mean_shap)
        if max_shap > 0:
            norm_shap = mean_shap / max_shap
        else:
            norm_shap = mean_shap
        selected_idx = [i for i in range(len(mean_shap)) if norm_shap[i] >= threshold]

        # Ensure at least 1 feature
        if len(selected_idx) == 0:
            selected_idx = [ranking[0]]

    else:
        raise ValueError(f"Unknown method: {method}")

    selected_names = [feature_names[i] for i in selected_idx]

    return selected_idx, selected_names


def compare_feature_sets(
    shap_features: List[int],
    mb_features: List[int],
    true_mb: Optional[List[int]] = None,
    n_total: int = None,
) -> Dict:
    """
    Compare SHAP-selected features with MB features.

    Args:
        shap_features: Indices selected by SHAP
        mb_features: Indices selected by JCCE MB
        true_mb: Ground truth MB indices (if known)
        n_total: Total number of features

    Returns:
        Dict with comparison metrics
    """
    shap_set = set(shap_features)
    mb_set = set(mb_features)

    # Jaccard similarity
    intersection = len(shap_set & mb_set)
    union = len(shap_set | mb_set)
    jaccard = intersection / union if union > 0 else 0

    result = {
        'shap_n_features': len(shap_features),
        'mb_n_features': len(mb_features),
        'overlap': intersection,
        'jaccard_similarity': jaccard,
        'shap_only': list(shap_set - mb_set),
        'mb_only': list(mb_set - shap_set),
        'both': list(shap_set & mb_set),
    }

    # Compare to ground truth if available
    if true_mb is not None:
        true_set = set(true_mb)

        # SHAP vs ground truth
        shap_tp = len(shap_set & true_set)
        shap_fp = len(shap_set - true_set)
        shap_fn = len(true_set - shap_set)
        shap_precision = shap_tp / (shap_tp + shap_fp) if (shap_tp + shap_fp) > 0 else 0
        shap_recall = shap_tp / (shap_tp + shap_fn) if (shap_tp + shap_fn) > 0 else 0
        shap_f1 = 2 * shap_precision * shap_recall / (shap_precision + shap_recall) if (shap_precision + shap_recall) > 0 else 0

        # MB vs ground truth
        mb_tp = len(mb_set & true_set)
        mb_fp = len(mb_set - true_set)
        mb_fn = len(true_set - mb_set)
        mb_precision = mb_tp / (mb_tp + mb_fp) if (mb_tp + mb_fp) > 0 else 0
        mb_recall = mb_tp / (mb_tp + mb_fn) if (mb_tp + mb_fn) > 0 else 0
        mb_f1 = 2 * mb_precision * mb_recall / (mb_precision + mb_recall) if (mb_precision + mb_recall) > 0 else 0

        result['ground_truth'] = {
            'shap': {
                'precision': shap_precision,
                'recall': shap_recall,
                'f1': shap_f1,
                'true_positives': list(shap_set & true_set),
                'false_positives': list(shap_set - true_set),
                'false_negatives': list(true_set - shap_set),
            },
            'mb': {
                'precision': mb_precision,
                'recall': mb_recall,
                'f1': mb_f1,
                'true_positives': list(mb_set & true_set),
                'false_positives': list(mb_set - true_set),
                'false_negatives': list(true_set - mb_set),
            },
        }

    return result


if __name__ == '__main__':
    # Quick test with LUCAS
    import sys
    sys.path.insert(0, '/home/user/Documentos/Pedro/dep')

    from pathlib import Path

    # Load LUCAS
    data_path = Path('/home/user/Documentos/Pedro/jcce/data/benchmarks/lucas/raw/lucas0_train.csv')
    df = pd.read_csv(data_path)

    feature_names = [c for c in df.columns if c != 'Lung_cancer']
    X = df[feature_names].values
    y = df['Lung_cancer'].values

    print("="*60)
    print("SHAP Feature Selection Test - LUCAS")
    print("="*60)

    # Compute SHAP
    shap_result = compute_shap_importance(X, y, feature_names)

    print("\nSHAP Feature Importance:")
    print(shap_result['importance_df'].to_string(index=False))

    # Select top-5 (same as true MB size)
    selected_idx, selected_names = select_features_shap(shap_result, method='top_k', k=5)

    print(f"\nTop-5 SHAP features: {selected_names}")
    print(f"Indices: {selected_idx}")

    # Compare to true MB
    true_mb = [0, 4, 8, 9, 10]
    true_mb_names = ['Smoking', 'Genetics', 'Fatigue', 'Allergy', 'Coughing']
    print(f"\nTrue MB: {true_mb_names}")

    comparison = compare_feature_sets(selected_idx, true_mb, true_mb=true_mb)

    print(f"\nSHAP vs Ground Truth:")
    print(f"  Precision: {comparison['ground_truth']['shap']['precision']:.3f}")
    print(f"  Recall: {comparison['ground_truth']['shap']['recall']:.3f}")
    print(f"  F1: {comparison['ground_truth']['shap']['f1']:.3f}")

    print("\n" + "="*60)
