"""Twins dataset loader for CATE evaluation (GANITE-preprocessed format).

The Twins dataset (Almond, Chay & Lee 2005, QJE) provides real twin-pair
counterfactuals for binary mortality outcomes — the only widely-used real-data
CATE benchmark with ground-truth counterfactuals (every twin's outcome is the
counterfactual for the other twin under a "heavier vs lighter" treatment).

The standard ML preprocessing is from Yoon, Jordon & van der Schaar 2018
(GANITE, ICLR 2018). Each row is one twin observation:
    - X: 30 covariates (gestational age, mother's age, race, education, ...)
    - T: binary treatment (1 if heavier twin, 0 if lighter)
    - Y_factual: observed binary mortality given T
    - Y_cf: counterfactual mortality (the other twin's outcome) — only known
            because of the twin-pair design

This file expects the data to live at:
    data/twins/twins_X.csv     — (n, 30) feature matrix
    data/twins/twins_T.csv     — (n,) binary treatment
    data/twins/twins_Yf.csv    — (n,) factual mortality
    data/twins/twins_Ycf.csv   — (n,) counterfactual mortality

If not found, prints download instructions and raises FileNotFoundError.

Returns the standard JCCE loader format: (X_with_T_appended, Y_factual, config).
The counterfactual data is stored in config['Y_counterfactual'] for PEHE
evaluation post-training.

Source links:
    - GANITE repo: https://github.com/jsyoon0823/GANITE
    - Almond-Chay-Lee 2005 source data: NBER linked-birth/infant-death
    - Standard ML preprocessing: GANITE preprocess script
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
TWINS_DIR = PROJECT_ROOT / "data" / "twins"


def _print_download_instructions():
    print("=" * 70)
    print("Twins dataset not found at:", TWINS_DIR)
    print("\nTo download (one of):")
    print("\n[Option 1 — GANITE preprocessed CSVs, recommended]:")
    print("  cd <jcce-repo-root>")
    print("  mkdir -p data/twins")
    print("  # GANITE's Twins data is at https://github.com/jsyoon0823/GANITE/tree/master/data")
    print("  # Download Twin_data.csv and split into 4 files:")
    print("  python -c \"")
    print("  import pandas as pd")
    print("  df = pd.read_csv('Twin_data.csv')")
    print("  X_cols = [c for c in df.columns if c not in ('treatment','y_factual','y_cfactual')]")
    print("  df[X_cols].to_csv('data/twins/twins_X.csv', index=False)")
    print("  df['treatment'].to_csv('data/twins/twins_T.csv', index=False, header=False)")
    print("  df['y_factual'].to_csv('data/twins/twins_Yf.csv', index=False, header=False)")
    print("  df['y_cfactual'].to_csv('data/twins/twins_Ycf.csv', index=False, header=False)")
    print("  \"")
    print("\n[Option 2 — direct from Mihaela van der Schaar lab repo]:")
    print("  git clone https://github.com/vanderschaarlab/mlforhealthlabpub.git")
    print("  # then look in alg/ganite or alg/cmgp for Twins CSVs")
    print("=" * 70)


def load_twins() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load Twins dataset for CATE evaluation.

    Returns:
        X: (n, d) feature matrix WITH treatment T appended as the last column
           — this matches the JCCE convention where T_idx is part of X
        Y: (n,) factual mortality (binary)
        config: dict including:
            - 'feature_names': X column names (last is 'treatment')
            - 'treatment_idx': index of T in X
            - 'task': 'classification'
            - 'has_true_dag': False
            - 'Y_counterfactual': (n,) counterfactual mortality — for PEHE eval
            - 'cate_ground_truth': (n,) per-row ground-truth ITE = Y_cf - Y_factual
                                    (sign-corrected for T)
    """
    required_files = [
        TWINS_DIR / "twins_X.csv",
        TWINS_DIR / "twins_T.csv",
        TWINS_DIR / "twins_Yf.csv",
        TWINS_DIR / "twins_Ycf.csv",
    ]
    if not all(p.exists() for p in required_files):
        _print_download_instructions()
        raise FileNotFoundError(f"Twins data not found in {TWINS_DIR}")

    X_df = pd.read_csv(required_files[0])
    T = pd.read_csv(required_files[1], header=None).values.ravel().astype(np.int32)
    Yf = pd.read_csv(required_files[2], header=None).values.ravel().astype(np.int32)
    Ycf = pd.read_csv(required_files[3], header=None).values.ravel().astype(np.int32)

    n = len(T)
    feature_names = list(X_df.columns) + ["treatment"]
    X_features = X_df.values.astype(np.float32)
    X_with_T = np.concatenate([X_features, T[:, None].astype(np.float32)], axis=1)

    # CATE ground truth per row:
    # ITE = Y_treated - Y_untreated
    # If T=1: factual=Y_treated, cf=Y_untreated → ITE = Yf - Ycf
    # If T=0: factual=Y_untreated, cf=Y_treated → ITE = Ycf - Yf
    cate_gt = np.where(T == 1, Yf - Ycf, Ycf - Yf).astype(np.float32)

    config = {
        "name": "Twins (Almond/Chay/Lee 2005, GANITE preprocessing)",
        "feature_names": feature_names,
        "treatment_idx": X_with_T.shape[1] - 1,
        "treatment_name": "heavier_twin",
        "task": "classification",
        "has_true_dag": False,
        "true_dag": None,
        "true_mb": None,
        "Y_counterfactual": Ycf,
        "cate_ground_truth": cate_gt,
        "n_samples": n,
        "n_features": X_with_T.shape[1],
        "ate_ground_truth": float(cate_gt.mean()),
        "pop_size": 50, "n_gen": 30, "max_iter": 200,  # JCCE Optuna defaults
    }

    return X_with_T, Yf.astype(np.float32), config


def evaluate_cate(Y_pred_T1: np.ndarray, Y_pred_T0: np.ndarray,
                   cate_ground_truth: np.ndarray) -> Dict:
    """Compute CATE evaluation metrics (PEHE, ATE error) for a trained model.

    Args:
        Y_pred_T1: (n,) predicted P(Y=1 | X, T=1) for each row
        Y_pred_T0: (n,) predicted P(Y=1 | X, T=0) for each row
        cate_ground_truth: (n,) ground-truth ITE per row

    Returns:
        dict with PEHE (root mean square error of ITE estimates),
        ATE_bias (estimated ATE − ground-truth ATE),
        ATE_estimated, ATE_ground_truth.
    """
    cate_pred = Y_pred_T1 - Y_pred_T0
    pehe = float(np.sqrt(np.mean((cate_pred - cate_ground_truth) ** 2)))
    ate_pred = float(cate_pred.mean())
    ate_gt = float(cate_ground_truth.mean())
    return {
        "PEHE": pehe,
        "ATE_estimated": ate_pred,
        "ATE_ground_truth": ate_gt,
        "ATE_bias": ate_pred - ate_gt,
        "ATE_abs_bias": abs(ate_pred - ate_gt),
    }
