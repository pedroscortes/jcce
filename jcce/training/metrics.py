"""
Evaluation metrics for causal representation learning.

This module provides metrics to evaluate:
1. Reconstruction quality
2. Disentanglement of latent factors
3. Causal structure recovery
"""

from typing import Dict

import jax.numpy as jnp
import numpy as np


def mean_correlation_coefficient(z_pred: jnp.ndarray, z_true: jnp.ndarray) -> float:
    """
    Mean Correlation Coefficient (MCC) for disentanglement.

    Measures if each predicted latent factor correlates with exactly one true factor.

    MCC = (1/K) * Σ_k max_j |corr(z_pred_k, z_true_j)|

    Perfect disentanglement → MCC = 1.0
    Random factors → MCC ≈ 0.0

    Args:
        z_pred: Predicted latent factors (n_samples, latent_dim)
        z_true: True latent factors (n_samples, latent_dim)

    Returns:
        mcc: Mean correlation coefficient [0, 1]
    """
    z_pred = np.array(z_pred)
    z_true = np.array(z_true)

    n_samples, latent_dim_pred = z_pred.shape
    _, latent_dim_true = z_true.shape

    # Compute correlation matrix between predicted and true factors
    # corr_matrix[i, j] = correlation between z_pred[:, i] and z_true[:, j]
    corr_matrix = np.zeros((latent_dim_pred, latent_dim_true))

    for i in range(latent_dim_pred):
        for j in range(latent_dim_true):
            # Handle zero variance case
            std_i = np.std(z_pred[:, i])
            std_j = np.std(z_true[:, j])

            if std_i < 1e-8 or std_j < 1e-8:
                # Zero variance - no correlation
                corr_matrix[i, j] = 0.0
            else:
                corr = np.corrcoef(z_pred[:, i], z_true[:, j])[0, 1]
                # Handle NaN from corrcoef
                if np.isnan(corr):
                    corr_matrix[i, j] = 0.0
                else:
                    corr_matrix[i, j] = np.abs(corr)

    # For each predicted factor, find the maximum correlation with any true factor
    max_corr_per_factor = np.max(corr_matrix, axis=1)

    # Average over all factors
    mcc = np.mean(max_corr_per_factor)

    # Final NaN check
    if np.isnan(mcc):
        return 0.0

    return float(mcc)


def mutual_information_gap(z_pred: jnp.ndarray, z_true: jnp.ndarray) -> float:
    """
    Mutual Information Gap (MIG) for disentanglement.

    Measures how much more one latent variable depends on a single ground truth factor
    compared to the second most dependent factor.

    Higher MIG → Better disentanglement

    Args:
        z_pred: Predicted latent factors (n_samples, latent_dim)
        z_true: True latent factors (n_samples, latent_dim)

    Returns:
        mig: Mutual information gap
    """
    z_pred = np.array(z_pred)
    z_true = np.array(z_true)

    n_samples, latent_dim_pred = z_pred.shape
    _, latent_dim_true = z_true.shape

    # Use the minimum dimension for MI calculation
    latent_dim = min(latent_dim_pred, latent_dim_true)

    # Discretize continuous variables for MI estimation
    def discretize(x, n_bins=20):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x, dtype=int)
        bins = np.linspace(x_min, x_max, n_bins + 1)
        return np.digitize(x, bins[:-1]) - 1

    # Compute mutual information matrix
    mi_matrix = np.zeros((latent_dim, latent_dim))

    for i in range(latent_dim):
        z_pred_discrete = discretize(z_pred[:, i])
        for j in range(latent_dim):
            z_true_discrete = discretize(z_true[:, j])

            # Compute MI using histogram
            mi = mutual_information_discrete(z_pred_discrete, z_true_discrete)
            mi_matrix[i, j] = mi

    # For each true factor j, find the two most informative predicted factors
    gaps = []
    for j in range(latent_dim):
        mi_j = mi_matrix[:, j]
        # Sort in descending order
        sorted_mi = np.sort(mi_j)[::-1]
        if len(sorted_mi) >= 2:
            gap = sorted_mi[0] - sorted_mi[1]
            gaps.append(gap)

    # Average gap
    mig = np.mean(gaps) if gaps else 0.0

    return float(mig)


def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information between discrete variables.

    MI(X, Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))

    Args:
        x: Discrete variable 1 (n_samples,)
        y: Discrete variable 2 (n_samples,)

    Returns:
        mi: Mutual information
    """
    # Compute joint histogram
    xy = np.stack([x, y], axis=0)
    joint_hist, _, _ = np.histogram2d(x, y, bins=(len(np.unique(x)), len(np.unique(y))))
    joint_hist = joint_hist / np.sum(joint_hist)  # Normalize to probability

    # Compute marginals
    px = np.sum(joint_hist, axis=1)
    py = np.sum(joint_hist, axis=0)

    # Compute MI
    mi = 0.0
    for i in range(joint_hist.shape[0]):
        for j in range(joint_hist.shape[1]):
            if joint_hist[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint_hist[i, j] * np.log(joint_hist[i, j] / (px[i] * py[j]))

    return mi


def sap_score(z_pred: jnp.ndarray, z_true: jnp.ndarray) -> float:
    """
    Separated Attribute Predictability (SAP) Score.

    Measures how well each true factor can be predicted from a single learned factor.

    Higher SAP → Better disentanglement

    Args:
        z_pred: Predicted latent factors (n_samples, latent_dim)
        z_true: True latent factors (n_samples, latent_dim)

    Returns:
        sap: SAP score
    """
    z_pred = np.array(z_pred)
    z_true = np.array(z_true)

    n_samples, latent_dim_pred = z_pred.shape
    _, latent_dim_true = z_true.shape

    # For each true factor, compute correlation with all predicted factors
    scores = []

    for j in range(latent_dim_true):
        # Correlations between z_true[:, j] and all predicted factors
        corrs = []
        for i in range(latent_dim_pred):
            # Handle zero variance case
            std_i = np.std(z_pred[:, i])
            std_j = np.std(z_true[:, j])

            if std_i < 1e-8 or std_j < 1e-8:
                corrs.append(0.0)
            else:
                corr = np.corrcoef(z_pred[:, i], z_true[:, j])[0, 1]
                if np.isnan(corr):
                    corrs.append(0.0)
                else:
                    corrs.append(np.abs(corr))

        corrs = np.array(corrs)
        # Sort in descending order
        sorted_corrs = np.sort(corrs)[::-1]

        if len(sorted_corrs) >= 2:
            # SAP = difference between top and second predictor
            sap_j = sorted_corrs[0] - sorted_corrs[1]
            scores.append(sap_j)

    sap = np.mean(scores) if scores else 0.0

    # Final NaN check
    if np.isnan(sap):
        return 0.0

    return float(sap)


def compute_r2_score(z_pred: jnp.ndarray, z_true: jnp.ndarray) -> float:
    """
    Compute R² score for latent factor recovery.

    Finds best permutation of predicted factors to match true factors,
    then computes R² for the alignment.

    Args:
        z_pred: Predicted latent factors (n_samples, latent_dim)
        z_true: True latent factors (n_samples, latent_dim)

    Returns:
        r2: R² score [0, 1]
    """
    z_pred = np.array(z_pred)
    z_true = np.array(z_true)

    n_samples, latent_dim_pred = z_pred.shape
    _, latent_dim_true = z_true.shape

    # Use minimum dimension for alignment
    latent_dim = min(latent_dim_pred, latent_dim_true)

    # Compute correlation matrix
    corr_matrix = np.zeros((latent_dim_pred, latent_dim_true))
    for i in range(latent_dim_pred):
        for j in range(latent_dim_true):
            # Handle zero variance case
            std_i = np.std(z_pred[:, i])
            std_j = np.std(z_true[:, j])

            if std_i < 1e-8 or std_j < 1e-8:
                corr_matrix[i, j] = 0.0
            else:
                corr = np.corrcoef(z_pred[:, i], z_true[:, j])[0, 1]
                if np.isnan(corr):
                    corr_matrix[i, j] = 0.0
                else:
                    corr_matrix[i, j] = np.abs(corr)

    # Find best permutation using greedy matching
    used_cols = set()
    permutation = []
    for i in range(latent_dim_pred):
        # For factor i, find best unused true factor
        best_j = -1
        best_corr = -1
        for j in range(latent_dim_true):
            if j not in used_cols and corr_matrix[i, j] > best_corr:
                best_corr = corr_matrix[i, j]
                best_j = j
        if best_j != -1:
            permutation.append(best_j)
            used_cols.add(best_j)
        else:
            # Fallback: use first unused column or repeat
            permutation.append(0)

    # Align z_pred to z_true using permutation
    # Only align dimensions up to min(latent_dim_pred, latent_dim_true)
    z_pred_aligned = np.zeros((n_samples, latent_dim_true))
    for i in range(min(len(permutation), latent_dim_true)):
        j = permutation[i]
        if j < latent_dim_true:
            z_pred_aligned[:, j] = z_pred[:, i]

    # Compute R² score
    ss_res = np.sum((z_true - z_pred_aligned) ** 2)
    ss_tot = np.sum((z_true - np.mean(z_true, axis=0)) ** 2)

    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    # Handle NaN and clip
    if np.isnan(r2):
        return 0.0

    return float(np.clip(r2, 0.0, 1.0))


def evaluate_disentanglement(z_pred: jnp.ndarray, z_true: jnp.ndarray) -> Dict[str, float]:
    """
    Compute all disentanglement metrics.

    Args:
        z_pred: Predicted latent factors (n_samples, latent_dim)
        z_true: True latent factors (n_samples, latent_dim)

    Returns:
        metrics: Dictionary with all disentanglement metrics
    """
    return {
        "mcc": mean_correlation_coefficient(z_pred, z_true),
        "mig": mutual_information_gap(z_pred, z_true),
        "sap": sap_score(z_pred, z_true),
        "r2": compute_r2_score(z_pred, z_true),
    }


def structural_hamming_distance(A_pred: jnp.ndarray, A_true: jnp.ndarray) -> float:
    """
    Structural Hamming Distance (SHD) between two DAGs.

    Counts the number of edge insertions, deletions, or flips needed to transform
    A_pred into A_true.

    Lower SHD → Better structure recovery

    Args:
        A_pred: Predicted adjacency matrix (n, n)
        A_true: True adjacency matrix (n, n)

    Returns:
        shd: Structural Hamming distance
    """
    A_pred = np.array(A_pred)
    A_true = np.array(A_true)

    # Binarize (threshold at small value)
    A_pred_bin = (np.abs(A_pred) > 1e-3).astype(int)
    A_true_bin = (np.abs(A_true) > 1e-3).astype(int)

    # Count differences
    diff = A_pred_bin - A_true_bin

    # Extra edges (false positives)
    extra_edges = np.sum(diff > 0)

    # Missing edges (false negatives)
    missing_edges = np.sum(diff < 0)

    # SHD = extra + missing (simplified, ignores edge reversals)
    shd = extra_edges + missing_edges

    return float(shd)


def graph_precision_recall_f1(
    A_pred: jnp.ndarray, A_true: jnp.ndarray, threshold: float = 1e-3
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for edge prediction.

    These are standard metrics for evaluating causal structure recovery:
    - Precision: Of predicted edges, how many are correct? (TP / (TP + FP))
    - Recall: Of true edges, how many did we find? (TP / (TP + FN))
    - F1: Harmonic mean of precision and recall

    Args:
        A_pred: Predicted adjacency matrix (n, n)
        A_true: True adjacency matrix (n, n)
        threshold: Threshold for binarizing adjacency matrices

    Returns:
        metrics: Dictionary with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'
    """
    A_pred = np.array(A_pred)
    A_true = np.array(A_true)

    # Binarize
    A_pred_bin = (np.abs(A_pred) > threshold).astype(int)
    A_true_bin = (np.abs(A_true) > threshold).astype(int)

    # Compute confusion matrix for edges
    # TP: Predicted edge exists and true edge exists
    # FP: Predicted edge exists but true edge doesn't
    # FN: True edge exists but not predicted
    # TN: Neither predicted nor true edge

    tp = np.sum((A_pred_bin == 1) & (A_true_bin == 1))
    fp = np.sum((A_pred_bin == 1) & (A_true_bin == 0))
    fn = np.sum((A_pred_bin == 0) & (A_true_bin == 1))
    tn = np.sum((A_pred_bin == 0) & (A_true_bin == 0))

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_reachability_matrix(A: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """
    Compute the reachability (transitive closure) matrix of a DAG.

    R[i,j] = 1 if there exists a directed path from i to j.

    Uses matrix exponentiation: R = (I + A)^n > 0

    Args:
        A: Adjacency matrix (n, n). A[i,j] > 0 means edge i -> j
        threshold: Threshold for binarizing A

    Returns:
        R: Reachability matrix (n, n). R[i,j] = 1 if i can reach j
    """
    n = A.shape[0]

    # Binarize
    A_bin = (np.abs(A) > threshold).astype(float)

    # Compute transitive closure via repeated squaring
    # R = A + A^2 + A^3 + ... + A^n (but we use (I + A)^n instead)
    R = np.eye(n) + A_bin

    # Matrix exponentiation: compute R^n
    power = 1
    while power < n:
        R = np.dot(R, R)
        R = (R > 0).astype(float)  # Binarize to avoid overflow
        power *= 2

    # Remove self-loops (diagonal)
    np.fill_diagonal(R, 0)

    return R


def structural_intervention_distance(
    A_pred: np.ndarray, A_true: np.ndarray, threshold: float = 1e-3
) -> int:
    """
    Structural Intervention Distance (SID) between two DAGs.

    SID counts the number of ordered pairs (i, j) where the intervention
    distribution P(Xj | do(Xi)) differs between the two graphs.

    This happens when:
    - In one graph, Xi can causally affect Xj (there's a directed path)
    - In the other graph, Xi cannot causally affect Xj

    Lower SID → Better causal structure recovery

    Reference:
        Peters & Bühlmann (2015). "Structural Intervention Distance for
        Evaluating Causal Graphs"

    Args:
        A_pred: Predicted adjacency matrix (n, n). A[i,j] > 0 means i -> j
        A_true: True adjacency matrix (n, n)
        threshold: Threshold for binarizing adjacency matrices

    Returns:
        sid: Structural Intervention Distance (non-negative integer)
    """
    A_pred = np.array(A_pred)
    A_true = np.array(A_true)

    # Handle size mismatch
    n_pred, n_true = A_pred.shape[0], A_true.shape[0]
    if n_pred != n_true:
        # Pad smaller matrix
        n = max(n_pred, n_true)
        if n_pred < n:
            A_pred_new = np.zeros((n, n))
            A_pred_new[:n_pred, :n_pred] = A_pred
            A_pred = A_pred_new
        if n_true < n:
            A_true_new = np.zeros((n, n))
            A_true_new[:n_true, :n_true] = A_true
            A_true = A_true_new

    # Compute reachability matrices
    R_pred = compute_reachability_matrix(A_pred, threshold)
    R_true = compute_reachability_matrix(A_true, threshold)

    # SID = number of pairs where reachability differs
    # Count (i,j) where R_pred[i,j] != R_true[i,j]
    sid = int(np.sum(R_pred != R_true))

    return sid


def structural_intervention_distance_normalized(
    A_pred: np.ndarray, A_true: np.ndarray, threshold: float = 1e-3
) -> float:
    """
    Normalized Structural Intervention Distance.

    SID normalized by the maximum possible value (n * (n-1)).

    Args:
        A_pred: Predicted adjacency matrix (n, n)
        A_true: True adjacency matrix (n, n)
        threshold: Threshold for binarizing adjacency matrices

    Returns:
        sid_norm: Normalized SID in [0, 1]. Lower is better.
    """
    A_pred = np.array(A_pred)
    A_true = np.array(A_true)

    n = max(A_pred.shape[0], A_true.shape[0])

    sid = structural_intervention_distance(A_pred, A_true, threshold)

    # Maximum possible SID is n * (n-1) (all off-diagonal pairs)
    max_sid = n * (n - 1)

    if max_sid == 0:
        return 0.0

    return float(sid) / float(max_sid)


def evaluate_structure_recovery(
    A_pred: jnp.ndarray, A_true: jnp.ndarray, compute_sid: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of graph structure recovery.

    Computes all standard causal discovery metrics:
    - Structural Hamming Distance (SHD) - lower is better
    - Structural Intervention Distance (SID) - lower is better
    - Precision, Recall, F1 - higher is better

    Args:
        A_pred: Predicted adjacency matrix (n, n)
        A_true: True adjacency matrix (n, n)
        compute_sid: Whether to compute SID (can be slow for large graphs)

    Returns:
        metrics: Dictionary with all structure recovery metrics
    """
    # SHD
    shd = structural_hamming_distance(A_pred, A_true)

    # Precision, Recall, F1
    prf_metrics = graph_precision_recall_f1(A_pred, A_true)

    # Combine
    metrics = {"shd": shd, **prf_metrics}

    # SID (optional, can be slow)
    if compute_sid:
        sid = structural_intervention_distance(A_pred, A_true)
        sid_norm = structural_intervention_distance_normalized(A_pred, A_true)
        metrics["sid"] = sid
        metrics["sid_normalized"] = sid_norm

    return metrics
