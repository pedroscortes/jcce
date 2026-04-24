"""
Metrics for evaluating causal structure learning.

Includes standard metrics for comparing learned vs ground truth DAGs:
- F1, Precision, Recall
- Structural Hamming Distance (SHD)
- True Positive Rate (TPR), False Positive Rate (FPR)
"""

import jax.numpy as jnp
import numpy as np

# Standard edge threshold for binarizing continuous adjacency matrices.
# Used across all modules that need to determine edge presence.
# - Structure metrics (SHD/F1): 0.3 — standard in NOTEARS/GOLEM literature
# - Bootstrap stability: 0.3 — same as structure metrics for consistency
# NOTE: DML parent/edge detection uses a lower threshold (0.01) intentionally,
# because DML should consider all plausible causal pathways, not just strong ones.
EDGE_THRESHOLD_DEFAULT = 0.3


def threshold_adjacency(A: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """
    Convert continuous adjacency to binary.

    Args:
        A: Continuous adjacency matrix
        threshold: Threshold value (edges with |A[i,j]| > threshold are kept)

    Returns:
        A_binary: Binary adjacency matrix
    """
    A_binary = (jnp.abs(A) > threshold).astype(float)
    # Remove self-loops
    A_binary = A_binary.at[jnp.diag_indices(A.shape[0])].set(0.0)
    return A_binary


def compute_structure_metrics(
    A_pred: jnp.ndarray,
    A_true: jnp.ndarray,
    threshold: float = 0.3
) -> dict:
    """
    Compute structure recovery metrics.

    Args:
        A_pred: Predicted adjacency matrix (can be continuous)
        A_true: Ground truth adjacency matrix (binary)
        threshold: Threshold for binarizing A_pred

    Returns:
        metrics: Dict with F1, precision, recall, SHD, etc.
    """
    # Binarize predicted adjacency
    A_pred_binary = threshold_adjacency(A_pred, threshold)
    A_true_binary = (jnp.abs(A_true) > 1e-6).astype(float)

    # Convert to numpy for easier computation
    A_pred_np = np.array(A_pred_binary)
    A_true_np = np.array(A_true_binary)

    # True positives, false positives, false negatives, true negatives
    TP = np.sum((A_pred_np == 1) & (A_true_np == 1))
    FP = np.sum((A_pred_np == 1) & (A_true_np == 0))
    FN = np.sum((A_pred_np == 0) & (A_true_np == 1))
    TN = np.sum((A_pred_np == 0) & (A_true_np == 0))

    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Structural Hamming Distance (SHD)
    # Number of edge insertions + deletions + reversals needed
    # For now, just count disagreements (simplified SHD)
    shd = FP + FN

    # True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = recall  # Same as recall
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'shd': int(shd),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'tp': int(TP),
        'fp': int(FP),
        'fn': int(FN),
        'tn': int(TN),
    }


def hard_threshold_adjacency(
    A: np.ndarray,
    threshold: float = EDGE_THRESHOLD_DEFAULT,
) -> np.ndarray:
    """
    Post-training hard thresholding: zero out weak edges.

    Unlike threshold_adjacency() which returns binary, this preserves
    continuous weights above the threshold. Use before DML/CV to reduce
    dense-graph artifacts.

    Args:
        A: Continuous adjacency matrix (n, n)
        threshold: Minimum absolute weight to keep

    Returns:
        A with entries below threshold zeroed out
    """
    A_out = np.array(A, dtype=np.float32)
    mask = np.abs(A_out) <= threshold
    A_out[mask] = 0.0
    np.fill_diagonal(A_out, 0.0)
    return A_out


def compute_sid(A_pred, A_true, threshold=0.3) -> int:
    """Structural Intervention Distance (Peters & Bühlmann 2015).

    Counts ordered pairs (i,j) where reachability differs between the two DAGs.
    Re-exports from jcce.training.metrics for convenience.

    Args:
        A_pred: Predicted adjacency matrix (n, n)
        A_true: True adjacency matrix (n, n)
        threshold: Threshold for binarizing adjacency matrices

    Returns:
        SID (non-negative integer). Lower is better.
    """
    from jcce.training.metrics import structural_intervention_distance
    return structural_intervention_distance(
        np.array(A_pred), np.array(A_true), threshold=threshold)


def compute_sid_normalized(A_pred, A_true, threshold=0.3) -> float:
    """Normalized SID in [0, 1]. Lower is better."""
    from jcce.training.metrics import structural_intervention_distance_normalized
    return structural_intervention_distance_normalized(
        np.array(A_pred), np.array(A_true), threshold=threshold)


def compute_varsortability(X, A_true, threshold=1e-6) -> float:
    """Reisach et al. (NeurIPS 2021). Fraction of edges j->i where var(Xj) < var(Xi).

    Convention: A_true[i,j] != 0 means j -> i.

    Args:
        X: (n_samples, d) data matrix
        A_true: (d, d) ground truth adjacency matrix
        threshold: Edge presence threshold

    Returns:
        Fraction of edges consistent with variance ordering (0.5 if no edges).
    """
    A_bin = (np.abs(np.asarray(A_true)) > threshold).astype(int)
    edges = list(zip(*np.where(A_bin != 0)))  # (i, j) pairs where j->i
    if not edges:
        return 0.5
    variances = np.var(np.asarray(X), axis=0)
    consistent = sum(1 for i, j in edges if variances[j] < variances[i])
    return consistent / len(edges)
