"""
Pareto Edge Stability (Phase B.1).

Computes edge frequency across Pareto-optimal solutions as a structural
uncertainty metric. Edges persistent across diverse accuracy-sparsity
tradeoffs are more reliable causal claims.

Reference: Meinshausen & Bühlmann, JRSSB 2010 (Stability Selection)

Usage:
    from jcce.analysis.pareto_stability import (
        compute_edge_stability,
        compare_stability_to_ground_truth,
        stability_sensitivity_analysis,
    )
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from jcce.utils.metrics import EDGE_THRESHOLD_DEFAULT


def compute_edge_stability(
    solutions: List[Dict],
    n_vars: int,
    threshold: float = EDGE_THRESHOLD_DEFAULT,
    key: str = 'structure_A_est',
) -> np.ndarray:
    """
    Compute edge frequency across Pareto solutions.

    P(Xi -> Xj) = (1/|P|) * sum_m I(|A_m[i,j]| > threshold)

    Args:
        solutions: List of enhanced_solution dicts with
                   sol['metrics'][key] containing adjacency matrices.
        n_vars: Number of variables (full A size = n_vars + 1 including Y).
        threshold: Edge weight threshold for presence.
        key: Key in sol['metrics'] for the adjacency matrix.

    Returns:
        (n_vars+1, n_vars+1) stability matrix with values in [0, 1].
    """
    n = n_vars + 1
    if not solutions:
        return np.zeros((n, n))

    stability = np.zeros((n, n))
    count = 0

    for sol in solutions:
        A = sol.get('metrics', {}).get(key)
        if A is None:
            continue
        A = np.array(A)
        if A.shape == (n, n):
            stability += (np.abs(A) > threshold).astype(float)
            count += 1

    if count == 0:
        return np.zeros((n, n))

    return stability / count


def compute_edge_stability_multi_threshold(
    solutions: List[Dict],
    n_vars: int,
    thresholds: Tuple[float, ...] = (0.1, 0.3, 0.5),
    key: str = 'structure_A_est',
) -> Dict[float, np.ndarray]:
    """
    Compute edge stability at multiple thresholds.

    Args:
        solutions: List of enhanced_solution dicts.
        n_vars: Number of variables.
        thresholds: Tuple of thresholds to evaluate.
        key: Key for adjacency matrix.

    Returns:
        Dict mapping threshold -> stability matrix.
    """
    return {
        t: compute_edge_stability(solutions, n_vars, threshold=t, key=key)
        for t in thresholds
    }


def get_stable_edges(
    stability: np.ndarray,
    min_frequency: float = 0.8,
) -> List[Tuple[int, int, float]]:
    """
    Extract edges above a minimum stability frequency.

    Args:
        stability: (n, n) stability matrix.
        min_frequency: Minimum P(edge) to be considered stable.

    Returns:
        List of (i, j, frequency) tuples, sorted by frequency descending.
    """
    edges = []
    n = stability.shape[0]
    for i in range(n):
        for j in range(n):
            if stability[i, j] >= min_frequency:
                edges.append((i, j, float(stability[i, j])))
    edges.sort(key=lambda x: -x[2])
    return edges


def compare_stability_to_ground_truth(
    stability: np.ndarray,
    true_graph: np.ndarray,
    min_frequency: float = 0.8,
) -> Dict[str, float]:
    """
    Compare stable edges to ground truth adjacency matrix.

    Args:
        stability: (n, n) stability matrix from compute_edge_stability().
        true_graph: (n, n) ground truth binary adjacency matrix.
        min_frequency: Threshold on stability for "stable" edge.

    Returns:
        Dict with precision, recall, f1, n_stable, n_true.
    """
    true_graph = np.array(true_graph)
    assert stability.shape == true_graph.shape, (
        f"Shape mismatch: stability {stability.shape} vs truth {true_graph.shape}"
    )

    # Stable = predicted edges
    predicted = (stability >= min_frequency).astype(float)
    true_binary = (np.abs(true_graph) > 0).astype(float)

    # Exclude diagonal
    np.fill_diagonal(predicted, 0)
    np.fill_diagonal(true_binary, 0)

    tp = np.sum(predicted * true_binary)
    fp = np.sum(predicted * (1 - true_binary))
    fn = np.sum((1 - predicted) * true_binary)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'n_stable': int(np.sum(predicted)),
        'n_true': int(np.sum(true_binary)),
    }


def stability_sensitivity_analysis(
    solutions: List[Dict],
    n_vars: int,
    thresholds: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
    frequencies: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    true_graph: Optional[np.ndarray] = None,
    key: str = 'structure_A_est',
) -> Dict[str, object]:
    """
    Sensitivity analysis: how threshold and frequency cutoffs affect results.

    Args:
        solutions: List of enhanced_solution dicts.
        n_vars: Number of variables.
        thresholds: Edge weight thresholds to evaluate.
        frequencies: Stability frequency cutoffs to evaluate.
        true_graph: Optional ground truth for precision/recall.
        key: Key for adjacency matrix.

    Returns:
        Dict with 'n_stable_edges' (threshold x frequency matrix),
        and optionally 'precision', 'recall', 'f1' matrices.
    """
    n_stable = np.zeros((len(thresholds), len(frequencies)))
    precision_mat = np.zeros_like(n_stable) if true_graph is not None else None
    recall_mat = np.zeros_like(n_stable) if true_graph is not None else None
    f1_mat = np.zeros_like(n_stable) if true_graph is not None else None

    for ti, t in enumerate(thresholds):
        stab = compute_edge_stability(solutions, n_vars, threshold=t, key=key)
        for fi, f in enumerate(frequencies):
            edges = get_stable_edges(stab, min_frequency=f)
            n_stable[ti, fi] = len(edges)

            if true_graph is not None:
                metrics = compare_stability_to_ground_truth(stab, true_graph, min_frequency=f)
                precision_mat[ti, fi] = metrics['precision']
                recall_mat[ti, fi] = metrics['recall']
                f1_mat[ti, fi] = metrics['f1']

    result = {
        'thresholds': list(thresholds),
        'frequencies': list(frequencies),
        'n_stable_edges': n_stable,
    }
    if true_graph is not None:
        result['precision'] = precision_mat
        result['recall'] = recall_mat
        result['f1'] = f1_mat

    return result
