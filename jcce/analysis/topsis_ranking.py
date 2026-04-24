"""
TOPSIS ranking for multi-criteria decision making on Pareto solutions.

Implements the Technique for Order of Preference by Similarity to Ideal
Solution (Hwang & Yoon, 1981). Used to select the "best" Pareto solution
when multiple competing objectives exist.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def topsis_rank(
    solutions: List[Dict],
    criteria: List[str] = ('balanced_accuracy', 'mb_sparsity'),
    weights: Optional[List[float]] = None,
    beneficial: Optional[List[bool]] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    TOPSIS ranking (Hwang & Yoon, 1981).

    Ranks solutions by closeness to the ideal point in normalized,
    weighted objective space.

    Args:
        solutions: List of solution dicts. Each must have a 'metrics' sub-dict
            containing the criteria keys.
        criteria: List of metric keys to rank on.
        weights: Importance weights per criterion (summed to 1 internally).
            If None, equal weights.
        beneficial: Per-criterion direction. True = maximize (default),
            False = minimize.

    Returns:
        (ranked_indices, scores): indices sorted best-first, and the
        closeness scores C_i in [0, 1] (higher = better).
    """
    n = len(solutions)
    if n == 0:
        return [], np.array([])
    if n == 1:
        return [0], np.array([1.0])

    m = len(criteria)

    # Default: all beneficial (maximize)
    if beneficial is None:
        beneficial = [True] * m

    # Default: equal weights
    if weights is None:
        weights = [1.0 / m] * m
    else:
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    # Build decision matrix (n solutions × m criteria)
    D = np.zeros((n, m))
    for i, sol in enumerate(solutions):
        metrics = sol.get('metrics', sol)
        for j, crit in enumerate(criteria):
            D[i, j] = float(metrics.get(crit, 0.0))

    # Vector normalization: x_ij / sqrt(sum_i x_ij^2)
    norms = np.sqrt(np.sum(D ** 2, axis=0))
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
    R = D / norms

    # Weighted normalized matrix
    W = np.array(weights)
    V = R * W

    # Ideal and anti-ideal points
    ideal = np.zeros(m)
    anti_ideal = np.zeros(m)
    for j in range(m):
        if beneficial[j]:
            ideal[j] = np.max(V[:, j])
            anti_ideal[j] = np.min(V[:, j])
        else:
            ideal[j] = np.min(V[:, j])
            anti_ideal[j] = np.max(V[:, j])

    # Euclidean distances to ideal and anti-ideal
    d_ideal = np.sqrt(np.sum((V - ideal) ** 2, axis=1))
    d_anti = np.sqrt(np.sum((V - anti_ideal) ** 2, axis=1))

    # Closeness coefficient
    denom = d_ideal + d_anti
    denom = np.where(denom == 0, 1.0, denom)
    C = d_anti / denom

    # Rank: highest C first
    ranked_indices = list(np.argsort(-C))
    return ranked_indices, C
