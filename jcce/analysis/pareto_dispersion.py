"""
Pareto Dispersion via Hellinger Distance (Phase B.4).

Thin wrapper around jcce.gbs.pareto_uncertainty to measure structural
diversity of Pareto fronts using GBS-inspired Hellinger metrics.

Three key metrics:
- Diameter: max pairwise Hellinger distance (maximum causal uncertainty)
- Mean dispersion: average structural disagreement
- Coverage: effective dimensionality of uncertainty

Usage:
    from jcce.analysis.pareto_dispersion import (
        extract_pareto_dags,
        pareto_hellinger_report,
    )
"""

import numpy as np
from typing import List, Dict, Optional


def extract_pareto_dags(
    solutions: List[Dict],
    key: str = 'structure_A_est',
) -> List[np.ndarray]:
    """
    Extract adjacency matrices from enhanced_solutions.

    Args:
        solutions: List of enhanced_solution dicts.
        key: Key in sol['metrics'] for adjacency matrix.

    Returns:
        List of (n, n) numpy arrays.
    """
    dags = []
    for sol in solutions:
        A = sol.get('metrics', {}).get(key)
        if A is not None:
            dags.append(np.array(A))
    return dags


def pareto_hellinger_report(
    solutions: List[Dict],
    key: str = 'structure_A_est',
    scale: float = 0.9,
    max_order: int = 2,
    n_mean: Optional[float] = None,
) -> Dict[str, object]:
    """
    Compute Hellinger-based dispersion report for Pareto solutions.

    Wraps jcce.gbs.pareto_uncertainty.pareto_uncertainty_report().

    Args:
        solutions: List of enhanced_solution dicts.
        key: Key for adjacency matrix.
        scale: Spectral radius for GBS encoding.
        max_order: Feature order (1, 2, or 3).
        n_mean: Mean photon number.

    Returns:
        Dict with 'diameter', 'mean_dispersion', 'coverage',
        'hellinger_matrix', 'n_dags', and 'feature_dim'.
    """
    from jcce.gbs.pareto_uncertainty import pareto_uncertainty_report

    dags = extract_pareto_dags(solutions, key=key)

    if len(dags) < 2:
        return {
            'diameter': 0.0,
            'mean_dispersion': 0.0,
            'coverage': 1 if dags else 0,
            'hellinger_matrix': np.zeros((len(dags), len(dags))),
            'n_dags': len(dags),
            'feature_dim': 0,
        }

    report = pareto_uncertainty_report(
        dags,
        scale=scale,
        max_order=max_order,
        n_mean=n_mean,
    )

    return report


def hellinger_structural_correlation(
    solutions: List[Dict],
    key: str = 'structure_A_est',
    scale: float = 0.9,
    max_order: int = 2,
) -> Dict[str, float]:
    """
    Compute correlation between Hellinger distance and SHD for Pareto solutions.

    Low correlation suggests Hellinger captures subgraph-level info that
    SHD misses (confirmed in Experiment 4).

    Args:
        solutions: List of enhanced_solution dicts.
        key: Key for adjacency matrix.
        scale: GBS encoding scale.
        max_order: Feature order.

    Returns:
        Dict with 'pearson_r', 'spearman_r', 'n_pairs'.
    """
    from jcce.gbs.pareto_uncertainty import dequantized_hellinger_matrix
    from jcce.gbs.gbs_utils import shd
    from scipy.stats import pearsonr, spearmanr

    dags = extract_pareto_dags(solutions, key=key)
    n = len(dags)

    if n < 3:
        return {'pearson_r': 0.0, 'spearman_r': 0.0, 'n_pairs': 0}

    H = dequantized_hellinger_matrix(dags, scale=scale, max_order=max_order)

    # Compute pairwise SHD
    hellinger_vals = []
    shd_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            hellinger_vals.append(H[i, j])
            shd_vals.append(shd(dags[i], dags[j]))

    if len(hellinger_vals) < 3:
        return {'pearson_r': 0.0, 'spearman_r': 0.0, 'n_pairs': len(hellinger_vals)}

    pr, _ = pearsonr(hellinger_vals, shd_vals)
    sr, _ = spearmanr(hellinger_vals, shd_vals)

    return {
        'pearson_r': float(pr),
        'spearman_r': float(sr),
        'n_pairs': len(hellinger_vals),
    }
