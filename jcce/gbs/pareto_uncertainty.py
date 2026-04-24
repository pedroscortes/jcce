"""
Application D: Hellinger Uncertainty Metrics for Pareto Fronts.

Quantifies structural uncertainty/diversity of a Pareto front by computing
pairwise Hellinger distances between the dequantized GBS feature distributions
of Pareto-optimal DAGs.

Uses dequantized features (no sampling) — polynomial time, exact.

References:
  - Manski (2003): Partial Identification of Probability Distributions
  - Schuld et al. (2020): GBS graph kernel
  - Oh et al. (2024): Classical simulability of GBS for non-negative matrices

Dependencies: numpy, gbs_utils (from this package)
"""

import numpy as np

from .gbs_utils import (
    dequantized_features,
    dequantized_hellinger,
    encode_dag_to_gbs,
    shd,
)


def pareto_gbs_features(
    pareto_dags: list[np.ndarray],
    scale: float = 0.9,
    n_mean: float | None = None,
    max_order: int = 2,
) -> list[np.ndarray]:
    """
    Compute dequantized GBS feature vectors for a list of Pareto-optimal DAGs.

    Args:
        pareto_dags: list of (d, d) directed adjacency matrices from Pareto front.
        scale: spectral radius for GBS encoding.
        n_mean: mean photon number. Defaults to d/2.
        max_order: feature order (1, 2, or 3).

    Returns:
        features: list of 1D feature arrays (one per DAG).
    """
    features = []
    for dag in pareto_dags:
        W = encode_dag_to_gbs(dag, scale=scale)
        f = dequantized_features(W, n_mean=n_mean, max_order=max_order)
        features.append(f)
    return features


def dequantized_hellinger_matrix(
    pareto_dags: list[np.ndarray],
    scale: float = 0.9,
    n_mean: float | None = None,
    max_order: int = 2,
) -> np.ndarray:
    """
    Compute pairwise Hellinger distance matrix for Pareto-optimal DAGs.

    Uses dequantized features (exact, no sampling).

    Args:
        pareto_dags: list of (d, d) directed adjacency matrices.
        scale: spectral radius for GBS encoding.
        n_mean: mean photon number.
        max_order: feature order.

    Returns:
        H: (n, n) symmetric distance matrix, H[i,i] = 0, H[i,j] in [0, 1].
    """
    n = len(pareto_dags)
    W_list = [encode_dag_to_gbs(dag, scale=scale) for dag in pareto_dags]

    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = dequantized_hellinger(W_list[i], W_list[j], n_mean=n_mean, max_order=max_order)
            H[i, j] = h
            H[j, i] = h
    return H


def hellinger_diameter(H: np.ndarray) -> float:
    """
    Structural spread of the Pareto front.

    The diameter is the maximum pairwise Hellinger distance:
    diam(Pareto) = max_{i,j} H(P_i, P_j)

    A large diameter indicates high structural diversity — the Pareto front
    contains DAGs with very different subgraph patterns.

    Args:
        H: (n, n) Hellinger distance matrix.

    Returns:
        diameter: maximum pairwise distance.
    """
    return float(np.max(H))


def mean_hellinger_dispersion(H: np.ndarray) -> float:
    """
    Average structural disagreement across the Pareto front.

    dispersion = mean_{i<j} H(P_i, P_j)

    Measures how spread out the Pareto solutions are in structural space.
    Low dispersion = all solutions are structurally similar.
    High dispersion = solutions disagree on graph structure.

    Args:
        H: (n, n) Hellinger distance matrix.

    Returns:
        dispersion: mean pairwise distance.
    """
    n = H.shape[0]
    if n < 2:
        return 0.0
    upper = H[np.triu_indices(n, k=1)]
    return float(np.mean(upper))


def hellinger_coverage(
    H: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Coverage of the structural space by the Pareto front.

    Measures the entropy of the distance distribution: high coverage means
    distances are spread across the full range, indicating the Pareto front
    samples diverse structural neighborhoods.

    coverage = H(dist) / log(n_bins)  (normalized entropy)

    Args:
        H: (n, n) Hellinger distance matrix.
        n_bins: number of histogram bins for entropy computation.

    Returns:
        coverage: normalized entropy in [0, 1].
    """
    n = H.shape[0]
    if n < 2:
        return 0.0

    upper = H[np.triu_indices(n, k=1)]
    if len(upper) == 0 or np.max(upper) < 1e-10:
        return 0.0

    counts, _ = np.histogram(upper, bins=n_bins, range=(0, np.max(upper) + 1e-10))
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_bins)

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def hellinger_vs_shd_correlation(
    pareto_dags: list[np.ndarray],
    H: np.ndarray | None = None,
    scale: float = 0.9,
    n_mean: float | None = None,
    max_order: int = 2,
) -> dict:
    """
    Mantel-like correlation between Hellinger distances and SHD distances.

    Tests whether the GBS-derived Hellinger distance correlates with the
    simple edge-counting SHD distance. High correlation validates that
    Hellinger captures structural differences. Low correlation means
    Hellinger captures information beyond edge-level disagreement.

    Args:
        pareto_dags: list of (d, d) directed adjacency matrices.
        H: precomputed Hellinger matrix (optional, computed if None).
        scale: spectral radius for GBS encoding.
        n_mean: mean photon number.
        max_order: feature order.

    Returns:
        dict with 'pearson_r', 'spearman_rho', 'shd_matrix', 'hellinger_matrix'.
    """
    from scipy.stats import pearsonr, spearmanr

    n = len(pareto_dags)

    if H is None:
        H = dequantized_hellinger_matrix(pareto_dags, scale, n_mean, max_order)

    # Compute SHD matrix
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = shd(pareto_dags[i], pareto_dags[j])
            S[i, j] = s
            S[j, i] = s

    # Extract upper triangles for correlation
    h_upper = H[np.triu_indices(n, k=1)]
    s_upper = S[np.triu_indices(n, k=1)]

    if len(h_upper) < 3 or np.std(h_upper) < 1e-10 or np.std(s_upper) < 1e-10:
        return {
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "shd_matrix": S,
            "hellinger_matrix": H,
        }

    r, _ = pearsonr(h_upper, s_upper)
    rho, _ = spearmanr(h_upper, s_upper)

    return {
        "pearson_r": float(r),
        "spearman_rho": float(rho),
        "shd_matrix": S,
        "hellinger_matrix": H,
    }


def pareto_uncertainty_report(
    pareto_dags: list[np.ndarray],
    scale: float = 0.9,
    n_mean: float | None = None,
    max_order: int = 2,
) -> dict:
    """
    Complete uncertainty analysis of a Pareto front.

    Computes all Application D metrics in a single call.

    Args:
        pareto_dags: list of (d, d) directed adjacency matrices.
        scale: spectral radius.
        n_mean: mean photon number.
        max_order: feature order.

    Returns:
        dict with keys:
            'hellinger_matrix': (n, n) pairwise Hellinger distances
            'diameter': max pairwise distance
            'mean_dispersion': mean pairwise distance
            'coverage': normalized entropy of distance distribution
            'pearson_r': correlation with SHD
            'spearman_rho': rank correlation with SHD
            'n_dags': number of Pareto DAGs
            'd': number of variables
    """
    n = len(pareto_dags)
    d = pareto_dags[0].shape[0] if n > 0 else 0

    H = dequantized_hellinger_matrix(pareto_dags, scale, n_mean, max_order)

    corr = hellinger_vs_shd_correlation(
        pareto_dags, H=H, scale=scale, n_mean=n_mean, max_order=max_order
    )

    return {
        "hellinger_matrix": H,
        "diameter": hellinger_diameter(H),
        "mean_dispersion": mean_hellinger_dispersion(H),
        "coverage": hellinger_coverage(H),
        "pearson_r": corr["pearson_r"],
        "spearman_rho": corr["spearman_rho"],
        "shd_matrix": corr["shd_matrix"],
        "n_dags": n,
        "d": d,
    }
