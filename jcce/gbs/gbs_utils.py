"""
Core GBS utilities: encoding, sampling, kernel computation, co-occurrence, Hellinger.

Dependencies: numpy, scipy, thewalrus (NO JAX).
Data exchange with JCCE via NumPy arrays (.npy files or in-memory).

References:
  - Schuld et al. (2020), PRA 101, 032314 — GBS graph kernel
  - Arrazola & Bromley (2018), PRL 121, 030503 — GBS for dense subgraphs
  - Hamilton et al. (2017) — Gaussian Boson Sampling
"""

import itertools

import numpy as np
from scipy import linalg
from thewalrus import threshold_detection_prob
from thewalrus.quantum import Covmat, gen_Qmat_from_graph
from thewalrus.samples import hafnian_sample_graph, torontonian_sample_graph

# =============================================================================
# Encoding functions: DAG/data → GBS graph matrix
# =============================================================================


def encode_dag_to_gbs(
    A_direct: np.ndarray,
    A_confound: np.ndarray | None = None,
    scale: float = 0.9,
) -> np.ndarray:
    """
    Encoding A: DAG adjacency → GBS graph matrix.

    Symmetrizes the directed adjacency, optionally adds bidirected (confound)
    edges, and normalizes spectral radius < 1 for valid GBS state.

    Args:
        A_direct: (d, d) directed adjacency matrix from GOLEM.
        A_confound: (d, d) symmetric bidirected/confound matrix (optional).
        scale: target spectral radius (default 0.9, must be < 1).

    Returns:
        W_tilde: (d, d) symmetric, non-negative, spectral radius < 1.
    """
    # Symmetrize: W = (|A| + |A|^T) / 2
    W = (np.abs(A_direct) + np.abs(A_direct).T) / 2.0

    # Add confound edges if provided (Grok's formula)
    if A_confound is not None:
        W = W + np.abs(A_confound)

    # Zero diagonal (no self-loops)
    np.fill_diagonal(W, 0.0)

    # Normalize spectral radius < 1
    W_tilde = _normalize_spectral_radius(W, scale)

    return W_tilde


def encode_dependency_to_gbs(
    X: np.ndarray,
    method: str = "partial_corr",
    scale: float = 0.9,
) -> np.ndarray:
    """
    Encoding B: Data → GBS graph matrix from statistical dependencies.

    Args:
        X: (n, d) data matrix.
        method: 'partial_corr' or 'correlation'.
        scale: target spectral radius.

    Returns:
        W_tilde: (d, d) symmetric, non-negative, spectral radius < 1.
    """
    if method == "partial_corr":
        W = _partial_correlation(X)
    elif method == "correlation":
        W = np.abs(np.corrcoef(X, rowvar=False))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'partial_corr' or 'correlation'.")

    np.fill_diagonal(W, 0.0)
    W_tilde = _normalize_spectral_radius(W, scale)
    return W_tilde


def encode_moralized_to_gbs(
    A_direct: np.ndarray,
    threshold: float = 0.1,
    scale: float = 0.9,
) -> np.ndarray:
    """
    Encoding C: DAG → moralized graph → GBS matrix (Gemini's collider fix).

    Moralization adds undirected edges between co-parents of each child,
    turning v-structures (A→C←B) into cliques that GBS can detect.

    Args:
        A_direct: (d, d) directed adjacency matrix.
        threshold: minimum edge weight to count as a parent.
        scale: target spectral radius.

    Returns:
        W_tilde: (d, d) symmetric moralized graph, spectral radius < 1.
    """
    d = A_direct.shape[0]
    A_abs = np.abs(A_direct)

    # Undirected skeleton
    M = (A_abs + A_abs.T) / 2.0

    # Moralize: add edges between co-parents
    for j in range(d):
        parents = np.where(A_abs[:, j] > threshold)[0]
        if len(parents) < 2:
            continue
        for i, pi in enumerate(parents):
            for pk in parents[i + 1 :]:
                # Edge weight = min of the two parent-child weights
                marry_weight = min(A_abs[pi, j], A_abs[pk, j])
                M[pi, pk] = max(M[pi, pk], marry_weight)
                M[pk, pi] = max(M[pk, pi], marry_weight)

    np.fill_diagonal(M, 0.0)
    W_tilde = _normalize_spectral_radius(M, scale)
    return W_tilde


# =============================================================================
# GBS Sampling
# =============================================================================


def sample_gbs(
    W_tilde: np.ndarray,
    n_samples: int = 5000,
    n_mean: float | None = None,
    max_photons: int = 12,
    mode: str = "threshold",
    parallel: bool = False,
) -> np.ndarray:
    """
    Sample photon patterns from GBS given a graph matrix.

    Args:
        W_tilde: (d, d) symmetric graph matrix with spectral radius < 1.
        n_samples: number of samples to draw.
        n_mean: mean photon number. If None, defaults to d/2.
        max_photons: maximum photon count per mode.
        mode: 'threshold' (binary click/no-click) or 'pnr' (photon number resolving).
        parallel: use dask parallelization.

    Returns:
        samples: (n_samples, d) array of photon patterns.
            threshold mode: binary {0, 1}
            pnr mode: non-negative integers
    """
    d = W_tilde.shape[0]
    if n_mean is None:
        n_mean = d / 2.0

    A = np.asarray(W_tilde, dtype=np.float64)

    if mode == "threshold":
        samples = torontonian_sample_graph(
            A,
            n_mean=n_mean,
            samples=n_samples,
            max_photons=max_photons,
            parallel=parallel,
        )
    elif mode == "pnr":
        samples = hafnian_sample_graph(
            A,
            n_mean=n_mean,
            samples=n_samples,
            cutoff=max_photons,
            max_photons=max_photons,
            parallel=parallel,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'threshold' or 'pnr'.")

    return samples


# =============================================================================
# Feature extraction from samples
# =============================================================================


def compute_orbit_features(samples: np.ndarray) -> np.ndarray:
    """
    Convert raw photon patterns to orbit (event) feature vector.

    An orbit groups patterns by total photon number k and number of occupied
    modes. This avoids the exponential raw pattern space.

    Args:
        samples: (n_samples, d) photon pattern array.

    Returns:
        features: (max_k+1, max_modes+1) normalized frequency array where
            features[k, m] = P(total photons = k AND occupied modes = m).
    """
    total_photons = samples.sum(axis=1)
    occupied_modes = (samples > 0).sum(axis=1)
    max_k = int(total_photons.max()) if len(total_photons) > 0 else 0
    max_m = int(occupied_modes.max()) if len(occupied_modes) > 0 else 0

    features = np.zeros((max_k + 1, max_m + 1))
    for k, m in zip(total_photons, occupied_modes):
        features[int(k), int(m)] += 1

    # Normalize to probabilities
    total = features.sum()
    if total > 0:
        features /= total

    return features


def compute_event_probabilities(samples: np.ndarray) -> np.ndarray:
    """
    Compute event (click pattern) probability distribution from samples.

    For threshold (binary) samples, each unique binary vector is an event.
    Returns probabilities as a dictionary mapping tuples → probabilities.

    Args:
        samples: (n_samples, d) binary or integer photon patterns.

    Returns:
        events: dict mapping pattern tuples → probabilities.
    """
    n = len(samples)
    if n == 0:
        return {}

    events = {}
    for row in samples:
        key = tuple(row)
        events[key] = events.get(key, 0) + 1

    for key in events:
        events[key] /= n

    return events


# =============================================================================
# GBS Graph Kernel (Schuld et al., 2020)
# =============================================================================


def gbs_kernel(
    W1: np.ndarray,
    W2: np.ndarray,
    n_samples: int = 5000,
    n_mean: float | None = None,
    max_photons: int = 12,
    mode: str = "threshold",
) -> float:
    """
    Compute GBS graph kernel K(G1, G2) = sum_S P(S|G1) * P(S|G2).

    The kernel measures overlap between the GBS pattern distributions
    of two graphs. High overlap → structurally similar.

    Args:
        W1, W2: (d, d) GBS graph matrices (symmetric, spectral radius < 1).
        n_samples: samples per graph for distribution estimation.
        n_mean: mean photon number (defaults to d/2).
        max_photons: max photon count per mode.
        mode: 'threshold' or 'pnr'.

    Returns:
        K: kernel value (non-negative float).
    """
    P1 = compute_event_probabilities(sample_gbs(W1, n_samples, n_mean, max_photons, mode))
    P2 = compute_event_probabilities(sample_gbs(W2, n_samples, n_mean, max_photons, mode))

    # K = sum_S P(S|G1) * P(S|G2) over shared events
    K = 0.0
    for event, p1 in P1.items():
        if event in P2:
            K += p1 * P2[event]

    return K


def gbs_kernel_matrix(
    W_list: list[np.ndarray],
    n_samples: int = 5000,
    n_mean: float | None = None,
    max_photons: int = 12,
    mode: str = "threshold",
) -> np.ndarray:
    """
    Compute pairwise GBS kernel matrix for a list of graphs.

    Samples each graph once, then computes all pairwise overlaps.
    More efficient than calling gbs_kernel() O(n^2) times.

    Args:
        W_list: list of (d, d) GBS graph matrices.
        n_samples: samples per graph.
        n_mean: mean photon number.
        max_photons: max photon count.
        mode: 'threshold' or 'pnr'.

    Returns:
        K: (n, n) symmetric positive semi-definite kernel matrix.
    """
    n = len(W_list)

    # Sample each graph once
    distributions = []
    for W in W_list:
        samples = sample_gbs(W, n_samples, n_mean, max_photons, mode)
        distributions.append(compute_event_probabilities(samples))

    # Compute pairwise overlaps
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            overlap = 0.0
            for event, pi in distributions[i].items():
                if event in distributions[j]:
                    overlap += pi * distributions[j][event]
            K[i, j] = overlap
            K[j, i] = overlap

    return K


# =============================================================================
# Co-occurrence matrix (for Application B: MB estimation)
# =============================================================================


def gbs_cooccurrence(
    W_tilde: np.ndarray,
    n_samples: int = 5000,
    n_mean: float | None = None,
    max_photons: int = 12,
    mode: str = "threshold",
) -> np.ndarray:
    """
    Compute co-occurrence matrix C[i,j] from GBS samples.

    C[i,j] = P(mode i AND mode j both detected in same sample).
    High co-occurrence → variables are in the same dense subgraph
    → candidate MB members.

    Args:
        W_tilde: (d, d) GBS graph matrix.
        n_samples: number of GBS samples.
        n_mean: mean photon number.
        max_photons: max photons per mode.
        mode: 'threshold' or 'pnr'.

    Returns:
        C: (d, d) symmetric co-occurrence matrix, values in [0, 1].
    """
    samples = sample_gbs(W_tilde, n_samples, n_mean, max_photons, mode)

    # Binary detection: mode detected if photon count > 0
    detected = (samples > 0).astype(np.float64)  # (n_samples, d)

    # Co-occurrence: C[i,j] = mean(detected_i AND detected_j)
    C = (detected.T @ detected) / len(samples)

    return C


# =============================================================================
# Hellinger distance
# =============================================================================


def hellinger_distance(P: dict, Q: dict) -> float:
    """
    Hellinger distance between two discrete distributions.

    H(P, Q) = (1/sqrt(2)) * sqrt(sum_x (sqrt(P(x)) - sqrt(Q(x)))^2)

    Properties: H(P,P) = 0, H(P,Q) = H(Q,P), 0 <= H <= 1, triangle inequality.

    Args:
        P, Q: dicts mapping events → probabilities.

    Returns:
        H: Hellinger distance in [0, 1].
    """
    all_events = set(P.keys()) | set(Q.keys())

    sum_sq = 0.0
    for event in all_events:
        p = P.get(event, 0.0)
        q = Q.get(event, 0.0)
        sum_sq += (np.sqrt(p) - np.sqrt(q)) ** 2

    return np.sqrt(sum_sq / 2.0)


def hellinger_matrix(distributions: list[dict]) -> np.ndarray:
    """
    Compute pairwise Hellinger distance matrix.

    Args:
        distributions: list of event probability dicts.

    Returns:
        H: (n, n) symmetric distance matrix, H[i,i] = 0.
    """
    n = len(distributions)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = hellinger_distance(distributions[i], distributions[j])
            H[i, j] = h
            H[j, i] = h
    return H


# =============================================================================
# Dequantized GBS Kernel (exact, polynomial-time)
# =============================================================================


def graph_to_gbs_state(
    W_tilde: np.ndarray,
    n_mean: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert GBS graph matrix to Gaussian state representation.

    Args:
        W_tilde: (d, d) symmetric graph matrix with spectral radius < 1.
        n_mean: mean photon number. Defaults to d/2.

    Returns:
        mu: (2d,) means vector (zero for undisplaced states).
        cov: (2d, 2d) Wigner covariance matrix in xp ordering.
    """
    d = W_tilde.shape[0]
    if n_mean is None:
        n_mean = d / 2.0
    Q = gen_Qmat_from_graph(np.asarray(W_tilde, dtype=np.float64), n_mean)
    cov = Covmat(Q, hbar=2)
    mu = np.zeros(2 * d)
    return mu, cov


def dequantized_features(
    W_tilde: np.ndarray,
    n_mean: float | None = None,
    max_order: int = 3,
) -> np.ndarray:
    """
    Compute dequantized GBS feature vector: exact marginal click probabilities.

    Instead of sampling, computes P(modes S click) exactly from the Gaussian
    state's covariance matrix using reduced states. For a subset of r modes,
    the cost is O(2^r * r^3) — trivial for r <= 6.

    Feature vector = [order-1 marginals] ⊕ [order-2 marginals] ⊕ [order-3 marginals]
    where order-k marginal = P(all k modes in subset click simultaneously).

    Args:
        W_tilde: (d, d) symmetric GBS graph matrix.
        n_mean: mean photon number. Defaults to d/2.
        max_order: maximum co-click order (1, 2, or 3). Default 3.
            order=1: d features (single-mode click probs)
            order=2: d + C(d,2) features
            order=3: d + C(d,2) + C(d,3) features

    Returns:
        features: 1D array of exact marginal click probabilities.
    """
    d = W_tilde.shape[0]
    mu, cov = graph_to_gbs_state(W_tilde, n_mean)

    features = []

    # Order 1: P(mode i clicks)
    for i in range(d):
        idx = [i, d + i]
        p = float(np.real(threshold_detection_prob(mu[idx], cov[np.ix_(idx, idx)], np.array([1]))))
        features.append(p)

    # Order 2: P(modes i,j both click)
    if max_order >= 2:
        for i, j in itertools.combinations(range(d), 2):
            idx = [i, j, d + i, d + j]
            p = float(
                np.real(threshold_detection_prob(mu[idx], cov[np.ix_(idx, idx)], np.array([1, 1])))
            )
            features.append(p)

    # Order 3: P(modes i,j,k all click)
    if max_order >= 3:
        for i, j, k in itertools.combinations(range(d), 3):
            idx = [i, j, k, d + i, d + j, d + k]
            p = float(
                np.real(
                    threshold_detection_prob(mu[idx], cov[np.ix_(idx, idx)], np.array([1, 1, 1]))
                )
            )
            features.append(p)

    return np.array(features)


def dequantized_kernel(
    W1: np.ndarray,
    W2: np.ndarray,
    n_mean: float | None = None,
    max_order: int = 3,
    normalize: bool = True,
) -> float:
    """
    Dequantized GBS kernel: exact, polynomial-time, no sampling.

    Computes K(G1, G2) = <φ(G1), φ(G2)> where φ(G) is the vector of
    marginal click probabilities from the GBS state encoding of G.

    This extracts the same mathematical structure as the Schuld et al.
    GBS graph kernel (matching/hafnian counting) without quantum sampling.
    Follows Ewin Tang's dequantization tradition.

    Complexity: O(d^max_order) per graph, vs O(N * 2^(d/2)) for sampling.

    Args:
        W1, W2: (d, d) GBS graph matrices.
        n_mean: mean photon number. Defaults to d/2.
        max_order: maximum co-click order (1, 2, or 3).
        normalize: if True, return cosine similarity K/sqrt(K11*K22).

    Returns:
        K: kernel value. If normalize=True, in [0, 1].
    """
    f1 = dequantized_features(W1, n_mean, max_order)
    f2 = dequantized_features(W2, n_mean, max_order)
    K = float(np.dot(f1, f2))

    if normalize:
        K11 = float(np.dot(f1, f1))
        K22 = float(np.dot(f2, f2))
        denom = np.sqrt(K11 * K22)
        K = K / denom if denom > 1e-10 else 0.0

    return K


def dequantized_kernel_matrix(
    W_list: list[np.ndarray],
    n_mean: float | None = None,
    max_order: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute pairwise dequantized GBS kernel matrix.

    Args:
        W_list: list of (d, d) GBS graph matrices.
        n_mean: mean photon number.
        max_order: maximum co-click order.
        normalize: if True, normalize to cosine similarity.

    Returns:
        K: (n, n) symmetric kernel matrix.
    """
    n = len(W_list)
    features = [dequantized_features(W, n_mean, max_order) for W in W_list]

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = float(np.dot(features[i], features[j]))
            K[j, i] = K[i, j]

    if normalize:
        diag = np.sqrt(np.diag(K))
        diag = np.where(diag > 1e-10, diag, 1.0)
        K = K / np.outer(diag, diag)

    return K


def dequantized_cooccurrence(
    W_tilde: np.ndarray,
    n_mean: float | None = None,
) -> np.ndarray:
    """
    Compute exact co-occurrence matrix from GBS state (no sampling).

    C[i,j] = P(mode i AND mode j both click), computed analytically
    from the reduced 2-mode Gaussian state.

    Args:
        W_tilde: (d, d) GBS graph matrix.
        n_mean: mean photon number. Defaults to d/2.

    Returns:
        C: (d, d) symmetric co-occurrence matrix with exact probabilities.
    """
    d = W_tilde.shape[0]
    mu, cov = graph_to_gbs_state(W_tilde, n_mean)

    C = np.zeros((d, d))
    for i in range(d):
        # Diagonal: single-mode click probability
        idx = [i, d + i]
        C[i, i] = float(
            np.real(threshold_detection_prob(mu[idx], cov[np.ix_(idx, idx)], np.array([1])))
        )

        # Off-diagonal: pairwise co-click probability
        for j in range(i + 1, d):
            idx = [i, j, d + i, d + j]
            p = float(
                np.real(threshold_detection_prob(mu[idx], cov[np.ix_(idx, idx)], np.array([1, 1])))
            )
            C[i, j] = p
            C[j, i] = p

    return C


def dequantized_hellinger(
    W1: np.ndarray,
    W2: np.ndarray,
    n_mean: float | None = None,
    max_order: int = 2,
) -> float:
    """
    Hellinger distance between dequantized GBS feature distributions.

    H(G1, G2) = (1/sqrt(2)) * ||sqrt(φ(G1)) - sqrt(φ(G2))||_2

    Args:
        W1, W2: (d, d) GBS graph matrices.
        n_mean: mean photon number.
        max_order: feature order (1, 2, or 3).

    Returns:
        H: Hellinger distance in [0, 1].
    """
    f1 = dequantized_features(W1, n_mean, max_order)
    f2 = dequantized_features(W2, n_mean, max_order)

    # Hellinger on the feature vectors treated as (unnormalized) distributions
    sum_sq = np.sum((np.sqrt(np.maximum(f1, 0)) - np.sqrt(np.maximum(f2, 0))) ** 2)
    return float(np.sqrt(sum_sq / 2.0))


# =============================================================================
# Utility: classical graph distance baselines
# =============================================================================


def shd(A1: np.ndarray, A2: np.ndarray, threshold: float = 0.1) -> int:
    """Structural Hamming Distance between two adjacency matrices."""
    E1 = (np.abs(A1) > threshold).astype(int)
    E2 = (np.abs(A2) > threshold).astype(int)
    return int(np.sum(E1 != E2))


def frobenius_distance(A1: np.ndarray, A2: np.ndarray) -> float:
    """Frobenius norm of difference between adjacency matrices."""
    return float(np.linalg.norm(A1 - A2, "fro"))


def spectral_distance(A1: np.ndarray, A2: np.ndarray) -> float:
    """Distance between sorted eigenvalue spectra of adjacency matrices."""
    ev1 = np.sort(np.real(np.linalg.eigvals(A1)))
    ev2 = np.sort(np.real(np.linalg.eigvals(A2)))
    return float(np.linalg.norm(ev1 - ev2))


def jaccard_edge_distance(A1: np.ndarray, A2: np.ndarray, threshold: float = 0.1) -> float:
    """1 - Jaccard index on edge sets. 0 = identical, 1 = disjoint."""
    E1 = set(zip(*np.where(np.abs(A1) > threshold)))
    E2 = set(zip(*np.where(np.abs(A2) > threshold)))
    if len(E1) == 0 and len(E2) == 0:
        return 0.0
    intersection = len(E1 & E2)
    union = len(E1 | E2)
    return 1.0 - intersection / union if union > 0 else 0.0


# =============================================================================
# Internal helpers
# =============================================================================


def _normalize_spectral_radius(W: np.ndarray, scale: float = 0.9) -> np.ndarray:
    """Normalize a symmetric matrix so its spectral radius equals `scale`."""
    if np.allclose(W, 0):
        return W

    # Spectral radius = max absolute eigenvalue (for symmetric = max singular value)
    spectral_radius = np.max(np.abs(linalg.eigvalsh(W)))

    if spectral_radius < 1e-10:
        return W

    alpha = scale / spectral_radius
    return alpha * W


def _partial_correlation(X: np.ndarray) -> np.ndarray:
    """
    Compute absolute partial correlation matrix from data.

    Partial correlation between Xi and Xj is the correlation after
    controlling for all other variables. Computed via precision matrix.

    Args:
        X: (n, d) data matrix.

    Returns:
        P: (d, d) absolute partial correlation matrix.
    """
    # Covariance → precision → partial correlation
    cov = np.cov(X, rowvar=False)

    # Regularize for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6

    prec = linalg.inv(cov)

    # Partial correlation: rho_ij = -P_ij / sqrt(P_ii * P_jj)
    d = prec.shape[0]
    diag = np.sqrt(np.diag(prec))
    P = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                P[i, j] = -prec[i, j] / (diag[i] * diag[j] + 1e-10)

    return np.abs(P)
