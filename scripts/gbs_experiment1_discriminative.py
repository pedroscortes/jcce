#!/usr/bin/env python3
"""
GBS Experiment 1: Kernel Discriminative Power

Tests whether the dequantized GBS kernel separates structurally different DAGs
better than classical distance measures.

Data:
  - 50 synthetic DAGs from each of 3 families:
      1. Erdos-Renyi (edge probability p=0.2)
      2. Scale-free (preferential attachment)
      3. Small-world (Watts-Strogatz rewired ring, k=4, p_rewire=0.3)
  - Dimensions: d=11 and d=30
  - Edge weights: uniform(0.3, 1.5)

Methods compared:
  1. Dequantized GBS kernel (order 2)
  2. Dequantized GBS kernel (order 3)
  3. SHD (Structural Hamming Distance)
  4. Frobenius norm
  5. Spectral distance
  6. Jaccard edge distance

Metrics:
  - Adjusted Rand Index (ARI): KMeans on embedded features vs true labels
  - Silhouette score on embedded features
  - Wall-clock timing per method

Reference: Roadmap Step 1.3 (prompts/gbs_mb_roadmap.md)
"""

import sys
import time
import warnings

import numpy as np

# Suppress sklearn warnings (MDS FutureWarning, convergence warnings)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Number of distinct clusters")

sys.path.insert(0, ".")

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import KernelPCA
    from sklearn.manifold import MDS
    from sklearn.metrics import adjusted_rand_score, silhouette_score
except ImportError as e:
    print(f"ERROR: sklearn not available: {e}")
    print("Install with: pip install scikit-learn")
    sys.exit(1)

from jcce.gbs.gbs_utils import (
    _normalize_spectral_radius,
    dequantized_features,
    encode_dag_to_gbs,
    frobenius_distance,
    jaccard_edge_distance,
    shd,
    spectral_distance,
)

# =============================================================================
# Robust GBS kernel computation (handles near-degenerate graphs)
# =============================================================================


def safe_dequantized_features(W_tilde: np.ndarray, n_mean=None, max_order: int = 3) -> np.ndarray:
    """
    Compute dequantized features with a fallback for near-degenerate matrices.

    If the singular values are too small for TheWalrus, adds a small uniform
    off-diagonal perturbation and re-normalizes. This preserves the structure
    while making the matrix non-degenerate.
    """
    try:
        return dequantized_features(W_tilde, n_mean=n_mean, max_order=max_order)
    except ValueError:
        # Add small perturbation to make the matrix non-degenerate
        d = W_tilde.shape[0]
        eps = 0.01
        W_perturbed = W_tilde + eps * (np.ones((d, d)) - np.eye(d))
        W_perturbed = _normalize_spectral_radius(W_perturbed, scale=0.9)
        return dequantized_features(W_perturbed, n_mean=n_mean, max_order=max_order)


def robust_dequantized_kernel_matrix(
    W_list: list,
    n_mean=None,
    max_order: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute pairwise dequantized GBS kernel matrix with robustness to
    near-degenerate matrices.
    """
    n = len(W_list)
    features = []
    for idx, W in enumerate(W_list):
        f = safe_dequantized_features(W, n_mean=n_mean, max_order=max_order)
        features.append(f)
        if (idx + 1) % 50 == 0 or idx == n - 1:
            print(f"    Features computed: {idx + 1}/{n}")

    # Pad features to same length (in case perturbation changed dimensions — shouldn't happen)
    max_len = max(len(f) for f in features)
    for i in range(n):
        if len(features[i]) < max_len:
            features[i] = np.pad(features[i], (0, max_len - len(features[i])))

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


# =============================================================================
# DAG generation
# =============================================================================


def generate_erdos_renyi(
    d: int, p: float = 0.2, rng: np.random.Generator = None, min_edges: int = 3
) -> np.ndarray:
    """
    Generate a random DAG from the Erdos-Renyi model.

    Uses a random topological ordering and adds edges with probability p.
    The random ordering ensures diverse structures without the edge-loss
    problem of permuting + re-extracting upper triangle.
    """
    if rng is None:
        rng = np.random.default_rng()

    while True:
        # Random topological ordering
        order = rng.permutation(d)
        A = np.zeros((d, d))
        # For each pair in topological order, add edge with probability p
        for idx_i in range(d):
            for idx_j in range(idx_i + 1, d):
                if rng.random() < p:
                    src, dst = order[idx_i], order[idx_j]
                    A[src, dst] = rng.uniform(0.3, 1.5)
        if np.sum(np.abs(A) > 0) >= min_edges:
            return A


def generate_scale_free(d: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a DAG with scale-free (preferential attachment) structure.

    Start with a 2-node connected graph. For each new node, connect it to
    existing nodes with probability proportional to (degree + 1).
    Edges are directed from existing to new (preserves DAG property).
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((d, d))

    # Start: node 0 -> node 1
    A[0, 1] = rng.uniform(0.3, 1.5)

    for new_node in range(2, d):
        # Compute attachment probabilities for existing nodes
        existing = list(range(new_node))
        degrees = np.array(
            [np.sum(A[v, :] > 0) + np.sum(A[:, v] > 0) + 1 for v in existing], dtype=float
        )
        probs = degrees / degrees.sum()

        # Number of parents: at least 1, up to min(3, existing)
        n_parents = min(max(1, int(rng.poisson(1.5))), min(3, len(existing)))

        # Select parents without replacement
        parents = rng.choice(existing, size=n_parents, replace=False, p=probs)

        for parent in parents:
            A[parent, new_node] = rng.uniform(0.3, 1.5)

    return A


def generate_small_world(
    d: int, k: int = 4, p_rewire: float = 0.3, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate a DAG with small-world (Watts-Strogatz) structure.

    1. Create a ring lattice where each node connects to k/2 nearest
       neighbors on each side (directed: lower index -> higher index).
    2. Rewire each edge with probability p_rewire to a random target.
    3. Keep only upper-triangular entries to ensure DAG property.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((d, d))
    half_k = k // 2

    # Ring lattice: connect each node i to its k nearest neighbors
    for i in range(d):
        for offset in range(1, half_k + 1):
            j = (i + offset) % d
            # Directed edge from lower to higher index
            src, dst = min(i, j), max(i, j)
            A[src, dst] = rng.uniform(0.3, 1.5)

    # Rewire edges
    for i in range(d):
        for j in range(i + 1, d):
            if A[i, j] > 0 and rng.random() < p_rewire:
                # Remove this edge and add to a random target
                A[i, j] = 0.0
                # Pick a new target (different from i, not already connected)
                candidates = [c for c in range(i + 1, d) if c != j and A[i, c] == 0]
                if candidates:
                    new_j = rng.choice(candidates)
                    A[i, new_j] = rng.uniform(0.3, 1.5)

    return A


def generate_dag_families(d: int, n_per_family: int = 50, seed: int = 42):
    """
    Generate DAGs from 3 structural families.

    Returns:
        dags: list of (d, d) adjacency matrices
        labels: list of integer labels (0=ER, 1=SF, 2=SW)
        family_names: list of family name strings
    """
    rng = np.random.default_rng(seed)
    family_names = ["Erdos-Renyi", "Scale-Free", "Small-World"]

    dags = []
    labels = []

    # Family 0: Erdos-Renyi
    for _ in range(n_per_family):
        dags.append(generate_erdos_renyi(d, p=0.2, rng=rng))
        labels.append(0)

    # Family 1: Scale-free
    for _ in range(n_per_family):
        dags.append(generate_scale_free(d, rng=rng))
        labels.append(1)

    # Family 2: Small-world
    for _ in range(n_per_family):
        dags.append(generate_small_world(d, k=4, p_rewire=0.3, rng=rng))
        labels.append(2)

    return dags, np.array(labels), family_names


# =============================================================================
# Pairwise distance matrices for classical methods
# =============================================================================


def compute_distance_matrix(dags, metric_fn):
    """Compute pairwise distance matrix using a given metric function."""
    n = len(dags)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_val = metric_fn(dags[i], dags[j])
            D[i, j] = d_val
            D[j, i] = d_val
    return D


def distance_to_kernel(D: np.ndarray) -> np.ndarray:
    """
    Convert distance matrix to kernel matrix using Gaussian (RBF) transformation.
    K = exp(-D / median(D)) where median is over non-zero distances.
    """
    # Use median of non-zero distances as bandwidth
    upper = D[np.triu_indices_from(D, k=1)]
    if len(upper) == 0 or np.all(upper == 0):
        return np.ones_like(D)
    median_d = np.median(upper[upper > 0])
    if median_d < 1e-10:
        median_d = 1.0
    K = np.exp(-D / median_d)
    return K


# =============================================================================
# Evaluation: embed + cluster + score
# =============================================================================


def evaluate_kernel_method(
    K: np.ndarray, labels: np.ndarray, n_clusters: int = 3, n_components: int = 2, seed: int = 42
):
    """
    Evaluate a kernel matrix: kernel PCA embedding -> KMeans -> ARI + silhouette.

    Returns dict with ARI, silhouette, and the embedding.
    """
    # Ensure kernel matrix is valid for KernelPCA
    # Add small regularization to diagonal for numerical stability
    K_reg = K.copy()
    K_reg += np.eye(K.shape[0]) * 1e-8

    try:
        kpca = KernelPCA(
            n_components=n_components,
            kernel="precomputed",
            random_state=seed,
        )
        embedding = kpca.fit_transform(K_reg)
    except Exception as e:
        print(f"    KernelPCA failed ({e}), falling back to MDS on distance from kernel")
        # Convert kernel to distance: D = sqrt(K_ii + K_jj - 2*K_ij)
        diag = np.diag(K_reg)
        D_from_K = np.sqrt(np.maximum(diag[:, None] + diag[None, :] - 2 * K_reg, 0))
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=seed,
            normalized_stress="auto",
        )
        embedding = mds.fit_transform(D_from_K)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    pred_labels = kmeans.fit_predict(embedding)

    ari = adjusted_rand_score(labels, pred_labels)

    # Silhouette needs at least 2 distinct labels
    n_unique = len(set(pred_labels))
    if n_unique < 2 or n_unique >= len(labels):
        sil = 0.0
    else:
        sil = silhouette_score(embedding, pred_labels)

    return {"ari": ari, "silhouette": sil, "embedding": embedding}


def evaluate_distance_method(
    D: np.ndarray, labels: np.ndarray, n_clusters: int = 3, n_components: int = 2, seed: int = 42
):
    """
    Evaluate a distance matrix: MDS embedding -> KMeans -> ARI + silhouette.
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
    )
    embedding = mds.fit_transform(D)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    pred_labels = kmeans.fit_predict(embedding)

    ari = adjusted_rand_score(labels, pred_labels)

    n_unique = len(set(pred_labels))
    if n_unique < 2 or n_unique >= len(labels):
        sil = 0.0
    else:
        sil = silhouette_score(embedding, pred_labels)

    return {"ari": ari, "silhouette": sil, "embedding": embedding}


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(d: int, n_per_family: int = 50, seed: int = 42):
    """Run the full discriminative power experiment for a given dimension."""
    print(f"\n{'=' * 70}")
    print(f"  Experiment 1: Kernel Discriminative Power  (d={d})")
    print(f"{'=' * 70}")

    # --- Generate DAGs ---
    print(f"\nGenerating {n_per_family} DAGs x 3 families (d={d})...")
    t0 = time.time()
    dags, labels, family_names = generate_dag_families(d, n_per_family, seed)
    t_gen = time.time() - t0
    n_total = len(dags)
    print(f"  Generated {n_total} DAGs in {t_gen:.2f}s")

    # Print edge density stats per family
    for fam_idx, fam_name in enumerate(family_names):
        fam_dags = [dags[i] for i in range(n_total) if labels[i] == fam_idx]
        densities = [np.sum(np.abs(dag) > 0.1) / (d * (d - 1)) for dag in fam_dags]
        print(
            f"  {fam_name}: avg edge density = {np.mean(densities):.3f} "
            f"(+/- {np.std(densities):.3f})"
        )

    # --- Encode DAGs for GBS ---
    print("\nEncoding DAGs to GBS graph matrices...")
    W_list = [encode_dag_to_gbs(dag, scale=0.9) for dag in dags]

    # --- Method evaluations ---
    results = {}

    # 1. Dequantized GBS kernel (order 2)
    print("\n[1/6] Dequantized GBS kernel (order 2)...")
    t0 = time.time()
    K_gbs2 = robust_dequantized_kernel_matrix(W_list, max_order=2, normalize=True)
    t_gbs2 = time.time() - t0
    print(f"  Kernel matrix computed in {t_gbs2:.2f}s")
    res = evaluate_kernel_method(K_gbs2, labels, seed=seed)
    results["GBS-deq (order 2)"] = {**res, "time": t_gbs2}
    print(f"  ARI = {res['ari']:.4f}, Silhouette = {res['silhouette']:.4f}")

    # 2. Dequantized GBS kernel (order 3)
    # For d=30, order 3 has C(30,3) = 4060 features — still tractable
    print("\n[2/6] Dequantized GBS kernel (order 3)...")
    t0 = time.time()
    K_gbs3 = robust_dequantized_kernel_matrix(W_list, max_order=3, normalize=True)
    t_gbs3 = time.time() - t0
    print(f"  Kernel matrix computed in {t_gbs3:.2f}s")
    res = evaluate_kernel_method(K_gbs3, labels, seed=seed)
    results["GBS-deq (order 3)"] = {**res, "time": t_gbs3}
    print(f"  ARI = {res['ari']:.4f}, Silhouette = {res['silhouette']:.4f}")

    # 3. SHD distance
    print("\n[3/6] SHD (Structural Hamming Distance)...")
    t0 = time.time()
    D_shd = compute_distance_matrix(dags, lambda a, b: float(shd(a, b)))
    t_shd = time.time() - t0
    print(f"  Distance matrix computed in {t_shd:.2f}s")
    res = evaluate_distance_method(D_shd, labels, seed=seed)
    results["SHD"] = {**res, "time": t_shd}
    print(f"  ARI = {res['ari']:.4f}, Silhouette = {res['silhouette']:.4f}")

    # 4. Frobenius norm
    print("\n[4/6] Frobenius norm distance...")
    t0 = time.time()
    D_frob = compute_distance_matrix(dags, frobenius_distance)
    t_frob = time.time() - t0
    print(f"  Distance matrix computed in {t_frob:.2f}s")
    res = evaluate_distance_method(D_frob, labels, seed=seed)
    results["Frobenius"] = {**res, "time": t_frob}
    print(f"  ARI = {res['ari']:.4f}, Silhouette = {res['silhouette']:.4f}")

    # 5. Spectral distance
    print("\n[5/6] Spectral distance...")
    t0 = time.time()
    D_spec = compute_distance_matrix(dags, spectral_distance)
    t_spec = time.time() - t0
    print(f"  Distance matrix computed in {t_spec:.2f}s")
    res = evaluate_distance_method(D_spec, labels, seed=seed)
    results["Spectral"] = {**res, "time": t_spec}
    print(f"  ARI = {res['ari']:.4f}, Silhouette = {res['silhouette']:.4f}")

    # 6. Jaccard edge distance
    print("\n[6/6] Jaccard edge distance...")
    t0 = time.time()
    D_jacc = compute_distance_matrix(dags, jaccard_edge_distance)
    t_jacc = time.time() - t0
    print(f"  Distance matrix computed in {t_jacc:.2f}s")
    res = evaluate_distance_method(D_jacc, labels, seed=seed)
    results["Jaccard"] = {**res, "time": t_jacc}
    print(f"  ARI = {res['ari']:.4f}, Silhouette = {res['silhouette']:.4f}")

    return results


def print_summary_table(all_results: dict):
    """Print a formatted summary table across dimensions."""
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Kernel Discriminative Power")
    print(f"{'=' * 70}")

    method_names = list(next(iter(all_results.values())).keys())
    dims = sorted(all_results.keys())

    # Header
    header = f"{'Method':<22}"
    for d in dims:
        header += f" | {'ARI':>7} {'Sil':>7} {'Time':>8}  "
    print(f"\n{header}")
    print(f"{'':22}", end="")
    for d in dims:
        print(f" | {'d=' + str(d):^25}", end="")
    print()
    print("-" * (22 + len(dims) * 29))

    # Rows
    for method in method_names:
        row = f"{method:<22}"
        for d in dims:
            r = all_results[d][method]
            row += f" | {r['ari']:>7.4f} {r['silhouette']:>7.4f} {r['time']:>7.2f}s "
        print(row)

    print()

    # Ranking
    for d in dims:
        ranked = sorted(all_results[d].items(), key=lambda x: x[1]["ari"], reverse=True)
        print(f"  d={d} ranking by ARI:")
        for rank, (method, r) in enumerate(ranked, 1):
            print(f"    {rank}. {method}: ARI={r['ari']:.4f}")


def main():
    print("GBS Experiment 1: Kernel Discriminative Power")
    print("=" * 50)
    print("Comparing dequantized GBS kernel vs classical distance metrics")
    print("for discriminating 3 DAG families (ER, Scale-Free, Small-World)")

    all_results = {}

    for d in [11, 30]:
        results = run_experiment(d, n_per_family=50, seed=42)
        all_results[d] = results

    print_summary_table(all_results)

    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    main()
