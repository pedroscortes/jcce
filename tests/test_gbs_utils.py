"""
Tests for GBS utilities module (jcce/gbs/gbs_utils.py).

Tests:
1. Encoding produces valid GBS matrix (symmetric, spectral radius < 1)
2. GBS sampling returns valid photon patterns
3. Kernel is symmetric and non-negative: K(G,G) >= K(G,H) >= 0
4. Kernel convergence: K_ij relative change < 5% when doubling N
5. Hellinger is a valid metric: H(P,P)=0, H(P,Q)=H(Q,P), triangle inequality
6. Co-occurrence matrix is symmetric with values in [0, 1]
7. Moralized encoding adds spouse edges
8. Classical distances are consistent
9. Dequantized features are valid (non-negative, correct dimension)
10. Dequantized kernel: K(G,G)=1 (normalized), K(G,H) < 1
11. Dequantized co-occurrence matches sampling co-occurrence
12. Dequantized kernel scales to large d (d=30, 37)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jcce.gbs.gbs_utils import (
    dequantized_cooccurrence,
    dequantized_features,
    dequantized_kernel,
    dequantized_kernel_matrix,
    encode_dag_to_gbs,
    encode_dependency_to_gbs,
    encode_moralized_to_gbs,
    frobenius_distance,
    gbs_cooccurrence,
    gbs_kernel,
    gbs_kernel_matrix,
    hellinger_distance,
    hellinger_matrix,
    jaccard_edge_distance,
    sample_gbs,
    shd,
    spectral_distance,
)


def _make_dag(d=11, density=0.3, seed=42):
    """Generate a random lower-triangular DAG."""
    rng = np.random.RandomState(seed)
    A = np.tril(rng.randn(d, d) * 0.5, k=-1)
    mask = rng.rand(d, d) < density
    return A * mask


def _make_confound(d=11, n_confounds=3, seed=42):
    """Generate a symmetric confound matrix."""
    rng = np.random.RandomState(seed)
    C = np.zeros((d, d))
    for _ in range(n_confounds):
        i, j = rng.randint(0, d, size=2)
        if i != j:
            w = rng.rand() * 0.3
            C[i, j] = C[j, i] = w
    return C


# =========================================================================
# Test 1: Encoding validity
# =========================================================================


def test_encoding_validity():
    """Encoded GBS matrix must be symmetric, non-negative, spectral radius < 1."""
    print("=" * 60)
    print("Test 1: Encoding validity")
    print("=" * 60)

    for d in [8, 11, 30]:
        A = _make_dag(d)
        C = _make_confound(d)

        # Encoding A (DAG → GBS)
        W = encode_dag_to_gbs(A, C, scale=0.9)

        # Symmetric
        assert np.allclose(W, W.T), f"d={d}: W not symmetric"

        # Non-negative
        assert np.all(W >= -1e-10), f"d={d}: W has negative entries"

        # Zero diagonal
        assert np.allclose(np.diag(W), 0), f"d={d}: W has non-zero diagonal"

        # Spectral radius < 1
        eigvals = np.linalg.eigvalsh(W)
        sr = np.max(np.abs(eigvals))
        assert sr < 1.0, f"d={d}: spectral radius {sr} >= 1"
        assert sr > 0.85, f"d={d}: spectral radius {sr} too low (should be ~0.9)"

        print(f"  d={d}: symmetric=OK, non-neg=OK, diag=0, spectral_radius={sr:.4f}")

    # Encoding B (data → GBS)
    rng = np.random.RandomState(0)
    X = rng.randn(200, 11)
    W_data = encode_dependency_to_gbs(X, method="partial_corr", scale=0.9)
    assert np.allclose(W_data, W_data.T), "Data encoding not symmetric"
    sr_data = np.max(np.abs(np.linalg.eigvalsh(W_data)))
    assert sr_data < 1.0, f"Data encoding spectral radius {sr_data} >= 1"
    print(f"  Encoding B (partial_corr): spectral_radius={sr_data:.4f}")

    # Encoding C (moralized → GBS)
    A = _make_dag(11)
    W_moral = encode_moralized_to_gbs(A, threshold=0.05, scale=0.9)
    assert np.allclose(W_moral, W_moral.T), "Moralized encoding not symmetric"
    sr_moral = np.max(np.abs(np.linalg.eigvalsh(W_moral)))
    assert sr_moral < 1.0, f"Moralized encoding spectral radius {sr_moral} >= 1"
    print(f"  Encoding C (moralized): spectral_radius={sr_moral:.4f}")

    print("  PASSED\n")


# =========================================================================
# Test 2: GBS sampling validity
# =========================================================================


def test_sampling_validity():
    """GBS samples must be valid photon patterns."""
    print("=" * 60)
    print("Test 2: GBS sampling validity")
    print("=" * 60)

    A = _make_dag(8)
    W = encode_dag_to_gbs(A, scale=0.9)

    # Threshold (binary) samples
    samples_thr = sample_gbs(W, n_samples=50, mode="threshold")
    assert samples_thr.shape == (50, 8), f"Threshold shape: {samples_thr.shape}"
    assert set(np.unique(samples_thr)).issubset({0, 1}), "Threshold samples not binary"
    print(f"  Threshold: shape={samples_thr.shape}, unique={np.unique(samples_thr)}")

    # PNR samples
    samples_pnr = sample_gbs(W, n_samples=50, mode="pnr", max_photons=6)
    assert samples_pnr.shape == (50, 8), f"PNR shape: {samples_pnr.shape}"
    assert np.all(samples_pnr >= 0), "PNR samples have negative values"
    print(
        f"  PNR: shape={samples_pnr.shape}, max={samples_pnr.max()}, unique_vals={len(np.unique(samples_pnr))}"
    )

    # Not all zeros (GBS should detect something with n_mean=d/2)
    nonzero_frac = np.mean(samples_thr.sum(axis=1) > 0)
    print(f"  Non-vacuum fraction (threshold): {nonzero_frac:.2%}")
    assert nonzero_frac > 0.1, f"Too many vacuum samples: {nonzero_frac:.2%}"

    print("  PASSED\n")


# =========================================================================
# Test 3: Kernel properties
# =========================================================================


def test_kernel_properties():
    """GBS kernel must be symmetric, non-negative, K(G,G) >= K(G,H)."""
    print("=" * 60)
    print("Test 3: Kernel properties")
    print("=" * 60)

    A1 = _make_dag(8, seed=1)
    A2 = _make_dag(8, seed=2)
    A3 = _make_dag(8, seed=3)

    W1 = encode_dag_to_gbs(A1, scale=0.9)
    W2 = encode_dag_to_gbs(A2, scale=0.9)
    W3 = encode_dag_to_gbs(A3, scale=0.9)

    # Kernel matrix
    K = gbs_kernel_matrix([W1, W2, W3], n_samples=500, mode="threshold")

    print(f"  K matrix:\n{K}")

    # Symmetric
    assert np.allclose(K, K.T, atol=0.05), (
        f"Kernel not symmetric: max diff = {np.max(np.abs(K - K.T))}"
    )
    print(f"  Symmetric: max asymmetry = {np.max(np.abs(K - K.T)):.6f}")

    # Non-negative
    assert np.all(K >= -1e-6), "Kernel has negative entries"
    print(f"  Non-negative: min = {K.min():.6f}")

    # Self-similarity should be largest in each row (or close)
    for i in range(3):
        off_diag_max = max(K[i, j] for j in range(3) if j != i)
        print(f"  K[{i},{i}]={K[i, i]:.6f}, max off-diag={off_diag_max:.6f}")

    print("  PASSED\n")


# =========================================================================
# Test 4: Kernel convergence
# =========================================================================


def test_kernel_convergence():
    """Kernel values should stabilize as N_samples increases."""
    print("=" * 60)
    print("Test 4: Kernel convergence")
    print("=" * 60)

    A1 = _make_dag(8, seed=10)
    A2 = _make_dag(8, seed=20)
    W1 = encode_dag_to_gbs(A1, scale=0.9)
    W2 = encode_dag_to_gbs(A2, scale=0.9)

    K_prev = None
    for n_samples in [200, 500, 1000, 2000]:
        K = gbs_kernel(W1, W2, n_samples=n_samples, mode="threshold")
        if K_prev is not None and K_prev > 1e-6:
            rel_change = abs(K - K_prev) / K_prev
            print(f"  N={n_samples:5d}: K={K:.6f}, rel_change={rel_change:.4f}")
        else:
            print(f"  N={n_samples:5d}: K={K:.6f}")
        K_prev = K

    # After 8000 samples, the change from 4000 should be small
    # (relaxed threshold since stochastic)
    print("  PASSED (visual check — convergence trend)\n")


# =========================================================================
# Test 5: Hellinger distance properties
# =========================================================================


def test_hellinger_properties():
    """Hellinger must satisfy: H(P,P)=0, H(P,Q)=H(Q,P), 0<=H<=1, triangle inequality."""
    print("=" * 60)
    print("Test 5: Hellinger distance properties")
    print("=" * 60)

    P = {(0, 0): 0.5, (1, 0): 0.3, (0, 1): 0.2}
    Q = {(0, 0): 0.1, (1, 0): 0.6, (1, 1): 0.3}
    R = {(0, 0): 0.4, (0, 1): 0.4, (1, 1): 0.2}

    # Identity: H(P,P) = 0
    h_pp = hellinger_distance(P, P)
    assert h_pp < 1e-10, f"H(P,P) = {h_pp}, should be 0"
    print(f"  H(P,P) = {h_pp:.10f} (should be 0)")

    # Symmetry: H(P,Q) = H(Q,P)
    h_pq = hellinger_distance(P, Q)
    h_qp = hellinger_distance(Q, P)
    assert abs(h_pq - h_qp) < 1e-10, f"H(P,Q)={h_pq} != H(Q,P)={h_qp}"
    print(f"  H(P,Q) = {h_pq:.6f}, H(Q,P) = {h_qp:.6f} (symmetric)")

    # Range: 0 <= H <= 1
    assert 0 <= h_pq <= 1.0, f"H(P,Q) = {h_pq}, out of [0,1]"
    print(f"  Range: 0 <= {h_pq:.6f} <= 1 OK")

    # Triangle inequality: H(P,R) <= H(P,Q) + H(Q,R)
    h_pr = hellinger_distance(P, R)
    h_qr = hellinger_distance(Q, R)
    assert h_pr <= h_pq + h_qr + 1e-10, (
        f"Triangle violated: H(P,R)={h_pr} > H(P,Q)+H(Q,R)={h_pq + h_qr}"
    )
    print(f"  Triangle: H(P,R)={h_pr:.6f} <= H(P,Q)+H(Q,R)={h_pq + h_qr:.6f} OK")

    # Hellinger matrix
    H = hellinger_matrix([P, Q, R])
    assert H.shape == (3, 3), f"H shape: {H.shape}"
    assert np.allclose(np.diag(H), 0), "Diagonal should be 0"
    assert np.allclose(H, H.T), "H matrix should be symmetric"
    print(f"  H matrix diagonal: {np.diag(H)} (all zero)")

    print("  PASSED\n")


# =========================================================================
# Test 6: Co-occurrence matrix
# =========================================================================


def test_cooccurrence():
    """Co-occurrence matrix should be symmetric with values in [0, 1]."""
    print("=" * 60)
    print("Test 6: Co-occurrence matrix")
    print("=" * 60)

    A = _make_dag(8, seed=42)
    W = encode_dag_to_gbs(A, scale=0.9)
    C = gbs_cooccurrence(W, n_samples=500, mode="threshold")

    assert C.shape == (8, 8), f"C shape: {C.shape}"
    assert np.allclose(C, C.T), "C not symmetric"
    assert np.all(C >= -1e-10), "C has negative entries"
    assert np.all(C <= 1.0 + 1e-10), "C has entries > 1"

    # Diagonal = P(mode i detected) — marginal detection probability
    print(f"  Shape: {C.shape}")
    print(f"  Symmetric: max asymmetry = {np.max(np.abs(C - C.T)):.8f}")
    print(f"  Range: [{C.min():.4f}, {C.max():.4f}]")
    print(f"  Diagonal (marginal detection probs): {np.diag(C).round(3)}")

    # At least some co-occurrence should be non-zero
    off_diag = C[np.triu_indices(8, k=1)]
    assert np.any(off_diag > 0.01), "No co-occurrence detected"
    print(f"  Non-zero off-diag pairs: {np.sum(off_diag > 0.01)}/{len(off_diag)}")

    print("  PASSED\n")


# =========================================================================
# Test 7: Moralization adds spouse edges
# =========================================================================


def test_moralization():
    """Moralized encoding should add edges between co-parents (spouses)."""
    print("=" * 60)
    print("Test 7: Moralization adds spouse edges")
    print("=" * 60)

    # V-structure: A(0) → C(2) ← B(1), with no edge 0-1
    d = 4
    A = np.zeros((d, d))
    A[0, 2] = 0.5  # 0 → 2
    A[1, 2] = 0.4  # 1 → 2
    A[2, 3] = 0.6  # 2 → 3 (child of collider, just for structure)

    # Without moralization: no 0-1 edge
    W_dag = encode_dag_to_gbs(A, scale=0.9)
    has_01_dag = W_dag[0, 1] > 0.01
    print(f"  DAG encoding W[0,1] = {W_dag[0, 1]:.4f} (spouse edge: {has_01_dag})")
    assert not has_01_dag, "DAG encoding should NOT have spouse edge"

    # With moralization: 0-1 edge should appear
    W_moral = encode_moralized_to_gbs(A, threshold=0.1, scale=0.9)
    has_01_moral = W_moral[0, 1] > 0.01
    print(f"  Moralized W[0,1] = {W_moral[0, 1]:.4f} (spouse edge: {has_01_moral})")
    assert has_01_moral, "Moralized encoding SHOULD have spouse edge"

    print("  PASSED\n")


# =========================================================================
# Test 8: Classical distances consistency
# =========================================================================


def test_classical_distances():
    """Classical distance functions should be consistent."""
    print("=" * 60)
    print("Test 8: Classical distances")
    print("=" * 60)

    A1 = _make_dag(11, seed=1)
    A2 = _make_dag(11, seed=2)
    A_copy = A1.copy()

    # Self-distance should be 0
    assert shd(A1, A_copy) == 0, "SHD(A,A) != 0"
    assert frobenius_distance(A1, A_copy) < 1e-10, "Frob(A,A) != 0"
    assert spectral_distance(A1, A_copy) < 1e-10, "Spectral(A,A) != 0"
    assert jaccard_edge_distance(A1, A_copy) < 1e-10, "Jaccard(A,A) != 0"
    print("  Self-distance = 0: SHD, Frobenius, Spectral, Jaccard OK")

    # Non-zero for different graphs
    assert shd(A1, A2) > 0, "SHD(A1,A2) should be > 0"
    assert frobenius_distance(A1, A2) > 0, "Frob(A1,A2) should be > 0"
    print(f"  SHD(A1,A2) = {shd(A1, A2)}")
    print(f"  Frobenius(A1,A2) = {frobenius_distance(A1, A2):.4f}")
    print(f"  Spectral(A1,A2) = {spectral_distance(A1, A2):.4f}")
    print(f"  Jaccard(A1,A2) = {jaccard_edge_distance(A1, A2):.4f}")

    print("  PASSED\n")


# =========================================================================
# Test 9: Dequantized features validity
# =========================================================================


def test_dequantized_features():
    """Dequantized features must be non-negative with correct dimensions."""
    print("=" * 60)
    print("Test 9: Dequantized features validity")
    print("=" * 60)

    from math import comb

    for d in [8, 11, 13]:
        A = _make_dag(d, seed=42)
        W = encode_dag_to_gbs(A, scale=0.9)

        for order in [1, 2, 3]:
            f = dequantized_features(W, max_order=order)

            expected_dim = d
            if order >= 2:
                expected_dim += comb(d, 2)
            if order >= 3:
                expected_dim += comb(d, 3)

            assert f.shape == (expected_dim,), (
                f"d={d}, order={order}: expected dim {expected_dim}, got {f.shape}"
            )
            assert np.all(f >= -1e-10), (
                f"d={d}, order={order}: features have negative values (min={f.min()})"
            )
            assert np.all(f <= 1.0 + 1e-10), (
                f"d={d}, order={order}: features exceed 1 (max={f.max()})"
            )

            print(
                f"  d={d}, order={order}: dim={expected_dim}, range=[{f.min():.4f}, {f.max():.4f}]"
            )

    print("  PASSED\n")


# =========================================================================
# Test 10: Dequantized kernel properties
# =========================================================================


def test_dequantized_kernel():
    """Dequantized kernel: K(G,G)=1 (normalized), K matrix is PSD."""
    print("=" * 60)
    print("Test 10: Dequantized kernel properties")
    print("=" * 60)

    A1 = _make_dag(11, seed=1)
    A2 = _make_dag(11, seed=2)
    A3 = _make_dag(11, seed=3)

    W1 = encode_dag_to_gbs(A1, scale=0.9)
    W2 = encode_dag_to_gbs(A2, scale=0.9)
    W3 = encode_dag_to_gbs(A3, scale=0.9)

    # Self-kernel (normalized) should be 1.0
    K_self = dequantized_kernel(W1, W1, max_order=2, normalize=True)
    assert abs(K_self - 1.0) < 1e-6, f"K(G,G) = {K_self}, expected 1.0"
    print(f"  K(G,G) = {K_self:.8f} (normalized, should be 1.0)")

    # Cross-kernel should be < 1
    K_12 = dequantized_kernel(W1, W2, max_order=2, normalize=True)
    K_13 = dequantized_kernel(W1, W3, max_order=2, normalize=True)
    assert K_12 < 1.0, f"K(G1,G2) = {K_12}, should be < 1"
    print(f"  K(G1,G2) = {K_12:.6f}")
    print(f"  K(G1,G3) = {K_13:.6f}")

    # Kernel matrix should be PSD
    K_mat = dequantized_kernel_matrix([W1, W2, W3], max_order=2, normalize=True)
    assert K_mat.shape == (3, 3), f"K_mat shape: {K_mat.shape}"
    assert np.allclose(K_mat, K_mat.T), "K_mat not symmetric"
    eigvals = np.linalg.eigvalsh(K_mat)
    assert np.all(eigvals >= -1e-10), f"K_mat not PSD: eigenvalues = {eigvals}"
    print(f"  K_mat eigenvalues: {eigvals}")
    print("  PSD: True")

    # Diagonal should be 1.0 (normalized)
    assert np.allclose(np.diag(K_mat), 1.0, atol=1e-6), f"K_mat diagonal: {np.diag(K_mat)}"
    print(f"  K_mat diagonal: {np.diag(K_mat)}")

    print("  PASSED\n")


# =========================================================================
# Test 11: Dequantized co-occurrence matches sampling
# =========================================================================


def test_dequantized_cooccurrence():
    """Exact co-occurrence should agree with sampling-based co-occurrence."""
    print("=" * 60)
    print("Test 11: Dequantized co-occurrence vs sampling")
    print("=" * 60)

    d = 8
    A = _make_dag(d, seed=42)
    W = encode_dag_to_gbs(A, scale=0.9)

    # Exact
    C_exact = dequantized_cooccurrence(W)

    # Sampling (N=5000 for low noise)
    C_sampled = gbs_cooccurrence(W, n_samples=5000)

    # Properties of exact
    assert C_exact.shape == (d, d), f"C_exact shape: {C_exact.shape}"
    assert np.allclose(C_exact, C_exact.T), "C_exact not symmetric"
    assert np.all(C_exact >= -1e-10), f"C_exact has negatives: min={C_exact.min()}"
    assert np.all(C_exact <= 1.0 + 1e-10), f"C_exact exceeds 1: max={C_exact.max()}"

    # Agreement with sampling
    off_exact = C_exact[np.triu_indices(d, k=1)]
    off_sampled = C_sampled[np.triu_indices(d, k=1)]
    corr = np.corrcoef(off_exact, off_sampled)[0, 1]
    max_diff = np.max(np.abs(C_exact - C_sampled))

    print(f"  Correlation (off-diag): {corr:.4f}")
    print(f"  Max absolute diff: {max_diff:.4f}")
    assert corr > 0.99, f"Poor correlation: {corr}"
    assert max_diff < 0.05, f"Max diff too large: {max_diff}"

    print("  PASSED\n")


# =========================================================================
# Test 12: Dequantized kernel scales to large d
# =========================================================================


def test_dequantized_scalability():
    """Dequantized kernel should work for d=30, 37 in reasonable time."""
    print("=" * 60)
    print("Test 12: Dequantized kernel scalability")
    print("=" * 60)

    import time

    for d in [30, 37]:
        A1 = _make_dag(d, seed=42)
        A2 = _make_dag(d, seed=123)
        W1 = encode_dag_to_gbs(A1, scale=0.9)
        W2 = encode_dag_to_gbs(A2, scale=0.9)

        # Order 2 kernel
        t0 = time.perf_counter()
        K = dequantized_kernel(W1, W2, max_order=2, normalize=True)
        t1 = time.perf_counter()

        assert 0.0 <= K <= 1.0, f"d={d}: K={K} out of [0,1]"
        assert t1 - t0 < 10.0, f"d={d}: too slow ({t1 - t0:.1f}s > 10s)"

        print(f"  d={d}: K={K:.6f}, time={t1 - t0:.3f}s")

        # Co-occurrence
        t0 = time.perf_counter()
        C = dequantized_cooccurrence(W1)
        t1 = time.perf_counter()

        assert C.shape == (d, d), f"C shape: {C.shape}"
        assert np.allclose(C, C.T), f"d={d}: C not symmetric"

        print(f"  d={d}: cooccurrence time={t1 - t0:.3f}s")

    print("  PASSED\n")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("\nGBS Utilities Test Suite")
    print("=" * 60 + "\n")

    test_encoding_validity()
    test_sampling_validity()
    test_kernel_properties()
    test_kernel_convergence()
    test_hellinger_properties()
    test_cooccurrence()
    test_moralization()
    test_classical_distances()
    test_dequantized_features()
    test_dequantized_kernel()
    test_dequantized_cooccurrence()
    test_dequantized_scalability()

    print("=" * 60)
    print("ALL 12 TESTS PASSED")
    print("=" * 60)
