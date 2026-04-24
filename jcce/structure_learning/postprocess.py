"""
Post-processing for continuous-weight adjacency matrices.

Continuous structure learning algorithms (NOTEARS, DAGMA, GOLEM) return
adjacency matrices with real-valued weights. These must be thresholded
to obtain binary DAGs suitable for Markov Blanket extraction and
downstream causal inference.

Problem:
    learn_with_dagma(X, key) returns A with values like:
        [0.000, 0.003, -0.001, 0.872, 0.421, 0.007, ...]
    The "real" edges (0.872, 0.421) are mixed with noise (0.003, -0.001).
    Without thresholding, extract_markov_blanket(A, target) with its
    default threshold=1e-6 will treat EVERY entry as an edge.

Solution:
    A_raw = learn_with_dagma(X, key)
    A_dag = postprocess_dag(A_raw)
    mb = extract_markov_blanket(A_dag, target)

Methods:
    - 'fixed':      |A[i,j]| > threshold
    - 'percentile': keep top k% of edges by magnitude
    - 'adaptive':   find natural gap in sorted weight distribution (recommended)
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np


def postprocess_dag(
    A: jnp.ndarray,
    method: str = "adaptive",
    threshold: float = 0.3,
    keep_fraction: float = 0.3,
    ensure_dag: bool = True,
    verbose: bool = False,
    protected_idx: Optional[int] = None,
) -> jnp.ndarray:
    """
    Threshold continuous adjacency matrix into a binary DAG.

    Args:
        A: (d, d) continuous-weight adjacency matrix from NOTEARS/DAGMA/GOLEM.
        method: Thresholding strategy:
            'fixed'      - keep edges with |A[i,j]| > threshold
            'percentile' - keep top (keep_fraction * 100)% of edges by magnitude
            'adaptive'   - find natural gap in weight distribution (recommended)
        threshold: Absolute cutoff for 'fixed' method (default: 0.3).
        keep_fraction: Fraction of candidate edges to keep for 'percentile'
                       method. 0.3 = keep top 30% (default: 0.3).
        ensure_dag: If True, remove any remaining cycles after thresholding
                    by iteratively dropping the weakest back-edge.
        verbose: Print diagnostics.
        protected_idx: If set, edges to/from this variable (e.g., Y) are kept
                       at a lower threshold (1e-3) to prevent empty Markov
                       blankets from aggressive thresholding.

    Returns:
        A_binary: (d, d) binary adjacency matrix {0, 1}.
                  Guaranteed acyclic if ensure_dag=True.
    """
    A_np = np.array(A, dtype=np.float64)
    d = A_np.shape[0]

    # --- Step 1: Zero diagonal (no self-loops) ---
    np.fill_diagonal(A_np, 0.0)

    # --- Step 2: Collect off-diagonal absolute weights ---
    weights = np.abs(A_np[A_np != 0])  # non-zero off-diagonal entries

    if len(weights) == 0:
        if verbose:
            print("postprocess_dag: adjacency is all zeros, returning empty DAG")
        return jnp.zeros((d, d), dtype=jnp.float32)

    # --- Step 3: Compute threshold based on method ---
    if method == "fixed":
        cutoff = threshold

    elif method == "percentile":
        # keep_fraction=0.3 means keep top 30% => threshold at 70th percentile
        all_abs = np.abs(A_np).flatten()
        # only consider non-zero entries for percentile computation
        nonzero_abs = all_abs[all_abs > 0]
        if len(nonzero_abs) == 0:
            cutoff = 0.0
        else:
            cutoff = float(np.percentile(nonzero_abs, (1.0 - keep_fraction) * 100))

    elif method == "adaptive":
        cutoff = _adaptive_threshold(A_np, verbose=verbose)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'fixed', 'percentile', or 'adaptive'.")

    if verbose:
        print(f"postprocess_dag: method='{method}', cutoff={cutoff:.6f}")
        print(
            f"  weight stats: min={weights.min():.6f}, "
            f"median={np.median(weights):.6f}, max={weights.max():.6f}, "
            f"n_nonzero={len(weights)}"
        )

    # --- Step 4: Apply threshold ---
    A_binary = (np.abs(A_np) > cutoff).astype(np.float64)
    np.fill_diagonal(A_binary, 0.0)

    # --- Step 4b: Protect edges to/from target variable ---
    # Continuous methods (DAGMA, NOTEARS, GOLEM) often produce weaker edges
    # to Y than X→X edges, causing the adaptive threshold to drop all Y-edges
    # and produce an empty Markov blanket. Use a much lower threshold (1e-3)
    # for edges involving the protected variable.
    if protected_idx is not None:
        Y_PROTECT_THRESH = 1e-3
        for i in range(d):
            if i == protected_idx:
                continue
            # Edge i -> protected
            if np.abs(A_np[i, protected_idx]) > Y_PROTECT_THRESH:
                A_binary[i, protected_idx] = 1.0
            # Edge protected -> i
            if np.abs(A_np[protected_idx, i]) > Y_PROTECT_THRESH:
                A_binary[protected_idx, i] = 1.0
        n_protected = int(A_binary[protected_idx, :].sum() + A_binary[:, protected_idx].sum())
        if verbose:
            print(
                f"  Y-protection (idx={protected_idx}): "
                f"{n_protected} edges preserved (threshold={Y_PROTECT_THRESH})"
            )

    n_edges_before = int(A_binary.sum())

    # --- Step 5: Ensure DAG (remove cycles) ---
    if ensure_dag:
        # Keep original weights for tie-breaking during cycle removal
        A_weighted = A_binary * np.abs(A_np)
        protected_nodes = {protected_idx} if protected_idx is not None else None
        A_weighted = _remove_cycles(A_weighted, protected_nodes=protected_nodes, verbose=verbose)
        A_binary = (np.abs(A_weighted) > 0).astype(np.float64)

    n_edges_after = int(A_binary.sum())

    if verbose:
        print(f"  edges after threshold: {n_edges_before}")
        if ensure_dag and n_edges_before != n_edges_after:
            print(
                f"  edges after cycle removal: {n_edges_after} "
                f"(removed {n_edges_before - n_edges_after})"
            )
        print(
            f"  final DAG density: {n_edges_after}/{d * (d - 1)} "
            f"= {n_edges_after / max(d * (d - 1), 1):.3f}"
        )

    return jnp.array(A_binary, dtype=jnp.float32)


def diagnose_weights(A: jnp.ndarray) -> dict:
    """
    Analyze the weight distribution of a continuous adjacency matrix.

    Useful for understanding what threshold to pick before calling
    postprocess_dag().

    Args:
        A: (d, d) continuous-weight adjacency matrix.

    Returns:
        Dictionary with:
            'n_vars': number of variables
            'n_nonzero': number of non-zero off-diagonal entries
            'max_possible_edges': d*(d-1)
            'weights_min', 'weights_max', 'weights_mean', 'weights_median':
                statistics over |A[i,j]| for non-zero entries
            'weights_std': standard deviation
            'suggested_fixed': a reasonable fixed threshold (mean + 1 std)
            'adaptive_cutoff': the cutoff that adaptive method would pick
            'gap_ratio': ratio of largest gap to median weight
                         (>2 means clear bimodal separation)
    """
    A_np = np.array(A, dtype=np.float64)
    d = A_np.shape[0]
    np.fill_diagonal(A_np, 0.0)

    abs_weights = np.abs(A_np).flatten()
    nonzero = abs_weights[abs_weights > 1e-12]

    if len(nonzero) == 0:
        return {
            "n_vars": d,
            "n_nonzero": 0,
            "max_possible_edges": d * (d - 1),
            "weights_min": 0.0,
            "weights_max": 0.0,
            "weights_mean": 0.0,
            "weights_median": 0.0,
            "weights_std": 0.0,
            "suggested_fixed": 0.0,
            "adaptive_cutoff": 0.0,
            "gap_ratio": 0.0,
        }

    sorted_w = np.sort(nonzero)
    gaps = np.diff(sorted_w)
    max_gap_idx = int(np.argmax(gaps)) if len(gaps) > 0 else 0
    max_gap = float(gaps[max_gap_idx]) if len(gaps) > 0 else 0.0
    adaptive_cutoff = (
        float((sorted_w[max_gap_idx] + sorted_w[max_gap_idx + 1]) / 2) if len(gaps) > 0 else 0.0
    )

    median_w = float(np.median(nonzero))

    return {
        "n_vars": d,
        "n_nonzero": len(nonzero),
        "max_possible_edges": d * (d - 1),
        "weights_min": float(nonzero.min()),
        "weights_max": float(nonzero.max()),
        "weights_mean": float(nonzero.mean()),
        "weights_median": median_w,
        "weights_std": float(nonzero.std()),
        "suggested_fixed": float(nonzero.mean() + nonzero.std()),
        "adaptive_cutoff": adaptive_cutoff,
        "gap_ratio": max_gap / median_w if median_w > 1e-12 else 0.0,
    }


def print_diagnosis(A: jnp.ndarray):
    """Print a human-readable weight diagnosis."""
    info = diagnose_weights(A)
    print(f"Weight Diagnosis ({info['n_vars']} variables)")
    print(f"  Non-zero entries: {info['n_nonzero']} / {info['max_possible_edges']}")
    print(
        f"  |weights|: min={info['weights_min']:.6f}, "
        f"median={info['weights_median']:.6f}, "
        f"max={info['weights_max']:.6f}, "
        f"std={info['weights_std']:.6f}"
    )
    print(
        f"  Gap ratio: {info['gap_ratio']:.2f} "
        f"({'clear separation' if info['gap_ratio'] > 2 else 'no clear separation'})"
    )
    print("  Suggested thresholds:")
    print(f"    adaptive: {info['adaptive_cutoff']:.6f}")
    print(f"    fixed (mean+std): {info['suggested_fixed']:.6f}")


# =============================================================================
# Internal: Adaptive threshold via largest-gap detection
# =============================================================================


def _adaptive_threshold(A_np: np.ndarray, verbose: bool = False) -> float:
    """
    Find a natural threshold by detecting the largest gap in sorted weights.

    Rationale: continuous structure learners produce a bimodal distribution
    of edge weights — a cluster of near-zero "noise" values and a cluster
    of larger "real edge" values. The largest gap in the sorted absolute
    weights separates these two clusters.

    If no clear gap exists (gap_ratio < 1.5), falls back to median
    thresholding, which keeps roughly half the edges.

    Args:
        A_np: (d, d) numpy adjacency matrix.
        verbose: Print gap analysis.

    Returns:
        cutoff: Threshold value. Edges with |A[i,j]| > cutoff are kept.
    """
    abs_vals = np.abs(A_np).flatten()
    nonzero = abs_vals[abs_vals > 1e-12]

    if len(nonzero) <= 1:
        return 0.0

    sorted_w = np.sort(nonzero)
    gaps = np.diff(sorted_w)

    if len(gaps) == 0:
        return 0.0

    max_gap_idx = int(np.argmax(gaps))
    max_gap = gaps[max_gap_idx]
    median_gap = float(np.median(gaps))

    # Gap ratio: how much bigger is the max gap compared to typical gaps
    gap_ratio = max_gap / median_gap if median_gap > 1e-12 else 0.0

    # Threshold at the midpoint of the largest gap
    cutoff = (sorted_w[max_gap_idx] + sorted_w[max_gap_idx + 1]) / 2

    if verbose:
        n_below = max_gap_idx + 1
        n_above = len(sorted_w) - n_below
        print(
            f"  adaptive: largest gap = {max_gap:.6f} at position {max_gap_idx}/{len(sorted_w) - 1}"
        )
        print(f"  adaptive: gap_ratio = {gap_ratio:.2f} (median_gap={median_gap:.6f})")
        print(f"  adaptive: {n_below} weights below cutoff, {n_above} weights above")

    # If no clear separation, fall back to median
    if gap_ratio < 1.5:
        fallback = float(np.median(nonzero))
        if verbose:
            print(
                f"  adaptive: gap_ratio < 1.5, weak separation -> "
                f"falling back to median={fallback:.6f}"
            )
        return fallback

    return float(cutoff)


# =============================================================================
# Internal: Cycle removal via weakest-back-edge deletion
# =============================================================================


def _remove_cycles(
    A_weighted: np.ndarray,
    protected_nodes: Optional[set] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Remove cycles by iteratively deleting the weakest edge in a cycle.

    Uses Kahn's algorithm to detect cycle nodes, then removes the weakest
    edge among them. Repeats until the graph is acyclic.

    Preserves the maximum number of strong edges. Edges TO or FROM
    protected nodes are never removed (e.g., outcome/target nodes).

    Args:
        A_weighted: (d, d) numpy adjacency matrix with non-negative weights
                    (already thresholded, weights = original |A[i,j]| values).
        protected_nodes: Set of node indices whose edges must not be removed.
            Typically the target/outcome node. Edges TO or FROM these nodes
            are skipped during weakest-edge search.
        verbose: Print cycle removal steps.

    Returns:
        A_dag: (d, d) acyclic weighted adjacency matrix.
    """
    A = A_weighted.copy()
    d = A.shape[0]
    removed = []
    if protected_nodes is None:
        protected_nodes = set()

    for iteration in range(d * d):  # upper bound on iterations
        # --- Kahn's algorithm: try topological sort ---
        in_degree = (np.abs(A) > 0).astype(int).sum(axis=0)
        queue = [i for i in range(d) if in_degree[i] == 0]
        order = []
        temp_in = in_degree.copy()

        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(d):
                if abs(A[node, j]) > 0:
                    temp_in[j] -= 1
                    if temp_in[j] == 0:
                        queue.append(j)

        if len(order) == d:
            break  # acyclic

        # --- Nodes involved in cycles ---
        cycle_nodes = set(range(d)) - set(order)

        # --- Find weakest edge among cycle nodes, skipping protected ---
        weakest_weight = float("inf")
        weakest_edge = None

        for i in cycle_nodes:
            for j in cycle_nodes:
                # Never remove edges to/from protected nodes
                if i in protected_nodes or j in protected_nodes:
                    continue
                w = abs(A[i, j])
                if w > 0 and w < weakest_weight:
                    weakest_weight = w
                    weakest_edge = (i, j)

        if weakest_edge is None:
            # All remaining cycle edges involve protected nodes — stop
            break

        A[weakest_edge[0], weakest_edge[1]] = 0.0
        removed.append((weakest_edge, weakest_weight))

    if verbose and removed:
        print(f"  cycle removal: removed {len(removed)} edges:")
        for (i, j), w in removed:
            print(f"    {i} -> {j} (weight={w:.6f})")

    return A


# =============================================================================
# Convenience: learn + postprocess in one call
# =============================================================================


def learn_and_postprocess(
    X: jnp.ndarray,
    key,
    algorithm: str = "dagma",
    postprocess_method: str = "adaptive",
    threshold: float = 0.3,
    keep_fraction: float = 0.3,
    verbose: bool = True,
    protected_idx: Optional[int] = None,
    **learn_kwargs,
) -> jnp.ndarray:
    """
    Run a structure learning algorithm and postprocess into a binary DAG.

    Convenience wrapper that chains learn_with_*() + postprocess_dag().

    Args:
        X: (n_samples, n_vars) data matrix.
        key: JAX random key.
        algorithm: 'dagma', 'notears', 'golem', 'golem_nv',
                   'pc', 'ges', 'directlingam'.
        postprocess_method: 'adaptive', 'fixed', or 'percentile'.
                            Ignored for PC/GES/DirectLiNGAM (already binary).
        threshold: For 'fixed' method.
        keep_fraction: For 'percentile' method.
        verbose: Print progress.
        protected_idx: If set, edges to/from this variable (e.g., Y) are
                       preserved at a lower threshold during postprocessing.
        **learn_kwargs: Passed to the underlying learn_with_*() function.

    Returns:
        A_binary: (d, d) binary DAG adjacency matrix.
    """
    # Algorithms that return binary matrices (no postprocessing needed)
    # FCI returns a PAG (int matrix with marks 0-3), not a binary DAG.
    # It needs pag_to_dag() conversion, not thresholding.
    binary_algorithms = {"pc", "ges", "directlingam"}
    pag_algorithms = {"fci"}

    if algorithm == "dagma":
        from jcce.structure_learning.dagma import learn_with_dagma

        A_raw = learn_with_dagma(X, key, verbose=verbose, **learn_kwargs)

    elif algorithm == "notears":
        from jcce.structure_learning.notears import learn_with_notears

        A_raw = learn_with_notears(X, key, verbose=verbose, **learn_kwargs)

    elif algorithm in ("golem", "golem_ev"):
        from jcce.structure_learning.golem import learn_with_golem

        A_raw = learn_with_golem(X, key, variant="ev", verbose=verbose, **learn_kwargs)

    elif algorithm == "golem_nv":
        from jcce.structure_learning.golem import learn_with_golem

        A_raw = learn_with_golem(X, key, variant="nv", verbose=verbose, **learn_kwargs)

    elif algorithm == "pc":
        from jcce.structure_learning.pc import learn_with_pc

        A_raw = learn_with_pc(X, key, verbose=verbose, **learn_kwargs)

    elif algorithm == "ges":
        from jcce.structure_learning.ges import learn_with_ges

        A_raw = learn_with_ges(X, key, verbose=verbose, **learn_kwargs)

    elif algorithm == "directlingam":
        from jcce.structure_learning.directlingam import learn_with_directlingam

        A_raw = learn_with_directlingam(X, key, verbose=verbose, **learn_kwargs)

    elif algorithm == "fci":
        from jcce.structure_learning.fci import learn_with_fci, pag_to_dag

        A_raw = learn_with_fci(X, key, verbose=verbose, **learn_kwargs)

    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Use: dagma, notears, golem, "
            f"golem_nv, pc, ges, directlingam, fci."
        )

    # PAG algorithms: convert PAG -> DAG (not thresholding)
    if algorithm in pag_algorithms:
        from jcce.structure_learning.fci import pag_to_dag

        A_dag = pag_to_dag(A_raw)
        if verbose:
            n_edges = int(jnp.sum(A_dag > 0))
            print(f"\n{algorithm.upper()} PAG -> conservative DAG: {n_edges} edges")
        return A_dag

    # Binary algorithms: skip postprocessing
    if algorithm in binary_algorithms:
        if verbose:
            n_edges = int(jnp.sum(A_raw > 0))
            print(f"\n{algorithm.upper()} returned binary DAG: {n_edges} edges")
        return A_raw

    # Continuous algorithms: apply postprocessing
    if verbose:
        print(f"\nPost-processing {algorithm.upper()} output...")
        print_diagnosis(A_raw)

    A_dag = postprocess_dag(
        A_raw,
        method=postprocess_method,
        threshold=threshold,
        keep_fraction=keep_fraction,
        ensure_dag=True,
        verbose=verbose,
        protected_idx=protected_idx,
    )

    if verbose:
        n_edges = int(jnp.sum(A_dag))
        d = A_dag.shape[0]
        print(f"\nFinal binary DAG: {n_edges} edges, {d} variables")

    return A_dag
