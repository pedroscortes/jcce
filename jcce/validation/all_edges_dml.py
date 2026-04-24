"""
All-Edges DML Validation for JCCE v16

Extends post-hoc DML validation from parents-of-Y only to EVERY edge in the
learned DAG. This makes the entire structure scientifically interpretable with
validated ATE estimates, confidence intervals, and significance flags.

For each edge i→j in A_est:
1. Treatment T = X[:, i] (binarized if continuous)
2. Outcome = X[:, j] if j != Y_idx, else Y
3. Covariates = backdoor-valid adjustment set for (i, j)
4. Run DML cross-fitting (AIPW scores)
5. Optionally run refutation suite
6. Apply BH-FDR correction across all edges

References:
- Chernozhukov et al. (2018) "Double/Debiased ML for Treatment and Structural Parameters"
- Benjamini & Hochberg (1995) "Controlling the False Discovery Rate"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
import warnings

from jcce.validation.dml_crossfitting import (
    DMLCrossFitter,
    DMLResult,
    create_simple_nuisance_functions,
)
from jcce.validation.refutation_suite import (
    RefutationSuite,
    RefutationSuiteResult,
    create_simple_effect_estimator,
)
from jcce.structure_learning.effect_estimation import (
    compute_valid_adjustment_sets,
    _get_descendants,
)


@dataclass
class EdgeEffectResult:
    """DML + refutation result for a single edge i→j in the DAG."""
    source_idx: int
    target_idx: int
    source_name: Optional[str]
    target_name: Optional[str]
    edge_weight: float
    dml_result: DMLResult
    refutation_result: Optional[RefutationSuiteResult]
    is_significant: bool
    is_significant_fdr: bool
    p_value: float

    @property
    def edge_key(self) -> str:
        src = self.source_name or f"X{self.source_idx}"
        tgt = self.target_name or f"X{self.target_idx}"
        return f"{src}->{tgt}"

    def to_dict(self) -> Dict:
        ci = self.dml_result.confidence_interval(0.05)
        result = {
            'source_idx': self.source_idx,
            'target_idx': self.target_idx,
            'source_name': self.source_name,
            'target_name': self.target_name,
            'edge_weight': self.edge_weight,
            'ate_mean': self.dml_result.ate_mean,
            'ate_std': self.dml_result.ate_std,
            'ci_95_lower': ci[0],
            'ci_95_upper': ci[1],
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'is_significant_fdr': self.is_significant_fdr,
        }
        if self.refutation_result is not None:
            result['refutation'] = self.refutation_result.to_dict()
        return result


@dataclass
class AllEdgesDMLResult:
    """Aggregated DML results for all edges in the learned DAG."""
    edge_results: List[EdgeEffectResult]
    n_edges: int = field(init=False)
    n_significant: int = field(init=False)
    n_significant_fdr: int = field(init=False)

    def __post_init__(self):
        self.n_edges = len(self.edge_results)
        self.n_significant = sum(
            1 for r in self.edge_results if r.is_significant
        )
        self.n_significant_fdr = sum(
            1 for r in self.edge_results if r.is_significant_fdr
        )

    def to_dict(self) -> Dict:
        return {
            'n_edges': self.n_edges,
            'n_significant': self.n_significant,
            'n_significant_fdr': self.n_significant_fdr,
            'edge_effects': [r.to_dict() for r in self.edge_results],
        }

    def to_storage_dict(self) -> Dict:
        """Return dict suitable for pkl storage (sol['all_edges_dml'])."""
        edge_effects = {}
        for r in self.edge_results:
            ci = r.dml_result.confidence_interval(0.05)
            edge_effects[r.edge_key] = {
                'ate': r.dml_result.ate_mean,
                'ci_95': [ci[0], ci[1]],
                'p_value': r.p_value,
                'sig': r.is_significant,
                'sig_fdr': r.is_significant_fdr,
            }
        return {
            'n_edges': self.n_edges,
            'n_significant': self.n_significant,
            'n_significant_fdr': self.n_significant_fdr,
            'edge_effects': edge_effects,
        }

    def to_causal_effects_dict(self) -> Dict[str, float]:
        """Return flat {edge_key: ate} dict for viz compatibility."""
        return {
            r.edge_key: r.dml_result.ate_mean
            for r in self.edge_results
        }

    def summary_table(self) -> str:
        """Generate human-readable summary table."""
        lines = [
            "",
            "  " + "=" * 100,
            "  ALL-EDGES DML RESULTS",
            "  " + "=" * 100,
            f"  {self.n_edges} edges tested, "
            f"{self.n_significant} significant (CI), "
            f"{self.n_significant_fdr} significant (FDR)",
            "",
            f"  {'Edge':<25} {'Weight':<9} {'ATE':<10} "
            f"{'95% CI':<22} {'p-value':<10} {'Sig':<5} {'FDR':<5}",
            "  " + "-" * 100,
        ]
        for r in self.edge_results:
            edge_label = r.edge_key[:24]
            ci = r.dml_result.confidence_interval(0.05)
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            sig_str = "YES" if r.is_significant else "no"
            fdr_str = "YES" if r.is_significant_fdr else "no"
            p_str = f"{r.p_value:.4f}" if r.p_value >= 0.0001 else "<.0001"

            lines.append(
                f"  {edge_label:<25} {r.edge_weight:<9.4f} "
                f"{r.dml_result.ate_mean:<10.4f} {ci_str:<22} "
                f"{p_str:<10} {sig_str:<5} {fdr_str:<5}"
            )

        lines.append("  " + "-" * 100)
        lines.append("")
        return "\n".join(lines)


def extract_all_edges(
    A: np.ndarray,
    threshold: float = 0.01,
) -> List[Tuple[int, int, float]]:
    """
    Extract all edges from adjacency matrix above threshold.

    Convention: A[i,j] means i → j.

    Args:
        A: Adjacency matrix (n_vars, n_vars)
        threshold: Minimum |A[i,j]| to consider as edge

    Returns:
        List of (source_idx, target_idx, edge_weight) sorted by weight desc.
    """
    abs_A = np.abs(A)
    edges = []
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and abs_A[i, j] > threshold:
                edges.append((i, j, float(abs_A[i, j])))
    edges.sort(key=lambda x: x[2], reverse=True)
    return edges


def _benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of raw p-values
        alpha: FDR level (default 0.05)

    Returns:
        List of booleans indicating significance after FDR correction
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keeping track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Find largest k such that p_(k) <= alpha * k / n
    significant = [False] * n
    max_significant_rank = -1
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        threshold = alpha * rank / n
        if p <= threshold:
            max_significant_rank = rank

    # All ranks up to max_significant_rank are significant
    if max_significant_rank > 0:
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if rank <= max_significant_rank:
                significant[orig_idx] = True

    return significant


def run_all_edges_dml(
    X: np.ndarray,
    Y: np.ndarray,
    A_est: np.ndarray,
    Y_idx: int,
    feature_names: Optional[List[str]] = None,
    edge_threshold: float = 0.01,
    n_dml_folds: int = 5,
    n_dml_repeats: int = 5,
    run_refutation: bool = True,
    n_refutation_sims: int = 100,
    fdr_alpha: float = 0.05,
    verbose: bool = True,
) -> Optional[AllEdgesDMLResult]:
    """
    Run DML cross-fitting on every edge in the learned DAG.

    For each edge i→j:
    - Treatment: X[:, i] (binarized if >10 unique values)
    - Outcome: X[:, j] if j != Y_idx, else Y
    - Covariates: backdoor-valid adjustment set for treatment i w.r.t. outcome j

    Args:
        X: Feature matrix (n_samples, n_features) — original features
        Y: Target vector (n_samples,)
        A_est: Learned adjacency matrix (n_vars, n_vars) where n_vars = n_features + 1
        Y_idx: Index of Y in the augmented matrix (usually n_features)
        feature_names: Optional feature names (length n_features)
        edge_threshold: Minimum |A[i,j]| to consider as edge
        n_dml_folds: Number of folds for DML cross-fitting
        n_dml_repeats: Number of repeated fold splits
        run_refutation: Whether to run refutation tests (slow)
        n_refutation_sims: MC simulations per refutation test
        fdr_alpha: FDR level for Benjamini-Hochberg correction
        verbose: Print progress

    Returns:
        AllEdgesDMLResult or None if no edges found
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y).flatten()
    A_np = np.array(A_est)
    n_features = X.shape[1]

    # Build augmented data matrix: [X | Y] for X→X edges that need Y as outcome
    # (Y column is at index Y_idx)

    # Extract all edges
    edges = extract_all_edges(A_np, threshold=edge_threshold)

    if not edges:
        if verbose:
            print("  All-edges DML: No edges above threshold. Skipping.")
        return None

    if verbose:
        print(f"\n  All-edges DML: Found {len(edges)} edges to validate")

    # Build name lookup
    def get_name(idx: int) -> Optional[str]:
        if idx == Y_idx:
            return "Y"
        if feature_names and idx < len(feature_names):
            return feature_names[idx]
        return None

    # Set up DML
    dml = DMLCrossFitter(
        n_splits=n_dml_folds,
        n_repeats=n_dml_repeats,
        random_state=42,
        stratify=True,
    )
    train_fn, predict_fn = create_simple_nuisance_functions()

    refutation_suite = None
    effect_estimator = None
    if run_refutation:
        refutation_suite = RefutationSuite(
            n_simulations=n_refutation_sims,
            subset_fraction=0.8,
            random_state=42,
        )
        effect_estimator = create_simple_effect_estimator()

    # Binarize adjacency for graph algorithms
    A_binary = (np.abs(A_np) > edge_threshold).astype(int)

    edge_results = []

    for edge_num, (src, tgt, weight) in enumerate(edges):
        src_name = get_name(src)
        tgt_name = get_name(tgt)
        edge_label = f"{src_name or f'X{src}'}->{tgt_name or f'X{tgt}'}"

        if verbose:
            print(f"  [{edge_num+1}/{len(edges)}] {edge_label} "
                  f"(weight={weight:.4f})...", end=" ")

        # Skip edges where source is beyond X columns
        if src >= n_features:
            if verbose:
                print("skip (source is Y)")
            continue

        # Determine outcome
        if tgt == Y_idx:
            outcome = Y.copy()
        elif tgt < n_features:
            outcome = X[:, tgt].copy()
        else:
            if verbose:
                print("skip (invalid target)")
            continue

        # Extract and binarize treatment
        T = X[:, src].copy()
        if len(np.unique(T)) > 10:
            T_binary = (T > np.median(T)).astype(float)
        else:
            T_binary = T.astype(float)

        # Check treatment has variation
        if len(np.unique(T_binary)) < 2:
            if verbose:
                print("skip (no treatment variation)")
            continue

        # Check outcome has variation
        if np.std(outcome) < 1e-10:
            if verbose:
                print("skip (no outcome variation)")
            continue

        # Compute adjustment set for this edge using backdoor criterion
        # For edge i→j, we need adjustment set w.r.t. outcome j
        adj_sets = compute_valid_adjustment_sets(
            A_np, Y_idx=tgt, threshold=edge_threshold
        )
        adj_set = adj_sets.get(src, set())

        if not adj_set:
            # Exogenous treatment (no parents): valid adjustment set is EMPTY.
            # No backdoor paths exist from an exogenous treatment, so adjusting
            # for non-descendants can only introduce collider bias (Pearl 2009).
            # DML with empty covariates reduces to cross-fitted diff-in-means.
            pass

        # Filter to valid feature indices only
        covariate_indices = sorted([
            idx for idx in adj_set
            if idx < n_features and idx != src
        ])

        # Build covariate matrix
        if covariate_indices:
            X_adj = X[:, covariate_indices]
        else:
            # No covariates (exogenous treatment). Provide constant column
            # so nuisance models can fit (Ridge needs ≥1 feature).
            # DML reduces to cross-fitted diff-in-means.
            X_adj = np.ones((X.shape[0], 1))

        # Run DML cross-fitting
        try:
            dml_result = dml.estimate_ate(
                X=X_adj, T=T_binary, Y=outcome,
                train_nuisance_fn=train_fn,
                predict_nuisance_fn=predict_fn,
                treatment_idx=src,
            )
        except Exception as e:
            if verbose:
                print(f"DML failed: {e}")
            continue

        # Compute p-value from t-test: ate / SE
        se = dml_result.standard_error
        if se > 0:
            t_stat = dml_result.ate_mean / se
            p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        else:
            p_value = 1.0

        # CI-based significance (before FDR)
        ci = dml_result.confidence_interval(0.05)
        ci_excludes_zero = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)

        # Refutation check
        refutation_result = None
        refutation_passes = True
        if refutation_suite is not None and effect_estimator is not None:
            try:
                original_effect = effect_estimator(X_adj, T_binary, outcome)
                refutation_result = refutation_suite.run_all(
                    X=X_adj, T=T_binary, Y=outcome,
                    estimate_effect_fn=effect_estimator,
                    original_effect=original_effect,
                )
                refutation_passes = refutation_result.pass_rate >= 0.5
            except Exception:
                pass  # Refutation failure is non-fatal

        is_significant = ci_excludes_zero and refutation_passes

        edge_results.append(EdgeEffectResult(
            source_idx=src,
            target_idx=tgt,
            source_name=src_name,
            target_name=tgt_name,
            edge_weight=weight,
            dml_result=dml_result,
            refutation_result=refutation_result,
            is_significant=is_significant,
            is_significant_fdr=False,  # set below after BH
            p_value=p_value,
        ))

        if verbose:
            sig_marker = "*" if is_significant else ""
            print(f"ATE={dml_result.ate_mean:.4f} "
                  f"p={p_value:.4f} {sig_marker}")

    if not edge_results:
        if verbose:
            print("  All-edges DML: All edge estimations failed. No results.")
        return None

    # Apply BH-FDR correction
    raw_p_values = [r.p_value for r in edge_results]
    fdr_flags = _benjamini_hochberg(raw_p_values, alpha=fdr_alpha)
    for r, is_sig_fdr in zip(edge_results, fdr_flags):
        r.is_significant_fdr = is_sig_fdr

    result = AllEdgesDMLResult(edge_results=edge_results)

    if verbose:
        print(result.summary_table())

    return result
