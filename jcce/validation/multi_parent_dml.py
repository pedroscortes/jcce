"""
Multi-Parent DML Validation for JCCE v16

Automatically discovers all parents of Y from the learned DAG and runs
DML cross-fitting + refutation tests on each discovered parent.

This replaces the old approach of hardcoding treatment_idx per dataset.
Instead, the framework discovers causal structure AND estimates effects.

For each parent of Y in the learned adjacency matrix:
1. Extract treatment T = X[:, parent_idx], binarize if continuous
2. Run DML cross-fitting (AIPW scores) for ATE estimation
3. Run refutation suite (placebo, RCC, subset stability, dummy outcome)
4. Determine significance: 95% CI excludes 0 AND refutation pass_rate >= 0.5

References:
- Chernozhukov et al. (2018) "Double/Debiased ML for Treatment and Structural Parameters"
- Sharma & Kiciman (2020) "DoWhy: An End-to-End Library for Causal Inference"
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

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


@dataclass
class ParentEffectResult:
    """DML + refutation result for a single discovered parent of Y."""

    feature_idx: int
    feature_name: Optional[str]
    edge_weight: float
    dml_result: DMLResult
    refutation_result: Optional[RefutationSuiteResult]
    is_known_treatment: bool
    is_significant: bool

    def to_dict(self) -> Dict:
        ci = self.dml_result.confidence_interval(0.05)
        result = {
            "feature_idx": self.feature_idx,
            "feature_name": self.feature_name,
            "edge_weight": self.edge_weight,
            "ate_mean": self.dml_result.ate_mean,
            "ate_std": self.dml_result.ate_std,
            "ci_95_lower": ci[0],
            "ci_95_upper": ci[1],
            "is_known_treatment": self.is_known_treatment,
            "is_significant": self.is_significant,
        }
        if self.refutation_result is not None:
            result["refutation"] = self.refutation_result.to_dict()
        return result


@dataclass
class MultiParentDMLResult:
    """Aggregated DML results for all discovered parents of Y."""

    parent_results: List[ParentEffectResult]
    n_parents_discovered: int = field(init=False)
    n_parents_significant: int = field(init=False)
    parent_indices: List[int] = field(init=False)

    def __post_init__(self):
        self.n_parents_discovered = len(self.parent_results)
        self.n_parents_significant = sum(1 for r in self.parent_results if r.is_significant)
        self.parent_indices = [r.feature_idx for r in self.parent_results]

    def to_dict(self) -> Dict:
        return {
            "n_parents_discovered": self.n_parents_discovered,
            "n_parents_significant": self.n_parents_significant,
            "parent_indices": self.parent_indices,
            "parent_effects": [r.to_dict() for r in self.parent_results],
        }

    def summary_table(self) -> str:
        """Generate human-readable summary table."""
        lines = [
            "",
            "  " + "=" * 90,
            "  DML RESULTS: Multi-Parent Effect Estimation",
            "  " + "=" * 90,
            f"  Discovered {self.n_parents_discovered} parents of Y, "
            f"{self.n_parents_significant} with significant effects",
            "",
            f"  {'Idx':<5} {'Feature':<20} {'Edge Wt':<9} {'ATE':<10} "
            f"{'95% CI':<22} {'Refutation':<12} {'Sig?':<5}",
            "  " + "-" * 90,
        ]
        for r in self.parent_results:
            name = (r.feature_name or f"X_{r.feature_idx}")[:19]
            ci = r.dml_result.confidence_interval(0.05)
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"

            if r.refutation_result is not None:
                ref_str = f"{r.refutation_result.n_passed}/{r.refutation_result.n_total} pass"
            else:
                ref_str = "skipped"

            sig_str = "YES" if r.is_significant else "no"
            known = " *" if r.is_known_treatment else ""

            lines.append(
                f"  {r.feature_idx:<5} {name:<20} {r.edge_weight:<9.4f} "
                f"{r.dml_result.ate_mean:<10.4f} {ci_str:<22} "
                f"{ref_str:<12} {sig_str}{known}"
            )

        lines.append("  " + "-" * 90)
        if any(r.is_known_treatment for r in self.parent_results):
            lines.append("  * = known treatment variable from dataset config")
        lines.append("")
        return "\n".join(lines)


def extract_parents_of_Y(
    A: np.ndarray,
    Y_idx: int,
    threshold: float = 0.01,
) -> List[Tuple[int, float]]:
    """
    Extract parent indices and their edge weights from learned adjacency matrix.

    Convention: A[i,j] means i -> j. Parents of Y = {i : A[i, Y_idx] > threshold}.

    Args:
        A: Learned adjacency matrix (n_vars, n_vars)
        Y_idx: Index of target variable Y
        threshold: Minimum edge weight to consider

    Returns:
        List of (parent_idx, edge_weight) sorted by weight descending.
    """
    weights = np.abs(A[:, Y_idx])
    parent_indices = np.where(weights > threshold)[0]
    parent_indices = [int(p) for p in parent_indices if p != Y_idx]
    parents_with_weights = [(p, float(weights[p])) for p in parent_indices]
    parents_with_weights.sort(key=lambda x: x[1], reverse=True)
    return parents_with_weights


def run_multi_parent_dml(
    X: np.ndarray,
    Y: np.ndarray,
    A_est: np.ndarray,
    Y_idx: int,
    feature_names: Optional[List[str]] = None,
    known_treatment_idx: Optional[int] = None,
    parent_threshold: float = 0.01,
    n_dml_folds: int = 5,
    run_refutation: bool = True,
    n_refutation_sims: int = 100,
    verbose: bool = True,
) -> Optional[MultiParentDMLResult]:
    """
    Run DML cross-fitting + refutation on all discovered parents of Y.

    Args:
        X: Feature matrix (n_samples, n_features) — original features, NOT augmented
        Y: Target vector (n_samples,)
        A_est: Learned adjacency matrix from GOLEM (n_vars, n_vars) where
               n_vars = n_features + 1 (augmented with Y)
        Y_idx: Index of Y in the augmented matrix (usually n_features)
        feature_names: Optional list of feature names (length n_features)
        known_treatment_idx: Optional known treatment index for comparison
        parent_threshold: Minimum |A[i, Y_idx]| to consider i as a parent
        n_dml_folds: Number of folds for DML cross-fitting
        run_refutation: Whether to run refutation tests (slow)
        n_refutation_sims: Number of MC simulations per refutation test
        verbose: Print progress

    Returns:
        MultiParentDMLResult or None if no parents discovered
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y).flatten()

    # Discover parents from learned DAG
    parents_with_weights = extract_parents_of_Y(A_est, Y_idx, parent_threshold)

    if not parents_with_weights:
        if verbose:
            print("  DML: No parents of Y discovered (empty MB parents). Skipping.")
        return None

    if verbose:
        print(f"\n  DML: Discovered {len(parents_with_weights)} parents of Y")
        for idx, w in parents_with_weights:
            name = feature_names[idx] if feature_names and idx < len(feature_names) else f"X_{idx}"
            print(f"    Parent {idx} ({name}): edge weight = {w:.4f}")

    # Set up DML and refutation
    # n_repeats >= 3 needed for proper CI estimation (each repeat is a
    # different random K-fold split, giving independent ATE estimates)
    dml = DMLCrossFitter(
        n_splits=n_dml_folds,
        n_repeats=5,
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

    parent_results = []

    for parent_idx, edge_weight in parents_with_weights:
        # Feature index in original X (not augmented)
        if parent_idx >= X.shape[1]:
            # This parent index is beyond X columns (shouldn't happen normally)
            continue

        name = None
        if feature_names and parent_idx < len(feature_names):
            name = feature_names[parent_idx]

        if verbose:
            print(
                f"\n  DML for parent {parent_idx} "
                f"({name or f'X_{parent_idx}'}), "
                f"edge weight = {edge_weight:.4f}..."
            )

        # Extract and binarize treatment
        T = X[:, parent_idx].copy()
        if len(np.unique(T)) > 10:
            T_binary = (T > np.median(T)).astype(float)
        else:
            T_binary = T.astype(float)

        # Check treatment has variation
        if len(np.unique(T_binary)) < 2:
            if verbose:
                print("    Skipping: no variation in treatment (all same value)")
            continue

        # Run DML cross-fitting
        try:
            dml_result = dml.estimate_ate(
                X=X,
                T=T_binary,
                Y=Y,
                train_nuisance_fn=train_fn,
                predict_nuisance_fn=predict_fn,
                treatment_idx=parent_idx,
            )
        except Exception as e:
            warnings.warn(f"DML failed for parent {parent_idx}: {e}")
            continue

        if verbose:
            ci = dml_result.confidence_interval(0.05)
            print(f"    ATE = {dml_result.ate_mean:.4f} +/- {dml_result.ate_std:.4f}")
            print(f"    95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

        # Run refutation
        refutation_result = None
        if refutation_suite is not None and effect_estimator is not None:
            try:
                original_effect = effect_estimator(X, T_binary, Y)
                refutation_result = refutation_suite.run_all(
                    X=X,
                    T=T_binary,
                    Y=Y,
                    estimate_effect_fn=effect_estimator,
                    original_effect=original_effect,
                )
                if verbose:
                    print(
                        f"    Refutation: {refutation_result.n_passed}/"
                        f"{refutation_result.n_total} tests passed"
                    )
            except Exception as e:
                warnings.warn(f"Refutation failed for parent {parent_idx}: {e}")

        # Determine significance
        ci = dml_result.confidence_interval(0.05)
        ci_excludes_zero = (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)
        refutation_passes = refutation_result is None or refutation_result.pass_rate >= 0.5
        is_significant = ci_excludes_zero and refutation_passes

        is_known = known_treatment_idx is not None and parent_idx == known_treatment_idx

        parent_results.append(
            ParentEffectResult(
                feature_idx=parent_idx,
                feature_name=name,
                edge_weight=edge_weight,
                dml_result=dml_result,
                refutation_result=refutation_result,
                is_known_treatment=is_known,
                is_significant=is_significant,
            )
        )

    if not parent_results:
        if verbose:
            print("  DML: All parent DML estimations failed. No results.")
        return None

    result = MultiParentDMLResult(parent_results=parent_results)

    if verbose:
        print(result.summary_table())

    return result
