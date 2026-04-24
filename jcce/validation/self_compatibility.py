"""
Self-Compatibility Check for Causal Structure Learning.

A learned DAG G is self-compatible if data generated from the fitted SCM
(using G) would lead the same algorithm to recover G again.
This is a fixed-point criterion: the algorithm's output is consistent
with its own assumptions.

Reference:
    Faller, P. M. et al. (2024). "Self-Compatibility: Evaluating Causal
    Discovery without Ground Truth." AISTATS.

Usage:
    from jcce.validation.self_compatibility import (
        self_compatibility_check,
        SelfCompatibilityResult,
    )
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class SelfCompatibilityResult:
    """Result of self-compatibility analysis."""

    # Core result
    compatibility_score: float  # Fraction of resamples recovering the DAG
    n_resamples: int  # Number of resample iterations completed
    n_compatible: int  # Number of resamples where DAG was recovered

    # Per-resample metrics
    per_resample: List[Dict]  # Per-resample F1, SHD, etc.

    # Aggregate
    mean_f1: float  # Mean F1(A_relearned, A_original) across resamples
    std_f1: float
    mean_shd: float
    edge_recovery_rate: float  # Fraction of edges recovered on average

    # Timing
    total_time: float

    def is_self_compatible(self, threshold: float = 0.8) -> bool:
        """Check if compatibility score exceeds threshold."""
        return self.compatibility_score >= threshold

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "SELF-COMPATIBILITY CHECK",
            "=" * 60,
            f"Resamples: {self.n_resamples}",
            f"Compatible: {self.n_compatible}/{self.n_resamples} ({self.compatibility_score:.1%})",
            f"Mean F1(relearned, original): {self.mean_f1:.3f} ± {self.std_f1:.3f}",
            f"Mean SHD: {self.mean_shd:.1f}",
            f"Edge recovery rate: {self.edge_recovery_rate:.1%}",
            f"Self-compatible: {'YES' if self.is_self_compatible() else 'NO'}",
            f"Time: {self.total_time:.1f}s",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "compatibility_score": self.compatibility_score,
            "n_resamples": self.n_resamples,
            "n_compatible": self.n_compatible,
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "mean_shd": self.mean_shd,
            "edge_recovery_rate": self.edge_recovery_rate,
            "is_self_compatible": self.is_self_compatible(),
            "total_time": self.total_time,
            "per_resample": self.per_resample,
        }


# ============================================================================
# SCM Fitting
# ============================================================================


def fit_linear_scm(
    data: np.ndarray,
    A_learned: np.ndarray,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Fit a linear SCM to data using the learned DAG structure.

    Model: X_i = sum_j A[i,j] * X_j + eps_i
    Estimates coefficients via OLS and noise variance from residuals.

    Args:
        data: (n, d) data matrix
        A_learned: (d, d) learned adjacency matrix
        threshold: Edge presence threshold

    Returns:
        Dict with 'A_fitted' (d, d) coefficients and 'noise_std' (d,)
    """
    n, d = data.shape
    A_bin = (np.abs(A_learned) > threshold).astype(float)
    np.fill_diagonal(A_bin, 0)

    A_fitted = np.zeros((d, d))
    noise_std = np.zeros(d)

    for i in range(d):
        parents = np.where(A_bin[i, :] > 0)[0]

        if len(parents) == 0:
            # Root node: X_i = eps_i
            noise_std[i] = max(np.std(data[:, i]), 1e-6)
            continue

        # OLS: X_i = sum_j beta_j * X_j + eps_i
        X_parents = data[:, parents]
        y_i = data[:, i]
        beta = np.linalg.lstsq(X_parents, y_i, rcond=None)[0]

        for idx, j in enumerate(parents):
            A_fitted[i, j] = beta[idx]

        residuals = y_i - X_parents @ beta
        noise_std[i] = max(np.std(residuals), 1e-6)

    return {
        "A_fitted": A_fitted,
        "noise_std": noise_std,
    }


def sample_from_fitted_scm(
    scm_params: Dict[str, Any],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Generate synthetic data from a fitted linear SCM via ancestral sampling.

    Args:
        scm_params: Output of fit_linear_scm()
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        X: (n_samples, d) synthetic data
    """
    A = scm_params["A_fitted"]
    noise_std = scm_params["noise_std"]
    d = A.shape[0]

    rng = np.random.RandomState(seed)

    # Topological sort
    topo_order = _topological_sort_numpy(A)

    # Ancestral sampling
    X = np.zeros((n_samples, d))
    for i in topo_order:
        parents = np.where(np.abs(A[i, :]) > 1e-10)[0]
        noise = rng.randn(n_samples) * noise_std[i]

        if len(parents) == 0:
            X[:, i] = noise
        else:
            X[:, i] = X[:, parents] @ A[i, parents] + noise

    return X


def _topological_sort_numpy(A: np.ndarray) -> List[int]:
    """Topological sort of a DAG using Kahn's algorithm."""
    d = A.shape[0]
    # In-degree: count parents (non-zero entries in row, excluding diagonal)
    in_degree = np.sum(np.abs(A) > 1e-10, axis=1).astype(int)
    np.fill_diagonal(A, 0)  # Safety

    queue = [i for i in range(d) if in_degree[i] == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)

        # Children: nodes that have node as parent (A[child, node] != 0)
        children = np.where(np.abs(A[:, node]) > 1e-10)[0]
        for child in children:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(int(child))

    if len(order) != d:
        # Not a DAG — fall back to natural ordering
        return list(range(d))

    return order


# ============================================================================
# Self-Compatibility Check
# ============================================================================


def self_compatibility_check(
    data: np.ndarray,
    A_learned: np.ndarray,
    learn_fn: Callable,
    learn_kwargs: Optional[Dict[str, Any]] = None,
    n_resamples: int = 10,
    edge_threshold: float = 0.3,
    recovery_threshold: float = 0.8,
    seed: int = 42,
    verbose: bool = False,
) -> SelfCompatibilityResult:
    """
    Self-compatibility check: is the learned DAG a fixed point?

    Algorithm:
    1. Fit linear SCM to (data, A_learned)
    2. For each resample m = 1..M:
       a. Sample D'_m ~ fitted_SCM(n=len(data))
       b. Relearn A_m = learn_fn(D'_m)
       c. Compare A_m to A_learned (F1, SHD)
    3. Compatibility = fraction of resamples where F1 >= threshold

    Args:
        data: (n, d) original data
        A_learned: (d, d) learned adjacency matrix to test
        learn_fn: Structure learning function.
            Signature: learn_fn(data, **kwargs) -> A_est (np.ndarray)
        learn_kwargs: Extra keyword arguments for learn_fn
        n_resamples: Number of synthetic resample iterations
        edge_threshold: Threshold for edge presence
        recovery_threshold: F1 threshold to count as "compatible"
        seed: Random seed
        verbose: Print progress

    Returns:
        SelfCompatibilityResult
    """
    data = np.asarray(data, dtype=np.float32)
    A_learned = np.asarray(A_learned)
    n_samples, d = data.shape

    if learn_kwargs is None:
        learn_kwargs = {}

    # 1. Fit SCM
    scm_params = fit_linear_scm(data, A_learned, threshold=edge_threshold)

    if verbose:
        n_edges = int(np.sum(np.abs(A_learned) > edge_threshold))
        print(
            f"Fitted linear SCM: {n_edges} edges, "
            f"noise_std range [{scm_params['noise_std'].min():.3f}, "
            f"{scm_params['noise_std'].max():.3f}]"
        )

    # Reference: binarized A_learned
    A_ref = (np.abs(A_learned) > edge_threshold).astype(float)
    np.fill_diagonal(A_ref, 0)
    n_ref_edges = int(np.sum(A_ref))

    per_resample = []
    n_compatible = 0
    total_start = time.time()

    for m in range(n_resamples):
        t0 = time.time()

        # 2a. Sample from fitted SCM
        D_synth = sample_from_fitted_scm(scm_params, n_samples, seed=seed + m)

        # 2b. Relearn structure
        try:
            A_relearned = learn_fn(D_synth, **learn_kwargs)
            A_relearned = np.array(A_relearned)
        except Exception as e:
            warnings.warn(f"Self-compat resample {m} failed: {e}")
            per_resample.append(
                {
                    "resample": m,
                    "f1": 0.0,
                    "shd": d * d,
                    "precision": 0.0,
                    "recall": 0.0,
                    "compatible": False,
                    "time": time.time() - t0,
                    "error": str(e),
                }
            )
            continue

        # 2c. Compare to original
        A_pred = (np.abs(A_relearned) > edge_threshold).astype(float)
        np.fill_diagonal(A_pred, 0)

        tp = np.sum(A_pred * A_ref)
        fp = np.sum(A_pred * (1 - A_ref))
        fn = np.sum((1 - A_pred) * A_ref)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        shd = int(fp + fn)

        compatible = f1 >= recovery_threshold
        if compatible:
            n_compatible += 1

        per_resample.append(
            {
                "resample": m,
                "f1": float(f1),
                "shd": shd,
                "precision": float(precision),
                "recall": float(recall),
                "compatible": compatible,
                "time": time.time() - t0,
                "n_edges_relearned": int(np.sum(A_pred)),
            }
        )

        if verbose:
            print(
                f"  Resample {m + 1}/{n_resamples}: F1={f1:.3f} SHD={shd} "
                f"{'COMPAT' if compatible else 'INCOMPAT'} ({per_resample[-1]['time']:.1f}s)"
            )

    total_time = time.time() - total_start
    n_completed = len([r for r in per_resample if "error" not in r])

    # Aggregates
    f1s = [r["f1"] for r in per_resample if "error" not in r]
    shds = [r["shd"] for r in per_resample if "error" not in r]
    recalls = [r["recall"] for r in per_resample if "error" not in r]

    result = SelfCompatibilityResult(
        compatibility_score=n_compatible / max(n_completed, 1),
        n_resamples=n_completed,
        n_compatible=n_compatible,
        per_resample=per_resample,
        mean_f1=float(np.mean(f1s)) if f1s else 0.0,
        std_f1=float(np.std(f1s)) if f1s else 0.0,
        mean_shd=float(np.mean(shds)) if shds else 0.0,
        edge_recovery_rate=float(np.mean(recalls)) if recalls else 0.0,
        total_time=total_time,
    )

    if verbose:
        print(f"\n{result.summary()}")

    return result
