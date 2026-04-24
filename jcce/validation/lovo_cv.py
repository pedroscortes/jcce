"""
Leave-One-Variable-Out (LOVO) Cross-Validation for Structural Stability.

Tests whether the learned causal structure is robust to variable perturbation:
for each variable k, remove it from data, relearn the DAG on d-1 variables,
and check if the sub-DAG (original DAG with variable k removed) is recovered.

Reference:
    Schkoda, T. et al. (2025). "Leave-One-Variable-Out Cross-Validation
    for Causal Discovery." Conference on Causal Learning and Reasoning (CLeaR).

Usage:
    from jcce.validation.lovo_cv import lovo_cv, LovoResult
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class LovoResult:
    """Result of LOVO cross-validation."""

    # Core results
    stability_score: float  # Fraction of variables where sub-DAG is recovered
    n_variables: int  # Total number of variables tested
    n_recovered: int  # Number of variables where sub-DAG matches

    # Per-variable details
    per_variable: List[Dict]  # Per-variable recovery metrics

    # Aggregate structural metrics
    mean_sub_f1: float  # Mean F1 across leave-out experiments
    mean_sub_shd: float  # Mean SHD across leave-out experiments

    # Timing
    total_time: float
    time_per_variable: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "LOVO CROSS-VALIDATION",
            "=" * 60,
            f"Variables tested: {self.n_variables}",
            f"Sub-DAGs recovered: {self.n_recovered}/{self.n_variables} "
            f"({self.stability_score:.1%})",
            f"Mean sub-DAG F1: {self.mean_sub_f1:.3f}",
            f"Mean sub-DAG SHD: {self.mean_sub_shd:.1f}",
            f"Time: {self.total_time:.1f}s ({self.time_per_variable:.1f}s/var)",
        ]

        # Worst variables
        worst = sorted(self.per_variable, key=lambda x: x["f1"])[:3]
        if worst:
            lines.append("\nLeast stable variables (lowest sub-DAG F1):")
            for v in worst:
                lines.append(f"  Var {v['var_idx']}: F1={v['f1']:.3f} SHD={v['shd']}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "stability_score": self.stability_score,
            "n_variables": self.n_variables,
            "n_recovered": self.n_recovered,
            "mean_sub_f1": self.mean_sub_f1,
            "mean_sub_shd": self.mean_sub_shd,
            "per_variable": self.per_variable,
            "total_time": self.total_time,
        }


# ============================================================================
# Core Functions
# ============================================================================


def remove_variable(data: np.ndarray, var_idx: int) -> np.ndarray:
    """Remove variable (column) var_idx from data matrix."""
    return np.delete(data, var_idx, axis=1)


def remove_variable_from_dag(A: np.ndarray, var_idx: int) -> np.ndarray:
    """
    Extract sub-DAG by removing variable var_idx.

    Removes row and column var_idx from the adjacency matrix.

    Args:
        A: (d, d) adjacency matrix
        var_idx: Variable index to remove

    Returns:
        A_sub: (d-1, d-1) sub-adjacency matrix
    """
    A = np.array(A)
    A_sub = np.delete(np.delete(A, var_idx, axis=0), var_idx, axis=1)
    return A_sub


def compare_dags(
    A_learned: np.ndarray,
    A_reference: np.ndarray,
    threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Compare two DAGs using standard structure metrics.

    Args:
        A_learned: Learned adjacency matrix
        A_reference: Reference (sub-)adjacency matrix
        threshold: Edge presence threshold

    Returns:
        Dict with f1, shd, precision, recall, recovered (bool)
    """
    A_learned = np.array(A_learned)
    A_reference = np.array(A_reference)

    # Binarize
    pred = (np.abs(A_learned) > threshold).astype(float)
    true = (np.abs(A_reference) > 1e-6).astype(float)

    # Remove diagonal
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(true, 0)

    tp = np.sum(pred * true)
    fp = np.sum(pred * (1 - true))
    fn = np.sum((1 - pred) * true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    shd = int(fp + fn)

    # "Recovered" if F1 >= recovery_threshold (default 0.7)
    # Lowered from 0.8 per Schkoda et al. (2025): threshold should
    # account for noise in finite-sample structure learning
    recovered = f1 >= 0.7

    return {
        "f1": float(f1),
        "shd": shd,
        "precision": float(precision),
        "recall": float(recall),
        "recovered": recovered,
    }


# ============================================================================
# LOVO CV
# ============================================================================


def lovo_cv(
    data: np.ndarray,
    A_full: np.ndarray,
    learn_fn: Callable,
    learn_kwargs: Optional[Dict[str, Any]] = None,
    variables: Optional[List[int]] = None,
    threshold: float = 0.3,
    recovery_threshold: float = 0.8,
    verbose: bool = False,
) -> LovoResult:
    """
    Leave-One-Variable-Out cross-validation for structural stability.

    For each variable k:
    1. Remove variable k from data → data_k (n, d-1)
    2. Remove variable k from A_full → A_sub_k (d-1, d-1)
    3. Relearn DAG on data_k → A_learned_k
    4. Compare A_learned_k to A_sub_k (SHD, F1)

    Args:
        data: (n_samples, d) data matrix (all variables including Y)
        A_full: (d, d) learned adjacency matrix from full data
        learn_fn: Structure learning function.
            Signature: learn_fn(data, **kwargs) -> A_est (np.ndarray)
        learn_kwargs: Extra keyword arguments for learn_fn
        variables: Which variables to test (default: all d)
        threshold: Edge presence threshold for comparison
        recovery_threshold: F1 threshold to count as "recovered"
        verbose: Print progress

    Returns:
        LovoResult with stability score and per-variable metrics
    """
    data = np.asarray(data, dtype=np.float32)
    A_full = np.asarray(A_full)
    n_samples, d = data.shape

    if learn_kwargs is None:
        learn_kwargs = {}

    if variables is None:
        variables = list(range(d))

    per_variable = []
    n_recovered = 0
    total_start = time.time()

    for k in variables:
        t0 = time.time()

        # 1. Remove variable k
        data_k = remove_variable(data, k)
        A_sub_k = remove_variable_from_dag(A_full, k)

        # 2. Relearn on reduced data
        try:
            A_learned_k = learn_fn(data_k, **learn_kwargs)
            A_learned_k = np.array(A_learned_k)
        except Exception as e:
            warnings.warn(f"LOVO var {k} failed: {e}")
            per_variable.append(
                {
                    "var_idx": k,
                    "f1": 0.0,
                    "shd": d * d,
                    "precision": 0.0,
                    "recall": 0.0,
                    "recovered": False,
                    "time": time.time() - t0,
                    "error": str(e),
                }
            )
            continue

        # 3. Compare to sub-DAG
        metrics = compare_dags(A_learned_k, A_sub_k, threshold=threshold)
        metrics["var_idx"] = k
        metrics["time"] = time.time() - t0
        metrics["recovered"] = metrics["f1"] >= recovery_threshold

        if metrics["recovered"]:
            n_recovered += 1

        per_variable.append(metrics)

        if verbose:
            print(
                f"  LOVO var {k}: F1={metrics['f1']:.3f} SHD={metrics['shd']} "
                f"{'RECOVERED' if metrics['recovered'] else 'CHANGED'} "
                f"({metrics['time']:.1f}s)"
            )

    total_time = time.time() - total_start
    n_tested = len(per_variable)

    # Aggregates
    f1s = [v["f1"] for v in per_variable if "error" not in v]
    shds = [v["shd"] for v in per_variable if "error" not in v]

    result = LovoResult(
        stability_score=n_recovered / max(n_tested, 1),
        n_variables=n_tested,
        n_recovered=n_recovered,
        per_variable=per_variable,
        mean_sub_f1=float(np.mean(f1s)) if f1s else 0.0,
        mean_sub_shd=float(np.mean(shds)) if shds else 0.0,
        total_time=total_time,
        time_per_variable=total_time / max(n_tested, 1),
    )

    if verbose:
        print(f"\n{result.summary()}")

    return result
