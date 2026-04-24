"""
Identifiability Diagnostics for JCCE v15.0

Provides tools to assess whether causal effects are identifiable given
the data and DAG structure:

1. Condition number of Markov Blanket design matrix (kappa)
2. SVD diagnostics (spectrum, collinear pairs, effective rank)
3. Hutchinson trace estimator for Effective Degrees of Freedom (EDF)
4. BIC variants (simple for linear, neural using Hutchinson EDF)
5. D-optimality penalty (JAX-differentiable, for regularizer in GOLEM loss)
6. IdentifiabilityReport dataclass for structured output

Key references:
- Belsley et al. (1980) "Regression Diagnostics" (condition number thresholds)
- Avron & Toledo (2011) "Randomized algorithms for estimating the trace" (Hutchinson)
- Schwarz (1978) "Estimating the Dimension of a Model" (BIC)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random as jax_random

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =============================================================================
# 1. Condition Number of MB Design Matrix
# =============================================================================


def compute_mb_condition_number(X: np.ndarray, mb_indices: List[int]) -> float:
    """
    Compute condition number kappa(X[:, MB]) = sigma_max / sigma_min via SVD.

    Args:
        X: Data matrix (n_samples, n_features)
        mb_indices: Indices of Markov Blanket features

    Returns:
        Condition number (kappa). Lower is better.
        kappa < 30 = well-conditioned, 30-100 = moderate, > 100 = ill-conditioned.
    """
    if len(mb_indices) == 0:
        return 1.0
    if len(mb_indices) == 1:
        return 1.0

    X_mb = np.asarray(X)[:, mb_indices]

    # Standardize columns to avoid scale artifacts
    stds = X_mb.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    X_mb = (X_mb - X_mb.mean(axis=0)) / stds

    singular_values = np.linalg.svd(X_mb, compute_uv=False)
    return float(singular_values[0] / (singular_values[-1] + 1e-10))


# =============================================================================
# 2. SVD Diagnostics
# =============================================================================


def compute_svd_diagnostics(
    X: np.ndarray,
    mb_indices: List[int],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full SVD analysis: spectrum, near-zero gaps, collinear pairs.

    Args:
        X: Data matrix (n_samples, n_features)
        mb_indices: Indices of Markov Blanket features
        feature_names: Optional names for features

    Returns:
        Dictionary with:
        - singular_values: SVD spectrum
        - condition_number: kappa
        - near_collinear_pairs: List of (feat_i, feat_j, correlation)
        - variance_explained_ratio: Fraction of variance per singular value
        - effective_rank: Number of singular values > 0.01 * max
    """
    if len(mb_indices) == 0:
        return {
            "singular_values": np.array([]),
            "condition_number": 1.0,
            "near_collinear_pairs": [],
            "variance_explained_ratio": np.array([]),
            "effective_rank": 0,
        }

    X_mb = np.asarray(X)[:, mb_indices]

    # Standardize
    stds = X_mb.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    X_mb_std = (X_mb - X_mb.mean(axis=0)) / stds

    # SVD
    U, singular_values, Vt = np.linalg.svd(X_mb_std, full_matrices=False)

    # Condition number
    condition_number = float(singular_values[0] / (singular_values[-1] + 1e-10))

    # Variance explained
    total_var = np.sum(singular_values**2)
    variance_explained_ratio = (singular_values**2) / (total_var + 1e-10)

    # Effective rank: count singular values > 1% of max
    threshold = 0.01 * singular_values[0]
    effective_rank = int(np.sum(singular_values > threshold))

    # Detect near-collinear pairs via correlation matrix
    if feature_names is None:
        feature_names = [f"X{idx}" for idx in mb_indices]

    near_collinear_pairs = []
    n_mb = len(mb_indices)
    if n_mb >= 2:
        corr_matrix = np.corrcoef(X_mb_std.T)
        for i in range(n_mb):
            for j in range(i + 1, n_mb):
                corr = abs(corr_matrix[i, j])
                if corr > 0.9:  # High correlation threshold
                    near_collinear_pairs.append(
                        (
                            feature_names[i] if i < len(feature_names) else f"X{mb_indices[i]}",
                            feature_names[j] if j < len(feature_names) else f"X{mb_indices[j]}",
                            float(corr),
                        )
                    )

    return {
        "singular_values": singular_values,
        "condition_number": condition_number,
        "near_collinear_pairs": near_collinear_pairs,
        "variance_explained_ratio": variance_explained_ratio,
        "effective_rank": effective_rank,
    }


# =============================================================================
# 3. Hutchinson Trace Estimator for EDF
# =============================================================================


def hutchinson_edf(
    predict_fn: Callable,
    X: Any,
    n_probes: int = 5,
    key: Any = None,
) -> float:
    """
    Estimate Effective Degrees of Freedom via Hutchinson trace estimator.

    EDF = trace(J) where J = d(predictions)/d(inputs).
    Uses Rademacher random vectors: trace(J) ~ (1/n_probes) * sum(v^T J v)
    Each probe uses one JVP (forward-mode AD via jax.jvp).

    Cost: n_probes forward passes (default 5 = negligible overhead).

    Args:
        predict_fn: Function X -> predictions (JAX-compatible)
        X: Input data (n_samples, n_features) as JAX array
        n_probes: Number of random probes (default 5)
        key: JAX PRNG key (optional, uses default seed if None)

    Returns:
        Estimated EDF (effective number of parameters).
    """
    if not HAS_JAX:
        raise RuntimeError("JAX is required for Hutchinson EDF estimation")

    X = jnp.asarray(X)
    n_samples = X.shape[0]

    if key is None:
        key = jax_random.PRNGKey(42)

    trace_estimates = []

    for i in range(n_probes):
        key, subkey = jax_random.split(key)

        # Rademacher random vector: +1 or -1 with equal probability
        v = jax_random.rademacher(subkey, shape=X.shape, dtype=X.dtype)

        # JVP: compute J @ v efficiently via forward-mode AD
        _, jvp_result = jax.jvp(predict_fn, (X,), (v,))

        # trace(J) ~ v^T @ (J @ v) = sum(v * jvp_result)
        trace_est = jnp.sum(v * jvp_result)
        trace_estimates.append(float(trace_est))

    # Average over probes
    edf = float(np.mean(trace_estimates))

    # EDF should be non-negative; clamp to avoid numerical artifacts
    return max(0.0, edf)


# =============================================================================
# 4. BIC Variants
# =============================================================================


def compute_simple_bic(
    residuals: np.ndarray,
    n_params: int,
    n_samples: int,
) -> float:
    """
    Classic BIC = n * log(RSS/n) + k * log(n).

    For linear processors where parameter count equals edge count.

    Args:
        residuals: Reconstruction residuals (n_samples,) or (n_samples, n_vars)
        n_params: Number of parameters (e.g., edge count)
        n_samples: Number of data points

    Returns:
        BIC value (lower is better)
    """
    residuals = np.asarray(residuals).flatten()
    n_samples = max(n_samples, 1)
    n_params = max(n_params, 1)

    rss = np.sum(residuals**2)
    # Avoid log(0)
    rss_per_n = max(rss / n_samples, 1e-20)

    return float(n_samples * np.log(rss_per_n) + n_params * np.log(n_samples))


def compute_neural_bic(
    residuals: np.ndarray,
    edf: float,
    n_samples: int,
) -> float:
    """
    Neural BIC using Hutchinson EDF as effective parameter count.

    BIC = n * log(RSS/n) + EDF * log(n)

    This replaces edge count with the effective degrees of freedom
    estimated via the Hutchinson trace estimator, giving a more
    accurate complexity penalty for neural processors.

    Args:
        residuals: Reconstruction residuals
        edf: Effective degrees of freedom (from hutchinson_edf)
        n_samples: Number of data points

    Returns:
        BIC value (lower is better)
    """
    residuals = np.asarray(residuals).flatten()
    n_samples = max(n_samples, 1)
    edf = max(edf, 1.0)

    rss = np.sum(residuals**2)
    rss_per_n = max(rss / n_samples, 1e-20)

    return float(n_samples * np.log(rss_per_n) + edf * np.log(n_samples))


def compute_bic_for_dag(
    X: np.ndarray,
    A: np.ndarray,
    processors: Any = None,
    processor_type: str = "linear",
    predict_fn: Optional[Callable] = None,
    key: Any = None,
) -> float:
    """
    Unified BIC: dispatches to simple_bic or neural_bic based on processor_type.

    Uses reconstruction residuals only (not classification or effect loss),
    consistent with BIC as a model selection criterion for data fit.

    Args:
        X: Data matrix (n_samples, n_features)
        A: Adjacency matrix (n_features, n_features)
        processors: Trained processor(s) (optional, for neural BIC)
        processor_type: 'linear', 'elm', 'mlp', 'gnn', 'mamba', 'transformer'
        predict_fn: Prediction function for neural processors (for EDF)
        key: JAX PRNG key for Hutchinson estimator

    Returns:
        BIC value (lower is better)
    """
    X = np.asarray(X)
    A = np.asarray(A)
    n_samples, n_features = X.shape

    # Handle augmented A matrices (n_features+1 × n_features+1, includes Y column)
    # Truncate to X-only submatrix for reconstruction
    if A.shape[0] > n_features:
        A = A[:n_features, :n_features]

    # Compute reconstruction residuals: X_hat = X @ A
    X_hat = X @ A
    residuals = X - X_hat

    # Count edges as base parameter count
    n_edges = int(np.sum(np.abs(A) > 1e-6))

    if processor_type == "linear" or predict_fn is None:
        return compute_simple_bic(residuals, n_edges, n_samples)
    else:
        # Neural processor: use Hutchinson EDF
        if HAS_JAX and predict_fn is not None:
            try:
                X_jax = jnp.asarray(X)
                edf = hutchinson_edf(predict_fn, X_jax, n_probes=5, key=key)
                return compute_neural_bic(residuals, edf, n_samples)
            except Exception:
                # Fallback to simple BIC with edge count
                return compute_simple_bic(residuals, n_edges, n_samples)
        else:
            return compute_simple_bic(residuals, n_edges, n_samples)


# =============================================================================
# 5. D-Optimality Penalty (JAX-differentiable)
# =============================================================================


def d_optimality_penalty(X_mb_jax: Any) -> Any:
    """
    D-optimality penalty: -log det(X_mb^T X_mb / n).

    Differentiable proxy for condition number. Lower value = better conditioned.
    Uses Cholesky decomposition for numerical stability.

    Suitable for adding to JAX loss functions as a regularizer.

    Args:
        X_mb_jax: Markov Blanket design matrix (n_samples, n_mb_features) as JAX array

    Returns:
        Scalar penalty value (JAX array, suitable for gradient computation)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX is required for D-optimality penalty")

    n = X_mb_jax.shape[0]
    p = X_mb_jax.shape[1]

    # Gram matrix
    G = X_mb_jax.T @ X_mb_jax / n

    # Add small ridge for positive-definiteness guarantee
    G = G + 1e-6 * jnp.eye(p)

    # log det via Cholesky: log det(G) = 2 * sum(log(diag(L)))
    L = jnp.linalg.cholesky(G)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    # Negative log-det: lower = better conditioned
    return -log_det


# =============================================================================
# 6. Identifiability Report
# =============================================================================


@dataclass
class IdentifiabilityReport:
    """Structured identifiability diagnostics for a single DAG."""

    condition_number: float
    singular_values: np.ndarray
    effective_rank: int
    near_collinear_pairs: List[Tuple[str, str, float]]
    bic: float
    edf: Optional[float] = None  # None for linear processors
    identifiability_grade: str = "unknown"

    # Grade thresholds: kappa < 30 = good, 30-100 = moderate, > 100 = poor

    def __post_init__(self):
        if self.identifiability_grade == "unknown":
            self.identifiability_grade = self._compute_grade()

    def _compute_grade(self) -> str:
        if self.condition_number < 30:
            return "good"
        elif self.condition_number < 100:
            return "moderate"
        else:
            return "poor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_number": self.condition_number,
            "singular_values": self.singular_values.tolist()
            if isinstance(self.singular_values, np.ndarray)
            else list(self.singular_values),
            "effective_rank": self.effective_rank,
            "near_collinear_pairs": self.near_collinear_pairs,
            "bic": self.bic,
            "edf": self.edf,
            "identifiability_grade": self.identifiability_grade,
        }

    def summary(self) -> str:
        lines = [
            f"  Condition Number (kappa): {self.condition_number:.1f}",
            f"  Effective Rank: {self.effective_rank}",
            f"  BIC: {self.bic:.2f}",
        ]
        if self.edf is not None:
            lines.append(f"  EDF (Hutchinson): {self.edf:.1f}")
        if self.near_collinear_pairs:
            lines.append(f"  Near-Collinear Pairs: {len(self.near_collinear_pairs)}")
            for f1, f2, corr in self.near_collinear_pairs[:3]:
                lines.append(f"    {f1} <-> {f2}: r={corr:.3f}")
        lines.append(f"  Grade: {self.identifiability_grade.upper()}")
        return "\n".join(lines)


def compute_full_identifiability_report(
    X: np.ndarray,
    A: np.ndarray,
    mb_indices: List[int],
    feature_names: Optional[List[str]] = None,
    processors: Any = None,
    processor_type: str = "linear",
    predict_fn: Optional[Callable] = None,
    key: Any = None,
) -> IdentifiabilityReport:
    """
    Compute full identifiability report for a single DAG.

    Args:
        X: Data matrix (n_samples, n_features)
        A: Adjacency matrix
        mb_indices: Markov Blanket feature indices
        feature_names: Optional feature names
        processors: Trained processors (optional)
        processor_type: Type of processor used
        predict_fn: Prediction function for neural BIC (optional)
        key: JAX PRNG key (optional)

    Returns:
        IdentifiabilityReport with all diagnostics
    """
    X = np.asarray(X)
    A = np.asarray(A)

    # SVD diagnostics (includes condition number)
    mb_names = None
    if feature_names is not None and len(mb_indices) > 0:
        mb_names = [feature_names[i] if i < len(feature_names) else f"X{i}" for i in mb_indices]

    svd_diag = compute_svd_diagnostics(X, mb_indices, feature_names=mb_names)

    # BIC
    bic = compute_bic_for_dag(
        X,
        A,
        processors=processors,
        processor_type=processor_type,
        predict_fn=predict_fn,
        key=key,
    )

    # EDF for neural processors
    edf = None
    if processor_type != "linear" and predict_fn is not None and HAS_JAX:
        try:
            X_jax = jnp.asarray(X)
            edf = hutchinson_edf(predict_fn, X_jax, n_probes=5, key=key)
        except Exception:
            pass

    return IdentifiabilityReport(
        condition_number=svd_diag["condition_number"],
        singular_values=svd_diag["singular_values"],
        effective_rank=svd_diag["effective_rank"],
        near_collinear_pairs=svd_diag["near_collinear_pairs"],
        bic=bic,
        edf=edf,
    )


# =============================================================================
# 7. Identifiability Summary (Cross-DAG)
# =============================================================================


@dataclass
class IdentifiabilitySummary:
    """Aggregated identifiability diagnostics across Pareto DAGs."""

    reports: List[IdentifiabilityReport]

    @property
    def n_dags(self) -> int:
        return len(self.reports)

    @property
    def bic_values(self) -> np.ndarray:
        return np.array([r.bic for r in self.reports])

    @property
    def condition_numbers(self) -> np.ndarray:
        return np.array([r.condition_number for r in self.reports])

    @property
    def best_bic_idx(self) -> int:
        if len(self.reports) == 0:
            return -1
        return int(np.argmin(self.bic_values))

    @property
    def grades(self) -> List[str]:
        return [r.identifiability_grade for r in self.reports]

    def to_dict(self) -> Dict[str, Any]:
        bics = self.bic_values
        kappas = self.condition_numbers
        return {
            "n_dags": self.n_dags,
            "bic_best": float(np.min(bics)) if len(bics) > 0 else float("nan"),
            "bic_median": float(np.median(bics)) if len(bics) > 0 else float("nan"),
            "bic_worst": float(np.max(bics)) if len(bics) > 0 else float("nan"),
            "condition_number_mean": float(np.mean(kappas)) if len(kappas) > 0 else float("nan"),
            "condition_number_max": float(np.max(kappas)) if len(kappas) > 0 else float("nan"),
            "grades": self.grades,
            "best_bic_dag_idx": self.best_bic_idx,
            "per_dag": [r.to_dict() for r in self.reports],
        }

    def summary(self) -> str:
        if len(self.reports) == 0:
            return "No identifiability reports available."

        bics = self.bic_values
        kappas = self.condition_numbers
        grade_counts = {}
        for g in self.grades:
            grade_counts[g] = grade_counts.get(g, 0) + 1

        lines = [
            "",
            "5. IDENTIFIABILITY DIAGNOSTICS",
            f"   DAGs analyzed: {self.n_dags}",
            f"   BIC landscape: best={np.min(bics):.1f}, median={np.median(bics):.1f}, worst={np.max(bics):.1f}",
            f"   Best BIC DAG: #{self.best_bic_idx}",
            f"   Condition numbers: mean={np.mean(kappas):.1f}, max={np.max(kappas):.1f}",
            "   Grades: " + ", ".join(f"{g}={c}" for g, c in sorted(grade_counts.items())),
        ]

        # Collinear features across DAGs
        all_collinear = set()
        for r in self.reports:
            for f1, f2, _ in r.near_collinear_pairs:
                all_collinear.add((f1, f2))
        if all_collinear:
            lines.append(f"   Collinear pairs found: {len(all_collinear)}")
            for f1, f2 in list(all_collinear)[:3]:
                lines.append(f"     {f1} <-> {f2}")

        return "\n".join(lines)
