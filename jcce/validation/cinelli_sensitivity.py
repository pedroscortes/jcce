"""
Cinelli & Hazlett (2020) Sensitivity Analysis for Unmeasured Confounding.

Implements the partial R²-based sensitivity framework for OLS/linear estimands.
Quantifies how strong an unobserved confounder would need to be to explain away
a causal effect — without specifying the confounder's distribution.

Key concepts:
- RV (Robustness Value): How much partial R² an unobserved confounder needs
  to reduce the estimate to zero
- Bias-adjusted bounds: Given assumed confounder strength, what is the maximum
  bias that could be introduced?

Reference:
    Cinelli, C. & Hazlett, C. (2020). "Making Sense of Sensitivity:
    Extending Omitted Variable Bias." JRSSB, 82(1), 39-67.

Usage:
    from jcce.validation.cinelli_sensitivity import (
        cinelli_sensitivity,
        SensitivityResult,
        robustness_value,
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SensitivityResult:
    """Result of Cinelli sensitivity analysis for one treatment variable."""

    treatment_name: str
    ate_estimate: float
    ate_se: float

    # Robustness value: confounder partial R² needed to nullify effect
    rv: float
    rv_alpha: float  # RV at significance level (accounts for SE)

    # Bias-adjusted estimates at benchmark confounder strengths
    benchmarks: Dict[
        str, Dict[str, float]
    ]  # name -> {r2_Y, r2_T, adj_estimate, adj_ci_lower, adj_ci_upper}

    # Raw partial R² values
    partial_r2_treatment: float  # R²_{Y~T|X}
    residual_dof: int

    def is_robust(self, alpha: float = 0.05) -> bool:
        """Check if the estimate is robust to moderate confounding."""
        return self.rv_alpha > 0.0

    def to_dict(self) -> Dict:
        return {
            "treatment_name": self.treatment_name,
            "ate_estimate": self.ate_estimate,
            "ate_se": self.ate_se,
            "rv": self.rv,
            "rv_alpha": self.rv_alpha,
            "partial_r2_treatment": self.partial_r2_treatment,
            "benchmarks": self.benchmarks,
            "is_robust": self.is_robust(),
        }

    def summary(self) -> str:
        lines = [
            f"Sensitivity Analysis: {self.treatment_name}",
            f"  ATE = {self.ate_estimate:.4f} (SE = {self.ate_se:.4f})",
            f"  Partial R²(Y~T|X) = {self.partial_r2_treatment:.4f}",
            f"  RV = {self.rv:.4f} (confounder needs this much partial R² to nullify)",
            f"  RV_α = {self.rv_alpha:.4f} (to lose significance)",
            f"  Robust: {'YES' if self.is_robust() else 'NO'}",
        ]
        if self.benchmarks:
            lines.append("  Benchmarks:")
            for name, b in self.benchmarks.items():
                lines.append(
                    f"    {name}: R²_Y={b['r2_Y']:.3f}, R²_T={b['r2_T']:.3f} "
                    f"-> ATE_adj={b['adj_estimate']:.4f} "
                    f"[{b['adj_ci_lower']:.4f}, {b['adj_ci_upper']:.4f}]"
                )
        return "\n".join(lines)


# ============================================================================
# Core Computations
# ============================================================================


def partial_r2(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> float:
    """
    Compute partial R²(Y ~ T | X).

    Partial R² = (SSR_restricted - SSR_full) / SSR_restricted
    where restricted model has X only, full model has X + T.

    Args:
        Y: Outcome (n,)
        T: Treatment (n,)
        X: Covariates (n, p)

    Returns:
        Partial R² in [0, 1]
    """
    n = len(Y)
    Y = Y.flatten()
    T = T.flatten()

    # Restricted model: Y ~ X
    X_with_intercept = np.column_stack([np.ones(n), X])
    beta_r = np.linalg.lstsq(X_with_intercept, Y, rcond=None)[0]
    resid_r = Y - X_with_intercept @ beta_r
    ssr_restricted = np.sum(resid_r**2)

    # Full model: Y ~ X + T
    X_full = np.column_stack([X_with_intercept, T])
    beta_f = np.linalg.lstsq(X_full, Y, rcond=None)[0]
    resid_f = Y - X_full @ beta_f
    ssr_full = np.sum(resid_f**2)

    if ssr_restricted < 1e-15:
        return 0.0

    return float((ssr_restricted - ssr_full) / ssr_restricted)


def robustness_value(
    partial_r2_T: float,
    ate: float,
    se: float,
    dof: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute Robustness Value (RV).

    RV_q: minimum confounder strength (partial R²) needed to reduce
    the point estimate to q * original_estimate. For q=0: reduce to zero.

    RV_α: minimum confounder strength to lose significance at level α.

    Based on Cinelli & Hazlett (2020) Proposition 2.

    Args:
        partial_r2_T: R²(Y~T|X)
        ate: ATE point estimate
        se: Standard error of ATE
        dof: Residual degrees of freedom
        alpha: Significance level

    Returns:
        (rv_0, rv_alpha): Robustness values for nullification and significance
    """
    from scipy import stats

    # RV for point estimate = 0 (Proposition 2, Eq. 5)
    # RV = 0.5 * (sqrt(f^4 + 4*f^2) - f^2)
    # where f = t-statistic = ate/se
    t_stat = abs(ate / se) if se > 1e-15 else 0.0
    f2 = t_stat**2 / dof if dof > 0 else t_stat**2

    rv_0 = 0.5 * (np.sqrt(f2**2 + 4 * f2) - f2) if f2 > 0 else 0.0

    # RV for significance (Proposition 2, Eq. 6; matches sensemakr R package)
    # f_alpha = max(|t| - t_crit, 0), then f2_alpha = f_alpha^2 / dof
    # Note: (|t| - t_crit)^2 != t^2 - t_crit^2 — the old code used the latter (wrong)
    t_crit = stats.t.ppf(1 - alpha / 2, dof) if dof > 0 else 1.96
    f_alpha = max(t_stat - t_crit, 0.0)
    f2_alpha = f_alpha**2 / dof if dof > 0 else f_alpha**2

    rv_alpha = 0.5 * (np.sqrt(f2_alpha**2 + 4 * f2_alpha) - f2_alpha) if f2_alpha > 0 else 0.0

    return float(rv_0), float(rv_alpha)


def bias_adjusted_estimate(
    ate: float,
    se: float,
    r2_Y_confounder: float,
    r2_T_confounder: float,
    partial_r2_T: float,
    dof: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute bias-adjusted ATE given assumed confounder strength.

    The maximum bias from an omitted variable with specified partial R² values is:
        bias = se * sqrt(dof) * sqrt(r2_Y_confounder * r2_T_confounder / (1 - r2_T_confounder))

    Based on Cinelli & Hazlett (2020) Theorem 1.

    Args:
        ate: Original ATE estimate
        se: Standard error
        r2_Y_confounder: R²(Y ~ U | X, T) — confounder's partial R² on Y
        r2_T_confounder: R²(T ~ U | X) — confounder's partial R² on T
        partial_r2_T: R²(Y ~ T | X) — treatment's partial R²
        dof: Residual degrees of freedom
        alpha: Significance level

    Returns:
        Dict with adj_estimate, adj_se, adj_ci_lower, adj_ci_upper, max_bias
    """
    from scipy import stats

    # Maximum bias (Theorem 1)
    if r2_T_confounder >= 1.0:
        r2_T_confounder = 0.999

    bias_factor = np.sqrt(r2_Y_confounder * r2_T_confounder / (1 - r2_T_confounder))
    max_bias = se * np.sqrt(max(dof, 1)) * bias_factor

    # Bias direction: worst case reduces magnitude
    if ate >= 0:
        adj_estimate = ate - max_bias
    else:
        adj_estimate = ate + max_bias

    # Adjusted SE (conservative)
    adj_se = se * np.sqrt(1 + r2_Y_confounder / (1 - r2_T_confounder))

    # CI
    t_crit = stats.t.ppf(1 - alpha / 2, max(dof, 1))
    adj_ci_lower = adj_estimate - t_crit * adj_se
    adj_ci_upper = adj_estimate + t_crit * adj_se

    return {
        "adj_estimate": float(adj_estimate),
        "adj_se": float(adj_se),
        "adj_ci_lower": float(adj_ci_lower),
        "adj_ci_upper": float(adj_ci_upper),
        "max_bias": float(max_bias),
        "r2_Y": float(r2_Y_confounder),
        "r2_T": float(r2_T_confounder),
    }


# ============================================================================
# Benchmark Generation
# ============================================================================


def compute_covariate_benchmarks(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k_multipliers: Tuple[float, ...] = (1.0, 2.0),
) -> Dict[str, Dict[str, float]]:
    """
    Use observed covariates as benchmarks for confounder strength.

    For each covariate X_j, compute its partial R² on Y and T,
    then use k * R²_j as hypothetical confounder strength.

    Args:
        Y: Outcome (n,)
        T: Treatment (n,)
        X: Covariates (n, p)
        feature_names: Names for covariates
        k_multipliers: How many times stronger the confounder might be

    Returns:
        Dict mapping benchmark_name -> {r2_Y, r2_T}
    """
    n, p = X.shape
    if feature_names is None:
        feature_names = [f"X{j}" for j in range(p)]

    benchmarks = {}

    for j in range(p):
        X_j = X[:, j]
        X_others = np.delete(X, j, axis=1)

        # R²(Y ~ X_j | X_others, T)
        X_base = np.column_stack([np.ones(n), X_others, T.flatten()])
        X_full = np.column_stack([X_base, X_j])
        beta_r = np.linalg.lstsq(X_base, Y.flatten(), rcond=None)[0]
        beta_f = np.linalg.lstsq(X_full, Y.flatten(), rcond=None)[0]
        ssr_r = np.sum((Y.flatten() - X_base @ beta_r) ** 2)
        ssr_f = np.sum((Y.flatten() - X_full @ beta_f) ** 2)
        r2_Y_j = (ssr_r - ssr_f) / max(ssr_r, 1e-15)

        # R²(T ~ X_j | X_others)
        X_base_t = np.column_stack([np.ones(n), X_others])
        X_full_t = np.column_stack([X_base_t, X_j])
        beta_r_t = np.linalg.lstsq(X_base_t, T.flatten(), rcond=None)[0]
        beta_f_t = np.linalg.lstsq(X_full_t, T.flatten(), rcond=None)[0]
        ssr_r_t = np.sum((T.flatten() - X_base_t @ beta_r_t) ** 2)
        ssr_f_t = np.sum((T.flatten() - X_full_t @ beta_f_t) ** 2)
        r2_T_j = (ssr_r_t - ssr_f_t) / max(ssr_r_t, 1e-15)

        for k in k_multipliers:
            label = f"{k:.0f}x {feature_names[j]}" if k != 1.0 else feature_names[j]
            benchmarks[label] = {
                "r2_Y": float(min(r2_Y_j * k, 0.99)),
                "r2_T": float(min(r2_T_j * k, 0.99)),
            }

    return benchmarks


# ============================================================================
# Main API
# ============================================================================


def cinelli_sensitivity(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    treatment_name: str = "T",
    feature_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    benchmark_multipliers: Tuple[float, ...] = (1.0, 2.0),
    custom_benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
) -> SensitivityResult:
    """
    Run Cinelli & Hazlett (2020) sensitivity analysis.

    Args:
        Y: Outcome variable (n,)
        T: Treatment variable (n,) — can be continuous or binary
        X: Covariates/confounders (n, p)
        treatment_name: Name for treatment variable
        feature_names: Names for covariates (for benchmark labels)
        alpha: Significance level
        benchmark_multipliers: How many times stronger hypothetical confounder is
        custom_benchmarks: Optional custom benchmark confounder strengths

    Returns:
        SensitivityResult with RV, bias-adjusted estimates, and benchmarks
    """
    Y = np.asarray(Y).flatten().astype(np.float64)
    T = np.asarray(T).flatten().astype(np.float64)
    X = np.asarray(X).astype(np.float64)
    n, p = X.shape

    # Fit OLS: Y ~ T + X
    X_design = np.column_stack([np.ones(n), T, X])
    beta = np.linalg.lstsq(X_design, Y, rcond=None)[0]
    ate = float(beta[1])  # Coefficient on T
    residuals = Y - X_design @ beta
    dof = n - X_design.shape[1]
    mse = np.sum(residuals**2) / max(dof, 1)
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)
    se = float(np.sqrt(mse * XtX_inv[1, 1]))

    # Partial R² of treatment
    pr2_T = partial_r2(Y, T, X)

    # Robustness values
    rv_0, rv_a = robustness_value(pr2_T, ate, se, dof, alpha)

    # Benchmarks
    if custom_benchmarks is not None:
        benchmarks_raw = custom_benchmarks
    else:
        benchmarks_raw = compute_covariate_benchmarks(Y, T, X, feature_names, benchmark_multipliers)

    # Compute bias-adjusted estimates for each benchmark
    benchmarks = {}
    for name, bm in benchmarks_raw.items():
        adj = bias_adjusted_estimate(ate, se, bm["r2_Y"], bm["r2_T"], pr2_T, dof, alpha)
        benchmarks[name] = adj

    return SensitivityResult(
        treatment_name=treatment_name,
        ate_estimate=ate,
        ate_se=se,
        rv=rv_0,
        rv_alpha=rv_a,
        benchmarks=benchmarks,
        partial_r2_treatment=pr2_T,
        residual_dof=dof,
    )
