"""
Metrics for Evaluating Causal Effect Estimation Quality.

This module implements standard metrics used in the causal inference literature
for evaluating treatment effect estimation, particularly on semi-synthetic
datasets where ground truth effects are known.

Primary Metrics:
- PEHE: Precision in Estimation of Heterogeneous Effects
- ATE Error: Error in Average Treatment Effect estimation

Secondary Metrics:
- Policy Risk: Quality of treatment decisions
- ATT Error: Error on treatment effect for the treated

References:
- Hill (2011): Bayesian Nonparametric Modeling for Causal Inference
- Shalit et al. (2017): IHDP benchmark
"""

from dataclasses import dataclass
from typing import Dict, Optional

import jax.numpy as jnp

# ============================================================================
# Primary Metrics
# ============================================================================


def compute_pehe(tau_pred: jnp.ndarray, tau_true: jnp.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effects (PEHE).

    PEHE = sqrt(mean((tau_pred - tau_true)^2))

    This is the primary metric for evaluating CATE estimation quality.
    Lower is better. Standard benchmark target: < 0.5

    Args:
        tau_pred: Predicted individual treatment effects (n,)
        tau_true: True individual treatment effects (n,)

    Returns:
        PEHE value (scalar)
    """
    return float(jnp.sqrt(jnp.mean((tau_pred - tau_true) ** 2)))


def compute_pehe_squared(tau_pred: jnp.ndarray, tau_true: jnp.ndarray) -> float:
    """
    Squared PEHE (for optimization, more stable gradients).

    PEHE^2 = mean((tau_pred - tau_true)^2)

    Args:
        tau_pred: Predicted individual treatment effects
        tau_true: True individual treatment effects

    Returns:
        Squared PEHE value
    """
    return float(jnp.mean((tau_pred - tau_true) ** 2))


def compute_ate_error(tau_pred: jnp.ndarray, ate_true: float) -> float:
    """
    Absolute error in Average Treatment Effect estimation.

    ATE Error = |mean(tau_pred) - ATE_true|

    Lower is better. Standard benchmark target: < 0.1

    Args:
        tau_pred: Predicted individual treatment effects
        ate_true: True average treatment effect

    Returns:
        Absolute ATE error
    """
    ate_pred = jnp.mean(tau_pred)
    return float(jnp.abs(ate_pred - ate_true))


def compute_ate_bias(tau_pred: jnp.ndarray, ate_true: float) -> float:
    """
    Signed bias in ATE (can be positive or negative).

    ATE Bias = mean(tau_pred) - ATE_true

    Positive: overestimating effect
    Negative: underestimating effect

    Args:
        tau_pred: Predicted individual treatment effects
        ate_true: True average treatment effect

    Returns:
        Signed ATE bias
    """
    ate_pred = jnp.mean(tau_pred)
    return float(ate_pred - ate_true)


# ============================================================================
# Secondary Metrics
# ============================================================================


def compute_att_error(tau_pred: jnp.ndarray, tau_true: jnp.ndarray, T: jnp.ndarray) -> float:
    """
    Error in Average Treatment Effect on the Treated (ATT).

    ATT = E[Y(1) - Y(0) | T=1]

    Args:
        tau_pred: Predicted individual treatment effects
        tau_true: True individual treatment effects
        T: Treatment assignments

    Returns:
        Absolute ATT error
    """
    treated_mask = T > 0.5
    n_treated = jnp.sum(treated_mask)

    if n_treated == 0:
        return 0.0

    att_pred = jnp.sum(tau_pred * treated_mask) / n_treated
    att_true = jnp.sum(tau_true * treated_mask) / n_treated

    return float(jnp.abs(att_pred - att_true))


def compute_policy_risk(
    tau_pred: jnp.ndarray, tau_true: jnp.ndarray, cost_ratio: float = 1.0
) -> float:
    """
    Policy risk: measures quality of treatment decisions.

    If tau_pred > 0, we recommend treatment.
    Policy risk = expected regret from wrong decisions.

    Args:
        tau_pred: Predicted CATE
        tau_true: True ITE
        cost_ratio: Cost of treating someone who shouldn't be treated
                    vs not treating someone who should be

    Returns:
        Policy risk value (lower is better)
    """
    # Our treatment decision based on predicted effects
    treat_decision = (tau_pred > 0).astype(jnp.float32)

    # Optimal decision based on true effects
    optimal_decision = (tau_true > 0).astype(jnp.float32)

    # False positives: we treat, but shouldn't (tau_true <= 0)
    false_positive = treat_decision * (1 - optimal_decision)
    fp_cost = cost_ratio * jnp.abs(tau_true) * false_positive

    # False negatives: we don't treat, but should (tau_true > 0)
    false_negative = (1 - treat_decision) * optimal_decision
    fn_cost = jnp.abs(tau_true) * false_negative

    return float(jnp.mean(fp_cost + fn_cost))


def compute_policy_value(tau_pred: jnp.ndarray, tau_true: jnp.ndarray) -> float:
    """
    Policy value: expected outcome improvement from following predictions.

    Higher is better.

    Args:
        tau_pred: Predicted CATE
        tau_true: True ITE

    Returns:
        Expected policy value
    """
    # Treatment decision based on predictions
    treat_decision = (tau_pred > 0).astype(jnp.float32)

    # Expected value: treat when tau_true > 0
    value = jnp.mean(tau_true * treat_decision)

    return float(value)


# ============================================================================
# Diagnostic Metrics
# ============================================================================


def compute_propensity_diagnostics(
    propensity_pred: jnp.ndarray, T: jnp.ndarray, propensity_true: Optional[jnp.ndarray] = None
) -> Dict[str, float]:
    """
    Compute diagnostics for propensity score estimation.

    Args:
        propensity_pred: Predicted P(T=1|X)
        T: True treatment assignments
        propensity_true: True propensity scores (if available)

    Returns:
        Dictionary of diagnostic metrics
    """
    p = propensity_pred.squeeze()

    diagnostics = {
        "propensity_mean": float(jnp.mean(p)),
        "propensity_std": float(jnp.std(p)),
        "propensity_min": float(jnp.min(p)),
        "propensity_max": float(jnp.max(p)),
        "propensity_variance": float(jnp.var(p)),
    }

    # Check for collapse (all predictions similar)
    diagnostics["propensity_collapsed"] = diagnostics["propensity_variance"] < 0.01

    # Propensity BCE (same as loss)
    eps = 1e-7
    p_clipped = jnp.clip(p, eps, 1 - eps)
    bce = -jnp.mean(T * jnp.log(p_clipped) + (1 - T) * jnp.log(1 - p_clipped))
    diagnostics["propensity_bce"] = float(bce)

    # AUC approximation (correlation with treatment)
    # This is a rough proxy; true AUC requires sklearn
    correlation = jnp.corrcoef(p, T)[0, 1]
    diagnostics["propensity_treatment_corr"] = (
        float(correlation) if not jnp.isnan(correlation) else 0.0
    )

    # Compare to true propensity if available
    if propensity_true is not None:
        p_true = propensity_true.squeeze()
        diagnostics["propensity_mse"] = float(jnp.mean((p - p_true) ** 2))
        diagnostics["propensity_mae"] = float(jnp.mean(jnp.abs(p - p_true)))

    return diagnostics


def compute_effect_heterogeneity(tau_pred: jnp.ndarray) -> Dict[str, float]:
    """
    Compute metrics about effect heterogeneity.

    Args:
        tau_pred: Predicted individual treatment effects

    Returns:
        Dictionary of heterogeneity metrics
    """
    return {
        "cate_mean": float(jnp.mean(tau_pred)),
        "cate_std": float(jnp.std(tau_pred)),
        "cate_min": float(jnp.min(tau_pred)),
        "cate_max": float(jnp.max(tau_pred)),
        "cate_iqr": float(jnp.percentile(tau_pred, 75) - jnp.percentile(tau_pred, 25)),
        "cate_positive_frac": float(jnp.mean(tau_pred > 0)),
    }


# ============================================================================
# Comprehensive Evaluation
# ============================================================================


@dataclass
class EffectEvaluationResults:
    """Container for comprehensive evaluation results."""

    # Primary metrics
    PEHE: float
    PEHE_squared: float
    ATE_error: float
    ATE_bias: float
    ATE_pred: float
    ATE_true: float

    # Secondary metrics
    ATT_error: float
    policy_risk: float
    policy_value: float

    # Heterogeneity metrics
    cate_std_pred: float
    cate_std_true: float

    # Propensity diagnostics
    propensity_diagnostics: Dict[str, float]


def evaluate_effect_estimation(
    tau_pred: jnp.ndarray,
    tau_true: jnp.ndarray,
    ate_true: float,
    T: jnp.ndarray,
    propensity_pred: Optional[jnp.ndarray] = None,
    propensity_true: Optional[jnp.ndarray] = None,
) -> EffectEvaluationResults:
    """
    Comprehensive evaluation of effect estimation quality.

    Args:
        tau_pred: Predicted individual treatment effects
        tau_true: True individual treatment effects
        ate_true: True average treatment effect
        T: Treatment assignments
        propensity_pred: Predicted propensity scores (optional)
        propensity_true: True propensity scores (optional)

    Returns:
        EffectEvaluationResults with all metrics
    """
    # Primary metrics
    pehe = compute_pehe(tau_pred, tau_true)
    pehe_squared = compute_pehe_squared(tau_pred, tau_true)
    ate_error = compute_ate_error(tau_pred, ate_true)
    ate_bias = compute_ate_bias(tau_pred, ate_true)
    ate_pred = float(jnp.mean(tau_pred))

    # Secondary metrics
    att_error = compute_att_error(tau_pred, tau_true, T)
    policy_risk = compute_policy_risk(tau_pred, tau_true)
    policy_value = compute_policy_value(tau_pred, tau_true)

    # Heterogeneity
    cate_std_pred = float(jnp.std(tau_pred))
    cate_std_true = float(jnp.std(tau_true))

    # Propensity diagnostics
    if propensity_pred is not None:
        prop_diag = compute_propensity_diagnostics(propensity_pred, T, propensity_true)
    else:
        prop_diag = {}

    return EffectEvaluationResults(
        PEHE=pehe,
        PEHE_squared=pehe_squared,
        ATE_error=ate_error,
        ATE_bias=ate_bias,
        ATE_pred=ate_pred,
        ATE_true=ate_true,
        ATT_error=att_error,
        policy_risk=policy_risk,
        policy_value=policy_value,
        cate_std_pred=cate_std_pred,
        cate_std_true=cate_std_true,
        propensity_diagnostics=prop_diag,
    )


def format_evaluation_report(
    results: EffectEvaluationResults, dataset_name: str = "unknown", method_name: str = "JCCE"
) -> str:
    """
    Format evaluation results as a printable report.

    Args:
        results: EffectEvaluationResults object
        dataset_name: Name of the dataset
        method_name: Name of the method

    Returns:
        Formatted string report
    """
    # Determine pass/fail for primary metrics
    pehe_status = "PASS" if results.PEHE < 0.5 else "FAIL"
    ate_status = "PASS" if results.ATE_error < 0.15 else "FAIL"

    report = f"""
================================================================================
EFFECT ESTIMATION RESULTS
================================================================================
Dataset: {dataset_name}
Method: {method_name}

PRIMARY METRICS:
  PEHE:           {results.PEHE:.4f} (target: < 0.5) [{pehe_status}]
  ATE Error:      {results.ATE_error:.4f} (target: < 0.15) [{ate_status}]
  ATE Bias:       {results.ATE_bias:+.4f}

EFFECT ESTIMATES:
  ATE (predicted): {results.ATE_pred:.4f}
  ATE (true):      {results.ATE_true:.4f}
  CATE std (pred): {results.cate_std_pred:.4f}
  CATE std (true): {results.cate_std_true:.4f}

SECONDARY METRICS:
  ATT Error:      {results.ATT_error:.4f}
  Policy Risk:    {results.policy_risk:.4f}
  Policy Value:   {results.policy_value:.4f}
"""

    if results.propensity_diagnostics:
        pd = results.propensity_diagnostics
        prop_status = "COLLAPSED" if pd.get("propensity_collapsed", False) else "OK"
        report += f"""
PROPENSITY DIAGNOSTICS:
  Mean:           {pd.get("propensity_mean", 0):.4f}
  Std:            {pd.get("propensity_std", 0):.4f}
  Variance:       {pd.get("propensity_variance", 0):.4f} [{prop_status}]
  BCE Loss:       {pd.get("propensity_bce", 0):.4f}
"""

    report += "=" * 80

    return report


# ============================================================================
# Benchmark Comparison Utilities
# ============================================================================


def compare_methods(results_dict: Dict[str, EffectEvaluationResults]) -> str:
    """
    Generate comparison table for multiple methods.

    Args:
        results_dict: Dictionary mapping method names to results

    Returns:
        Formatted comparison table
    """
    header = f"{'Method':<20} {'PEHE':>10} {'ATE Error':>12} {'Policy Risk':>12}"
    separator = "-" * 56

    rows = [header, separator]

    for method_name, results in results_dict.items():
        row = f"{method_name:<20} {results.PEHE:>10.4f} {results.ATE_error:>12.4f} {results.policy_risk:>12.4f}"
        rows.append(row)

    rows.append(separator)

    return "\n".join(rows)
