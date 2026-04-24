#!/usr/bin/env python3
"""
Validate JCCE's AIPW/DML implementation against EconML.

One-time validation script. Generates synthetic data with known ATE,
runs both our DMLCrossFitter and EconML's LinearDML, compares:
- Point estimates (ATE)
- Confidence interval width
- Coverage (does CI contain true ATE?)

Usage:
    uv run python scripts/validate_aipw_against_econml.py
    uv run python scripts/validate_aipw_against_econml.py --n-sims 50

Requirements:
    pip install econml  (not a core dependency)
"""

import argparse
import numpy as np
import sys
from typing import Dict, Tuple


def generate_synthetic_ate_data(
    n: int = 1000,
    p: int = 5,
    true_ate: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate data with known ATE for validation.

    DGP: Y = true_ate * T + X @ beta + noise
         T = sigmoid(X @ gamma + noise) > 0.5

    Args:
        n: Number of samples
        p: Number of covariates
        true_ate: Ground truth average treatment effect
        seed: Random seed

    Returns:
        (X, T, Y, true_ate)
    """
    rng = np.random.RandomState(seed)

    X = rng.randn(n, p)
    beta = rng.randn(p) * 0.3
    gamma = rng.randn(p) * 0.5

    # Treatment assignment (confounded by X)
    propensity = 1.0 / (1.0 + np.exp(-(X @ gamma + rng.randn(n) * 0.5)))
    T = (propensity > 0.5).astype(float)

    # Outcome (depends on T and X)
    Y = true_ate * T + X @ beta + rng.randn(n) * 0.3

    return X, T, Y, true_ate


def run_jcce_dml(X, T, Y, treatment_idx=0) -> Dict:
    """Run JCCE's DMLCrossFitter."""
    from jcce.validation.dml_crossfitting import (
        DMLCrossFitter,
        create_simple_nuisance_functions,
    )

    dml = DMLCrossFitter(n_splits=5, n_repeats=5, random_state=42, stratify=True)
    train_fn, predict_fn = create_simple_nuisance_functions()

    result = dml.estimate_ate(
        X=X, T=T, Y=Y,
        train_nuisance_fn=train_fn,
        predict_nuisance_fn=predict_fn,
        treatment_idx=treatment_idx,
    )

    ci = result.confidence_interval(0.05)
    return {
        'ate': result.ate_mean,
        'std': result.ate_std,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'ci_width': ci[1] - ci[0],
    }


def run_econml_dml(X, T, Y) -> Dict:
    """Run EconML's LinearDRLearner (doubly-robust for binary treatment)."""
    try:
        from econml.dr import LinearDRLearner
        from sklearn.linear_model import LogisticRegressionCV, LassoCV
    except ImportError:
        print("ERROR: econml not installed. Run: pip install econml")
        sys.exit(1)

    est = LinearDRLearner(
        model_propensity=LogisticRegressionCV(max_iter=1000),
        model_regression=LassoCV(),
        random_state=42,
    )
    est.fit(Y, T, X=X)

    ate = float(est.ate(X))
    ci = est.ate_interval(X, alpha=0.05)
    ci_lower, ci_upper = float(ci[0]), float(ci[1])

    return {
        'ate': ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate AIPW against EconML')
    parser.add_argument('--n-sims', type=int, default=20,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Samples per simulation')
    parser.add_argument('--true-ate', type=float, default=0.5,
                        help='Ground truth ATE')
    args = parser.parse_args()

    print("AIPW VALIDATION: JCCE vs EconML")
    print("=" * 60)
    print(f"Monte Carlo simulations: {args.n_sims}")
    print(f"Samples per sim: {args.n_samples}")
    print(f"True ATE: {args.true_ate}")
    print()

    jcce_ates, econml_ates = [], []
    jcce_coverage, econml_coverage = 0, 0
    jcce_widths, econml_widths = [], []

    for sim in range(args.n_sims):
        X, T, Y, true_ate = generate_synthetic_ate_data(
            n=args.n_samples, true_ate=args.true_ate, seed=sim
        )

        # JCCE
        jcce_result = run_jcce_dml(X, T, Y)
        jcce_ates.append(jcce_result['ate'])
        jcce_widths.append(jcce_result['ci_width'])
        if jcce_result['ci_lower'] <= true_ate <= jcce_result['ci_upper']:
            jcce_coverage += 1

        # EconML
        econml = run_econml_dml(X, T, Y)
        econml_ates.append(econml['ate'])
        econml_widths.append(econml['ci_width'])
        if econml['ci_lower'] <= true_ate <= econml['ci_upper']:
            econml_coverage += 1

        if (sim + 1) % 5 == 0:
            print(f"  Sim {sim+1}/{args.n_sims}: "
                  f"JCCE={jcce_result['ate']:.4f} EconML={econml['ate']:.4f}")

    # Summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'JCCE':<15} {'EconML':<15}")
    print("-" * 55)
    print(f"{'Mean ATE':<25} {np.mean(jcce_ates):<15.4f} {np.mean(econml_ates):<15.4f}")
    print(f"{'Std ATE':<25} {np.std(jcce_ates):<15.4f} {np.std(econml_ates):<15.4f}")
    print(f"{'Bias (ATE - true)':<25} {np.mean(jcce_ates)-args.true_ate:<15.4f} "
          f"{np.mean(econml_ates)-args.true_ate:<15.4f}")
    print(f"{'RMSE':<25} "
          f"{np.sqrt(np.mean((np.array(jcce_ates)-args.true_ate)**2)):<15.4f} "
          f"{np.sqrt(np.mean((np.array(econml_ates)-args.true_ate)**2)):<15.4f}")
    print(f"{'Coverage (95% CI)':<25} "
          f"{jcce_coverage/args.n_sims:<15.1%} "
          f"{econml_coverage/args.n_sims:<15.1%}")
    print(f"{'Mean CI width':<25} {np.mean(jcce_widths):<15.4f} {np.mean(econml_widths):<15.4f}")
    print()

    # Agreement
    corr = np.corrcoef(jcce_ates, econml_ates)[0, 1]
    mean_abs_diff = np.mean(np.abs(np.array(jcce_ates) - np.array(econml_ates)))
    print(f"Correlation (JCCE vs EconML): {corr:.4f}")
    print(f"Mean |ATE_jcce - ATE_econml|: {mean_abs_diff:.4f}")
    print()

    # Verdict
    if corr > 0.9 and mean_abs_diff < 0.1:
        print("VERDICT: JCCE and EconML implementations AGREE closely.")
    elif corr > 0.7:
        print("VERDICT: JCCE and EconML show MODERATE agreement. "
              "Differences may be due to nuisance model choices.")
    else:
        print("WARNING: JCCE and EconML show LOW agreement. Investigate!")

    print("=" * 60)


if __name__ == '__main__':
    main()
