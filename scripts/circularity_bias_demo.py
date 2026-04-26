#!/usr/bin/env python3
"""
Circularity bias demonstration: PC+DML on same data vs.\ honest 70/30 split.

Shows empirically that reusing the same data for DAG discovery and ATE
estimation biases the resulting effect estimates and undercovers their
confidence intervals, while a JCCE-style 70/30 split preserves nominal
coverage.

Setup:
  - Synthetic linear SEM with known ATE
  - Method A (Naive): PC on 100% data -> DML on same 100%
  - Method B (Honest): PC on 70% (X_struct) -> DML on 30% (X_effect)
  - 100 Monte Carlo replicates
  - Compare: bias, CI width, empirical coverage of true ATE

Usage:
    uv run python scripts/circularity_bias_demo.py
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

OUTPUT = Path('results/circularity_demo')
OUTPUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.RandomState(42)


# ============================================================
# Synthetic SEM with known ATE
# ============================================================
def generate_sem(n: int, true_ate_X1Y: float = 0.5,
                 seed: int = 0, n_noise: int = 15) -> tuple[np.ndarray, np.ndarray, list]:
    """
    DGP: linear Gaussian SEM with a true causal structure plus many
    weakly-correlated noise variables that PC might spuriously include
    as Y's parents under finite-sample uncertainty.

    True structure (5 informative variables):
        X0 ~ N(0, 1)                             (exogenous confounder)
        X1 = 0.7*X0 + N(0, 0.5^2)                (treatment; causal parent of Y)
        X2 = 0.5*X1 + 0.4*X0 + N(0, 0.5^2)       (mediator)
        X3 ~ N(0, 1)                             (irrelevant)
        X4 = 0.6*X2 + N(0, 0.5^2)                (downstream of mediator)
        Y  = ate*X1 + 0.3*X2 + 0.2*X0 + N(0, 0.3^2)

    Plus `n_noise` candidate features X5..X(5+n_noise-1), each a weak
    linear combination of {X0, Y} plus Gaussian noise, designed to be
    occasionally significant under Fisher's-z at alpha=0.05. These create
    ambiguity for PC's parent search and amplify post-selection effects.

    True ATE of X1 on Y is exactly `true_ate_X1Y`.
    """
    rs = np.random.RandomState(seed)
    X0 = rs.randn(n)
    X1 = 0.7 * X0 + 0.5 * rs.randn(n)
    X2 = 0.5 * X1 + 0.4 * X0 + 0.5 * rs.randn(n)
    X3 = rs.randn(n)
    X4 = 0.6 * X2 + 0.5 * rs.randn(n)
    Y_signal = true_ate_X1Y * X1 + 0.3 * X2 + 0.2 * X0
    Y = Y_signal + 0.3 * rs.randn(n)

    # Noise candidates with weak correlations to (X0, Y) to confuse PC
    noise_cols = []
    for k in range(n_noise):
        # weak loadings with Y (post-hoc data leakage candidate)
        a, b = rs.uniform(-0.15, 0.15), rs.uniform(-0.10, 0.10)
        nk = a * X0 + b * Y + rs.randn(n)
        noise_cols.append(nk)

    cols = [X0, X1, X2, X3, X4] + noise_cols
    X = np.stack(cols, axis=1)
    feature_names = [f'X{i}' for i in range(5 + n_noise)]
    return X, Y, feature_names


# ============================================================
# PC + DML on a candidate parent set
# ============================================================
def run_pc(X: np.ndarray) -> np.ndarray:
    """Run PC algorithm and return adjacency."""
    import jax.numpy as jnp
    from jax import random
    from jcce.structure_learning.pc import learn_with_pc
    A = learn_with_pc(jnp.array(X.astype(np.float32)), random.PRNGKey(0),
                      alpha=0.05, verbose=False)
    return np.array(A).astype(int)


def get_parents_of_target(A_full: np.ndarray, target_idx: int) -> list:
    """For an adjacency where target is the last column (joint XY space).

    target_idx in joint-space indexing (so target_idx = d for X augmented with Y).
    """
    parents = np.where(A_full[:, target_idx] != 0)[0].tolist()
    return [p for p in parents if p != target_idx]


def dml_ate(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_eval: np.ndarray, Y_eval: np.ndarray,
    treatment_idx: int, parent_idxs: list,
    n_folds: int = 5,
) -> tuple[float, float, float]:
    """
    Standard partial-DML / partialling-out estimator.

    Estimate E[Y | covariates], E[T | covariates] via cross-fitted random
    forests. Residualise Y and T, regress residuals to obtain ATE and an
    influence-function-based standard error.

    Args:
        X_train, Y_train: data used to fit nuisance functions
        X_eval, Y_eval:   data used to compute residuals + ATE
        treatment_idx:    column index of T in X
        parent_idxs:      indices of T's parents (used as covariates)

    Returns:
        ate_hat, ci_lower, ci_upper (95%)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    # Covariates: parents of T (excluding T itself)
    cov_idxs = [i for i in parent_idxs if i != treatment_idx]
    if not cov_idxs:
        # No parents → ATE is just OLS coefficient of T on Y
        from sklearn.linear_model import LinearRegression
        T = X_eval[:, treatment_idx]
        ols = LinearRegression().fit(T.reshape(-1, 1), Y_eval)
        ate = float(ols.coef_[0])
        # Approximate SE from residuals
        pred = ols.predict(T.reshape(-1, 1))
        sigma2 = np.var(Y_eval - pred, ddof=1)
        var_T = np.var(T, ddof=1) * len(T)
        se = float(np.sqrt(sigma2 / var_T))
        return ate, ate - 1.96 * se, ate + 1.96 * se

    Z_eval = X_eval[:, cov_idxs]
    T_eval = X_eval[:, treatment_idx]

    # Cross-fitted nuisance estimation. Using OLS for nuisance: with a linear
    # SEM, this matches the parametric truth and stabilises finite-sample SEs.
    from sklearn.linear_model import LinearRegression
    n_eval = len(X_eval)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    Y_resid = np.zeros(n_eval)
    T_resid = np.zeros(n_eval)

    for tr_idx, te_idx in kf.split(np.arange(n_eval)):
        m_y = LinearRegression().fit(Z_eval[tr_idx], Y_eval[tr_idx])
        m_t = LinearRegression().fit(Z_eval[tr_idx], T_eval[tr_idx])
        Y_resid[te_idx] = Y_eval[te_idx] - m_y.predict(Z_eval[te_idx])
        T_resid[te_idx] = T_eval[te_idx] - m_t.predict(Z_eval[te_idx])

    # ATE = sum(Y_resid * T_resid) / sum(T_resid^2)
    denom = float(np.sum(T_resid ** 2))
    if denom == 0:
        return float('nan'), float('nan'), float('nan')
    ate_hat = float(np.sum(Y_resid * T_resid) / denom)

    # Influence-function SE: sqrt(Var(psi_i) / n) where
    # psi_i = (Y_resid_i - ate * T_resid_i) * T_resid_i / E[T_resid^2]
    psi = (Y_resid - ate_hat * T_resid) * T_resid / (denom / n_eval)
    se = float(np.std(psi, ddof=1) / np.sqrt(n_eval))
    return ate_hat, ate_hat - 1.96 * se, ate_hat + 1.96 * se


# ============================================================
# Two pipelines
# ============================================================
def naive_pipeline(X: np.ndarray, Y: np.ndarray, treatment_idx: int = 1):
    """PC on 100% data -> DML on the same 100% data."""
    XY = np.concatenate([X, Y.reshape(-1, 1)], axis=1)
    A = run_pc(XY)
    parents = get_parents_of_target(A, target_idx=XY.shape[1] - 1)
    parents = [p for p in parents if p < X.shape[1]]
    if treatment_idx not in parents:
        parents.append(treatment_idx)
    return dml_ate(X, Y, X, Y, treatment_idx, parents)


def honest_pipeline(X: np.ndarray, Y: np.ndarray, treatment_idx: int = 1,
                    test_size: float = 0.3, seed: int = 0):
    """PC on 70% (struct) -> DML on the held-out 30% (effect)."""
    X_s, X_e, Y_s, Y_e = train_test_split(X, Y, test_size=test_size,
                                           random_state=seed)
    XY_s = np.concatenate([X_s, Y_s.reshape(-1, 1)], axis=1)
    A = run_pc(XY_s)
    parents = get_parents_of_target(A, target_idx=XY_s.shape[1] - 1)
    parents = [p for p in parents if p < X.shape[1]]
    if treatment_idx not in parents:
        parents.append(treatment_idx)
    return dml_ate(X_s, Y_s, X_e, Y_e, treatment_idx, parents)


# ============================================================
# Monte Carlo experiment
# ============================================================
def run_experiment(n_per_replicate: int = 1000,
                   n_replicates: int = 100,
                   true_ate: float = 0.5):
    print(f"Monte Carlo: n={n_per_replicate}, reps={n_replicates}, "
          f"true ATE={true_ate}")
    naive_ates, naive_ci_lo, naive_ci_hi = [], [], []
    honest_ates, honest_ci_lo, honest_ci_hi = [], [], []

    for r in range(n_replicates):
        if r % 10 == 0:
            print(f"  rep {r}/{n_replicates}")
        X, Y, _ = generate_sem(n_per_replicate, true_ate, seed=r)
        try:
            n_ate, n_lo, n_hi = naive_pipeline(X, Y, treatment_idx=1)
        except Exception as e:
            print(f"  naive failed rep {r}: {e}")
            continue
        try:
            h_ate, h_lo, h_hi = honest_pipeline(X, Y, treatment_idx=1, seed=r)
        except Exception as e:
            print(f"  honest failed rep {r}: {e}")
            continue
        naive_ates.append(n_ate); naive_ci_lo.append(n_lo); naive_ci_hi.append(n_hi)
        honest_ates.append(h_ate); honest_ci_lo.append(h_lo); honest_ci_hi.append(h_hi)

    naive_ates = np.array(naive_ates)
    honest_ates = np.array(honest_ates)
    naive_widths = np.array(naive_ci_hi) - np.array(naive_ci_lo)
    honest_widths = np.array(honest_ci_hi) - np.array(honest_ci_lo)
    naive_cov = np.mean((np.array(naive_ci_lo) <= true_ate) &
                         (true_ate <= np.array(naive_ci_hi)))
    honest_cov = np.mean((np.array(honest_ci_lo) <= true_ate) &
                          (true_ate <= np.array(honest_ci_hi)))

    return {
        'true_ate': true_ate,
        'n_per_replicate': n_per_replicate,
        'n_replicates_completed': len(naive_ates),
        'naive_ate_mean': float(np.mean(naive_ates)),
        'naive_ate_std': float(np.std(naive_ates)),
        'naive_bias': float(np.mean(naive_ates) - true_ate),
        'naive_ci_width_mean': float(np.mean(naive_widths)),
        'naive_coverage': float(naive_cov),
        'honest_ate_mean': float(np.mean(honest_ates)),
        'honest_ate_std': float(np.std(honest_ates)),
        'honest_bias': float(np.mean(honest_ates) - true_ate),
        'honest_ci_width_mean': float(np.mean(honest_widths)),
        'honest_coverage': float(honest_cov),
    }


def main():
    results = run_experiment(n_per_replicate=1000, n_replicates=200,
                              true_ate=0.5)
    print('\n=== Results ===')
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    out_path = OUTPUT / 'circularity_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Brief interpretation
    print('\n=== Interpretation ===')
    print(f"  Naive bias:     {results['naive_bias']:+.4f}")
    print(f"  Honest bias:    {results['honest_bias']:+.4f}")
    print(f"  Naive CI width: {results['naive_ci_width_mean']:.4f}")
    print(f"  Honest CI width:{results['honest_ci_width_mean']:.4f}")
    print(f"  Naive coverage of true ATE (95% nominal): {results['naive_coverage']:.2%}")
    print(f"  Honest coverage of true ATE (95% nominal): {results['honest_coverage']:.2%}")


if __name__ == '__main__':
    main()
