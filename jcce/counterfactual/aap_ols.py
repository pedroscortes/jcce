"""OLS-augmented AAP counterfactual head.

The Sprint-1 + Option C diagnostics established that the trained
DAG-Attention's per-variable processor for ``j=Y`` collapses to a
constant function of its inputs. Day 1 of the architectural-fix
mini-Sprint (``aap_ols_diagnostic.py``) confirmed that a closed-form
OLS regressor on the same input distribution recovers a meaningful
CATE on synthetic SCM (``+0.255`` vs the trained Transformer's
``-0.000``; ``R^2 = 0.614``). The OLS regressor is a deployable
post-hoc replacement for ``f_Y`` at AAP-inference time.

This module wraps that replacement into the ``AAPCounterfactual`` API
so callers get the standard ``cate``, ``predict_counterfactual``, and
``round_trip`` methods with minimal change. Only ``f_Y`` is replaced;
the trained DAG-Attention processors for the X variables continue to
drive abduction and the topological forward pass.

The OLS predictor is fit once at construction time on the observed
``(X_features, Y)`` pair, using the same input weighting (``X *
|A[:, Y_idx]|``) that the AAP cascade and ``AAPCounterfactual``
default ``_f_j`` use. The fit closed-form solution is cached in the
instance, so subsequent calls to ``cate`` are O(n).

This is intentionally a *post-hoc* head, not a re-training: it
validates the "data has signal that f_Y missed" diagnosis as a
shippable AAP path, and it keeps the rest of the JCCE pipeline
untouched (no main-owned file changes). Trained jointly, a parallel
linear residual head could in principle do better; that experiment
is the Day 2 / Day 3 plan if this post-hoc head proves the concept.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .aap import AAPCounterfactual


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form OLS with intercept. Returns ``(beta, intercept)``."""
    X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coefs, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coefs[:-1].astype(np.float32), float(coefs[-1])


def _fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-4, maxiter: int = 200,
) -> tuple[np.ndarray, float]:
    """Logistic regression via L-BFGS. Returns ``(beta, intercept)``.

    The fit minimises $-\\sum y \\log \\sigma(X\\beta + b) + (1 - y) \\log
    \\sigma(-(X\\beta + b)) + \\lambda \\|\\beta\\|_2^2$. The L2 ridge with
    a tiny $\\lambda$ stabilises rank-deficient inputs (e.g., when
    ``X * |A|`` has near-collinear weighted columns).

    Parameters
    ----------
    X : (n, d) ndarray
    y : (n,) ndarray of 0/1 labels
    l2 : float
        L2 ridge strength on ``beta`` (not on intercept).
    maxiter : int
    """
    from scipy.optimize import minimize

    n, d = X.shape
    y = y.astype(np.float64)
    X = X.astype(np.float64)

    def neg_log_lik_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        beta = params[:-1]
        b = params[-1]
        logits = X @ beta + b
        # Stable log(1 + exp(-z)) = -log(sigmoid(z))
        log1pexp_neg = np.logaddexp(0.0, -logits)
        log1pexp_pos = np.logaddexp(0.0, logits)
        nll = float(np.sum(y * log1pexp_neg + (1.0 - y) * log1pexp_pos))
        nll += float(0.5 * l2 * np.sum(beta ** 2))
        # Gradient: dL/dz_i = sigmoid(z_i) - y_i; dL/dbeta = X^T grad; dL/db = sum(grad)
        p = 1.0 / (1.0 + np.exp(-logits))
        residual = p - y
        gbeta = X.T @ residual + l2 * beta
        gb = float(np.sum(residual))
        return nll, np.concatenate([gbeta, [gb]])

    init = np.zeros(d + 1)
    result = minimize(
        neg_log_lik_and_grad, init, jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    return result.x[:-1].astype(np.float32), float(result.x[-1])


class AAPCounterfactualOLS(AAPCounterfactual):
    """AAPCounterfactual with the trained ``f_Y`` swapped for closed-form OLS.

    The OLS predictor is fit at construction time on the observed
    ``(X_features, Y)`` pair, using the same input weighting that the
    default ``_f_j`` would apply to the Y variable.

    Parameters mirror :class:`AAPCounterfactual` plus the observed data
    used to fit the OLS regressor.
    """

    def __init__(
        self,
        processor,
        A: jnp.ndarray,
        processor_params: list,
        Y_idx: int,
        X_features: np.ndarray | jnp.ndarray,
        Y: np.ndarray | jnp.ndarray,
        edge_threshold: float = 0.05,
        enforce_hard_parents: bool = False,
    ):
        super().__init__(
            processor=processor,
            A=A,
            processor_params=processor_params,
            Y_idx=Y_idx,
            edge_threshold=edge_threshold,
            enforce_hard_parents=enforce_hard_parents,
        )
        # Use the same input weighting the default _f_j would apply for Y.
        # We use the "learned" weighting (continuous |A|), not the STE hard mask
        # — the OLS replacement is conceptually the dense-pathway predictor that
        # the constant Transformer f_Y failed to be, so dropping non-parents
        # before OLS fitting would defeat its purpose.
        weights = np.abs(np.asarray(A)[: self.n_features, self.Y_idx])
        X_np = np.asarray(X_features, dtype=np.float32)
        Y_np = np.asarray(Y, dtype=np.float32).reshape(-1)
        X_weighted = X_np * weights[None, :]
        self.ols_beta, self.ols_intercept = _fit_ols(X_weighted, Y_np)
        self.ols_weights = weights.astype(np.float32)

        # Cache R^2 on the fit set for diagnostics.
        Y_hat = X_weighted @ self.ols_beta + self.ols_intercept
        ss_res = float(np.sum((Y_np - Y_hat) ** 2))
        ss_tot = float(np.sum((Y_np - np.mean(Y_np)) ** 2))
        self.ols_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    def _f_j(self, X_features: jnp.ndarray, j: int, A_use: jnp.ndarray) -> jnp.ndarray:
        """Override f_Y to use OLS; delegate other variables to the base class."""
        if j == self.Y_idx:
            X_jnp = jnp.asarray(X_features)
            beta = jnp.asarray(self.ols_beta)
            weights = jnp.asarray(self.ols_weights)
            X_weighted = X_jnp * weights[jnp.newaxis, :]
            return X_weighted @ beta + self.ols_intercept
        return super()._f_j(X_features, j, A_use)


class AAPCounterfactualLogistic(AAPCounterfactual):
    """AAPCounterfactual with the trained ``f_Y`` swapped for logistic regression.

    Same construction protocol as :class:`AAPCounterfactualOLS`, but fits
    $P(Y=1) = \\sigma(X_w \\beta + b)$ via L-BFGS instead of OLS. The
    counterfactual prediction returned is the **logit** $X_w \\beta + b$,
    matching the conventional logit-scale CATE used by the JCCE pipeline
    and by binary-Y SCM data-generating processes. This avoids the
    linear-probability attenuation that bottlenecks
    :class:`AAPCounterfactualOLS`'s magnitude on synthetic.

    The downstream ``AAPCounterfactual.cate`` returns the logit
    difference; downstream consumers can apply ``jax.nn.sigmoid`` to
    convert to probability deltas if a probability-scale ``|dP|`` is
    needed.
    """

    def __init__(
        self,
        processor,
        A: jnp.ndarray,
        processor_params: list,
        Y_idx: int,
        X_features: np.ndarray | jnp.ndarray,
        Y: np.ndarray | jnp.ndarray,
        edge_threshold: float = 0.05,
        enforce_hard_parents: bool = False,
        l2: float = 1e-4,
    ):
        super().__init__(
            processor=processor,
            A=A,
            processor_params=processor_params,
            Y_idx=Y_idx,
            edge_threshold=edge_threshold,
            enforce_hard_parents=enforce_hard_parents,
        )
        weights = np.abs(np.asarray(A)[: self.n_features, self.Y_idx])
        X_np = np.asarray(X_features, dtype=np.float32)
        Y_np = np.asarray(Y, dtype=np.float32).reshape(-1)
        X_weighted = X_np * weights[None, :]
        self.logistic_beta, self.logistic_intercept = _fit_logistic(X_weighted, Y_np, l2=l2)
        self.logistic_weights = weights.astype(np.float32)

        # Cache classification accuracy and AUC-like score on fit set.
        logits = X_weighted @ self.logistic_beta + self.logistic_intercept
        p = 1.0 / (1.0 + np.exp(-logits))
        Y_pred = (p > 0.5).astype(np.float32)
        self.logistic_accuracy = float(np.mean(Y_pred == Y_np))
        # Mean log-likelihood as a R^2-like quality scalar.
        log1pexp_neg = np.logaddexp(0.0, -logits)
        log1pexp_pos = np.logaddexp(0.0, logits)
        nll = float(np.mean(Y_np * log1pexp_neg + (1.0 - Y_np) * log1pexp_pos))
        # Null model NLL: predict marginal probability.
        p0 = float(np.mean(Y_np))
        eps = 1e-9
        nll_null = -p0 * np.log(p0 + eps) - (1.0 - p0) * np.log(1.0 - p0 + eps)
        self.logistic_pseudo_r2 = 1.0 - nll / max(nll_null, 1e-12)

    def _f_j(self, X_features: jnp.ndarray, j: int, A_use: jnp.ndarray) -> jnp.ndarray:
        """Override f_Y to return the logistic logit; delegate others to the base."""
        if j == self.Y_idx:
            X_jnp = jnp.asarray(X_features)
            beta = jnp.asarray(self.logistic_beta)
            weights = jnp.asarray(self.logistic_weights)
            X_weighted = X_jnp * weights[jnp.newaxis, :]
            return X_weighted @ beta + self.logistic_intercept
        return super()._f_j(X_features, j, A_use)
