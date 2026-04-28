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
