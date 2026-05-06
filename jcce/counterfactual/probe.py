"""Probe API for diagnosing per-variable predictor collapse in JCCE pipelines.

The architectural-fix mini-Sprint (Day 1-3 of the AAP work; commits 11d75a0,
0310b39, ddbaf09 on origin/aap, and the full ablation in 7b42789) found that
JCCE's joint loss systematically drives the per-variable Y predictor f_Y to a
near-constant function of its inputs, regardless of processor architecture
(DAG-Attention and LinearHead both collapse identically). This module exports
the reusable diagnostic functions used to surface that collapse:

- :func:`make_f_Y` constructs the per-variable forward callable from trained
  JCCE artifacts under one of four input-weighting conventions.
- :func:`autodiff_sensitivity` is the gradient probe — mean ``|df/dx_var|``.
- :func:`grid_sweep` is the function-shape probe — sweeps ``x_var`` across a
  grid and reports mean output at each value.

Together they answer "is the trained predictor degenerate at this variable?"
in a processor-agnostic way: if both probes return zero across the
``{learned, hard, unit, training}`` weighting modes, the predictor has
collapsed to a constant function of its inputs.

The four weighting modes correspond to the candidate explanations the probe
was built to disambiguate:

- ``learned``  matches the AAP-cascade input: ``X * |A[:, Y_idx]|``.
- ``hard``     applies a strict 0/1 mask: ``X * (|A[:, Y_idx]| > tau)``.
- ``unit``     bypasses the |A| weighting: ``X * 1``. Tests architecture-only.
- ``training`` matches the Y-reconstruction additive baseline used during
               training (``|A| + 1/(n_features + 1)``); diagnoses train/eval
               distribution mismatch.

"""

from __future__ import annotations

from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np


__all__ = ["make_f_Y", "autodiff_sensitivity", "grid_sweep"]


def make_f_Y(
    processor,
    A: jnp.ndarray,
    params_Y,
    Y_idx: int,
    n_features: int,
    weighting: str = "learned",
    edge_threshold: float = 0.05,
    skip_centering: bool = True,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:
    """Construct the trained ``f_Y`` callable under a chosen input-weighting mode.

    JCCE's per-variable forward at AAP time is conventionally
    ``f_j(X) = processor.forward(X * |A[:, j]|, params[j], A=A_struct, ...)``.
    This builds an analogous callable for ``j == Y_idx`` with explicit control
    over the input weighting, so the diagnostic probes can isolate whether a
    constant output traces to (a) the ``|A|`` weighting attenuating the signal,
    (b) train/inference distribution mismatch, or (c) the processor's
    prediction head itself.

    Parameters
    ----------
    processor : adapter
        Any JCCE processor adapter with ``.forward(X, params, ...)``. The
        function detects ``DAGAttentionAdapter`` and ``GNNAdapter`` for their
        adapter-specific calling conventions; other adapters fall through to
        the default signature.
    A : (d_+, d_+) jnp.ndarray
        Learned adjacency matrix, where ``d_+ = n_features + 1``.
    params_Y
        Per-variable parameters for the Y processor (typically ``params[Y_idx]``).
    Y_idx : int
        Index of the Y variable in the ``d_+`` layout. Must equal
        ``n_features`` for JCCE's outcome-as-sink convention.
    n_features : int
        Number of X variables (``d_+ - 1``).
    weighting : {'learned', 'hard', 'unit', 'training'}
        How to weight the input columns before the per-variable forward.
        See module docstring for semantics.
    edge_threshold : float
        Magnitude threshold for the ``'hard'`` mode.
    skip_centering : bool
        Forwarded to ``processor.forward``; ``True`` keeps the classification
        logit (matches AAPCounterfactual's behavior for ``j == Y_idx``),
        ``False`` mean-centers the output (matches the X-reconstruction path).

    Returns
    -------
    f_Y : Callable[[jnp.ndarray], jnp.ndarray]
        Maps ``(n_samples, n_features)`` to ``(n_samples,)`` predictions.
    weights : jnp.ndarray
        ``(n_features,)`` array of weights actually applied; useful for logging.

    Raises
    ------
    ValueError
        If ``weighting`` is not one of the recognized modes.
    """
    A_struct = A[:n_features, :n_features]
    weights_abs = jnp.abs(A[:n_features, Y_idx])

    if weighting == "learned":
        weights = weights_abs
    elif weighting == "hard":
        weights = (weights_abs > edge_threshold).astype(A.dtype)
    elif weighting == "unit":
        weights = jnp.ones(n_features, dtype=A.dtype)
    elif weighting == "training":
        # Mirror the Y-reconstruction additive baseline in jcce_learner.py
        # (line ~4897): weights_Y_recon = |A[:, Y_idx]| + 1/n_v, where
        # n_v = n_features + 1 includes Y. This is the input distribution
        # the per-variable Y processor saw during training.
        n_v = n_features + 1
        weights = weights_abs + 1.0 / n_v
    else:
        raise ValueError(
            f"unknown weighting {weighting!r}; expected one of "
            "{'learned', 'hard', 'unit', 'training'}"
        )

    proc_name = processor.__class__.__name__

    def f_Y(X_features: jnp.ndarray) -> jnp.ndarray:
        X_weighted = X_features * weights[jnp.newaxis, :]
        if proc_name == "DAGAttentionAdapter":
            return processor.forward(
                X_weighted, params_Y, A=A_struct, skip_centering=skip_centering,
            )
        if proc_name == "GNNAdapter":
            A_norm = A_struct / (jnp.sum(jnp.abs(A_struct), axis=0, keepdims=True) + 1e-8)
            return processor.forward(
                X_weighted, params_Y, A=A_norm, skip_centering=skip_centering,
            )
        return processor.forward(X_weighted, params_Y, skip_centering=skip_centering)

    return f_Y, weights


def autodiff_sensitivity(
    f_Y: Callable[[jnp.ndarray], jnp.ndarray],
    X: jnp.ndarray,
    var_idx: int,
) -> float:
    """Gradient probe: mean ``|df_Y/dX[:, var_idx]|`` evaluated on the observed batch.

    Returns the average per-sample slope ``E_i [|df_Y(X[i])/dX[i, var_idx]|]``,
    obtained by computing ``jax.grad`` of ``sum_i f_Y(X[i])`` (whose gradient
    at sample ``i`` w.r.t. ``X[i, var_idx]`` is exactly the per-sample slope)
    and then averaging ``|grad|`` across samples. A near-zero result is the
    smoking-gun signature of a collapsed (constant) f_Y; combine with
    :func:`grid_sweep` for a function-shape confirmation.

    Parameters
    ----------
    f_Y : Callable
        Forward function from :func:`make_f_Y` (or any equivalent
        ``(n_samples, n_features) -> (n_samples,)`` map).
    X : (n_samples, n_features) jnp.ndarray
        Input batch at which to evaluate the gradient.
    var_idx : int
        Column index whose sensitivity to probe.

    Returns
    -------
    float
        Mean absolute per-sample slope. For linear ``f_Y(X) = X * w``, returns
        ``|w[var_idx]|`` exactly; for collapsed (constant) ``f_Y``, returns 0.
    """
    def loss(X_in: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(f_Y(X_in))

    grads = jax.grad(loss)(X)
    return float(jnp.mean(jnp.abs(grads[:, var_idx])))


def grid_sweep(
    f_Y: Callable[[jnp.ndarray], jnp.ndarray],
    X: jnp.ndarray,
    var_idx: int,
    grid: Iterable[float],
) -> np.ndarray:
    """Function-shape probe: vary ``X[:, var_idx]`` across a grid and report mean f_Y.

    Complements :func:`autodiff_sensitivity` by reporting the *shape* of f_Y
    as a function of the probed variable, not just the gradient at a single
    point. A flat sweep (constant across grid) confirms collapse; a monotone
    sweep confirms input dependence.

    Parameters
    ----------
    f_Y : Callable
        Forward function as in :func:`autodiff_sensitivity`.
    X : (n_samples, n_features) jnp.ndarray
        Observed input batch; columns other than ``var_idx`` stay at their
        observed values throughout the sweep.
    var_idx : int
        Column index whose values are replaced.
    grid : Iterable[float]
        Values to substitute into ``X[:, var_idx]``.

    Returns
    -------
    np.ndarray
        ``(len(grid),)`` array of mean ``f_Y`` at each grid value.
    """
    means: list[float] = []
    for v in grid:
        X_set = X.at[:, var_idx].set(jnp.full((X.shape[0],), float(v)))
        means.append(float(jnp.mean(f_Y(X_set))))
    return np.asarray(means)
