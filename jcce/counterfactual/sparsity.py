"""Sparsity primitives for AAP training.

The AAP cascade loss in ``jcce_learner.py`` originally computed
``X_cascade * |A|`` to weight each parent contribution in the
per-variable forward. Continuous magnitudes are sparsity-permissive:
a dense processor such as DAG-Attention can amplify even small
non-parent weights to fit Y, so the loss bypasses the structural
pathway. The synthetic-SCM CATE collapse to ~0 (Sprint C) and the
Heart Disease ``mean |delta P(Y)| = 0.011`` (Sprint D) are both
consistent with this dense-compensation pathology.

``ste_hard_parents`` replaces the continuous weighting with a hard
0/1 mask in the forward pass while preserving identity gradients
w.r.t. ``|A|`` in the backward pass via a straight-through estimator.
This forces the AAP forward to use only above-threshold parents,
while letting the structure-learning gradient continue to grow or
shrink edges.

``edge_set_agreement`` reports how well a learned adjacency recovers
a known ground-truth edge set, used to validate sparsity-trained
models on synthetic SCMs where ``A_true`` is available.

References
----------
Bengio, Leonard, Courville (2013) — "Estimating or Propagating
Gradients Through Stochastic Neurons for Conditional Computation"
(the straight-through estimator).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def ste_hard_parents(weights: jnp.ndarray, threshold: float) -> jnp.ndarray:
    """Straight-through hard mask on parent weights.

    Forward pass returns a 0/1 mask, ``hard = (|weights| > threshold)``.
    Backward pass behaves as identity w.r.t. ``|weights|``: the gradient
    flows back through ``|weights|`` as if no thresholding occurred.

    Parameters
    ----------
    weights : jnp.ndarray
        Edge weights, signed or non-negative. The hard mask is computed on
        ``|weights|``.
    threshold : float
        Magnitude threshold; entries with ``|weights| <= threshold`` are
        zeroed in the forward but still receive gradient in the backward.

    Returns
    -------
    jnp.ndarray
        Same shape and dtype as ``|weights|``. Forward values are in
        $\\{0, 1\\}$; autodiff treats it as a soft pass-through of
        ``|weights|``.

    Notes
    -----
    The standard STE trick is $\\mathrm{stop\\_gradient}(\\mathrm{hard}
    - \\mathrm{soft}) + \\mathrm{soft}$: the forward equals
    $\\mathrm{hard}$ since ``stop_gradient`` propagates value unchanged,
    and the backward gradient equals $\\partial \\mathrm{soft} /
    \\partial \\mathrm{weights}$.
    """
    weights_abs = jnp.abs(weights)
    hard_mask = (weights_abs > threshold).astype(weights_abs.dtype)
    return jax.lax.stop_gradient(hard_mask - weights_abs) + weights_abs


def edge_set_agreement(
    A_learned: jnp.ndarray | np.ndarray,
    A_true: jnp.ndarray | np.ndarray,
    threshold_learned: float = 0.05,
    threshold_true: float = 1e-6,
) -> dict[str, float]:
    """Compare a learned adjacency to a ground-truth one.

    Parameters
    ----------
    A_learned : array
        Learned adjacency matrix.
    A_true : array
        Ground-truth adjacency matrix, same shape.
    threshold_learned : float, default 0.05
        Magnitude threshold for binarizing the learned matrix.
    threshold_true : float, default 1e-6
        Magnitude threshold for binarizing the true matrix; the small
        non-zero default lets clean SCM ``A_true`` matrices binarize
        cleanly to $\\{0, 1\\}$.

    Returns
    -------
    dict
        ``accuracy`` (fraction of $(i, j)$ where presence/absence agrees;
        Sprint-1 gate metric), ``precision``, ``recall``, ``f1`` on edge
        presence, plus ``n_true_edges`` and ``n_learned_edges``.

    Notes
    -----
    For sparse graphs the trivial all-zero baseline can attain high
    ``accuracy`` (mostly true negatives), so ``f1`` and ``recall`` are
    the more informative metrics in practice.
    """
    learned_bin = (np.abs(np.asarray(A_learned)) > threshold_learned).astype(np.int64)
    true_bin = (np.abs(np.asarray(A_true)) > threshold_true).astype(np.int64)

    accuracy = float((learned_bin == true_bin).mean())

    tp = int(np.sum((learned_bin == 1) & (true_bin == 1)))
    fp = int(np.sum((learned_bin == 1) & (true_bin == 0)))
    fn = int(np.sum((learned_bin == 0) & (true_bin == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = precision + recall
    f1 = (2 * precision * recall) / denom if denom > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_true_edges": int(true_bin.sum()),
        "n_learned_edges": int(learned_bin.sum()),
    }
