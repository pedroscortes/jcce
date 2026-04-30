"""Attention-extraction probes for trained JCCE processors.

Two diagnostic tools that operate on already-trained JCCE pipelines (no
re-training, runs on CPU):

1. :func:`extract_topomamba_hidden_attention` — per Ali, Zimerman, Wolf
   (ACL 2025) "The Hidden Attention of Mamba Models", Mamba's selective
   scan implies an attention matrix. We approximate it via input-gradient
   analysis (mathematically equivalent to the unrolled scan for linear
   SSM stages; first-order approximation otherwise). Output: per-variable
   attention from Y position to each X position. Compare to JCCE's |A|.

2. :func:`extract_dagattn_causal_graph` — per Rohekar, Gurwicz, Nisimov
   (NeurIPS 2023, Intel Labs) "Causal Interpretation of Self-Attention in
   Pre-Trained Transformers", attention matrices encode causal structure.
   We extract the deepest layer's attention pattern, threshold to binary
   edges, and report a learned graph G_attn. Compare to JCCE's |A| via
   edge Jaccard / agreement metrics.

Together these probes test whether high-capacity processors (DAG-Attention,
TopoMamba) learn a causal structure consistent with JCCE's explicit |A|, or
whether they game the constraint by routing through non-target features
(documented as "high-capacity gaming pattern" in §6 / §7.2.4 of the paper).
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def extract_topomamba_hidden_attention(
    processor,
    A: jnp.ndarray,
    params: list,
    X: jnp.ndarray,
    Y_idx: int,
) -> jnp.ndarray:
    """Per-input attention from TopoMamba's Y output, via input-gradient analysis.

    The Mamba selective scan ``h_t = Ā_t h_{t-1} + B̄_t x_t`` unrolls as
    ``y_t = sum_i α_ti · x_i`` where α_ti is the implicit attention weight
    (Ali et al. ACL 2025). For our purposes — diagnosing whether TopoMamba's
    Y output attends to the variables JCCE's |A[:, Y_idx]| says it should
    attend to — the gradient ``|∂y_Y / ∂x_i|`` is mathematically equivalent
    to ``|α_{Y,i}|`` for the linear part of the recurrence and a first-order
    approximation for the SiLU-gated nonlinear part.

    Parameters
    ----------
    processor : adapter
        Trained JCCE processor (must accept ``A=...`` kwarg if SSM/Attn class).
    A : (n_total, n_total) jnp.ndarray
        Trained adjacency. Used to weight input via ``|A[:, Y_idx]| + 0.01``,
        matching JCCE's training-time forward.
    params : list
        Trained per-variable processor params.
    X : (n_samples, n_features) jnp.ndarray
        Data on which to compute the attention.
    Y_idx : int
        Y position index. Must equal ``n_features`` (JCCE convention).

    Returns
    -------
    attention : (n_features,) jnp.ndarray
        ``|α_{Y, i}|`` averaged over the batch. Each entry is the implicit
        attention weight from input variable i to the Y output.
    """
    n_features = X.shape[1]
    weights_Y = jnp.abs(A[:n_features, Y_idx]) + 0.01

    def model_fn(x_in):
        x_w = x_in * weights_Y[None, :]
        pname = processor.__class__.__name__
        if pname in ("DAGAttentionAdapter", "TopoMambaAdapter", "CausalMambaAdapter"):
            Y_logit = processor.forward(
                x_w, params[Y_idx],
                A=A[:n_features, :n_features], skip_centering=True,
            )
        else:
            Y_logit = processor.forward(x_w, params[Y_idx], skip_centering=True)
        return jnp.sum(Y_logit)

    grad_x = jax.grad(model_fn)(jnp.array(X, dtype=jnp.float32))
    attention = jnp.mean(jnp.abs(grad_x), axis=0)  # (n_features,)
    return attention


def compare_attention_to_A(
    attention: jnp.ndarray,
    A: jnp.ndarray,
    Y_idx: int,
    threshold_attn: float = None,
    threshold_A: float = 0.05,
) -> Dict[str, float]:
    """Compare an extracted attention vector against JCCE's |A[:, Y_idx]|.

    Two binary-edge comparisons:
    - JCCE edges: ``|A[:n_features, Y_idx]| > threshold_A``
    - Attention edges: ``attention > threshold_attn`` (default = mean(attention))

    Returns
    -------
    metrics : dict
        - ``correlation``: Pearson correlation between ``attention`` and ``|A[:, Y_idx]|``
        - ``jaccard``: Jaccard index of binarized edge sets
        - ``n_edges_attn``, ``n_edges_A``: edge counts
        - ``intersection``, ``union``: counts for Jaccard
        - ``threshold_attn``: actual threshold used (mean if None passed)
    """
    n_features = len(attention)
    A_Y = jnp.abs(A[:n_features, Y_idx])

    # Pearson correlation between continuous values
    a_np = np.asarray(attention)
    A_Y_np = np.asarray(A_Y)
    if a_np.std() > 1e-10 and A_Y_np.std() > 1e-10:
        correlation = float(np.corrcoef(a_np, A_Y_np)[0, 1])
    else:
        correlation = float("nan")

    # Binary edge agreement
    if threshold_attn is None:
        threshold_attn = float(np.mean(a_np))

    attn_edges = a_np > threshold_attn
    A_edges = A_Y_np > threshold_A

    n_edges_attn = int(attn_edges.sum())
    n_edges_A = int(A_edges.sum())
    intersection = int((attn_edges & A_edges).sum())
    union = int((attn_edges | A_edges).sum())
    jaccard = intersection / max(union, 1)

    return {
        "correlation": correlation,
        "jaccard": float(jaccard),
        "n_edges_attn": n_edges_attn,
        "n_edges_A": n_edges_A,
        "intersection": intersection,
        "union": union,
        "threshold_attn": float(threshold_attn),
    }


def extract_dagattn_causal_graph(
    processor,
    A: jnp.ndarray,
    params: list,
    X: jnp.ndarray,
    Y_idx: int,
    edge_threshold: float = 0.10,
) -> Tuple[jnp.ndarray, Dict]:
    """Extract a learned causal graph from DAG-Attention's attention matrix.

    Per Rohekar, Gurwicz, Nisimov (NeurIPS 2023): self-attention's matrix
    encodes a structural equation model over input symbols. The deepest
    layer's attention is most informative about causal structure.

    For our purposes — diagnosing whether DAG-Attention learns the same
    causal graph JCCE's |A| represents — we:
    1. Run a forward pass with ``return_attn=True``
    2. Aggregate the deepest layer's attention across heads + batch → (N, N)
    3. Threshold to binary edges → ``G_attn``
    4. Compare to JCCE's |A|

    This is a "soft CLEANN" — it uses the attention pattern directly rather
    than running constraint-based discovery (PC/FCI) on partial correlations.
    Faster and sufficient for the diagnostic purpose.

    Parameters
    ----------
    processor : DAGAttentionAdapter
        Must be a trained DAG-Attention adapter.
    A : (n_total, n_total) jnp.ndarray
    params : list
    X : (n_samples, n_features) jnp.ndarray
    Y_idx : int
    edge_threshold : float
        Attention values above this threshold are kept as edges.

    Returns
    -------
    G_attn : (n_features, n_features) jnp.ndarray
        Binary adjacency: 1 where attention > threshold, 0 elsewhere.
    metrics : dict
        - ``attn_dense_mean``: mean of dense (n_features × n_features) attention
        - ``edges_in_G_attn``: count of edges in G_attn
        - ``edges_in_JCCE_A``: count of edges in |A| > 0.05
        - ``jaccard_with_A``: Jaccard between G_attn and JCCE's |A|
        - ``correlation_with_A``: Pearson on the dense matrices

    Notes
    -----
    Only works for ``DAGAttentionAdapter``. For other processor types, raises
    ``NotImplementedError``.
    """
    pname = processor.__class__.__name__
    if pname != "DAGAttentionAdapter":
        raise NotImplementedError(
            f"extract_dagattn_causal_graph only supports DAGAttentionAdapter; "
            f"got {pname}. For TopoMamba, use extract_topomamba_hidden_attention "
            f"(returns gradient-based attention for the Y row only)."
        )

    n_features = X.shape[1]
    weights_Y = jnp.abs(A[:n_features, Y_idx]) + 0.01
    X_w = jnp.array(X, dtype=jnp.float32) * weights_Y[None, :]

    # Run forward with return_attn=True; processor.forward should expose this
    try:
        result = processor.forward(
            X_w, params[Y_idx],
            A=A[:n_features, :n_features],
            skip_centering=True,
            return_attn=True,
        )
        if isinstance(result, tuple):
            _, attn_list = result
            # attn_list: list of (B, H, N, N) for each layer
            attn = attn_list[-1]  # deepest layer
        else:
            raise ValueError("return_attn=True did not return a tuple")
    except (TypeError, ValueError) as e:
        # Fallback: use input-gradient as proxy if return_attn not available
        # (gradient-based attention from Y to each X position)
        attention_Y = extract_topomamba_hidden_attention(processor, A, params, X, Y_idx)
        # Build a dense matrix where row Y_idx has the gradient attention
        G_attn = np.zeros((n_features, n_features), dtype=np.float32)
        attn_np = np.asarray(attention_Y)
        attn_norm = attn_np / max(attn_np.max(), 1e-10)
        # Fill the Y row (last row when Y_idx = n_features, but we'd be at n_features-1 ... actually Y is at n_features which is OOB for n_features × n_features)
        # Treat the "attention to Y" as a column of the matrix
        # G_attn[i, j] symmetric proxy: edge if attention[i] high (everyone attends to Y)
        # For diagnostic purposes, just return the per-feature attention as Y-row
        G_attn_y_row = (attn_norm > edge_threshold).astype(np.float32)
        # No information about other positions; return empty G_attn with Y-row populated symbolically
        metrics = {
            "fallback": True,
            "attn_dense_mean": float(attn_np.mean()),
            "Y_attention_max": float(attn_np.max()),
            "edges_to_Y_attn": int(G_attn_y_row.sum()),
        }
        return jnp.array(G_attn), metrics

    # attn is (B, H, N, N) — aggregate across heads + batch
    A_attn = jnp.mean(attn, axis=(0, 1))  # (N, N)
    A_attn_np = np.asarray(A_attn)

    # Build binary G_attn
    G_attn = (A_attn_np > edge_threshold).astype(np.float32)

    # Compare to JCCE's |A|
    A_jcce = np.abs(np.asarray(A[:n_features, :n_features]))
    A_jcce_edges = (A_jcce > 0.05).astype(np.float32)
    G_attn_features = G_attn[:n_features, :n_features] if G_attn.shape[0] > n_features else G_attn

    if G_attn_features.shape == A_jcce_edges.shape:
        intersection = int((G_attn_features * A_jcce_edges).sum())
        union = int(((G_attn_features + A_jcce_edges) > 0).sum())
        jaccard = intersection / max(union, 1)
        if A_attn_np[:n_features, :n_features].std() > 1e-10 and A_jcce.std() > 1e-10:
            correlation = float(np.corrcoef(
                A_attn_np[:n_features, :n_features].flatten(),
                A_jcce.flatten(),
            )[0, 1])
        else:
            correlation = float("nan")
    else:
        jaccard = float("nan")
        correlation = float("nan")

    metrics = {
        "fallback": False,
        "attn_dense_mean": float(A_attn_np.mean()),
        "edges_in_G_attn": int(G_attn_features.sum()),
        "edges_in_JCCE_A": int(A_jcce_edges.sum()),
        "jaccard_with_A": float(jaccard),
        "correlation_with_A": correlation,
    }
    return jnp.array(G_attn_features), metrics
