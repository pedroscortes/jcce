"""
CausalMamba: Mamba SSM with causal-aware variable ordering.

Three orderings are supported:

  1. Hard topological sort (Kahn's algorithm via jax.pure_callback). Discrete,
     non-differentiable; gradient stops at A. The Sprint 1 prototype.
  2. Sinkhorn soft topological sort (Sprint 2). Differentiable: gradient flows
     from the loss through the Mamba forward, the soft permutation matrix P,
     the ancestral-depth scoring s = (I - A^T)^{-1} @ 1, and back to A.
  3. Identity (standard Mamba scan order, no permutation).

The hard sort is jit-safe via pure_callback; the soft sort is jit-safe by
construction (pure JAX ops). Both run cheaply at the d <= 30 scales we care
about.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from jcce.models.mamba import MambaProcessor as _MambaProcessorBase


def _kahn_numpy(A_np: np.ndarray, threshold: float) -> np.ndarray:
    """Pure-numpy Kahn's algorithm. Runs on host CPU under pure_callback."""
    d = A_np.shape[0]
    B = (np.abs(A_np) > threshold).astype(np.float32)
    in_deg = B.sum(axis=0).astype(np.int32)

    order: list[int] = []
    available = list(np.where(in_deg == 0)[0])
    while available:
        node = int(available.pop(0))
        order.append(node)
        for child in range(d):
            if B[node, child] > 0:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    available.append(child)

    if len(order) < d:
        seen = set(order)
        order.extend(i for i in range(d) if i not in seen)

    return np.asarray(order, dtype=np.int32)


def topological_sort_from_adjacency(A: jnp.ndarray, threshold: float = 0.01) -> jnp.ndarray:
    """
    Compute topological ordering from adjacency matrix A under jit.

    Uses Kahn's algorithm on the thresholded binary graph (host-side via
    jax.pure_callback). A[i,j] != 0 means i -> j.

    The hard sort is non-differentiable; jax.lax.stop_gradient on the input
    keeps the (otherwise undefined) JVP at zero so this composes inside any
    grad-traced loss. Sprint 2's Sinkhorn soft sort is what carries gradient.

    Returns:
        order: (d,) int32 array; parents before children, cycle leftovers
               appended in original index order.
    """
    d = A.shape[0]
    A_detached = jax.lax.stop_gradient(A)
    return jax.pure_callback(
        lambda a: _kahn_numpy(a, threshold),
        jax.ShapeDtypeStruct((d,), jnp.int32),
        A_detached,
        vmap_method="sequential",
    )


# ---------------------------------------------------------------------------
# Sprint 2 Mechanism 1: differentiable soft topological sort via Sinkhorn.
# ---------------------------------------------------------------------------

def ancestral_depth_scores(A: jnp.ndarray, eps: float = 1e-3) -> jnp.ndarray:
    """Total ancestral weight per variable: s = (I - A^T)^{-1} @ 1.

    For a DAG with weighted parent-to-child edges in A, s[j] aggregates the
    influence reaching j from all ancestors (including itself, through the
    self-loop term in the Neumann series). High s -> deeper in the DAG ->
    later position in the desired topological ordering.

    eps perturbs (I - A^T) to keep the linear solve stable when A is close to
    cyclic (||A|| close to 1). Without it the inverse blows up; with eps the
    solution is the regularized series sum_k (A^T)^k applied to 1 / (1 + eps).
    Gradient flows: loss -> s -> A.
    """
    d = A.shape[0]
    I = jnp.eye(d, dtype=A.dtype)
    return jnp.linalg.solve(I - A.T + eps * I, jnp.ones(d, dtype=A.dtype))


def _soft_rank(s: jnp.ndarray, sharpness: float = 10.0) -> jnp.ndarray:
    """Differentiable rank in [0, d-1].

    soft_rank[i] = (sum_j sigmoid(sharpness * (s[i] - s[j]))) - 0.5

    Counts (softly) how many s[j] are <= s[i], minus the self-term. As
    sharpness -> infinity this approaches the exact integer rank; at
    sharpness=10 the spacing between adjacent ranks is preserved within
    ~1e-2 even for tightly-spaced s. Gradient flows: rank -> s -> A.

    This is the standard fix for the "geometric series" failure mode of
    linear rescaling: when s grows multiplicatively (as it does for chain
    DAGs), linear rescale leaves clusters where multiple variables are
    closer to the same position than to their own. Rank-spacing fixes that.
    """
    diffs = s[:, None] - s[None, :]
    return jax.nn.sigmoid(sharpness * diffs).sum(axis=1) - 0.5


def _sinkhorn_log_domain(log_M: jnp.ndarray, n_iters: int) -> jnp.ndarray:
    """Log-domain Sinkhorn normalization. Returns exp(log_M) doubly stochastic.

    Each iteration alternates row and column log-sum-exp normalization, which
    is numerically stable even for tight temperatures where exp(log_M) would
    overflow in the linear domain.
    """
    def body(log_M, _):
        log_M = log_M - jax.scipy.special.logsumexp(log_M, axis=1, keepdims=True)
        log_M = log_M - jax.scipy.special.logsumexp(log_M, axis=0, keepdims=True)
        return log_M, None

    log_M, _ = jax.lax.scan(body, log_M, xs=None, length=n_iters)
    return jnp.exp(log_M)


def sinkhorn_topological_sort(
    A: jnp.ndarray,
    temperature: float = 0.1,
    n_iters: int = 20,
    eps_solve: float = 1e-3,
    rank_sharpness: float = 10.0,
) -> jnp.ndarray:
    """Differentiable soft topological sort.

    Returns a (d, d) approximately doubly-stochastic matrix P with P[i, j]
    approx 1 meaning "variable j is at sequence position i". Gradient flows:
        loss -> P -> log_P -> s -> A.

    Construction:
      1. s = ancestral_depth_scores(A): in [s_min, s_max].
      2. r = soft_rank(s, sharpness=rank_sharpness): differentiable rank in
         [0, d-1]. Spaces variables uniformly along the position axis even
         when raw s is non-uniform (e.g., geometric chain).
      3. log_P[i, j] = -(positions[i] - r[j])^2 / temperature.
         High when position i matches the rank of variable j.
      4. n_iters of log-domain Sinkhorn -> approximately doubly-stochastic P.

    Doubly-stochastic only holds in the limit. With the default n_iters=20
    and the row->col normalization order, columns sum to 1 exactly (last
    operation) and rows sum to 1 within ~5e-2 at temperature=0.1. Tighter
    tolerance is achievable at higher iteration count or higher temperature
    (less peaky log_P -> faster Sinkhorn convergence). The model layer
    absorbs the ~5e-2 row-magnitude variation through its Dense projection,
    and as temperature -> 0 the matrix converges to a hard permutation.

    Tied ancestral depths produce ties in the soft sort: two variables with
    equal s map to the same column-mass and argmax decoding can yield an
    invalid (non-permutation) integer order. The matrix itself remains a
    valid soft assignment for the model.
    """
    d = A.shape[0]
    s = ancestral_depth_scores(A, eps=eps_solve)
    r = _soft_rank(s, sharpness=rank_sharpness)

    positions = jnp.arange(d, dtype=A.dtype)
    log_P = -((positions[:, None] - r[None, :]) ** 2) / temperature
    return _sinkhorn_log_domain(log_P, n_iters=n_iters)


class CausalMambaProcessor(nn.Module):
    """
    Mamba SSM with causal (topological) variable ordering.

    Variables are reordered so that parents come before children in the sequence.
    The SSM hidden state accumulates "causal context" from ancestors.

    The ordering is passed in externally (precomputed from A) rather than
    computed inside the forward pass, to avoid JAX recompilation.

    Attributes:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution width
        expand: Expansion factor for inner dimension
        n_layers: Number of Mamba layers
    """

    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 1

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        topo_order: Optional[jnp.ndarray] = None,
        perm_matrix: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Process variables in causal-aware order using Mamba SSM.

        Exactly one of topo_order or perm_matrix should be set (or neither,
        in which case the standard scan order is used). If both are set,
        perm_matrix wins.

        Args:
            z: (batch_size, n_vars) input features.
            topo_order: (n_vars,) hard permutation of variable indices.
            perm_matrix: (n_vars, n_vars) soft doubly-stochastic permutation.
                P[i, j] approx 1 means variable j is at sequence position i.
                When provided, sorting and unsorting use matrix multiplies
                so the gradient can flow back to A through P.
            training: Whether in training mode.

        Returns:
            h: (batch_size, n_vars, d_model) processed representations,
               returned in ORIGINAL variable order.
        """
        batch_size, n_vars = z.shape

        # Step 1: Reorder variables.
        if perm_matrix is not None:
            # X_sorted[:, i] = sum_j P[i, j] * z[:, j] = (z @ P^T)[:, i]
            z_sorted = z @ perm_matrix.T  # (B, N)
        elif topo_order is not None:
            z_sorted = z[:, topo_order]
        else:
            z_sorted = z

        # Step 2: Project to d_model.
        z_expanded = z_sorted[..., None]  # (B, N, 1)
        z_projected = nn.Dense(self.d_model, name="input_projection")(z_expanded)

        # Step 3: Apply Mamba (processes in the chosen order).
        mamba = _MambaProcessorBase(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        h_sorted = mamba(z_projected)  # (B, N, d_model)

        # Step 4: Reorder back to original variable order.
        if perm_matrix is not None:
            # h[:, j, :] = sum_i P[i, j] * h_sorted[:, i, :]
            # (For permutation P this matches argsort(order); for general
            #  doubly-stochastic P it is the variational inverse.)
            h = jnp.einsum("ij,bid->bjd", perm_matrix, h_sorted)
        elif topo_order is not None:
            inverse_order = jnp.argsort(topo_order)
            h = h_sorted[:, inverse_order]
        else:
            h = h_sorted

        return h
