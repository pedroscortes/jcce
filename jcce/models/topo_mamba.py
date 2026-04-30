"""
TopoMamba: Mamba SSM with topological-sort + DAG-gating mechanisms.

Renamed from "CausalMamba" 2026-04-28 due to naming collisions with two
unrelated 2025 papers: Zhan & Cheng (Berkeley, arXiv:2510.17318, NLP rumor
detection) and Bae & Cha (arXiv:2511.16191, fMRI BOLD causality). "TopoMamba"
more precisely captures what the mechanisms do — Mechanism 1 (Sinkhorn) is
literally a differentiable topological sort, Mechanism 2 (DAG-gating) selects
A's columns by topological position. Backward-compat alias
``CausalMambaProcessor = TopoMambaProcessor`` is preserved at the bottom of
this module.

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


# ---------------------------------------------------------------------------
# Sprint 3 Mechanism 2: DAG-gated state transitions.
#
# Path (A) per the multi-chat coordination policy: duplicate the relevant
# selective-scan + MambaBlock + MambaProcessor structure here so the base
# Mamba implementation in jcce/models/mamba.py stays untouched. That base
# is shared with classification_processors.py and processor_wrappers.py
# (two paths via MambaProcessorWrapper); patching its scan signature would
# ripple beyond CausalMamba.
#
# The gated step is:
#   h_new = deltaA_t * gate_t * h + deltaB_u_t
# with gate_t in R^{d_state} computed per skill spec from A's columns
# selected by the variable at position t.
# ---------------------------------------------------------------------------

import math  # local import: only the gated path uses it


def _gated_selective_scan(u, delta, A, B, C, D, gate):
    """Selective scan with a per-position multiplicative gate on the hidden state.

    Mirrors jcce.models.mamba.selective_scan but accepts a (seq_len, d_state)
    gate that multiplies h_{t-1} element-wise inside the recurrence:
        h_new = deltaA_t * gate_t * h + deltaB_u_t

    The gate is broadcast against (batch, d_inner, d_state) so the same
    d_state-shaped gate applies across batch and inner-channel axes.

    Parameters mirror selective_scan; ``gate`` is shape (seq_len, d_state).
    """
    batch, seq_len, d_in = u.shape
    d_state = A.shape[1]

    delta = jnp.clip(delta, 0.001, 0.1)
    deltaA = jnp.exp(jnp.einsum("bld,dn->bldn", delta, A))
    deltaB_u = jnp.einsum("bld,bln,bld->bldn", delta, B, u)

    def scan_fn(h, inputs):
        deltaA_t, deltaB_u_t, C_t, gate_t = inputs
        # gate_t: (d_state,) -> broadcasts over (batch, d_inner, d_state).
        h_new = deltaA_t * gate_t * h + deltaB_u_t
        y_t = jnp.einsum("bdn,bn->bd", h_new, C_t)
        return h_new, y_t

    h_0 = jnp.zeros((batch, d_in, d_state))

    _, y = jax.lax.scan(
        scan_fn,
        h_0,
        (
            deltaA.transpose(1, 0, 2, 3),
            deltaB_u.transpose(1, 0, 2, 3),
            C.transpose(1, 0, 2),
            gate,  # already (seq_len, d_state)
        ),
    )

    y = y.transpose(1, 0, 2)
    return y + u * D


class _GatedMambaBlock(nn.Module):
    """Mamba block with a per-position gate inserted into the SSM recurrence.

    Direct copy of jcce.models.mamba.MambaBlock parameter setup; only the
    selective_scan call is replaced by _gated_selective_scan. ``__call__``
    accepts an extra ``gate`` argument of shape (seq_len, d_state).
    """

    d_model: int
    d_state: int = 8
    d_conv: int = 4
    expand: int = 1

    def setup(self):
        self.d_inner = self.expand * self.d_model
        dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d = nn.Conv(
            features=self.d_inner,
            kernel_size=(self.d_conv,),
            feature_group_count=self.d_inner,
            padding="VALID",
            use_bias=True,
        )
        self.x_proj = nn.Dense(dt_rank + 2 * self.d_state, use_bias=False)
        self.dt_proj = nn.Dense(
            self.d_inner,
            use_bias=True,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.constant(0.1),
        )

        A_init = jnp.repeat(
            jnp.arange(1, self.d_state + 1)[None, :], self.d_inner, axis=0
        )
        self.A_log = self.param("A_log", lambda key: jnp.log(A_init))
        self.D = self.param("D", lambda key: jnp.ones(self.d_inner))
        self.out_proj = nn.Dense(self.d_model, use_bias=False)

    def __call__(self, x, gate):
        batch, seq_len, d_model = x.shape

        x_and_z = self.in_proj(x)
        x_proj, z = jnp.split(x_and_z, 2, axis=-1)

        x_conv_input = jnp.pad(
            x_proj,
            ((0, 0), (self.d_conv - 1, 0), (0, 0)),
            mode="constant",
        )
        x_conv = self.conv1d(x_conv_input)
        x_conv = nn.silu(x_conv)

        x_ssm = self.x_proj(x_conv)
        dt_rank = math.ceil(self.d_model / 16)
        dt, B, C = jnp.split(x_ssm, [dt_rank, dt_rank + self.d_state], axis=-1)
        dt = self.dt_proj(dt)
        dt = nn.softplus(dt)

        A = -jnp.exp(self.A_log)

        y = _gated_selective_scan(x_conv, dt, A, B, C, self.D, gate)

        y = y * nn.silu(z)
        return self.out_proj(y)


class _GatedMambaProcessor(nn.Module):
    """Stack of _GatedMambaBlock layers. Mirrors MambaProcessor; threads gate
    through every layer (residual connections preserved).
    """

    d_model: int = 32
    n_layers: int = 1
    d_state: int = 8
    d_conv: int = 4
    expand: int = 1

    @nn.compact
    def __call__(self, z_sequence: jnp.ndarray, gate: jnp.ndarray) -> jnp.ndarray:
        h = z_sequence
        for i in range(self.n_layers):
            block = _GatedMambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                name=f"gated_mamba_block_{i}",
            )
            h = block(h, gate) + h
        return h


def _compute_dag_gate(
    A: jnp.ndarray,
    topo_order: Optional[jnp.ndarray],
    perm_matrix: Optional[jnp.ndarray],
    d_state: int,
    name: str = "dag_gate",
    identity_floor: float = 0.1,
) -> jnp.ndarray:
    """Return the per-position DAG gate of shape (n_vars, d_state).

    For each sequence position t, the gate is computed from the parent
    signature of "the variable at position t":

      hard sort:  parent_signature[t, :] = A[:, topo_order[t]]
      soft sort:  parent_signature[t, :] = sum_j P[t, j] * A[:, j]
                                        = (P @ A.T)[t, :]
      identity:   parent_signature[t, :] = A[:, t]   (= A.T[t, :])

    Then ``Dense(d_state)(parent_signature)`` gives the (n_vars, d_state)
    pre-gate logits, and the identity floor prevents total state collapse:
        gate = identity_floor + (1 - identity_floor) * sigmoid(logits)

    Must be called from inside a flax @nn.compact context (the Dense layer
    is registered as a parameter of the calling module).
    """
    n_vars = A.shape[0]
    if perm_matrix is not None:
        # Row t of parent_signature is sum_j P[t, j] * A[:, j].
        # Equivalent matrix form: (P @ A^T) since A^T[j, :] = A[:, j].
        parent_signature = perm_matrix @ A.T
    elif topo_order is not None:
        # parent_signature[t, :] = A[:, topo_order[t]] = A.T[topo_order[t], :]
        parent_signature = A.T[topo_order]
    else:
        parent_signature = A.T  # identity

    gate_logits = nn.Dense(d_state, name=name)(parent_signature)
    return identity_floor + (1.0 - identity_floor) * jax.nn.sigmoid(gate_logits)


class TopoMambaProcessor(nn.Module):
    """
    Mamba SSM with topological (causal-DAG-derived) variable ordering.

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
    enable_gating: bool = False
    # Q3.1.F (2026-04-30): Direction 1 — Treatment-Conditioned SSM.
    # When t_idx is set (>= 0), the input projection is modulated by a learned
    # function of z[:, t_idx]. Mechanism: each batch sample's projected input
    # is scaled by its T value, so the SSM's selective parameters (Δ, B, C)
    # depend on T per sample. The model cannot route around T because T is
    # in the input-projection equation, not just one of the input features.
    # No-op when t_idx is None or t_idx < 0.
    t_idx: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        topo_order: Optional[jnp.ndarray] = None,
        perm_matrix: Optional[jnp.ndarray] = None,
        A: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Process variables in causal-aware order using Mamba SSM.

        Exactly one of topo_order or perm_matrix should be set (or neither,
        in which case the standard scan order is used). If both are set,
        perm_matrix wins.

        When ``enable_gating=True`` and A is provided, the Sprint 3
        Mechanism 2 DAG-gated SSM is used: a per-position gate in
        R^{d_state} multiplies h_{t-1} inside the recurrence, computed from
        A's columns selected (hard) or soft-mixed (Sinkhorn) by the
        variable at position t.

        Args:
            z: (batch_size, n_vars) input features.
            topo_order: (n_vars,) hard permutation of variable indices.
            perm_matrix: (n_vars, n_vars) soft doubly-stochastic permutation.
                P[i, j] approx 1 means variable j is at sequence position i.
                When provided, sorting and unsorting use matrix multiplies
                so the gradient can flow back to A through P.
            A: (n_vars, n_vars) adjacency. Required when enable_gating=True.
                Ignored otherwise (the ordering is already encoded in
                topo_order or perm_matrix).
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

        # Q3.1.F (2026-04-30): Direction 1 — Treatment-Conditioned input
        # projection. When t_idx is configured, modulate the projected input
        # by a learned function of z[:, t_idx]. Each batch sample gets a
        # T-dependent scaling factor, forcing the SSM's selective parameters
        # to depend on T per sample.
        if self.t_idx is not None and self.t_idx >= 0:
            t_values = z[:, self.t_idx:self.t_idx + 1]  # (B, 1) — pre-sort z
            t_modulation = nn.Dense(self.d_model, name="t_modulation")(t_values)
            # Multiplicative gate: 1.0 + tanh(t_mod) → keeps modulation
            # bounded ([0, 2] effective range) and reduces to no-op when
            # t_modulation = 0 at init.
            t_gate = 1.0 + jnp.tanh(t_modulation)  # (B, d_model)
            z_projected = z_projected * t_gate[:, None, :]  # (B, N, d_model)

        # Step 3: Apply Mamba — either gated (Mechanism 2) or standard.
        if self.enable_gating and A is not None:
            gate = _compute_dag_gate(
                A,
                topo_order=topo_order,
                perm_matrix=perm_matrix,
                d_state=self.d_state,
                name="dag_gate",
            )
            mamba = _GatedMambaProcessor(
                d_model=self.d_model,
                n_layers=self.n_layers,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
            )
            h_sorted = mamba(z_projected, gate)
        else:
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


# ============================================================================
# Backward-compatibility alias (renamed 2026-04-28; see module docstring)
# ============================================================================

CausalMambaProcessor = TopoMambaProcessor
