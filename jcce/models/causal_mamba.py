"""
CausalMamba: Mamba SSM with topological variable ordering.

Key difference from standard MambaProcessor:
Variables are reordered according to the topological sort of the learned DAG A
before being fed to the SSM. This means parents are processed before children,
so the SSM hidden state carries "causal context" from ancestors to descendants.

The topological sort runs host-side via jax.pure_callback so the forward path
is jit-safe; the order is recomputed every call (cheap for d <= 30).

This is a NEW processor variant — the original MambaProcessor is unchanged.
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

    Returns:
        order: (d,) int32 array; parents before children, cycle leftovers
               appended in original index order.
    """
    d = A.shape[0]
    return jax.pure_callback(
        lambda a: _kahn_numpy(a, threshold),
        jax.ShapeDtypeStruct((d,), jnp.int32),
        A,
        vmap_method="sequential",
    )


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
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Process variables in topological order using Mamba SSM.

        Args:
            z: (batch_size, n_vars) input features
            topo_order: (n_vars,) topological ordering of variables.
                        If None, uses identity ordering (standard Mamba behavior).
            training: Whether in training mode

        Returns:
            h: (batch_size, n_vars, d_model) processed representations
               in ORIGINAL variable order (reordered back after Mamba)
        """
        batch_size, n_vars = z.shape

        # Step 1: Reorder variables by topological sort (if provided)
        if topo_order is not None:
            z_sorted = z[:, topo_order]  # (B, N) reordered
        else:
            z_sorted = z

        # Step 2: Project to d_model
        z_expanded = z_sorted[..., None]  # (B, N, 1)
        z_projected = nn.Dense(self.d_model, name="input_projection")(z_expanded)

        # Step 3: Apply Mamba (processes in topological order)
        mamba = _MambaProcessorBase(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        h_sorted = mamba(z_projected)  # (B, N, d_model)

        # Step 4: Reorder back to original variable order
        if topo_order is not None:
            # Inverse permutation
            inverse_order = jnp.argsort(topo_order)
            h = h_sorted[:, inverse_order]
        else:
            h = h_sorted

        return h
