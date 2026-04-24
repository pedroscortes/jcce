"""
CausalMamba: Mamba SSM with topological variable ordering.

Key difference from standard MambaProcessor:
Variables are reordered according to the topological sort of the learned DAG A
before being fed to the SSM. This means parents are processed before children,
so the SSM hidden state carries "causal context" from ancestors to descendants.

The topological order is recomputed periodically (every reorder_interval epochs),
not every step, to avoid JAX recompilation.

This is a NEW processor variant — the original MambaProcessor is unchanged.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from jcce.models.mamba import MambaProcessor as _MambaProcessorBase


def topological_sort_from_adjacency(A: jnp.ndarray, threshold: float = 0.01) -> jnp.ndarray:
    """
    Compute topological ordering from adjacency matrix A.

    Uses Kahn's algorithm on the thresholded binary graph.
    A[i,j] != 0 means i -> j (i is parent of j).

    Args:
        A: (d, d) adjacency matrix (continuous weights)
        threshold: minimum absolute weight to consider an edge

    Returns:
        order: (d,) array of variable indices in topological order
               (parents before children)
    """
    d = A.shape[0]
    # Binary adjacency
    B = (jnp.abs(A) > threshold).astype(jnp.float32)

    # In-degree for each node
    in_degree = jnp.sum(B, axis=0).astype(jnp.int32)

    # Kahn's algorithm (must be done in Python, not JAX-traced)
    # Convert to numpy for the algorithm
    import numpy as np
    in_deg = np.array(in_degree)
    B_np = np.array(B)

    order = []
    available = list(np.where(in_deg == 0)[0])

    while available:
        node = available.pop(0)
        order.append(node)
        for child in range(d):
            if B_np[node, child] > 0:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    available.append(child)

    # If not all nodes visited (cycle), append remaining in original order
    if len(order) < d:
        remaining = [i for i in range(d) if i not in order]
        order.extend(remaining)

    return jnp.array(order, dtype=jnp.int32)


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
        z_projected = nn.Dense(self.d_model, name='input_projection')(z_expanded)

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
