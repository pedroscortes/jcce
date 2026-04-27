"""
Abduction-Action-Prediction counterfactual generation for DAG-Attention.

Implements Pearl's three-step counterfactual computation on the SCM defined
by a learned adjacency matrix A and per-variable structural equations f_j
(provided by a processor adapter).

Step 1 — Abduction. Given observed (X), infer noise terms:
    U_j = X_j - f_j(X[parents(j)] under A)

Step 2 — Action. Modify the SCM:
    do(X_T = t_prime)
    A_modified = A with column t_idx zeroed (sever incoming edges into T)

Step 3 — Prediction. Forward-propagate in topological order under A_modified,
using the abducted U_j as the noise terms.

Counterfactual ATE: CATE(x) = E[Y(t=1) - Y(t=0) | X=x], computed via
sample-wise difference Y_pred(t=1) - Y_pred(t=0) using the same noise.

Currently scoped to inference (eager mode). Loss-integration into JIT'd
training loop is deferred to a separate sprint and would require a
jit-safe topological sort (already available via causal_mamba.topological_sort_from_adjacency).
"""

from __future__ import annotations

from collections import deque

import jax.numpy as jnp
import numpy as np


def _topological_sort(dag: np.ndarray, threshold: float = 0.05) -> list[int]:
    """Kahn's algorithm in pure numpy. Cycle nodes appended in original order."""
    n = dag.shape[0]
    binary = (np.abs(dag) > threshold).astype(np.int64)
    in_degree = binary.sum(axis=0)
    queue = deque(int(i) for i in range(n) if in_degree[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in range(n):
            if binary[node, child] > 0:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    if len(order) < n:
        # Cycle: append remaining nodes in index order to keep output complete.
        seen = set(order)
        order.extend(i for i in range(n) if i not in seen)
    return order


class AAPCounterfactual:
    """Abduction-Action-Prediction counterfactual head over a learned SCM.

    The SCM is defined by:
    - A: (d_+, d_+) learned adjacency matrix
    - processor: adapter providing per-variable forward f_j via processor.forward
    - processor_params: list of per-variable parameter dicts

    Notes
    -----
    Per-variable processors (e.g., DAGAttentionAdapter) are not analytically
    invertible, so noise inference uses empirical residuals
    U_j = X_j - f_j(X[parents]). This is the standard practical workaround
    when the structural equations are non-invertible neural networks.

    The topological sort uses a CPU-side numpy implementation via
    jcce.counterfactual.causal_constraints.topological_sort. For training-loop
    integration (JIT'd loss path), use causal_mamba.topological_sort_from_adjacency
    which wraps the same algorithm in jax.pure_callback.
    """

    def __init__(
        self,
        processor,
        A: jnp.ndarray,
        processor_params: list,
        Y_idx: int,
        edge_threshold: float = 0.05,
    ):
        self.processor = processor
        self.A = jnp.asarray(A)
        self.params = processor_params
        self.Y_idx = int(Y_idx)
        self.edge_threshold = float(edge_threshold)
        self.d_plus = int(self.A.shape[0])
        self._proc_name = processor.__class__.__name__

    # -------- Abduction --------

    def _f_j(self, X_curr: jnp.ndarray, j: int, A_use: jnp.ndarray) -> jnp.ndarray:
        """Forward through f_j given current X state and adjacency A_use.

        Returns predicted scalar X_j of shape (n_samples,).
        """
        weights = jnp.abs(A_use[:, j])  # (d_+,)
        X_weighted = X_curr * weights[jnp.newaxis, :]  # (n_samples, d_+)

        # Dispatch by adapter type — mirrors learn_structure's per-variable forward.
        if self._proc_name == "GNNAdapter":
            A_norm = A_use / (jnp.sum(jnp.abs(A_use), axis=0, keepdims=True) + 1e-8)
            return self.processor.forward(X_weighted, self.params[j], A=A_norm)
        if self._proc_name in ("DAGAttentionAdapter", "CausalMambaAdapter"):
            return self.processor.forward(X_weighted, self.params[j], A=A_use)
        return self.processor.forward(X_weighted, self.params[j])

    def abduct(self, X: jnp.ndarray) -> jnp.ndarray:
        """Infer noise terms U_j = X_j - f_j(X[parents(j)]) for every variable.

        Parameters
        ----------
        X : (n_samples, d_+) jnp.ndarray
            Observed data including the outcome at column Y_idx.

        Returns
        -------
        U : (n_samples, d_+) jnp.ndarray
            Per-variable empirical residuals (noise terms).
        """
        X = jnp.asarray(X)
        n_samples = X.shape[0]
        U_cols = []
        for j in range(self.d_plus):
            f_j_pred = self._f_j(X, j, self.A)
            # Predictions might be 1D (n_samples,) or scalar — coerce to (n_samples,).
            f_j_pred = jnp.broadcast_to(jnp.atleast_1d(f_j_pred), (n_samples,))
            residual = X[:, j] - f_j_pred
            U_cols.append(residual)
        return jnp.stack(U_cols, axis=1)  # (n_samples, d_+)

    # -------- Action + Prediction --------

    def predict_counterfactual(
        self,
        X: jnp.ndarray,
        t_idx: int,
        t_prime: float | jnp.ndarray,
        U: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict X under do(X[t_idx] = t_prime) using abducted noise U.

        Parameters
        ----------
        X : (n_samples, d_+) jnp.ndarray
            Observed data; used as initial state for non-intervened variables
            and as fallback for variables outside the topological reach of t_idx.
        t_idx : int
            Index of the intervened variable.
        t_prime : float or (n_samples,) array
            Counterfactual treatment value(s). Scalar broadcasts.
        U : (n_samples, d_+) jnp.ndarray
            Abducted noise terms from abduct(X).

        Returns
        -------
        X_pred : (n_samples, d_+) jnp.ndarray
            Counterfactual predicted state for every variable.
        """
        X = jnp.asarray(X)
        U = jnp.asarray(U)
        n_samples = X.shape[0]

        # Action: zero incoming edges into t_idx (intervention severs parents).
        A_mod = self.A.at[:, t_idx].set(0.0)

        # Topological order under MODIFIED adjacency. Use eager numpy sort.
        order = _topological_sort(np.asarray(A_mod), threshold=self.edge_threshold)

        # Initialize counterfactual state to observed (will be overwritten below
        # for variables in topo order). This handles vars with no parents, which
        # are not affected by the intervention and equal their observed values.
        X_pred = X

        # Prepare scalar / array t_prime
        t_prime_arr = jnp.broadcast_to(jnp.asarray(t_prime, dtype=X.dtype), (n_samples,))

        for j in order:
            if j == t_idx:
                X_pred = X_pred.at[:, j].set(t_prime_arr)
                continue
            # Predict using f_j with current state and modified A; add noise.
            f_j_pred = self._f_j(X_pred, j, A_mod)
            f_j_pred = jnp.broadcast_to(jnp.atleast_1d(f_j_pred), (n_samples,))
            X_pred = X_pred.at[:, j].set(f_j_pred + U[:, j])

        return X_pred

    # -------- Convenience: CATE --------

    def cate(
        self,
        X: jnp.ndarray,
        t_idx: int,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> jnp.ndarray:
        """Sample-wise CATE via paired counterfactuals using the same noise.

        Returns (n_samples,) array of Y(t1) - Y(t0).
        """
        U = self.abduct(X)
        Y0 = self.predict_counterfactual(X, t_idx, t0, U)[:, self.Y_idx]
        Y1 = self.predict_counterfactual(X, t_idx, t1, U)[:, self.Y_idx]
        return Y1 - Y0

    # -------- Round-trip self-test --------

    def round_trip(self, X: jnp.ndarray, t_idx: int) -> jnp.ndarray:
        """Sanity check: abduct(X), then predict using observed treatment.

        Returns the per-sample MSE between observed and predicted X. Should
        be near zero (within float tolerance) when the noise is correctly
        captured and the topological forward pass is consistent.
        """
        X = jnp.asarray(X)
        U = self.abduct(X)
        T_observed = X[:, t_idx]
        X_pred = self.predict_counterfactual(X, t_idx, T_observed, U)
        return jnp.mean((X_pred - X) ** 2, axis=1)  # (n_samples,)
