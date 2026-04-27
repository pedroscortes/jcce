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
    """Abduction-Action-Prediction counterfactual head over a learned JCCE SCM.

    JCCE convention: data has shape (n_samples, n_features) for the X variables
    only. The outcome Y is supplied separately. The adjacency matrix A has
    shape (n_features+1, n_features+1) including Y at row/column Y_idx; the
    per-variable processors take (n_samples, n_features) input and predict a
    scalar per sample for one variable index. Y is treated as a sink node
    (no outgoing edges) by JCCE's enforce_outcome_sink option.

    The SCM is defined by:
    - A: (d_+, d_+) learned adjacency matrix where d_+ = n_features + 1
    - processor: adapter providing per-variable forward f_j via processor.forward
    - processor_params: list of per-variable parameter dicts (length d_+)

    Notes
    -----
    Per-variable processors (e.g., DAGAttentionAdapter) are not analytically
    invertible, so noise inference uses empirical residuals
    U_j = obs_j - f_j(X_features[parents]). This is the standard practical
    workaround when the structural equations are non-invertible neural networks.

    Y has no children, so the counterfactual forward only modifies X features
    in topological order; Y is then predicted from the final X state. This
    matches JCCE's outcome-as-sink invariant.
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
        self.n_features = self.d_plus - 1
        self._proc_name = processor.__class__.__name__
        if self.Y_idx != self.n_features:
            raise ValueError(
                f"Y_idx must equal n_features (= d_+ - 1 = {self.n_features}); got {self.Y_idx}."
            )

    # -------- Per-variable forward --------

    def _f_j(self, X_features: jnp.ndarray, j: int, A_use: jnp.ndarray) -> jnp.ndarray:
        """Forward through f_j given current X-feature state and adjacency A_use.

        X_features : (n_samples, n_features) — only the X variables (no Y col).
        Returns predicted scalar for variable j of shape (n_samples,).

        Centering policy (mirrors learn_structure's per-variable forward):
        - For X variables (j < Y_idx): output is mean-centered (default
          adapter behavior; reconstruction targets vary around mean).
        - For Y (j == Y_idx): skip_centering=True so the output retains
          its bias as a classification logit. Otherwise the centering
          would zero out f_Y's mean signal and counterfactual predictions
          would all collapse to the same value.
        """
        weights = jnp.abs(A_use[: self.n_features, j])  # (n_features,)
        X_weighted = X_features * weights[jnp.newaxis, :]  # (n_samples, n_features)

        A_struct = A_use[: self.n_features, : self.n_features]
        skip_centering = j == self.Y_idx

        if self._proc_name == "GNNAdapter":
            A_norm = A_struct / (jnp.sum(jnp.abs(A_struct), axis=0, keepdims=True) + 1e-8)
            return self.processor.forward(
                X_weighted, self.params[j], A=A_norm, skip_centering=skip_centering,
            )
        if self._proc_name in ("DAGAttentionAdapter", "CausalMambaAdapter"):
            return self.processor.forward(
                X_weighted, self.params[j], A=A_struct, skip_centering=skip_centering,
            )
        return self.processor.forward(X_weighted, self.params[j], skip_centering=skip_centering)

    # -------- Abduction --------

    def abduct(self, X_features: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """Infer noise terms U_j for every variable (X features and Y).

        Parameters
        ----------
        X_features : (n_samples, n_features) jnp.ndarray
            Observed X variables (no Y column).
        Y : (n_samples,) jnp.ndarray
            Observed outcome.

        Returns
        -------
        U : (n_samples, d_+) jnp.ndarray
            Empirical residuals; column Y_idx holds U_Y.
        """
        X_features = jnp.asarray(X_features)
        Y = jnp.asarray(Y).reshape(-1)
        n_samples = X_features.shape[0]
        U_cols = []
        for j in range(self.n_features):
            f_j_pred = self._f_j(X_features, j, self.A)
            f_j_pred = jnp.broadcast_to(jnp.atleast_1d(f_j_pred), (n_samples,))
            U_cols.append(X_features[:, j] - f_j_pred)
        # Y residual
        f_Y_pred = self._f_j(X_features, self.Y_idx, self.A)
        f_Y_pred = jnp.broadcast_to(jnp.atleast_1d(f_Y_pred), (n_samples,))
        U_cols.append(Y - f_Y_pred)
        return jnp.stack(U_cols, axis=1)  # (n_samples, d_+)

    # -------- Action + Prediction --------

    def predict_counterfactual(
        self,
        X_features: jnp.ndarray,
        Y: jnp.ndarray,
        t_idx: int,
        t_prime: float | jnp.ndarray,
        U: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict (X_features, Y) under do(X[t_idx] = t_prime) using abducted noise.

        Parameters
        ----------
        X_features : (n_samples, n_features) jnp.ndarray
            Observed X variables, used as initial state.
        Y : (n_samples,) jnp.ndarray
            Observed Y, used to abduct U_Y if U is not provided.
        t_idx : int
            Index of the intervened variable; must be < n_features.
        t_prime : float or (n_samples,) array
            Counterfactual treatment value(s).
        U : optional (n_samples, d_+) jnp.ndarray
            Pre-computed noise terms (skip abduction if provided).

        Returns
        -------
        X_pred : (n_samples, n_features) jnp.ndarray
        Y_pred : (n_samples,) jnp.ndarray
        """
        if t_idx == self.Y_idx:
            raise ValueError("Cannot intervene on Y (sink node).")

        X_features = jnp.asarray(X_features)
        Y = jnp.asarray(Y).reshape(-1)
        n_samples = X_features.shape[0]

        if U is None:
            U = self.abduct(X_features, Y)

        # Action: zero incoming edges into t_idx (intervention severs parents).
        A_mod = self.A.at[:, t_idx].set(0.0)

        # Topological order over X features only (Y is sink, predicted last).
        # Slice A to X-X submatrix for the sort.
        A_xx = np.asarray(A_mod[: self.n_features, : self.n_features])
        order = _topological_sort(A_xx, threshold=self.edge_threshold)

        # Initialize state to observed.
        X_pred = X_features

        # Prepare scalar/array t_prime
        t_prime_arr = jnp.broadcast_to(jnp.asarray(t_prime, dtype=X_features.dtype), (n_samples,))

        for j in order:
            if j == t_idx:
                X_pred = X_pred.at[:, j].set(t_prime_arr)
                continue
            f_j_pred = self._f_j(X_pred, j, A_mod)
            f_j_pred = jnp.broadcast_to(jnp.atleast_1d(f_j_pred), (n_samples,))
            X_pred = X_pred.at[:, j].set(f_j_pred + U[:, j])

        # Predict Y from final X state using f_Y + U_Y noise.
        Y_pred = self._f_j(X_pred, self.Y_idx, A_mod)
        Y_pred = jnp.broadcast_to(jnp.atleast_1d(Y_pred), (n_samples,)) + U[:, self.Y_idx]

        return X_pred, Y_pred

    # -------- Convenience: CATE --------

    def cate(
        self,
        X_features: jnp.ndarray,
        Y: jnp.ndarray,
        t_idx: int,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> jnp.ndarray:
        """Sample-wise CATE via paired counterfactuals using shared noise."""
        U = self.abduct(X_features, Y)
        _, Y0 = self.predict_counterfactual(X_features, Y, t_idx, t0, U=U)
        _, Y1 = self.predict_counterfactual(X_features, Y, t_idx, t1, U=U)
        return Y1 - Y0

    # -------- Round-trip self-test --------

    def round_trip(
        self, X_features: jnp.ndarray, Y: jnp.ndarray, t_idx: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Abduct then predict with observed treatment; should recover (X, Y)."""
        X_features = jnp.asarray(X_features)
        Y = jnp.asarray(Y).reshape(-1)
        U = self.abduct(X_features, Y)
        T_observed = X_features[:, t_idx]
        X_pred, Y_pred = self.predict_counterfactual(X_features, Y, t_idx, T_observed, U=U)
        x_mse = jnp.mean((X_pred - X_features) ** 2, axis=1)
        y_mse = (Y_pred - Y) ** 2
        return x_mse, y_mse
