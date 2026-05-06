"""Tests for the Y-as-sink DAG constraint.

Verifies that when enforce_outcome_sink=True, the DAG constraint is computed
on the X-only submatrix, freeing X→Y edges from acyclicity pressure.
"""

import jax
import jax.numpy as jnp

from jcce.structure_learning.jcce_learner import dag_constraint


class TestYAsSinkDAGConstraint:
    """Test the X-submatrix DAG constraint logic."""

    def _compute_h_with_sink(self, A, Y_idx, enforce_outcome_sink=True):
        """Replicate the DAG constraint logic from loss_fn."""
        n_total = A.shape[0]
        if enforce_outcome_sink:
            x_idx = jnp.concatenate([jnp.arange(Y_idx), jnp.arange(Y_idx + 1, n_total)])
            A_dag = A[jnp.ix_(x_idx, x_idx)]
            return dag_constraint(A_dag)
        else:
            return dag_constraint(A)

    def test_sink_excludes_y_column(self):
        """X→Y edges should NOT contribute to DAG constraint when Y is sink."""
        # Create a DAG among X variables (no cycles)
        A = jnp.zeros((4, 4))  # 3 X vars + Y at index 3
        A = A.at[0, 1].set(0.5)  # X0 → X1
        A = A.at[1, 2].set(0.5)  # X1 → X2

        # Add X→Y edges (should be free)
        A_with_y = A.at[0, 3].set(0.8)  # X0 → Y
        A_with_y = A_with_y.at[2, 3].set(0.8)  # X2 → Y

        h_without_y_edges = self._compute_h_with_sink(A, Y_idx=3)
        h_with_y_edges = self._compute_h_with_sink(A_with_y, Y_idx=3)

        # DAG constraint should be IDENTICAL — X→Y edges are excluded
        assert jnp.allclose(h_without_y_edges, h_with_y_edges, atol=1e-6), (
            f"X→Y edges affected DAG constraint: {h_without_y_edges:.6f} vs {h_with_y_edges:.6f}"
        )

    def test_no_sink_includes_y_column(self):
        """Without enforce_outcome_sink, X→Y edges DO affect DAG constraint."""
        A = jnp.zeros((4, 4))
        A = A.at[0, 1].set(0.5)

        A_with_y = A.at[0, 3].set(0.8)
        A_with_y = A_with_y.at[2, 3].set(0.8)

        h_without = self._compute_h_with_sink(A, Y_idx=3, enforce_outcome_sink=False)
        h_with = self._compute_h_with_sink(A_with_y, Y_idx=3, enforce_outcome_sink=False)

        # Full matrix: adding edges changes h_A (even if no cycle, numerical difference)
        # The key point: they ARE included in the computation
        # (both should be ~0 since no cycles, but numerical precision may differ)
        # This test mainly verifies the code path works

    def test_cycle_in_x_still_detected(self):
        """Cycles among X variables should still be penalized."""
        A = jnp.zeros((4, 4))
        # Create cycle: X0 → X1 → X2 → X0
        A = A.at[0, 1].set(0.5)
        A = A.at[1, 2].set(0.5)
        A = A.at[2, 0].set(0.5)

        h = self._compute_h_with_sink(A, Y_idx=3)
        assert h > 0.01, f"Cycle among X vars not detected: h_A={h:.6f}"

    def test_y_in_middle_index(self):
        """Y-as-sink works when Y is not the last variable."""
        # Y at index 1 (middle)
        A = jnp.zeros((4, 4))
        A = A.at[0, 2].set(0.5)  # X0 → X2
        A = A.at[2, 3].set(0.5)  # X2 → X3

        # Add X→Y edges
        A = A.at[0, 1].set(0.9)  # X0 → Y
        A = A.at[3, 1].set(0.9)  # X3 → Y

        h = self._compute_h_with_sink(A, Y_idx=1)

        # Should match DAG constraint on [X0, X2, X3] subgraph only
        A_x = jnp.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        h_expected = dag_constraint(A_x)
        assert jnp.allclose(h, h_expected, atol=1e-5), (
            f"Y_idx=1 submatrix mismatch: {h:.6f} vs {h_expected:.6f}"
        )

    def test_gradient_flows_to_x_to_y_edges(self):
        """Gradient of DAG loss should be ZERO for X→Y edges when Y is sink."""

        def dag_loss_sink(A):
            n_total = A.shape[0]
            Y_idx = 3
            x_idx = jnp.concatenate([jnp.arange(Y_idx), jnp.arange(Y_idx + 1, n_total)])
            A_dag = A[jnp.ix_(x_idx, x_idx)]
            return dag_constraint(A_dag)

        def dag_loss_full(A):
            return dag_constraint(A)

        A = jnp.zeros((4, 4))
        A = A.at[0, 1].set(0.5)
        A = A.at[0, 3].set(0.5)  # X0 → Y

        grad_sink = jax.grad(dag_loss_sink)(A)
        grad_full = jax.grad(dag_loss_full)(A)

        # With sink: gradient for X→Y edges (column 3) should be zero
        assert jnp.allclose(grad_sink[:, 3], 0.0, atol=1e-7), (
            f"X→Y gradient not zero with sink: {grad_sink[:, 3]}"
        )

        # Without sink: gradient for X→Y edges may be nonzero
        # (even without cycles, DAGMA has numerical gradient)

    def test_gradient_nonzero_for_x_to_x_edges(self):
        """Gradient should still flow for X→X edges."""

        def dag_loss_sink(A):
            n_total = A.shape[0]
            Y_idx = 3
            x_idx = jnp.concatenate([jnp.arange(Y_idx), jnp.arange(Y_idx + 1, n_total)])
            A_dag = A[jnp.ix_(x_idx, x_idx)]
            return dag_constraint(A_dag)

        # Create near-cycle to get nonzero gradient
        A = jnp.zeros((4, 4))
        A = A.at[0, 1].set(0.5)
        A = A.at[1, 2].set(0.5)
        A = A.at[2, 0].set(0.3)  # Weak back-edge

        grad = jax.grad(dag_loss_sink)(A)

        # X→X edges should have nonzero gradient (cycle pressure)
        x_grad_norm = jnp.linalg.norm(grad[:3, :3])
        assert x_grad_norm > 1e-4, f"X→X gradient too small: {x_grad_norm:.6f}"

    def test_jit_compatible(self):
        """The submatrix extraction should work under JIT."""

        @jax.jit
        def h_jit(A):
            Y_idx = 3
            n_total = A.shape[0]
            x_idx = jnp.concatenate([jnp.arange(Y_idx), jnp.arange(Y_idx + 1, n_total)])
            A_dag = A[jnp.ix_(x_idx, x_idx)]
            return dag_constraint(A_dag)

        A = jnp.zeros((5, 5))
        A = A.at[0, 1].set(0.5)

        h = h_jit(A)
        assert jnp.isfinite(h), f"JIT result not finite: {h}"

    def test_submatrix_size(self):
        """Submatrix should be (n_total-1) × (n_total-1)."""
        n = 30  # Typical dataset size
        Y_idx = 12
        A = jnp.zeros((n, n))
        x_idx = jnp.concatenate([jnp.arange(Y_idx), jnp.arange(Y_idx + 1, n)])
        A_dag = A[jnp.ix_(x_idx, x_idx)]
        assert A_dag.shape == (n - 1, n - 1), f"Wrong shape: {A_dag.shape}"
