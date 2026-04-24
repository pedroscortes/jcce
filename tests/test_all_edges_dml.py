"""
Unit tests for All-Edges DML validation.

Tests:
1. Edge extraction from adjacency matrix
2. Adjustment set computation for X→X edges
3. DML runs for X→X edge with known effect
4. DML runs for X→Y edge (equivalence check)
5. BH-FDR correction
6. Empty DAG graceful handling
7. Integration test with 5-node DAG
"""

import unittest

import numpy as np

from jcce.validation.all_edges_dml import (
    _benjamini_hochberg,
    extract_all_edges,
    run_all_edges_dml,
)


class TestExtractAllEdges(unittest.TestCase):
    """Test 1: Edge extraction from adjacency matrix."""

    def test_chain_graph(self):
        """3-node chain A→B→C should have exactly 2 edges."""
        # A[i,j] = i→j
        A = np.zeros((3, 3))
        A[0, 1] = 0.5  # A→B
        A[1, 2] = 0.3  # B→C
        edges = extract_all_edges(A, threshold=0.01)
        self.assertEqual(len(edges), 2)
        edge_pairs = [(e[0], e[1]) for e in edges]
        self.assertIn((0, 1), edge_pairs)
        self.assertIn((1, 2), edge_pairs)
        # No transitive edge A→C
        self.assertNotIn((0, 2), edge_pairs)

    def test_threshold_filtering(self):
        """Edges below threshold should be excluded."""
        A = np.zeros((3, 3))
        A[0, 1] = 0.5
        A[1, 2] = 0.005  # Below threshold
        edges = extract_all_edges(A, threshold=0.01)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][:2], (0, 1))

    def test_sorted_by_weight(self):
        """Edges should be sorted by weight descending."""
        A = np.zeros((4, 4))
        A[0, 1] = 0.1
        A[1, 2] = 0.5
        A[2, 3] = 0.3
        edges = extract_all_edges(A, threshold=0.01)
        weights = [e[2] for e in edges]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_no_self_loops(self):
        """Self-loops should be excluded."""
        A = np.eye(3) * 0.5
        edges = extract_all_edges(A, threshold=0.01)
        self.assertEqual(len(edges), 0)


class TestAdjustmentSets(unittest.TestCase):
    """Test 2: Adjustment sets for X→X edges."""

    def test_chain_adjustment(self):
        """For A→B→C: treatment A on outcome B has empty adj set (exogenous).
        Treatment B on outcome C has adj set = {A}."""
        from jcce.structure_learning.effect_estimation import (
            compute_valid_adjustment_sets,
        )

        # A[i,j] = i→j, 3 nodes (A=0, B=1, C=2)
        A = np.zeros((3, 3))
        A[0, 1] = 0.5  # A→B
        A[1, 2] = 0.3  # B→C

        # For outcome B (idx=1): treatment A (idx=0)
        adj_sets_B = compute_valid_adjustment_sets(A, Y_idx=1, threshold=0.01)
        adj_A = adj_sets_B.get(0, set())
        # A is exogenous — no parents, so empty set
        self.assertEqual(adj_A, set())

        # For outcome C (idx=2): treatment B (idx=1)
        adj_sets_C = compute_valid_adjustment_sets(A, Y_idx=2, threshold=0.01)
        adj_B = adj_sets_C.get(1, set())
        # Parents(B) = {A=0}, Descendants(B) = {C=2}
        # Valid = {A} - {C} - {B, C} = {A=0}
        self.assertEqual(adj_B, {0})


class TestDMLXtoX(unittest.TestCase):
    """Test 3: DML runs for X→X edge with known effect."""

    def test_known_linear_effect(self):
        """Synthetic data: X0 → X1 with ATE ~ 2.0."""
        np.random.seed(42)
        n = 500
        X0 = np.random.binomial(1, 0.5, n).astype(float)
        X1 = 2.0 * X0 + np.random.normal(0, 0.5, n)
        Y = np.random.normal(0, 1, n)  # Y unrelated

        X = np.column_stack([X0, X1])
        # A_est: 3 vars (X0, X1, Y). X0→X1
        A_est = np.zeros((3, 3))
        A_est[0, 1] = 0.5  # X0→X1

        result = run_all_edges_dml(
            X=X,
            Y=Y,
            A_est=A_est,
            Y_idx=2,
            feature_names=["X0", "X1"],
            n_dml_folds=3,
            n_dml_repeats=3,
            run_refutation=False,
            verbose=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.n_edges, 1)
        ate = result.edge_results[0].dml_result.ate_mean
        # True ATE = 2.0, allow tolerance
        self.assertAlmostEqual(ate, 2.0, delta=0.5)


class TestDMLXtoY(unittest.TestCase):
    """Test 4: DML runs for X→Y edge."""

    def test_known_treatment_effect(self):
        """Synthetic data: X0 → Y with ATE ~ 1.5."""
        np.random.seed(42)
        n = 500
        X0 = np.random.binomial(1, 0.5, n).astype(float)
        Y = 1.5 * X0 + np.random.normal(0, 0.5, n)
        X1 = np.random.normal(0, 1, n)  # Noise feature

        X = np.column_stack([X0, X1])
        A_est = np.zeros((3, 3))
        A_est[0, 2] = 0.8  # X0→Y (Y_idx=2)

        result = run_all_edges_dml(
            X=X,
            Y=Y,
            A_est=A_est,
            Y_idx=2,
            feature_names=["X0", "X1"],
            n_dml_folds=3,
            n_dml_repeats=3,
            run_refutation=False,
            verbose=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.n_edges, 1)
        ate = result.edge_results[0].dml_result.ate_mean
        self.assertAlmostEqual(ate, 1.5, delta=0.5)
        # Edge should target Y
        self.assertEqual(result.edge_results[0].target_name, "Y")


class TestFDRCorrection(unittest.TestCase):
    """Test 5: Benjamini-Hochberg FDR correction."""

    def test_known_p_values(self):
        """Given known p-values, verify BH flags correct edges."""
        # 5 tests, alpha=0.05
        p_values = [0.001, 0.01, 0.03, 0.20, 0.80]
        flags = _benjamini_hochberg(p_values, alpha=0.05)
        # Sorted: 0.001 <= 0.05*1/5=0.01 (yes)
        #         0.01  <= 0.05*2/5=0.02 (yes)
        #         0.03  <= 0.05*3/5=0.03 (yes)
        #         0.20  <= 0.05*4/5=0.04 (no)
        #         0.80  <= 0.05*5/5=0.05 (no)
        self.assertEqual(flags, [True, True, True, False, False])

    def test_empty(self):
        """Empty p-values → empty result."""
        self.assertEqual(_benjamini_hochberg([]), [])

    def test_all_significant(self):
        """All very small p-values → all significant."""
        p_values = [0.001, 0.002, 0.003]
        flags = _benjamini_hochberg(p_values, alpha=0.05)
        self.assertTrue(all(flags))

    def test_none_significant(self):
        """All large p-values → none significant."""
        p_values = [0.5, 0.6, 0.9]
        flags = _benjamini_hochberg(p_values, alpha=0.05)
        self.assertFalse(any(flags))


class TestEmptyDAG(unittest.TestCase):
    """Test 6: Empty DAG graceful handling."""

    def test_no_edges(self):
        """Empty adjacency matrix → None result."""
        X = np.random.randn(100, 3)
        Y = np.random.randn(100)
        A_est = np.zeros((4, 4))

        result = run_all_edges_dml(
            X=X,
            Y=Y,
            A_est=A_est,
            Y_idx=3,
            verbose=False,
        )
        self.assertIsNone(result)


class TestIntegration(unittest.TestCase):
    """Test 7: Integration test with 5-node DAG."""

    def test_five_node_dag(self):
        """5-node DAG: X0→X1→X2→Y, X3→Y. Verify result structure."""
        np.random.seed(42)
        n = 400

        X0 = np.random.binomial(1, 0.5, n).astype(float)
        X1 = 1.0 * X0 + np.random.normal(0, 0.5, n)
        X2 = 0.8 * X1 + np.random.normal(0, 0.5, n)
        X3 = np.random.binomial(1, 0.3, n).astype(float)
        Y = 0.5 * X2 + 1.2 * X3 + np.random.normal(0, 0.5, n)

        X = np.column_stack([X0, X1, X2, X3])
        # A_est: 5 vars (X0..X3, Y=4)
        A_est = np.zeros((5, 5))
        A_est[0, 1] = 0.4  # X0→X1
        A_est[1, 2] = 0.3  # X1→X2
        A_est[2, 4] = 0.5  # X2→Y
        A_est[3, 4] = 0.6  # X3→Y

        result = run_all_edges_dml(
            X=X,
            Y=Y,
            A_est=A_est,
            Y_idx=4,
            feature_names=["X0", "X1", "X2", "X3"],
            n_dml_folds=3,
            n_dml_repeats=3,
            run_refutation=False,
            verbose=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.n_edges, 4)

        # Verify result structure
        d = result.to_dict()
        self.assertIn("n_edges", d)
        self.assertIn("n_significant", d)
        self.assertIn("n_significant_fdr", d)
        self.assertIn("edge_effects", d)
        self.assertEqual(len(d["edge_effects"]), 4)

        # Verify storage dict
        storage = result.to_storage_dict()
        self.assertIn("edge_effects", storage)
        for key in storage["edge_effects"]:
            self.assertIn("->", key)
            edge_data = storage["edge_effects"][key]
            self.assertIn("ate", edge_data)
            self.assertIn("ci_95", edge_data)
            self.assertIn("p_value", edge_data)
            self.assertIn("sig", edge_data)
            self.assertIn("sig_fdr", edge_data)

        # Verify causal effects dict
        effects_dict = result.to_causal_effects_dict()
        self.assertEqual(len(effects_dict), 4)
        self.assertIn("X0->X1", effects_dict)
        self.assertIn("X2->Y", effects_dict)

        # Verify summary table doesn't crash
        table = result.summary_table()
        self.assertIn("ALL-EDGES DML", table)

        # Verify per-edge result structure
        for er in result.edge_results:
            self.assertIsInstance(er.p_value, float)
            self.assertGreaterEqual(er.p_value, 0)
            self.assertLessEqual(er.p_value, 1)
            self.assertIsInstance(er.is_significant, bool)
            self.assertIsInstance(er.is_significant_fdr, bool)


if __name__ == "__main__":
    unittest.main()
