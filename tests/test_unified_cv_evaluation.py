"""
Tests for v15 Unified CV Evaluation.

Covers:
1. UnifiedCVResult dataclass (fields, summary, to_dict)
2. Helper functions (Jaccard, edge frequency, SHD, MB F1)
3. evaluate_pareto_solution_cv basic behavior
4. Structure stability metrics
5. Integration with ground truth (LUCAS-style)
"""

import numpy as np
import pytest

from jcce.validation.unified_cv_evaluation import (
    UnifiedCVResult,
    _compute_jaccard,
    _compute_mb_jaccard_mean,
    _compute_edge_frequency,
    _compute_shd,
    _compute_mb_f1,
)


# =============================================================================
# 1. UnifiedCVResult Tests
# =============================================================================

class TestUnifiedCVResult:
    """Test the UnifiedCVResult dataclass."""

    def _make_result(self) -> UnifiedCVResult:
        return UnifiedCVResult(
            accuracy_per_fold=[0.8, 0.85, 0.82, 0.78, 0.81],
            precision_per_fold=[0.77, 0.82, 0.80, 0.75, 0.78],
            recall_per_fold=[0.73, 0.78, 0.76, 0.71, 0.74],
            f1_per_fold=[0.75, 0.80, 0.78, 0.73, 0.76],
            balanced_acc_per_fold=[0.79, 0.84, 0.81, 0.77, 0.80],
            roc_auc_per_fold=[0.85, 0.88, 0.86, 0.83, 0.85],
            accuracy_mean=0.812,
            accuracy_std=0.025,
            precision_mean=0.784,
            precision_std=0.025,
            recall_mean=0.744,
            recall_std=0.025,
            f1_mean=0.764,
            f1_std=0.026,
            balanced_acc_mean=0.802,
            balanced_acc_std=0.024,
            roc_auc_mean=0.854,
            roc_auc_std=0.017,
            mb_per_fold=[[0, 4], [0, 4, 8], [0, 4], [4, 8], [0, 4]],
            mb_jaccard_mean=0.7,
            edge_frequency={(0, 11): 0.8, (4, 11): 1.0, (8, 11): 0.4},
        )

    def test_result_fields(self):
        """Verify all fields are accessible."""
        result = self._make_result()
        assert len(result.accuracy_per_fold) == 5
        assert result.accuracy_mean == pytest.approx(0.812)
        assert result.mb_jaccard_mean == pytest.approx(0.7)
        assert len(result.edge_frequency) == 3

    def test_result_to_dict(self):
        """Verify to_dict returns all keys."""
        result = self._make_result()
        d = result.to_dict()
        assert 'accuracy_mean' in d
        assert 'f1_std' in d
        assert 'mb_per_fold' in d
        assert 'edge_frequency' in d
        assert d['n_folds'] == 5

    def test_result_summary_string(self):
        """Verify summary produces non-empty string."""
        result = self._make_result()
        s = result.summary()
        assert 'UNIFIED CV EVALUATION' in s
        assert '0.812' in s or '0.8120' in s

    def test_result_with_ground_truth(self):
        """Verify SHD and MB F1 fields."""
        result = self._make_result()
        result.shd_per_fold = [5, 6, 4, 7, 5]
        result.mb_f1_per_fold = [0.8, 0.7, 0.8, 0.6, 0.8]
        d = result.to_dict()
        assert 'shd_per_fold' in d
        assert 'mb_f1_per_fold' in d
        s = result.summary()
        assert 'SHD' in s
        assert 'MB F1' in s


# =============================================================================
# 2. Helper Function Tests
# =============================================================================

class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""

    def test_identical_sets(self):
        assert _compute_jaccard({0, 1, 2}, {0, 1, 2}) == 1.0

    def test_disjoint_sets(self):
        assert _compute_jaccard({0, 1}, {2, 3}) == 0.0

    def test_partial_overlap(self):
        # Intersection={1}, Union={0,1,2} → 1/3
        assert _compute_jaccard({0, 1}, {1, 2}) == pytest.approx(1/3)

    def test_empty_sets(self):
        assert _compute_jaccard(set(), set()) == 1.0

    def test_one_empty(self):
        assert _compute_jaccard({0, 1}, set()) == 0.0


class TestMBJaccardMean:
    """Test average pairwise Jaccard across folds."""

    def test_all_identical(self):
        mbs = [[0, 4], [0, 4], [0, 4]]
        assert _compute_mb_jaccard_mean(mbs) == pytest.approx(1.0)

    def test_all_different(self):
        mbs = [[0], [1], [2]]
        assert _compute_mb_jaccard_mean(mbs) == pytest.approx(0.0)

    def test_mixed(self):
        mbs = [[0, 1], [0, 1, 2], [0, 1]]
        # Pairs: (0,1)=2/3, (0,2)=1.0, (1,2)=2/3
        # Mean = (2/3 + 1.0 + 2/3) / 3 = 7/9
        assert _compute_mb_jaccard_mean(mbs) == pytest.approx(7/9, rel=1e-6)

    def test_single_fold(self):
        assert _compute_mb_jaccard_mean([[0, 1]]) == 1.0


class TestEdgeFrequency:
    """Test edge frequency computation across folds."""

    def test_all_folds_same(self):
        A = np.array([[0, 0.5], [0, 0]])
        freq = _compute_edge_frequency([A, A, A])
        assert freq[(0, 1)] == pytest.approx(1.0)

    def test_one_fold_different(self):
        A1 = np.array([[0, 0.5], [0, 0]])
        A2 = np.array([[0, 0], [0, 0]])
        freq = _compute_edge_frequency([A1, A1, A2])
        assert freq[(0, 1)] == pytest.approx(2/3)

    def test_empty(self):
        freq = _compute_edge_frequency([])
        assert freq == {}


class TestSHD:
    """Test Structural Hamming Distance."""

    def test_identical_dags(self):
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        assert _compute_shd(A, A) == 0

    def test_one_edge_difference(self):
        A_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        A_est = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # Missing edge
        assert _compute_shd(A_est, A_true) == 1

    def test_extra_edge(self):
        A_true = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        A_est = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])  # Extra edge
        assert _compute_shd(A_est, A_true) == 1


class TestMBF1:
    """Test Markov Blanket F1 computation."""

    def test_perfect_recovery(self):
        assert _compute_mb_f1([0, 4, 8], [0, 4, 8]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _compute_mb_f1([0, 1], [2, 3]) == pytest.approx(0.0)

    def test_partial(self):
        # Predicted={0,4}, True={0,4,8}
        # TP=2, Precision=2/2=1.0, Recall=2/3, F1=2*(1.0*2/3)/(1.0+2/3)
        expected_f1 = 2 * (1.0 * 2/3) / (1.0 + 2/3)
        assert _compute_mb_f1([0, 4], [0, 4, 8]) == pytest.approx(expected_f1)

    def test_both_empty(self):
        assert _compute_mb_f1([], []) == 1.0

    def test_one_empty(self):
        assert _compute_mb_f1([], [0, 1]) == 0.0
        assert _compute_mb_f1([0, 1], []) == 0.0


# =============================================================================
# 3. evaluate_pareto_solution_cv Tests (requires JAX)
# =============================================================================

# These tests are integration tests that require JAX and the full GOLEM stack.
# They use small synthetic data to verify the CV pipeline runs correctly.

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestEvaluateParetoCVBasic:
    """Basic integration tests for evaluate_pareto_solution_cv."""

    @pytest.fixture
    def synthetic_data(self):
        """Create small synthetic dataset for testing."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p).astype(np.float32)
        # Y depends on features 0 and 2
        logits = 0.5 * X[:, 0] + 0.3 * X[:, 2]
        Y = (logits > 0).astype(int)
        return X, Y

    def test_cv_returns_valid_result(self, synthetic_data):
        """Run CV on synthetic data, verify UnifiedCVResult fields."""
        from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

        X, Y = synthetic_data
        n_features = X.shape[1]

        hyperparams = {
            'lambda_1': 0.02,
            'lambda_2': 0.01,
            'lambda_class': 1.0,
            'lr': 0.001,
        }
        A_init = np.zeros((n_features + 1, n_features + 1), dtype=np.float32)

        result = evaluate_pareto_solution_cv(
            X=X, Y=Y,
            hyperparams=hyperparams,
            processor_type='elm',
            A_init=A_init,
            n_folds=3,
            golem_max_iter=10,  # Very few iterations for speed
            verbose=False,
        )

        assert isinstance(result, UnifiedCVResult)
        assert len(result.accuracy_per_fold) == 3
        assert len(result.f1_per_fold) == 3
        assert 0 <= result.accuracy_mean <= 1
        assert result.accuracy_std >= 0
        assert 0 <= result.roc_auc_mean <= 1
        assert result.total_time is not None and result.total_time > 0

    def test_cv_metrics_reasonable(self, synthetic_data):
        """Verify metrics are within reasonable ranges."""
        from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

        X, Y = synthetic_data
        n_features = X.shape[1]

        hyperparams = {
            'lambda_1': 0.02,
            'lambda_2': 0.01,
            'lambda_class': 1.0,
            'lr': 0.001,
        }
        A_init = np.zeros((n_features + 1, n_features + 1), dtype=np.float32)

        result = evaluate_pareto_solution_cv(
            X=X, Y=Y,
            hyperparams=hyperparams,
            processor_type='mlp',
            A_init=A_init,
            n_folds=3,
            golem_max_iter=10,
            verbose=False,
        )

        # All accuracies should be between 0 and 1
        for acc in result.accuracy_per_fold:
            assert 0 <= acc <= 1

        # Std should be non-negative
        assert result.accuracy_std >= 0
        assert result.f1_std >= 0

    def test_cv_structure_stability(self, synthetic_data):
        """Verify Jaccard similarity is computed correctly."""
        from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

        X, Y = synthetic_data
        n_features = X.shape[1]

        hyperparams = {
            'lambda_1': 0.02,
            'lambda_2': 0.01,
            'lambda_class': 1.0,
            'lr': 0.001,
        }
        A_init = np.zeros((n_features + 1, n_features + 1), dtype=np.float32)

        result = evaluate_pareto_solution_cv(
            X=X, Y=Y,
            hyperparams=hyperparams,
            processor_type='elm',
            A_init=A_init,
            n_folds=3,
            golem_max_iter=10,
            verbose=False,
        )

        # Jaccard should be between 0 and 1
        assert 0 <= result.mb_jaccard_mean <= 1

        # MB per fold should have 3 entries
        assert len(result.mb_per_fold) == 3

    def test_cv_with_ground_truth(self, synthetic_data):
        """Verify SHD and MB F1 are computed when ground truth provided."""
        from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

        X, Y = synthetic_data
        n_features = X.shape[1]

        # Create a dummy true DAG (6x6 for 5 features + Y)
        true_dag = np.zeros((n_features + 1, n_features + 1))
        true_dag[0, 5] = 1  # Feature 0 -> Y
        true_dag[2, 5] = 1  # Feature 2 -> Y
        true_mb = [0, 2]

        hyperparams = {
            'lambda_1': 0.02,
            'lambda_2': 0.01,
            'lambda_class': 1.0,
            'lr': 0.001,
        }
        A_init = np.zeros((n_features + 1, n_features + 1), dtype=np.float32)

        result = evaluate_pareto_solution_cv(
            X=X, Y=Y,
            hyperparams=hyperparams,
            processor_type='elm',
            A_init=A_init,
            n_folds=3,
            golem_max_iter=10,
            verbose=False,
            true_mb=true_mb,
            true_dag=true_dag,
        )

        # SHD and MB F1 should be computed
        assert result.shd_per_fold is not None
        assert len(result.shd_per_fold) == 3
        assert result.mb_f1_per_fold is not None
        assert len(result.mb_f1_per_fold) == 3

        # SHD should be non-negative integers
        for shd in result.shd_per_fold:
            assert shd >= 0
            assert isinstance(shd, int)

        # MB F1 should be between 0 and 1
        for f1 in result.mb_f1_per_fold:
            assert 0 <= f1 <= 1

    def test_cv_folds_independent(self, synthetic_data):
        """Verify test folds don't overlap."""
        from sklearn.model_selection import StratifiedKFold

        X, Y = synthetic_data
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        all_test_indices = []
        for _, test_idx in kfold.split(X, Y):
            all_test_indices.append(set(test_idx))

        # Verify no overlap between test folds
        for i in range(len(all_test_indices)):
            for j in range(i + 1, len(all_test_indices)):
                overlap = all_test_indices[i] & all_test_indices[j]
                assert len(overlap) == 0, f"Folds {i} and {j} overlap: {overlap}"

        # Verify all samples are covered
        all_covered = set()
        for idx_set in all_test_indices:
            all_covered |= idx_set
        assert len(all_covered) == len(Y)
