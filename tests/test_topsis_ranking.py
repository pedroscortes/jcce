"""Tests for TOPSIS ranking."""

import numpy as np
import pytest

from jcce.analysis.topsis_ranking import topsis_rank


def _make_sol(bacc, sparsity):
    return {'metrics': {'balanced_accuracy': bacc, 'mb_sparsity': sparsity}}


class TestTopsisRank:
    """TOPSIS ranking unit tests."""

    def test_basic_ranking(self):
        """Best BAcc + sparsity solution should rank first."""
        solutions = [
            _make_sol(0.9, 0.8),   # good on both
            _make_sol(0.5, 0.3),   # bad on both
            _make_sol(0.7, 0.6),   # medium
        ]
        indices, scores = topsis_rank(solutions)
        assert indices[0] == 0
        assert indices[-1] == 1
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_equal_solutions(self):
        """Identical solutions should get equal scores."""
        solutions = [_make_sol(0.8, 0.5)] * 3
        indices, scores = topsis_rank(solutions)
        assert len(indices) == 3
        np.testing.assert_allclose(scores, scores[0], atol=1e-10)

    def test_single_solution(self):
        """Single solution should get score 1.0."""
        solutions = [_make_sol(0.7, 0.4)]
        indices, scores = topsis_rank(solutions)
        assert indices == [0]
        assert scores[0] == 1.0

    def test_custom_weights(self):
        """Heavier weight on BAcc should prefer high-BAcc solution."""
        solutions = [
            _make_sol(0.9, 0.2),   # high BAcc, low sparsity
            _make_sol(0.5, 0.9),   # low BAcc, high sparsity
        ]
        # Weight BAcc heavily
        indices, _ = topsis_rank(solutions, weights=[0.9, 0.1])
        assert indices[0] == 0

        # Weight sparsity heavily
        indices, _ = topsis_rank(solutions, weights=[0.1, 0.9])
        assert indices[0] == 1

    def test_minimize_criterion(self):
        """Minimize criterion should prefer lower values."""
        solutions = [
            _make_sol(0.8, 10.0),  # high "cost"
            _make_sol(0.8, 2.0),   # low "cost"
        ]
        # Second criterion is a cost to minimize
        indices, scores = topsis_rank(
            solutions,
            criteria=['balanced_accuracy', 'mb_sparsity'],
            beneficial=[True, False],
        )
        assert indices[0] == 1
        assert scores[1] > scores[0]
