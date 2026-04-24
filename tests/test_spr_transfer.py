"""
Unit tests for SPR transfer (Phase D.2).

Tests hyperparameter distance, candidate finding, adjacency SPR, and processor SPR.
"""

import numpy as np
import pytest

from jcce.structure_learning.spr_transfer import (
    hyperparameter_distance,
    find_spr_candidate,
    spr_adjacency,
    spr_processor_params,
)


class TestHyperparameterDistance:
    def test_same_config_distance_is_zero(self):
        """Identical configs have distance 0."""
        config = {
            'processor_type': 'mlp',
            'lambda_1': 0.01,
            'lambda_2': 0.1,
            'lr': 0.001,
        }
        assert hyperparameter_distance(config, config) == 0.0

    def test_different_processor_returns_inf(self):
        """Different processor types → infinite distance."""
        config_a = {
            'processor_type': 'mlp',
            'lambda_1': 0.01,
            'lambda_2': 0.1,
            'lr': 0.001,
        }
        config_b = {
            'processor_type': 'elm',
            'lambda_1': 0.01,
            'lambda_2': 0.1,
            'lr': 0.001,
        }
        assert hyperparameter_distance(config_a, config_b) == float('inf')

    def test_closer_config_has_smaller_distance(self):
        """Config with closer HPs has smaller distance."""
        base = {
            'processor_type': 'mlp',
            'lambda_1': 0.01,
            'lambda_2': 0.1,
            'lr': 0.001,
        }
        close = {
            'processor_type': 'mlp',
            'lambda_1': 0.012,
            'lambda_2': 0.12,
            'lr': 0.0012,
        }
        far = {
            'processor_type': 'mlp',
            'lambda_1': 0.1,
            'lambda_2': 0.8,
            'lr': 0.008,
        }
        d_close = hyperparameter_distance(base, close)
        d_far = hyperparameter_distance(base, far)
        assert d_close < d_far


class TestFindSprCandidate:
    def test_empty_artifact_store_returns_none(self):
        """No artifacts → no candidate."""
        config = {
            'processor_type': 'mlp',
            'lambda_1': 0.01,
            'lambda_2': 0.1,
            'lr': 0.001,
        }
        result = find_spr_candidate(config, {}, max_distance=0.5)
        assert result is None

    def test_finds_closest_match(self):
        """Returns trial number of closest matching config."""
        base_config = {
            'processor_type': 'mlp',
            'lambda_1': 0.01,
            'lambda_2': 0.1,
            'lr': 0.001,
        }
        artifacts = {
            0: {'config': {
                'processor_type': 'mlp',
                'lambda_1': 0.1,  # far
                'lambda_2': 0.5,
                'lr': 0.005,
            }},
            1: {'config': {
                'processor_type': 'mlp',
                'lambda_1': 0.012,  # close
                'lambda_2': 0.11,
                'lr': 0.0011,
            }},
            2: {'config': {
                'processor_type': 'elm',  # wrong processor → inf
                'lambda_1': 0.01,
                'lambda_2': 0.1,
                'lr': 0.001,
            }},
        }
        result = find_spr_candidate(base_config, artifacts, max_distance=1.0)
        assert result == 1

    def test_rejects_beyond_max_distance(self):
        """Returns None when all candidates exceed max_distance."""
        config = {
            'processor_type': 'mlp',
            'lambda_1': 0.01,
            'lambda_2': 0.1,
            'lr': 0.001,
        }
        artifacts = {
            0: {'config': {
                'processor_type': 'mlp',
                'lambda_1': 0.4,  # very far
                'lambda_2': 0.9,
                'lr': 0.009,
            }},
        }
        result = find_spr_candidate(config, artifacts, max_distance=0.01)
        assert result is None


class TestSprAdjacency:
    def test_preserves_shape(self):
        """Output has same shape as input."""
        A = np.random.rand(6, 6)
        A_new = spr_adjacency(A, rng=np.random.default_rng(42))
        assert A_new.shape == A.shape

    def test_preserves_zero_diagonal(self):
        """Output has zero diagonal."""
        A = np.random.rand(6, 6)
        A_new = spr_adjacency(A, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(np.diag(A_new), 0.0)

    def test_shrinks_toward_half(self):
        """With lambda_A < 1, logit-space shrinkage pulls values toward 0.5."""
        rng = np.random.default_rng(42)
        # Create A with very strong edges (close to 1)
        A_strong = np.full((5, 5), 0.95)
        np.fill_diagonal(A_strong, 0.0)

        # Apply SPR with moderate shrinkage and NO noise
        A_new = spr_adjacency(
            A_strong, lambda_A=0.5, noise_scale=0.0, rng=rng
        )

        # Off-diagonal should be closer to 0.5 than original
        mask = ~np.eye(5, dtype=bool)
        dist_old = np.abs(A_strong[mask] - 0.5).mean()
        dist_new = np.abs(A_new[mask] - 0.5).mean()
        assert dist_new < dist_old

    def test_output_in_valid_range(self):
        """Output values are in [0, 1]."""
        A = np.random.rand(6, 6) * 0.8 + 0.1  # [0.1, 0.9]
        A_new = spr_adjacency(A, rng=np.random.default_rng(42))
        assert np.all(A_new >= 0.0)
        assert np.all(A_new <= 1.0)


class TestSprProcessorParams:
    def test_shrinks_magnitude(self):
        """SPR with lambda < 1 reduces parameter magnitude on average."""
        rng = np.random.default_rng(42)
        old_params = [
            {'flat_params': rng.normal(0, 1, size=(50,)).astype(np.float32),
             'tree_def': 'mock', 'shapes': [(10, 5)]},
            {'flat_params': rng.normal(0, 1, size=(50,)).astype(np.float32),
             'tree_def': 'mock', 'shapes': [(10, 5)]},
        ]

        new_params = spr_processor_params(
            old_params,
            lambda_theta=0.3,
            noise_scale_multiplier=0.0,  # no noise → pure shrinkage
            n_inputs=10,
            rng=np.random.default_rng(42),
        )

        # Magnitude should decrease
        old_mag = sum(np.abs(p['flat_params']).mean() for p in old_params)
        new_mag = sum(np.abs(p['flat_params']).mean() for p in new_params)
        assert new_mag < old_mag

    def test_preserves_metadata(self):
        """Metadata fields (tree_def, shapes, n_inputs) pass through unchanged."""
        old_params = [
            {'flat_params': np.ones(10, dtype=np.float32),
             'tree_def': 'my_tree_def',
             'shapes': [(5, 2)],
             'n_inputs': 5},
        ]
        new_params = spr_processor_params(
            old_params, rng=np.random.default_rng(42)
        )
        assert new_params[0]['tree_def'] == 'my_tree_def'
        assert new_params[0]['shapes'] == [(5, 2)]
        assert new_params[0]['n_inputs'] == 5

    def test_handles_no_flat_params(self):
        """Params without flat_params are passed through unchanged."""
        old_params = [{'metadata_only': True}]
        new_params = spr_processor_params(
            old_params, rng=np.random.default_rng(42)
        )
        assert new_params[0] == {'metadata_only': True}
