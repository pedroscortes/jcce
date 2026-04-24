"""
Tests for Phase 3: Warm-Start Cache integration with single-GPU NSGA-II.

Tests the ImprovedWarmStartCache standalone behavior and its integration
into UnifiedSCDProblem.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np


class TestCacheAddGetRoundTrip(unittest.TestCase):
    """Test 1: Cache add/get round-trip."""

    def test_add_and_get(self):
        from jcce.structure_learning.warm_start_cache import ImprovedWarmStartCache

        cache = ImprovedWarmStartCache(max_size=50)
        config = {'processor_type': 'mlp', 'lambda_1_idx': 2, 'lr_idx': 1}
        A = np.random.randn(5, 5) * 0.1
        fitness = 0.85

        cache.add(config, A, fitness, generation=0, processor_type='mlp')

        # Get should return a copy of A
        A_retrieved = cache.get(config)
        self.assertIsNotNone(A_retrieved)
        np.testing.assert_array_almost_equal(A_retrieved, A)

        # Should be a copy, not the same object
        self.assertFalse(A_retrieved is A)


class TestSampleFromTop20(unittest.TestCase):
    """Test 2: Sample from top 20% by fitness."""

    def test_sample_top_20_percent(self):
        from jcce.structure_learning.warm_start_cache import ImprovedWarmStartCache

        cache = ImprovedWarmStartCache(max_size=50)

        # Add 10 entries with increasing fitness
        for i in range(10):
            config = {'processor_type': 'mlp', 'lambda_1_idx': i}
            A = np.eye(5) * (i + 1) * 0.01
            fitness = 0.5 + i * 0.05  # 0.5 to 0.95
            cache.add(config, A, fitness, generation=0, processor_type='mlp')

        # Top 20% = top 2 entries (fitness 0.90 and 0.95)
        # Sample many times and check all come from top entries
        sampled_fitnesses = set()
        for _ in range(50):
            A_sampled = cache.sample()
            self.assertIsNotNone(A_sampled)
            # Find matching entry by A matrix content
            for entry in cache.cache.values():
                if np.allclose(A_sampled, entry.A_matrix):
                    sampled_fitnesses.add(entry.fitness)

        # All sampled entries should have fitness >= 0.90 (top 20%)
        for f in sampled_fitnesses:
            self.assertGreaterEqual(f, 0.90)


class TestProcessorTypeMatchingPreference(unittest.TestCase):
    """Test 3: Processor-type matching preference in get()."""

    def test_prefers_same_processor_type(self):
        from jcce.structure_learning.warm_start_cache import ImprovedWarmStartCache

        cache = ImprovedWarmStartCache(max_size=50)

        # Add entry for 'mlp' with lower fitness
        config_mlp = {'processor_type': 'mlp', 'lambda_1_idx': 0}
        A_mlp = np.ones((5, 5)) * 0.1
        cache.add(config_mlp, A_mlp, fitness=0.75, generation=0, processor_type='mlp')

        # Add entry for 'transformer' with higher fitness
        config_trans = {'processor_type': 'transformer', 'lambda_1_idx': 1}
        A_trans = np.ones((5, 5)) * 0.2
        cache.add(config_trans, A_trans, fitness=0.95, generation=0, processor_type='transformer')

        # Query with new mlp config (different hash, same processor_type)
        query_config = {'processor_type': 'mlp', 'lambda_1_idx': 99}
        A_result = cache.get(query_config)

        # Should prefer mlp match despite lower fitness
        self.assertIsNotNone(A_result)
        np.testing.assert_array_almost_equal(A_result, A_mlp)


class TestEvictionAtMaxSize(unittest.TestCase):
    """Test 4: Eviction at max_size=50."""

    def test_eviction(self):
        from jcce.structure_learning.warm_start_cache import ImprovedWarmStartCache

        max_size = 10  # Small for testing
        cache = ImprovedWarmStartCache(max_size=max_size)

        # Add more entries than max_size
        for i in range(15):
            config = {'processor_type': 'mlp', 'lambda_1_idx': i, 'lr_idx': i}
            A = np.random.randn(5, 5) * 0.1
            fitness = 0.5 + i * 0.03  # 0.5 to 0.92
            cache.add(config, A, fitness, generation=0, processor_type='mlp')

        # Cache should not exceed max_size
        self.assertLessEqual(len(cache.cache), max_size)

        # Evictions should have happened
        self.assertGreater(cache.stats['evictions'], 0)


class TestSkipLowFitness(unittest.TestCase):
    """Test 5: Skip low fitness (<0.5)."""

    def test_low_fitness_not_cached(self):
        from jcce.structure_learning.warm_start_cache import ImprovedWarmStartCache

        cache = ImprovedWarmStartCache(max_size=50)

        config = {'processor_type': 'mlp', 'lambda_1_idx': 0}
        A = np.random.randn(5, 5) * 0.1

        # Low fitness should not be cached
        cache.add(config, A, fitness=0.3, generation=0, processor_type='mlp')
        self.assertEqual(len(cache.cache), 0)

        # Adequate fitness should be cached
        cache.add(config, A, fitness=0.6, generation=0, processor_type='mlp')
        self.assertEqual(len(cache.cache), 1)


class TestIntegrationUnifiedSCDProblem(unittest.TestCase):
    """Test 6: Integration - UnifiedSCDProblem with enable_warm_start=True."""

    def test_warm_start_enabled_on_problem(self):
        """Verify warm_start_cache is initialized and wired up."""
        import jax.numpy as jnp
        from jcce.structure_learning.nsga2_search import UnifiedSCDProblem

        n_vars = 3
        n_samples = 20
        X = jnp.ones((n_samples, n_vars))
        Y = jnp.zeros(n_samples)

        problem = UnifiedSCDProblem(
            X_train=X,
            Y_train=Y,
            n_vars=n_vars,
            stage1_max_iter=5,
            use_v7=True,
            enable_warm_start=True,
            warm_start_prob=0.3,
        )

        # Cache should be initialized
        self.assertIsNotNone(problem.warm_start_cache)
        self.assertEqual(problem.warm_start_cache.max_size, 50)
        self.assertEqual(problem.warm_start_prob, 0.3)

        # Without warm_start, cache should be None
        problem_no_ws = UnifiedSCDProblem(
            X_train=X,
            Y_train=Y,
            n_vars=n_vars,
            stage1_max_iter=5,
            use_v7=True,
            enable_warm_start=False,
        )
        self.assertIsNone(problem_no_ws.warm_start_cache)


if __name__ == '__main__':
    unittest.main()
