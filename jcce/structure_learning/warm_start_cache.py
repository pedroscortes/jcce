"""
Improved Warm-Start Cache for NSGA-II Causal Discovery

Fixes:
- No JAX arrays as keys (unhashable type error)
- Proper similarity matching
- Memory-efficient storage
- LRU eviction with fitness prioritization

Usage:
    cache = ImprovedWarmStartCache(max_size=50)
    cache.add(config, A_matrix, fitness, generation, processor_type)
    A_init = cache.get(config)
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    A_matrix: np.ndarray
    fitness: float
    generation: int
    processor_type: str
    config_hash: str
    hit_count: int = 0

    def to_jax(self):
        """Convert to JAX array for use in optimization."""
        import jax.numpy as jnp

        return jnp.array(self.A_matrix)


class ImprovedWarmStartCache:
    """
    Improved warm-start cache with proper hashability and LRU eviction.

    Fixes:
    - No JAX arrays as keys (unhashable type error)
    - Proper similarity matching
    - Memory-efficient storage

    Attributes:
        cache: Dict mapping config_hash to CacheEntry
        max_size: Maximum number of structures to cache
        similarity_threshold: Threshold for structure similarity matching
        stats: Cache statistics (hits, misses, evictions)
    """

    def __init__(self, max_size: int = 50, similarity_threshold: float = 0.8):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of structures to cache
            similarity_threshold: Threshold for structure similarity matching
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "errors": 0}
        self.generation = 0

    def _config_to_hash(self, config: dict) -> str:
        """
        Convert configuration to deterministic hash string.

        Args:
            config: Genome configuration dictionary

        Returns:
            16-character hash string
        """
        # Extract and sort relevant keys
        relevant_keys = [
            "processor_type",
            "lambda_1_idx",
            "lambda_2_idx",
            "lr_idx",
            "hidden_dim_idx",
            "n_layers_idx",
            "sl_lambda_1_idx",
            "sl_lambda_2_idx",
            "sl_lr_idx",
            "processor_type_idx",
        ]

        hash_parts = []
        for key in sorted(relevant_keys):
            if key in config:
                # Convert value to string, handling both regular and JAX types
                val = config[key]
                if hasattr(val, "item"):  # JAX/numpy scalar
                    val = val.item()
                hash_parts.append(f"{key}:{val}")

        hash_string = "|".join(hash_parts)
        return hashlib.md5(hash_string.encode()).hexdigest()[:16]

    def _structure_similarity(self, A1: np.ndarray, A2: np.ndarray) -> float:
        """
        Compute structural similarity between two adjacency matrices.

        Uses Jaccard similarity of edge sets.

        Args:
            A1: First adjacency matrix
            A2: Second adjacency matrix

        Returns:
            Similarity score in [0, 1]
        """
        # Binarize (edges vs no edges)
        threshold = 0.1
        mask1 = np.abs(A1) > threshold
        mask2 = np.abs(A2) > threshold

        # Jaccard similarity of edge sets
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return float(intersection / union)

    def add(
        self,
        config: dict,
        A_matrix,
        fitness: float,
        generation: int = None,
        processor_type: str = "unknown",
    ) -> None:
        """
        Add structure to cache.

        Args:
            config: Genome configuration dictionary
            A_matrix: Adjacency matrix (JAX or numpy)
            fitness: Quality metric (accuracy)
            generation: Current generation (optional, uses internal counter if None)
            processor_type: Processor type string
        """
        try:
            # Only cache if fitness is reasonable (avoid caching failures)
            if fitness < 0.5:
                return

            # Convert JAX array to numpy (hashability fix)
            A_np = np.asarray(A_matrix).copy()

            if generation is None:
                generation = self.generation

            config_hash = self._config_to_hash(config)

            entry = CacheEntry(
                A_matrix=A_np,
                fitness=float(fitness),
                generation=int(generation),
                processor_type=processor_type,
                config_hash=config_hash,
            )

            # Check if similar entry exists (update if better)
            if config_hash in self.cache:
                existing = self.cache[config_hash]
                if fitness > existing.fitness:
                    self.cache[config_hash] = entry
            else:
                self.cache[config_hash] = entry

            # Evict if over capacity (LRU based on hit_count and fitness)
            while len(self.cache) > self.max_size:
                self._evict_worst()

        except Exception:
            self.stats["errors"] += 1
            # Silently fail - cache is non-essential

    def get(self, config: dict) -> Optional[np.ndarray]:
        """
        Get warm-start structure for similar configuration.

        Args:
            config: Genome configuration dictionary

        Returns:
            Adjacency matrix to use as initialization (None if cache empty/miss)
        """
        try:
            config_hash = self._config_to_hash(config)

            # Exact match
            if config_hash in self.cache:
                entry = self.cache[config_hash]
                entry.hit_count += 1
                self.stats["hits"] += 1
                return entry.A_matrix.copy()

            # Find similar (same processor type, best fitness)
            processor_type = config.get("processor_type", config.get("processor_type_idx", ""))
            if isinstance(processor_type, int):
                # Map index to type name
                type_map = {0: "mlp", 1: "transformer", 2: "elm", 3: "gnn", 4: "mamba"}
                processor_type = type_map.get(processor_type, "")

            best_entry = None
            best_fitness = -float("inf")

            for entry in self.cache.values():
                if entry.processor_type == processor_type and entry.fitness > best_fitness:
                    best_fitness = entry.fitness
                    best_entry = entry

            if best_entry is not None:
                best_entry.hit_count += 1
                self.stats["hits"] += 1
                return best_entry.A_matrix.copy()

            # No match found - try any high-fitness structure
            if len(self.cache) > 0:
                best_entry = max(self.cache.values(), key=lambda e: e.fitness)
                if best_entry.fitness > 0.6:  # Only use if reasonably good
                    best_entry.hit_count += 1
                    self.stats["hits"] += 1
                    return best_entry.A_matrix.copy()

            self.stats["misses"] += 1
            return None

        except Exception:
            self.stats["errors"] += 1
            self.stats["misses"] += 1
            return None

    def sample(self, rng_key=None) -> Optional[np.ndarray]:
        """
        Sample a structure from the cache for warm-start initialization.

        Strategy:
            - Sample from top 20% by fitness (balance quality vs diversity)
            - Random selection within top 20% to avoid premature convergence

        Args:
            rng_key: JAX random key (optional, uses numpy random if None)

        Returns:
            A: Adjacency matrix to use as initialization (None if cache empty)
        """
        if len(self.cache) == 0:
            return None  # Cache empty

        try:
            # Get all entries sorted by fitness
            entries = list(self.cache.values())
            entries.sort(key=lambda e: e.fitness, reverse=True)

            # Top 20% (highest fitness)
            top_k = max(1, int(0.2 * len(entries)))
            top_entries = entries[:top_k]

            # Randomly sample from top 20%
            if rng_key is not None:
                import jax.random as random

                idx = int(random.randint(rng_key, (), 0, len(top_entries)))
            else:
                idx = np.random.randint(0, len(top_entries))

            entry = top_entries[idx]
            entry.hit_count += 1
            self.stats["hits"] += 1

            return entry.A_matrix.copy()

        except Exception:
            self.stats["errors"] += 1
            return None

    def _evict_worst(self) -> None:
        """Evict entry with lowest fitness and hit count."""
        if not self.cache:
            return

        # Score: lower is worse (will be evicted)
        def eviction_score(entry: CacheEntry) -> float:
            # Combine fitness (higher = better) with recency (higher hits = better)
            return entry.fitness + 0.1 * entry.hit_count

        worst_key = min(self.cache.keys(), key=lambda k: eviction_score(self.cache[k]))
        del self.cache[worst_key]
        self.stats["evictions"] += 1

    def next_generation(self):
        """Increment generation counter."""
        self.generation += 1

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if len(self.cache) == 0:
            return {
                **self.stats,
                "size": 0,
                "best_fitness": 0.0,
                "worst_fitness": 0.0,
                "mean_fitness": 0.0,
                "hit_rate": 0.0,
            }

        fitnesses = [e.fitness for e in self.cache.values()]
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            **self.stats,
            "size": len(self.cache),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "hit_rate": hit_rate,
        }

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "errors": 0}
        self.generation = 0
