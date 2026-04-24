"""
Global adjacency prior via exponential moving average (EMA).

Maintains a running EMA over adjacency matrices from feasible Pareto
solutions, providing a structured initialization bias for new trials.

Thread-safe for use with Optuna n_jobs > 1.
"""

import threading
from typing import Optional

import numpy as np


class AdjacencyPrior:
    """
    EMA prior over continuous adjacency matrices from good GOLEM solutions.

    Starts uninformative (0.5 off-diagonal) and converges toward historically
    strong edge patterns as feasible trials accumulate.

    Args:
        n_total: Number of nodes (n_vars + 1 for Y)
        beta: EMA decay factor (higher = slower adaptation)
        min_prob: Floor for edge probabilities after clipping
        max_prob: Ceiling for edge probabilities after clipping
        fitness_threshold: Minimum fitness to accept an update
    """

    def __init__(
        self,
        n_total: int,
        beta: float = 0.9,
        min_prob: float = 0.05,
        max_prob: float = 0.95,
        fitness_threshold: float = 0.6,
    ):
        self._n_total = n_total
        self._beta = beta
        self._min_prob = min_prob
        self._max_prob = max_prob
        self._fitness_threshold = fitness_threshold

        # Start uninformative: 0.5 off-diagonal, 0 on diagonal
        self._A_tilde = np.full((n_total, n_total), 0.5)
        np.fill_diagonal(self._A_tilde, 0.0)

        self._n_updates = 0
        self._lock = threading.Lock()

    @property
    def A_tilde(self) -> np.ndarray:
        """Current EMA adjacency matrix (read-only copy)."""
        with self._lock:
            return self._A_tilde.copy()

    @property
    def n_updates(self) -> int:
        with self._lock:
            return self._n_updates

    def update(self, A_pareto: np.ndarray, fitness: float) -> None:
        """
        Update the EMA prior with a feasible solution's adjacency matrix.

        Args:
            A_pareto: Continuous adjacency matrix from a feasible trial
            fitness: Combined fitness score (e.g. 0.7*bacc + 0.3*sparsity)
        """
        if fitness < self._fitness_threshold:
            return

        A_clipped = np.clip(A_pareto, 0.0, 1.0)

        with self._lock:
            self._A_tilde = self._beta * self._A_tilde + (1.0 - self._beta) * A_clipped
            self._A_tilde = np.clip(self._A_tilde, self._min_prob, self._max_prob)
            np.fill_diagonal(self._A_tilde, 0.0)
            self._n_updates += 1

    def sample_A_init(
        self,
        noise_scale: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample an initial adjacency matrix from the prior.

        Cold-start (< 5 updates): returns small random values matching
        GOLEM's default initialization.
        Warm (>= 5 updates): returns prior + Gaussian noise.

        Args:
            noise_scale: Std dev of additive Gaussian noise
            rng: NumPy random generator (default: fresh one)

        Returns:
            (n_total, n_total) initial adjacency matrix
        """
        if rng is None:
            rng = np.random.default_rng()

        with self._lock:
            n_updates = self._n_updates
            A_base = self._A_tilde.copy()

        if n_updates < 5:
            # Cold-start: small random values matching GOLEM default
            A_init = rng.normal(0.0, 0.1, size=(self._n_total, self._n_total))
            np.fill_diagonal(A_init, 0.0)
            return A_init

        # Warm: prior + noise
        noise = rng.normal(0.0, noise_scale, size=(self._n_total, self._n_total))
        A_init = np.clip(A_base + noise, 0.0, 1.0)
        np.fill_diagonal(A_init, 0.0)
        return A_init

    def get_stats(self) -> dict:
        """Return summary statistics of the current prior."""
        with self._lock:
            A = self._A_tilde.copy()
            n_updates = self._n_updates

        mask = ~np.eye(self._n_total, dtype=bool)
        off_diag = A[mask]
        return {
            "n_updates": n_updates,
            "mean_edge_prob": float(off_diag.mean()),
            "std_edge_prob": float(off_diag.std()),
            "min_edge_prob": float(off_diag.min()),
            "max_edge_prob": float(off_diag.max()),
            "n_strong_edges": int((off_diag > 0.7).sum()),
            "n_weak_edges": int((off_diag < 0.3).sum()),
        }
