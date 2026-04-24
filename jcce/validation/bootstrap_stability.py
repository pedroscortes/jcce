"""
Bootstrap DAG Stability for JCCE.

Gold standard for structural uncertainty quantification without ground truth.
B bootstrap resamples → retrain GOLEM → edge frequency analysis.

References:
- Friedman et al. (1999) "Learning Bayesian Network Structure from Massive Datasets"
- Meinshausen & Bühlmann (2010) "Stability Selection" (JRSSB)
- Scutari (2013) "Identifying Significant Edges in Graphical Models of Molecular Networks"

Usage:
    from jcce.validation.bootstrap_stability import (
        bootstrap_dag_stability,
        BootstrapStabilityResult,
    )
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BootstrapStabilityResult:
    """Result of bootstrap DAG stability analysis."""

    # Core results
    edge_frequency: np.ndarray  # (n_total, n_total) — P(edge present)
    n_bootstraps: int  # Number of completed bootstrap samples
    n_requested: int  # Number of requested samples (B)

    # Edge statistics
    n_stable_edges: int  # Edges with frequency >= stability_threshold
    n_total_edges_ever: int  # Edges appearing in at least one bootstrap
    stability_threshold: float  # Threshold used for "stable"

    # Per-bootstrap metadata
    n_edges_per_bootstrap: List[int]  # Edge count per resample
    h_A_per_bootstrap: List[float]  # DAG constraint per resample
    bacc_per_bootstrap: List[float]  # Balanced accuracy per resample

    # Timing
    total_time: float
    time_per_bootstrap: float

    def get_stable_edges(
        self,
        min_frequency: float = 0.5,
    ) -> List[Tuple[int, int, float]]:
        """Return edges with frequency >= min_frequency as (i, j, freq) triples."""
        n = self.edge_frequency.shape[0]
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and self.edge_frequency[i, j] >= min_frequency:
                    edges.append((i, j, float(self.edge_frequency[i, j])))
        edges.sort(key=lambda x: x[2], reverse=True)
        return edges

    def to_dict(self) -> Dict:
        return {
            "edge_frequency": self.edge_frequency.tolist(),
            "n_bootstraps": self.n_bootstraps,
            "n_requested": self.n_requested,
            "n_stable_edges": self.n_stable_edges,
            "n_total_edges_ever": self.n_total_edges_ever,
            "stability_threshold": self.stability_threshold,
            "n_edges_per_bootstrap": self.n_edges_per_bootstrap,
            "total_time": self.total_time,
            "time_per_bootstrap": self.time_per_bootstrap,
        }

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BOOTSTRAP DAG STABILITY",
            "=" * 60,
            f"Bootstraps: {self.n_bootstraps}/{self.n_requested}",
            f"Stability threshold: {self.stability_threshold}",
            f"Stable edges (freq >= {self.stability_threshold}): {self.n_stable_edges}",
            f"Total edges ever seen: {self.n_total_edges_ever}",
            f"Edges per bootstrap: {np.mean(self.n_edges_per_bootstrap):.1f} "
            f"± {np.std(self.n_edges_per_bootstrap):.1f}",
            f"Time: {self.total_time:.1f}s ({self.time_per_bootstrap:.1f}s/bootstrap)",
        ]

        # Top stable edges
        stable = self.get_stable_edges(self.stability_threshold)
        if stable:
            lines.append(f"\nTop stable edges (freq >= {self.stability_threshold}):")
            for i, j, freq in stable[:10]:
                lines.append(f"  {i} -> {j}: {freq:.3f}")
            if len(stable) > 10:
                lines.append(f"  ... and {len(stable) - 10} more")

        lines.append("=" * 60)
        return "\n".join(lines)


def bootstrap_dag_stability(
    X: np.ndarray,
    Y: np.ndarray,
    hyperparams: Dict[str, Any],
    processor_type: str,
    B: int = 200,
    max_iter: int = 100,
    A_init: Optional[np.ndarray] = None,
    edge_threshold: float = 0.3,
    stability_threshold: float = 0.5,
    seed: int = 42,
    task: str = "classification",
    verbose: bool = False,
) -> BootstrapStabilityResult:
    """
    Bootstrap DAG stability analysis.

    For each of B bootstrap resamples:
    1. Resample (X, Y) with replacement
    2. Retrain GOLEM (warm-start from A_init for speed)
    3. Binarize learned A and record edge presence

    Edge frequency = fraction of bootstraps where |A[i,j]| > edge_threshold.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Target variable (n_samples,)
        hyperparams: GOLEM hyperparameters (lambda_1, lambda_2, etc.)
        processor_type: Processor type ('elm', 'mlp', etc.)
        B: Number of bootstrap resamples (recommended: 100-200)
        max_iter: GOLEM iterations per bootstrap (fewer than full training)
        A_init: Ignored (cold-start used per Meinshausen & Bühlmann 2010 to
                avoid selection bias from full-data solution)
        edge_threshold: Threshold for edge presence (|A[i,j]| > threshold)
        stability_threshold: Minimum frequency to consider an edge "stable"
        seed: Random seed for reproducibility
        task: 'classification' or 'regression'
        verbose: Print progress

    Returns:
        BootstrapStabilityResult with edge frequency matrix and diagnostics
    """
    import jax
    import jax.numpy as jnp
    from jax import random

    from jcce.structure_learning.jcce_learner import (
        create_processor,
        learn_structure,
    )

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y).flatten()
    n_samples, n_features = X.shape
    n_total = n_features + 1  # Including Y

    rng = np.random.RandomState(seed)
    key = random.PRNGKey(seed)

    # Pre-allocate
    edge_counts = np.zeros((n_total, n_total))
    n_edges_per_bootstrap = []
    h_A_per_bootstrap = []
    bacc_per_bootstrap = []
    n_completed = 0

    total_start = time.time()

    for b in range(B):
        # Resample with replacement
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]
        Y_boot = Y[idx]

        key, proc_key, fold_key = random.split(key, 3)

        processor = create_processor(
            processor_type,
            key=proc_key,
            n_features=n_features,
            **hyperparams.get("processor_config", {}),
        )

        # Cold-start each bootstrap resample per Meinshausen & Bühlmann (2010):
        # warm-starting from full-data A biases edge frequencies toward the
        # original solution, inflating apparent stability.
        A_warm = None

        try:
            A_est, _, _, metrics = learn_structure(
                data=jnp.array(X_boot),
                Y=jnp.array(Y_boot).reshape(-1, 1).astype(jnp.float32),
                Y_idx=n_features,
                processor=processor,
                key=fold_key,
                processor_type=processor_type,
                lambda_1=hyperparams.get("lambda_1", 0.02),
                lambda_2_init=hyperparams.get("lambda_2", 0.01),
                lambda_class=hyperparams.get("lambda_class", 1.0),
                lr=hyperparams.get("lr", 0.001),
                max_iter=max_iter,
                patience=25,  # Cold-start needs more patience
                verbose=0,
                A_init=A_warm,
                effect_hidden_dim=hyperparams.get("effect_hidden_dim", 64),
                effect_embed_dim=hyperparams.get("effect_embed_dim", 16),
                lambda_effect=hyperparams.get("lambda_effect", 0.5),
                effect_warmup_iter=hyperparams.get("effect_warmup_iter", 20),
                lambda_confound_sparse=hyperparams.get("lambda_confound_sparse", 0.02),
                lambda_bow=hyperparams.get("lambda_bow", 0.3),
                task=task,
            )

            A_np = np.array(A_est)
            edges_present = (np.abs(A_np) > edge_threshold).astype(float)
            np.fill_diagonal(edges_present, 0)
            edge_counts += edges_present

            n_edges = int(np.sum(edges_present))
            n_edges_per_bootstrap.append(n_edges)
            h_A_per_bootstrap.append(float(metrics.get("h_A", 0.0)))
            bacc_per_bootstrap.append(float(metrics.get("classification_balanced_accuracy", 0.0)))
            n_completed += 1

            if verbose and (b + 1) % max(1, B // 10) == 0:
                elapsed = time.time() - total_start
                rate = elapsed / (b + 1)
                print(
                    f"  Bootstrap {b + 1}/{B}: {n_edges} edges, {elapsed:.0f}s ({rate:.1f}s/iter)"
                )

        except Exception as e:
            warnings.warn(f"Bootstrap {b + 1} failed: {e}")
            continue

        # Cleanup periodically
        if (b + 1) % 20 == 0:
            jax.clear_caches()

    total_time = time.time() - total_start

    # Compute frequency
    edge_frequency = edge_counts / max(n_completed, 1)

    # Count stable edges
    n_stable = int(np.sum((edge_frequency >= stability_threshold) & ~np.eye(n_total, dtype=bool)))
    n_total_edges_ever = int(np.sum((edge_frequency > 0) & ~np.eye(n_total, dtype=bool)))

    result = BootstrapStabilityResult(
        edge_frequency=edge_frequency,
        n_bootstraps=n_completed,
        n_requested=B,
        n_stable_edges=n_stable,
        n_total_edges_ever=n_total_edges_ever,
        stability_threshold=stability_threshold,
        n_edges_per_bootstrap=n_edges_per_bootstrap,
        h_A_per_bootstrap=h_A_per_bootstrap,
        bacc_per_bootstrap=bacc_per_bootstrap,
        total_time=total_time,
        time_per_bootstrap=total_time / max(n_completed, 1),
    )

    if verbose:
        print(f"\n{result.summary()}")

    return result
