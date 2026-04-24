"""
Shrink-Perturb-Repeat (SPR) transfer for JCCE warm-starting.

Implements SPR (Ash & Adams, 2020): θ_new = λ·θ_old + N(0, σ²),
applied to both adjacency matrices (in logit space) and processor
parameters (in weight space with LeCun scaling).

Optimizer state is NOT transferred (fresh Adam moments per Ash & Adams).
"""

from typing import Any, Dict, List, Optional

import numpy as np


# Log-space bounds for HP distance normalization (from PROCESSOR_SEARCH_SPACE)
_HP_LOG_RANGES = {
    'lambda_1': (np.log(0.005), np.log(0.5)),
    'lambda_2': (np.log(0.001), np.log(1.0)),
    'lr': (np.log(0.0001), np.log(0.01)),
}


def hyperparameter_distance(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> float:
    """
    Compute distance between two trial configs in normalized log-space.

    Returns inf if processor types differ (SPR only within same processor).
    Only considers continuous params (lambda_1, lambda_2, lr) — categorical
    processor-specific params are ignored.

    Args:
        config_a: First config dict from suggest_hyperparams()
        config_b: Second config dict from suggest_hyperparams()

    Returns:
        Normalized L2 distance, or inf if processor types differ
    """
    if config_a.get('processor_type') != config_b.get('processor_type'):
        return float('inf')

    sq_sum = 0.0
    for key, (lo, hi) in _HP_LOG_RANGES.items():
        val_a = config_a.get(key, 0.0)
        val_b = config_b.get(key, 0.0)
        if val_a <= 0 or val_b <= 0:
            continue
        span = hi - lo
        if span == 0:
            continue
        norm_a = (np.log(val_a) - lo) / span
        norm_b = (np.log(val_b) - lo) / span
        sq_sum += (norm_a - norm_b) ** 2

    return float(np.sqrt(sq_sum))


def find_spr_candidate(
    config: Dict[str, Any],
    artifact_store: Dict[int, Dict[str, Any]],
    max_distance: float = 0.5,
) -> Optional[int]:
    """
    Find the closest completed trial suitable for SPR transfer.

    Searches the artifact store for the trial with the same processor type
    and smallest hyperparameter distance below max_distance.

    Args:
        config: Current trial's config from suggest_hyperparams()
        artifact_store: Module-level _trial_artifacts dict
        max_distance: Maximum HP distance to consider a match

    Returns:
        Trial number of best candidate, or None if no match
    """
    best_trial = None
    best_dist = max_distance

    for trial_num, artifacts in artifact_store.items():
        stored_config = artifacts.get('config')
        if stored_config is None:
            continue
        dist = hyperparameter_distance(config, stored_config)
        if dist < best_dist:
            best_dist = dist
            best_trial = trial_num

    return best_trial


def spr_adjacency(
    A_old: np.ndarray,
    lambda_A: float = 0.85,
    noise_scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply SPR to an adjacency matrix in logit space.

    Logit-space shrinkage preserves strong edges while allowing
    exploration near the 0.5 boundary.

    Args:
        A_old: Previous trial's continuous adjacency matrix
        lambda_A: Shrinkage factor (1.0 = full transfer, 0.0 = random)
        noise_scale: Std dev of Gaussian noise in logit space
        rng: NumPy random generator

    Returns:
        New adjacency matrix after SPR
    """
    if rng is None:
        rng = np.random.default_rng()

    # Clip to valid logit range
    A_clipped = np.clip(A_old, 1e-6, 1.0 - 1e-6)

    # Convert to logit space
    logit_A = np.log(A_clipped / (1.0 - A_clipped))

    # SPR: shrink + perturb in logit space
    noise = rng.normal(0.0, noise_scale, size=logit_A.shape)
    logit_new = lambda_A * logit_A + noise

    # Convert back via sigmoid
    A_new = 1.0 / (1.0 + np.exp(-logit_new))

    # Enforce constraints
    A_new = np.clip(A_new, 0.0, 1.0)
    np.fill_diagonal(A_new, 0.0)

    return A_new


def spr_processor_params(
    old_proc_params: List[Dict[str, Any]],
    lambda_theta: float = 0.7,
    noise_scale_multiplier: float = 1.0,
    n_inputs: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """
    Apply SPR to processor parameters.

    Only transforms `flat_params` arrays; metadata (tree_def, shapes,
    n_inputs) is passed through unchanged. Uses LeCun-scaled noise
    (sqrt(1/n_inputs)) for appropriate perturbation magnitude.

    Optimizer state is NOT transferred (Ash & Adams 2020 recommendation).

    Args:
        old_proc_params: List of param dicts from previous trial
        lambda_theta: Shrinkage factor for weights
        noise_scale_multiplier: Scales the LeCun noise std
        n_inputs: Number of input features (for LeCun scaling)
        rng: NumPy random generator

    Returns:
        New processor params with SPR-transformed flat_params
    """
    if rng is None:
        rng = np.random.default_rng()

    lecun_std = np.sqrt(1.0 / max(n_inputs, 1))

    new_params = []
    for p in old_proc_params:
        new_p = dict(p)  # shallow copy

        if 'flat_params' in new_p:
            flat = np.asarray(new_p['flat_params'], dtype=np.float32)
            noise = rng.normal(
                0.0,
                lecun_std * noise_scale_multiplier,
                size=flat.shape,
            ).astype(np.float32)
            new_p['flat_params'] = lambda_theta * flat + noise

        new_params.append(new_p)

    return new_params
