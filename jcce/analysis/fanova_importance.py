"""
fANOVA Hyperparameter Importance (Phase B.3).

Wraps Optuna's fANOVA importance evaluator to identify which
hyperparameters matter most for causal discovery quality.

Key insight: After importance stabilizes (~100 trials), freeze
unimportant params (<5%) and narrow ranges of important ones.

Usage:
    from jcce.analysis.fanova_importance import (
        compute_param_importances,
        track_importance_stability,
        get_frozen_params,
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import optuna
    from optuna.importance import get_param_importances, FanovaImportanceEvaluator
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


def compute_param_importances(
    study,
    target_index: int = 0,
    evaluator: Optional[object] = None,
) -> Dict[str, float]:
    """
    Compute hyperparameter importances via fANOVA.

    Args:
        study: Optuna study object.
        target_index: Which objective to analyze (0=bacc, 1=sparsity).
        evaluator: Optional custom evaluator. Defaults to FanovaImportanceEvaluator.

    Returns:
        Dict mapping param_name -> importance (0-1, sums to 1).
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is required for fANOVA importance analysis")

    if evaluator is None:
        evaluator = FanovaImportanceEvaluator(seed=42)

    def target_fn(trial):
        return trial.values[target_index]

    return get_param_importances(
        study,
        evaluator=evaluator,
        target=target_fn,
    )


def compute_importances_both_objectives(
    study,
    evaluator: Optional[object] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute importances for both objectives.

    Args:
        study: Optuna study object.
        evaluator: Optional evaluator.

    Returns:
        Dict with 'balanced_accuracy' and 'sparsity' importance dicts.
    """
    return {
        'balanced_accuracy': compute_param_importances(study, target_index=0, evaluator=evaluator),
        'sparsity': compute_param_importances(study, target_index=1, evaluator=evaluator),
    }


def track_importance_stability(
    study,
    check_every: int = 50,
    target_index: int = 0,
) -> List[Dict[str, float]]:
    """
    Track importance estimates at regular intervals during optimization.

    Useful for checking when importance estimates have stabilized.

    Args:
        study: Completed Optuna study.
        check_every: Compute importance every N completed trials.
        target_index: Which objective.

    Returns:
        List of importance dicts, one per checkpoint.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is required")

    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    completed.sort(key=lambda t: t.number)

    snapshots = []
    evaluator = FanovaImportanceEvaluator(seed=42)

    for i in range(check_every, len(completed) + 1, check_every):
        # Create a partial study view
        partial_study = optuna.create_study(
            directions=study.directions,
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        for trial in completed[:i]:
            partial_study.add_trial(trial)

        try:
            imp = compute_param_importances(
                partial_study, target_index=target_index, evaluator=evaluator
            )
            snapshots.append(imp)
        except Exception:
            # Not enough trials or other fANOVA failure
            snapshots.append({})

    return snapshots


def importance_convergence(
    snapshots: List[Dict[str, float]],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Measure convergence of importance estimates across snapshots.

    For each of the top-k params (by final importance), compute the
    coefficient of variation of importance across the last 3 snapshots.

    Args:
        snapshots: List of importance dicts from track_importance_stability().
        top_k: Number of top params to track.

    Returns:
        Dict mapping param_name -> CV (coefficient of variation).
        Lower CV = more stable importance estimate.
    """
    if len(snapshots) < 3:
        return {}

    # Use final snapshot to identify top params
    final = snapshots[-1]
    if not final:
        return {}

    sorted_params = sorted(final.keys(), key=lambda k: -final[k])[:top_k]

    recent = snapshots[-3:]
    result = {}
    for param in sorted_params:
        values = [snap.get(param, 0.0) for snap in recent]
        mean = np.mean(values)
        std = np.std(values)
        cv = std / (mean + 1e-10)
        result[param] = float(cv)

    return result


def get_frozen_params(
    importances: Dict[str, float],
    threshold: float = 0.05,
    study=None,
) -> Dict[str, object]:
    """
    Identify parameters with importance below threshold for freezing.

    Args:
        importances: Dict from compute_param_importances().
        threshold: Importance below this is "unimportant" (default 5%).
        study: Optional study to extract median values for frozen params.

    Returns:
        Dict mapping param_name -> suggested frozen value (median from study,
        or None if study not provided).
    """
    frozen = {}
    for param, imp in importances.items():
        if imp < threshold:
            if study is not None:
                # Get median value from completed trials
                values = []
                for trial in study.trials:
                    if trial.state.name == 'COMPLETE' and param in trial.params:
                        values.append(trial.params[param])
                if values:
                    if isinstance(values[0], (int, float)):
                        frozen[param] = float(np.median(values))
                    else:
                        # Categorical: use mode
                        from collections import Counter
                        frozen[param] = Counter(values).most_common(1)[0][0]
                else:
                    frozen[param] = None
            else:
                frozen[param] = None

    return frozen
