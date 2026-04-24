"""
Hypervolume Computation (Phase B.2).

Dominated hypervolume for 2D Pareto fronts, plus anytime performance
curves (hypervolume vs wall-clock time) for comparing optimization
strategies.

Usage:
    from jcce.analysis.hypervolume import (
        compute_hypervolume_2d,
        compute_anytime_hypervolume,
        compare_anytime_curves,
    )
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def compute_hypervolume_2d(
    points: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """
    Compute dominated hypervolume for 2D maximization objectives.

    O(n log n) via sorted sweep.

    Args:
        points: (n, 2) array of objective values (higher = better).
        ref_point: (2,) reference point (lower bound).

    Returns:
        Dominated hypervolume area.
    """
    if len(points) == 0:
        return 0.0

    # Filter out points dominated by ref point
    valid = np.all(points > ref_point, axis=1)
    points = points[valid]

    if len(points) == 0:
        return 0.0

    # Sort by first objective descending
    sorted_idx = np.argsort(-points[:, 0])
    points = points[sorted_idx]

    hv = 0.0
    prev_y = ref_point[1]

    for p in points:
        if p[1] > prev_y:
            hv += (p[0] - ref_point[0]) * (p[1] - prev_y)
            prev_y = p[1]

    return hv


def extract_pareto_front_2d(
    points: np.ndarray,
) -> np.ndarray:
    """
    Extract non-dominated points from a 2D point set (maximization).

    Args:
        points: (n, 2) array of objective values.

    Returns:
        (k, 2) array of non-dominated points.
    """
    if len(points) == 0:
        return points

    # Sort by first objective descending
    sorted_idx = np.argsort(-points[:, 0])
    sorted_pts = points[sorted_idx]

    pareto = [sorted_pts[0]]
    max_y = sorted_pts[0, 1]

    for p in sorted_pts[1:]:
        if p[1] > max_y:
            pareto.append(p)
            max_y = p[1]

    return np.array(pareto)


def compute_anytime_hypervolume(
    study,
    ref_point: np.ndarray = np.array([0.0, 0.0]),
    objective_indices: Tuple[int, int] = (0, 1),
) -> Dict[str, np.ndarray]:
    """
    Compute hypervolume over wall-clock time from an Optuna study.

    Tracks how the Pareto front's dominated hypervolume grows as
    trials complete.

    Args:
        study: Optuna study object (multi-objective).
        ref_point: (2,) reference point for hypervolume.
        objective_indices: Which objectives to use (default: first two).

    Returns:
        Dict with:
            'timestamps': (n,) array of wall-clock seconds from study start.
            'hypervolumes': (n,) array of cumulative hypervolume at each trial.
            'n_pareto': (n,) array of cumulative Pareto front size.
    """
    trials = [t for t in study.trials
              if t.state.name == 'COMPLETE' and t.values is not None]

    if not trials:
        return {
            'timestamps': np.array([]),
            'hypervolumes': np.array([]),
            'n_pareto': np.array([]),
        }

    # Sort by completion time
    trials.sort(key=lambda t: t.datetime_complete or t.datetime_start)

    start_time = min(
        t.datetime_start for t in trials if t.datetime_start is not None
    )

    timestamps = []
    hypervolumes = []
    n_pareto_list = []
    all_points = []

    i0, i1 = objective_indices

    for trial in trials:
        obj = np.array([trial.values[i0], trial.values[i1]])
        all_points.append(obj)

        pts = np.array(all_points)
        pareto = extract_pareto_front_2d(pts)
        hv = compute_hypervolume_2d(pareto, ref_point)

        dt = trial.datetime_complete or trial.datetime_start
        elapsed = (dt - start_time).total_seconds()

        timestamps.append(elapsed)
        hypervolumes.append(hv)
        n_pareto_list.append(len(pareto))

    return {
        'timestamps': np.array(timestamps),
        'hypervolumes': np.array(hypervolumes),
        'n_pareto': np.array(n_pareto_list),
    }


def compute_anytime_hypervolume_from_solutions(
    solutions: List[Dict],
    ref_point: np.ndarray = np.array([0.0, 0.0]),
    bacc_key: str = 'classification_balanced_accuracy',
    sparsity_key: str = 'mb_sparsity',
    time_key: str = 'wall_time',
) -> Dict[str, np.ndarray]:
    """
    Compute anytime hypervolume from a list of enhanced_solutions with timestamps.

    For use with NSGA-II results or any solver that provides per-solution timing.

    Args:
        solutions: List of dicts with sol['metrics'][bacc_key], sol['metrics'][sparsity_key].
        ref_point: Reference point.
        bacc_key: Key for balanced accuracy metric.
        sparsity_key: Key for sparsity metric.
        time_key: Key for wall-clock time (in sol['metrics'] or sol).

    Returns:
        Same format as compute_anytime_hypervolume().
    """
    if not solutions:
        return {
            'timestamps': np.array([]),
            'hypervolumes': np.array([]),
            'n_pareto': np.array([]),
        }

    # Sort by time
    def get_time(sol):
        t = sol.get('metrics', {}).get(time_key)
        if t is None:
            t = sol.get(time_key, 0.0)
        return float(t)

    sorted_sols = sorted(solutions, key=get_time)

    timestamps = []
    hypervolumes = []
    n_pareto_list = []
    all_points = []

    for sol in sorted_sols:
        m = sol.get('metrics', {})
        bacc = m.get(bacc_key, 0.0)
        sparsity = m.get(sparsity_key, 0.0)
        all_points.append([bacc, sparsity])

        pts = np.array(all_points)
        pareto = extract_pareto_front_2d(pts)
        hv = compute_hypervolume_2d(pareto, ref_point)

        timestamps.append(get_time(sol))
        hypervolumes.append(hv)
        n_pareto_list.append(len(pareto))

    return {
        'timestamps': np.array(timestamps),
        'hypervolumes': np.array(hypervolumes),
        'n_pareto': np.array(n_pareto_list),
    }


def compare_anytime_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    target_hv: Optional[float] = None,
) -> Dict[str, object]:
    """
    Compare anytime hypervolume curves from different methods.

    Args:
        curves: Dict mapping method_name -> anytime curve dict
                (from compute_anytime_hypervolume).
        target_hv: Optional target hypervolume for time-to-target comparison.

    Returns:
        Dict with:
            'final_hv': method -> final hypervolume
            'final_n_pareto': method -> final Pareto size
            'time_to_target': method -> seconds to reach target_hv (if provided)
            'total_time': method -> total wall-clock time
    """
    result = {
        'final_hv': {},
        'final_n_pareto': {},
        'total_time': {},
    }

    if target_hv is not None:
        result['time_to_target'] = {}

    for name, curve in curves.items():
        hvs = curve['hypervolumes']
        ts = curve['timestamps']

        result['final_hv'][name] = float(hvs[-1]) if len(hvs) > 0 else 0.0
        result['final_n_pareto'][name] = int(curve['n_pareto'][-1]) if len(curve['n_pareto']) > 0 else 0
        result['total_time'][name] = float(ts[-1]) if len(ts) > 0 else 0.0

        if target_hv is not None:
            reached = np.where(hvs >= target_hv)[0]
            if len(reached) > 0:
                result['time_to_target'][name] = float(ts[reached[0]])
            else:
                result['time_to_target'][name] = float('inf')

    return result
