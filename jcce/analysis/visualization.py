"""
Pareto Front Visualization (Phase B.5).

Plots for analyzing Pareto fronts, edge stability, hypervolume
anytime curves, and fANOVA importance.

All functions return matplotlib Figure objects for flexibility
(save, display, embed in notebooks).

Usage:
    from jcce.analysis.visualization import (
        plot_pareto_front,
        plot_edge_stability_heatmap,
        plot_anytime_hypervolume,
        plot_importance_bar,
        plot_stability_sensitivity,
    )
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# Processor color scheme (consistent across all plots)
PROCESSOR_COLORS = {
    "elm": "#1f77b4",
    "gnn": "#ff7f0e",
    "mlp": "#2ca02c",
    "transformer": "#d62728",
    "mamba": "#9467bd",
    "unknown": "#7f7f7f",
}


def _check_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualization")


def plot_pareto_front(
    solutions: List[Dict],
    bacc_key: str = "classification_balanced_accuracy",
    sparsity_key: str = "mb_sparsity",
    processor_key: str = "processor_type",
    title: str = "Pareto Front: Balanced Accuracy vs Sparsity",
    figsize: Tuple[int, int] = (8, 6),
) -> "plt.Figure":
    """
    Processor-colored scatter plot of Pareto solutions.

    Args:
        solutions: List of enhanced_solution dicts.
        bacc_key: Metric key for balanced accuracy.
        sparsity_key: Metric key for sparsity.
        processor_key: Metric key for processor type.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=figsize)

    # Group by processor
    proc_points = {}
    for sol in solutions:
        m = sol.get("metrics", {})
        proc = m.get(processor_key, "unknown")
        bacc = m.get(bacc_key, 0.0)
        sparsity = m.get(sparsity_key, 0.0)

        if proc not in proc_points:
            proc_points[proc] = {"x": [], "y": []}
        proc_points[proc]["x"].append(bacc)
        proc_points[proc]["y"].append(sparsity)

    for proc, pts in sorted(proc_points.items()):
        color = PROCESSOR_COLORS.get(proc, "#7f7f7f")
        ax.scatter(
            pts["x"],
            pts["y"],
            c=color,
            label=proc,
            s=60,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Balanced Accuracy", fontsize=12)
    ax.set_ylabel("MB Sparsity", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Processor", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_edge_stability_heatmap(
    stability: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Edge Stability Heatmap",
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> "plt.Figure":
    """
    d x d heatmap of edge stability frequencies.

    Args:
        stability: (n, n) stability matrix from compute_edge_stability().
        feature_names: Optional variable names for axis labels.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap name.
        vmin, vmax: Color scale range.

    Returns:
        matplotlib Figure.
    """
    _check_mpl()

    n = stability.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        stability, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest"
    )

    if feature_names is not None and len(feature_names) == n:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(feature_names, fontsize=9)
    else:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

    ax.set_xlabel("To (child)", fontsize=11)
    ax.set_ylabel("From (parent)", fontsize=11)
    ax.set_title(title, fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("P(edge present)", fontsize=11)

    fig.tight_layout()
    return fig


def plot_anytime_hypervolume(
    curves: Dict[str, Dict[str, np.ndarray]],
    title: str = "Anytime Hypervolume",
    figsize: Tuple[int, int] = (8, 5),
    xlabel: str = "Wall-clock time (s)",
) -> "plt.Figure":
    """
    Plot hypervolume vs wall-clock time for multiple methods.

    Args:
        curves: Dict mapping method_name -> anytime curve dict
                with 'timestamps' and 'hypervolumes' arrays.
        title: Plot title.
        figsize: Figure size.
        xlabel: X-axis label.

    Returns:
        matplotlib Figure.
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=figsize)

    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, (name, curve) in enumerate(sorted(curves.items())):
        ts = curve["timestamps"]
        hvs = curve["hypervolumes"]
        color = colors[i % len(colors)]
        ax.plot(ts, hvs, label=name, color=color, linewidth=2)
        ax.scatter(ts, hvs, color=color, s=15, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Dominated Hypervolume", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_importance_bar(
    importances: Dict[str, float],
    title: str = "Hyperparameter Importance (fANOVA)",
    figsize: Tuple[int, int] = (10, 5),
    top_k: int = 15,
    threshold: float = 0.05,
) -> "plt.Figure":
    """
    Horizontal bar chart of hyperparameter importances.

    Args:
        importances: Dict from compute_param_importances().
        title: Plot title.
        figsize: Figure size.
        top_k: Show only top K params.
        threshold: Draw vertical line at this importance level.

    Returns:
        matplotlib Figure.
    """
    _check_mpl()

    # Sort by importance
    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]
    names = [p[0] for p in reversed(sorted_params)]
    values = [p[1] for p in reversed(sorted_params)]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#2ca02c" if v >= threshold else "#d62728" for v in values]
    ax.barh(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=13)

    if threshold > 0:
        ax.axvline(
            x=threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({threshold:.0%})",
        )
        ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig


def plot_stability_sensitivity(
    sensitivity: Dict[str, object],
    metric: str = "n_stable_edges",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 5),
) -> "plt.Figure":
    """
    Plot sensitivity analysis results (threshold vs frequency grid).

    Args:
        sensitivity: Dict from stability_sensitivity_analysis().
        metric: Which metric to plot ('n_stable_edges', 'precision', 'recall', 'f1').
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _check_mpl()

    data = sensitivity.get(metric)
    if data is None:
        raise ValueError(f"Metric '{metric}' not found in sensitivity results")

    thresholds = sensitivity["thresholds"]
    frequencies = sensitivity["frequencies"]

    if title is None:
        title = f"Stability Sensitivity: {metric}"

    fig, ax = plt.subplots(figsize=figsize)

    for ti, t in enumerate(thresholds):
        ax.plot(frequencies, data[ti, :], marker="o", label=f"threshold={t}", linewidth=1.5)

    ax.set_xlabel("Min Stability Frequency", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, title="Edge threshold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
