"""
DAG Visualization.

Publication-quality causal DAG visualization using Graphviz.
Supports:
- Directed edges (A_direct) with ATE labels
- Bidirected edges (A_confound) for latent confounders
- Markov Blanket highlighting
- Treatment/Outcome node styling

Usage:
    from jcce.visualization.dag_viz import visualize_jcce_dag

    visualize_jcce_dag(
        A_direct=metrics['A_direct'],
        A_confound=metrics.get('A_confound'),
        causal_effects=metrics.get('causal_effects', {}),
        markov_blanket=metrics['markov_blanket'],
        feature_names=['X0', 'X1', ..., 'Y'],
        treatment_idx=0,
        outcome_idx=-1,
        output_path='dag.pdf'
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import graphviz

    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: graphviz not installed. Install with: pip install graphviz")


# =============================================================================
# Color Schemes
# =============================================================================

# Publication-friendly color scheme (colorblind-safe)
COLORS = {
    "treatment": "#2E86AB",  # Blue - treatment variable
    "outcome": "#A23B72",  # Magenta - outcome variable
    "markov_blanket": "#F18F01",  # Orange - MB variables
    "other": "#C5C3C6",  # Gray - non-MB variables
    "directed_edge": "#1B1B1E",  # Black - directed edges
    "bidirected_edge": "#E94F37",  # Red - bidirected/confounded edges
    "edge_label": "#495057",  # Dark gray - edge labels
}

# Alternative schemes
COLORS_MINIMAL = {
    "treatment": "#000000",
    "outcome": "#000000",
    "markov_blanket": "#666666",
    "other": "#AAAAAA",
    "directed_edge": "#000000",
    "bidirected_edge": "#000000",
    "edge_label": "#333333",
}

# Academic style - gray nodes, black text, sober look (matches lucas_sol4_sparsest.png)
COLORS_ACADEMIC = {
    "treatment": "#C0C0C0",  # Light gray
    "outcome": "#C0C0C0",  # Light gray (same as others)
    "markov_blanket": "#C0C0C0",  # Light gray
    "other": "#C0C0C0",  # Light gray
    "directed_edge": "#000000",  # Black
    "bidirected_edge": "#000000",  # Black
    "edge_label": "#000000",  # Black
    "node_border": "#000000",  # Black border
}

COLORS_NATURE = {
    "treatment": "#E64B35",  # Nature red
    "outcome": "#4DBBD5",  # Nature cyan
    "markov_blanket": "#00A087",  # Nature green
    "other": "#B09C85",  # Nature tan
    "directed_edge": "#3C5488",  # Nature blue
    "bidirected_edge": "#F39B7F",  # Nature salmon
    "edge_label": "#3C5488",
}


# =============================================================================
# Main Visualization Function
# =============================================================================


def visualize_jcce_dag(
    A_direct: Union[np.ndarray, List[List[float]]],
    feature_names: List[str],
    A_confound: Optional[Union[np.ndarray, List[List[float]]]] = None,
    causal_effects: Optional[Dict[str, float]] = None,
    markov_blanket: Optional[List[int]] = None,
    treatment_idx: Optional[int] = None,  # None = no special treatment node
    outcome_idx: Optional[int] = None,  # None = no special outcome node
    highlight_nodes: Optional[List[int]] = None,  # Custom nodes to highlight
    threshold: float = 0.01,
    output_path: Optional[str] = None,
    output_format: str = "pdf",
    title: Optional[str] = None,
    show_edge_weights: bool = False,
    show_ate: bool = True,
    color_scheme: str = "default",
    layout: str = "dot",
    rankdir: str = "TB",
    node_shape: str = "ellipse",
    fontname: str = "Helvetica",
    fontsize: str = "12",
    width: Optional[str] = None,
    height: Optional[str] = None,
    dpi: str = "300",
) -> "graphviz.Digraph":
    """
    Create publication-quality DAG visualization from JCCE output.

    Two modes of operation:

    1. GENERAL CAUSAL DISCOVERY (treatment_idx=None, outcome_idx=None):
       - All nodes treated equally
       - Shows learned causal structure between all variables
       - Markov Blanket or highlight_nodes used for emphasis

    2. SUPERVISED / TREATMENT-EFFECT MODE (treatment_idx and/or outcome_idx set):
       - Specific nodes highlighted as treatment/outcome
       - ATE labels shown on edges to outcome
       - Useful for IHDP-like benchmarks with known treatment

    Args:
        A_direct: Directed adjacency matrix (n_vars, n_vars). A[i,j] > 0 means i -> j
        feature_names: List of variable names ['X0', 'X1', ..., 'Y']
        A_confound: Optional bidirected adjacency matrix (symmetric). A[i,j] > 0 means i <-> j
        causal_effects: Dict of ATE estimates {'X0->Y': 0.3, 'X1->X2': 0.1, ...}
        markov_blanket: List of feature indices in the Markov Blanket (highlighted)
        treatment_idx: Index of treatment variable (None = no treatment designation)
        outcome_idx: Index of outcome variable (None = no outcome designation)
        highlight_nodes: Custom list of node indices to highlight (alternative to MB)
        threshold: Edge weight threshold for inclusion
        output_path: Path to save the figure (without extension)
        output_format: Output format ('pdf', 'png', 'svg')
        title: Optional title for the graph
        show_edge_weights: Show raw edge weights on directed edges
        show_ate: Show ATE values on edges (requires causal_effects dict)
        color_scheme: 'default', 'minimal', or 'nature'
        layout: Graphviz layout engine ('dot', 'neato', 'fdp', 'circo', 'twopi')
        rankdir: Rank direction ('TB'=top-bottom, 'LR'=left-right, 'BT', 'RL')
        node_shape: Node shape ('ellipse', 'circle', 'box', 'diamond')
        fontname: Font family
        fontsize: Font size
        width: Graph width in inches
        height: Graph height in inches
        dpi: Resolution for raster output

    Returns:
        graphviz.Digraph object (can be further customized)
    """
    if not HAS_GRAPHVIZ:
        raise ImportError("graphviz package required. Install with: pip install graphviz")

    # Convert to numpy arrays
    A_direct = np.array(A_direct)
    n_vars = A_direct.shape[0]

    if A_confound is not None:
        A_confound = np.array(A_confound)

    # Handle negative outcome_idx (only if outcome_idx is specified)
    if outcome_idx is not None and outcome_idx < 0:
        outcome_idx = n_vars + outcome_idx

    # Select color scheme
    if color_scheme == "minimal":
        colors = COLORS_MINIMAL
    elif color_scheme == "nature":
        colors = COLORS_NATURE
    elif color_scheme == "academic":
        colors = COLORS_ACADEMIC
    else:
        colors = COLORS

    # Create directed graph
    dot = graphviz.Digraph(
        name="jcce_dag",
        engine=layout,
        format=output_format,
    )

    # Graph attributes — wider spacing for label visibility.
    graph_attrs = {
        "rankdir": rankdir,
        "splines": "true",  # Curved edges
        "overlap": "false",
        "fontname": fontname,
        "fontsize": fontsize,
        "dpi": dpi,
        "bgcolor": "white",  # or transparent
        "ranksep": "0.8",  # Vertical spacing between ranks
        "nodesep": "0.6",  # Horizontal spacing between nodes
    }
    if width:
        graph_attrs["size"] = f"{width},{height}" if height else width
    if title:
        graph_attrs["label"] = title
        graph_attrs["labelloc"] = "t"
        graph_attrs["fontsize"] = "14"

    dot.attr(**graph_attrs)

    # Default node attributes
    dot.attr(
        "node",
        shape=node_shape,
        style="filled",
        fontname=fontname,
        fontsize=fontsize,
        penwidth="1.5",
    )

    # Default edge attributes
    dot.attr(
        "edge",
        fontname=fontname,
        fontsize="10",
        penwidth="1.2",
    )

    # Add nodes with appropriate styling
    mb_set = set(markov_blanket) if markov_blanket else set()
    highlight_set = set(highlight_nodes) if highlight_nodes else set()

    # Academic scheme uses black text on all nodes
    is_academic = color_scheme == "academic"

    for i, name in enumerate(feature_names):
        # Determine node color based on role
        if treatment_idx is not None and i == treatment_idx:
            # Explicit treatment designation
            fillcolor = colors["treatment"]
            fontcolor = "black" if is_academic else "white"
            penwidth = "2.5" if not is_academic else "1.5"
            node_label = name  # No "(Treatment)" label
        elif outcome_idx is not None and i == outcome_idx:
            # Explicit outcome designation
            fillcolor = colors["outcome"]
            fontcolor = "black" if is_academic else "white"
            penwidth = "2.5" if not is_academic else "1.5"
            node_label = name  # No "(Outcome)" label
        elif i in highlight_set:
            # Custom highlighted nodes
            fillcolor = colors["markov_blanket"]
            fontcolor = "black" if is_academic else "white"
            penwidth = "2.0" if not is_academic else "1.5"
            node_label = name
        elif i in mb_set:
            # Markov Blanket nodes (when no explicit treatment/outcome)
            fillcolor = colors["markov_blanket"]
            fontcolor = "black" if is_academic else "white"
            penwidth = "2.0" if not is_academic else "1.5"
            node_label = name
        else:
            # Regular nodes
            fillcolor = colors["other"]
            fontcolor = "black"
            penwidth = "1.0" if not is_academic else "1.5"
            node_label = name

        dot.node(
            name,
            label=node_label,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            penwidth=penwidth,
        )

    # Add directed edges
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            weight = A_direct[i, j]
            if abs(weight) > threshold:
                # Edge label
                label_parts = []

                # Add edge weight label
                if show_ate:
                    # First try causal_effects dict
                    ate = None
                    if causal_effects:
                        source_name = feature_names[i]
                        target_name = feature_names[j]
                        possible_keys = [
                            f"{source_name}->{target_name}",  # X0->Y
                            f"{source_name}→{target_name}",  # X0→Y (unicode arrow)
                            f"X{i}->X{j}",  # X0->X1
                            f"X{i}->{target_name}",  # X0->Y
                        ]
                        for key in possible_keys:
                            if key in causal_effects:
                                ate = causal_effects[key]
                                break

                    # Fall back to raw A_direct weight if no causal effect found
                    if ate is None:
                        ate = weight

                    # Always use 2 decimal places
                    label_parts.append(f"{ate:.2f}")

                # Add raw weight if requested
                if show_edge_weights:
                    label_parts.append(f"w={weight:.2f}")

                edge_label = "\n".join(label_parts) if label_parts else ""

                dot.edge(
                    feature_names[i],
                    feature_names[j],
                    label=edge_label,
                    color=colors["directed_edge"],
                    fontcolor=colors["edge_label"],
                    arrowhead="normal",
                )

    # Add bidirected edges (confounders)
    # Skip bidirected edge if directed edge already exists between same nodes
    # (having both is causally inconsistent - prefer directed over bidirected)
    if A_confound is not None:
        # Only process upper triangle (symmetric matrix)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                weight = A_confound[i, j]
                if abs(weight) > threshold:
                    # Check if directed edge exists in either direction
                    has_directed = (
                        abs(A_direct[i, j]) > threshold or abs(A_direct[j, i]) > threshold
                    )
                    if has_directed:
                        continue  # Skip bidirected if directed exists

                    # Bidirected edge: use both arrowheads, same color as directed (black)
                    dot.edge(
                        feature_names[i],
                        feature_names[j],
                        color=colors["directed_edge"],  # Same black as directed edges
                        style="dashed",
                        dir="both",
                        arrowhead="normal",
                        arrowtail="normal",
                        constraint="false",  # Don't affect layout ranking
                    )

    # Save if path provided
    if output_path:
        # Remove extension if provided
        output_path = str(output_path)
        for ext in [".pdf", ".png", ".svg", ".gv"]:
            if output_path.endswith(ext):
                output_path = output_path[: -len(ext)]

        dot.render(output_path, cleanup=True)
        print(f"DAG saved to: {output_path}.{output_format}")

    return dot


# =============================================================================
# Convenience Functions
# =============================================================================


def visualize_from_metrics(
    metrics: Dict,
    feature_names: List[str],
    treatment_idx: int = 0,
    output_path: Optional[str] = None,
    **kwargs,
) -> "graphviz.Digraph":
    """
    Create DAG visualization directly from a JCCE metrics dict.

    Args:
        metrics: Metrics dict from learn_structure()
        feature_names: List of variable names
        treatment_idx: Index of treatment variable
        output_path: Path to save figure
        **kwargs: Additional arguments to visualize_jcce_dag()

    Returns:
        graphviz.Digraph object
    """
    return visualize_jcce_dag(
        A_direct=metrics.get("A_direct", metrics.get("A_weights", [])),
        A_confound=metrics.get("A_confound"),
        causal_effects=metrics.get("causal_effects", {}),
        markov_blanket=metrics.get("markov_blanket", []),
        feature_names=feature_names,
        treatment_idx=treatment_idx,
        output_path=output_path,
        **kwargs,
    )


def visualize_pareto_solution(
    pareto_metrics: Dict,
    feature_names: List[str],
    treatment_idx: int = 0,
    output_path: Optional[str] = None,
    **kwargs,
) -> "graphviz.Digraph":
    """
    Create DAG visualization from a Pareto solution's metrics.

    Args:
        pareto_metrics: Single solution from results['pareto_metrics']
        feature_names: List of variable names
        treatment_idx: Index of treatment variable
        output_path: Path to save figure
        **kwargs: Additional arguments to visualize_jcce_dag()

    Returns:
        graphviz.Digraph object
    """
    # Extract adjacency matrix from structure_A_est
    A_direct = pareto_metrics.get(
        "structure_A_est", np.zeros((len(feature_names), len(feature_names)))
    )

    return visualize_jcce_dag(
        A_direct=A_direct,
        A_confound=pareto_metrics.get("A_confound"),
        causal_effects=pareto_metrics.get("causal_effects", {}),
        markov_blanket=pareto_metrics.get("mb_indices", []),
        feature_names=feature_names,
        treatment_idx=treatment_idx,
        output_path=output_path,
        **kwargs,
    )


def create_legend(
    output_path: str = "dag_legend",
    output_format: str = "pdf",
    color_scheme: str = "default",
) -> "graphviz.Digraph":
    """
    Create a standalone legend for JCCE DAG figures.

    Args:
        output_path: Path to save legend
        output_format: Output format
        color_scheme: Color scheme to use

    Returns:
        graphviz.Digraph object
    """
    if not HAS_GRAPHVIZ:
        raise ImportError("graphviz package required")

    if color_scheme == "minimal":
        colors = COLORS_MINIMAL
    elif color_scheme == "nature":
        colors = COLORS_NATURE
    else:
        colors = COLORS

    dot = graphviz.Digraph(
        name="legend",
        format=output_format,
    )

    dot.attr(rankdir="TB", label="Legend", labelloc="t", fontsize="14")
    dot.attr("node", shape="ellipse", style="filled", fontsize="10")

    # Node types
    with dot.subgraph(name="cluster_nodes") as c:
        c.attr(label="Node Types", style="rounded")
        c.node("T", "Treatment", fillcolor=colors["treatment"], fontcolor="white")
        c.node("Y", "Outcome", fillcolor=colors["outcome"], fontcolor="white")
        c.node("MB", "Markov Blanket", fillcolor=colors["markov_blanket"], fontcolor="white")
        c.node("O", "Other", fillcolor=colors["other"], fontcolor="black")

    # Edge types
    with dot.subgraph(name="cluster_edges") as c:
        c.attr(label="Edge Types", style="rounded")
        c.node("A", "", shape="point", width="0")
        c.node("B", "", shape="point", width="0")
        c.node("C", "", shape="point", width="0")
        c.node("D", "", shape="point", width="0")
        c.edge("A", "B", label="Directed\n(causal)", color=colors["directed_edge"])
        c.edge(
            "C",
            "D",
            label="Bidirected\n(confounded)",
            color=colors["directed_edge"],
            style="dashed",
            dir="both",
        )  # Same color as directed

    dot.render(output_path, cleanup=True)
    print(f"Legend saved to: {output_path}.{output_format}")

    return dot


# =============================================================================
# Test Functions
# =============================================================================


def create_test_data_simple() -> Tuple[np.ndarray, np.ndarray, Dict, List[int], List[str]]:
    """
    Create simple test data for DAG visualization.

    Ground truth structure:
        X0 (Treatment) -> Y
        X1 -> X0
        X1 -> Y
        X2 -> Y
        X0 <-> X2 (confounded)
    """
    n_vars = 4  # X0, X1, X2, Y

    # Directed edges
    A_direct = np.zeros((n_vars, n_vars))
    A_direct[0, 3] = 0.5  # X0 -> Y (treatment effect)
    A_direct[1, 0] = 0.3  # X1 -> X0
    A_direct[1, 3] = 0.2  # X1 -> Y
    A_direct[2, 3] = 0.15  # X2 -> Y

    # Bidirected edges (confounding)
    A_confound = np.zeros((n_vars, n_vars))
    A_confound[0, 2] = 0.25  # X0 <-> X2
    A_confound[2, 0] = 0.25  # Symmetric

    # Causal effects
    causal_effects = {
        "X0->Y": 0.52,
        "X1->Y": 0.18,
        "X2->Y": 0.14,
    }

    # Markov Blanket of Y
    markov_blanket = [0, 1, 2]  # X0, X1, X2 are all in MB

    # Feature names
    feature_names = ["X0", "X1", "X2", "Y"]

    return A_direct, A_confound, causal_effects, markov_blanket, feature_names


def create_test_data_general() -> Tuple[np.ndarray, np.ndarray, Dict, List[str]]:
    """
    Create test data for GENERAL causal discovery (no treatment/outcome).

    This represents discovering causal structure among variables
    without any pre-designated treatment or outcome.

    Structure:
        Age -> Income
        Age -> Education
        Education -> Income
        Gender -> Income
        Age <-> Gender (confounded by genetics/culture)
    """
    n_vars = 4  # Age, Education, Gender, Income
    feature_names = ["Age", "Education", "Gender", "Income"]

    # Directed edges (causal relationships)
    A_direct = np.zeros((n_vars, n_vars))
    A_direct[0, 1] = 0.4  # Age -> Education
    A_direct[0, 3] = 0.3  # Age -> Income
    A_direct[1, 3] = 0.5  # Education -> Income (strongest)
    A_direct[2, 3] = 0.2  # Gender -> Income

    # Bidirected edges (latent confounders)
    A_confound = np.zeros((n_vars, n_vars))
    A_confound[0, 2] = 0.15  # Age <-> Gender
    A_confound[2, 0] = 0.15

    # Causal effects (for ALL edges, not just to outcome)
    causal_effects = {
        "Age->Education": 0.42,
        "Age->Income": 0.28,
        "Education->Income": 0.55,
        "Gender->Income": 0.18,
    }

    return A_direct, A_confound, causal_effects, feature_names


def create_test_data_ihdp() -> Tuple[np.ndarray, np.ndarray, Dict, List[int], List[str]]:
    """
    Create IHDP-like test data for DAG visualization.

    Simulates a subset of IHDP structure:
        X0 (Treatment) -> Y
        X5 -> Y (strong covariate)
        X0 <-> X5 (confounded - both affected by unobserved socioeconomic status)
        Multiple other covariates in MB
    """
    n_vars = 8  # X0-X6 + Y

    # Directed edges
    A_direct = np.zeros((n_vars, n_vars))
    A_direct[0, 7] = 0.30  # X0 -> Y (treatment)
    A_direct[5, 7] = 0.45  # X5 -> Y (strongest covariate)
    A_direct[1, 7] = 0.12  # X1 -> Y
    A_direct[2, 7] = 0.08  # X2 -> Y
    A_direct[3, 0] = 0.20  # X3 -> X0 (affects treatment assignment)
    A_direct[4, 5] = 0.15  # X4 -> X5
    A_direct[6, 7] = 0.10  # X6 -> Y

    # Bidirected edges (confounding)
    A_confound = np.zeros((n_vars, n_vars))
    A_confound[0, 5] = 0.35  # X0 <-> X5 (treatment confounded with strongest predictor)
    A_confound[5, 0] = 0.35
    A_confound[0, 7] = 0.20  # X0 <-> Y (residual confounding)
    A_confound[7, 0] = 0.20

    # Causal effects
    causal_effects = {
        "X0->Y": 0.30,
        "X1->Y": 0.12,
        "X2->Y": 0.08,
        "X5->Y": 0.45,
        "X6->Y": 0.10,
    }

    # Markov Blanket
    markov_blanket = [0, 1, 2, 5, 6]

    # Feature names
    feature_names = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "Y"]

    return A_direct, A_confound, causal_effects, markov_blanket, feature_names


def run_visualization_tests(output_dir: str = "dag_test_outputs"):
    """
    Run visualization test using general causal discovery pattern.

    Args:
        output_dir: Directory to save test outputs
    """
    if not HAS_GRAPHVIZ:
        print("ERROR: graphviz package not installed. Run: pip install graphviz")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JCCE DAG Visualization Test")
    print("=" * 60)

    # General Causal Discovery pattern
    print("\n[Test] General Causal Discovery")
    A_direct, A_confound, causal_effects, names = create_test_data_general()

    dot = visualize_jcce_dag(
        A_direct=A_direct,
        A_confound=A_confound,
        causal_effects=causal_effects,
        feature_names=names,
        output_path=str(output_path / "test_dag"),
        output_format="png",
        show_ate=True,
    )
    print(f"  Saved: {output_path}/test_dag.png")

    # PDF render
    dot = visualize_jcce_dag(
        A_direct=A_direct,
        A_confound=A_confound,
        causal_effects=causal_effects,
        feature_names=names,
        output_path=str(output_path / "test_dag"),
        output_format="pdf",
        show_ate=True,
    )
    print(f"  Saved: {output_path}/test_dag.pdf")

    print("\n" + "=" * 60)
    print(f"Check outputs in: {output_path.absolute()}")
    print("=" * 60)


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JCCE DAG Visualization")
    parser.add_argument("--test", action="store_true", help="Run visualization tests")
    parser.add_argument(
        "--output-dir", type=str, default="dag_test_outputs", help="Output directory for test files"
    )

    args = parser.parse_args()

    if args.test:
        run_visualization_tests(args.output_dir)
    else:
        print("JCCE DAG Visualization Module")
        print("Run with --test to generate test outputs")
        print("\nUsage in code:")
        print("  from jcce.visualization.dag_viz import visualize_jcce_dag")
        print("  dot = visualize_jcce_dag(A_direct, feature_names, ...)")
