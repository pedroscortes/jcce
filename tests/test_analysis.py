"""
Unit tests for Phase B: Analysis Infrastructure.

Tests pareto_stability, hypervolume, fanova_importance,
pareto_dispersion, and visualization modules.
"""

import numpy as np

# ============================================================================
# Helpers
# ============================================================================


def _make_solutions(n_solutions=5, n_vars=5, seed=42):
    """Create synthetic enhanced_solutions for testing."""
    rng = np.random.RandomState(seed)
    n = n_vars + 1  # includes Y

    solutions = []
    for i in range(n_solutions):
        A = rng.randn(n, n) * 0.3
        # Make some edges clearly present
        A[0, 1] = 0.5 + rng.rand() * 0.3
        A[2, 3] = 0.4 + rng.rand() * 0.2
        # Make diagonal zero
        np.fill_diagonal(A, 0)

        proc_type = ["elm", "gnn", "mlp", "transformer", "mamba"][i % 5]

        solutions.append(
            {
                "metrics": {
                    "structure_A_est": A,
                    "classification_balanced_accuracy": 0.5 + rng.rand() * 0.4,
                    "mb_sparsity": 0.3 + rng.rand() * 0.5,
                    "processor_type": proc_type,
                    "h_A": rng.rand() * 0.05,
                    "wall_time": float(i * 10 + rng.rand() * 5),
                }
            }
        )

    return solutions


def _make_true_graph(n_vars=5):
    """Create a ground truth adjacency matrix."""
    n = n_vars + 1
    true_graph = np.zeros((n, n))
    true_graph[0, 1] = 1.0  # Edge 0->1
    true_graph[2, 3] = 1.0  # Edge 2->3
    true_graph[1, 4] = 1.0  # Edge 1->4
    return true_graph


# ============================================================================
# Tests: Pareto Stability (B.1)
# ============================================================================


class TestParetoStability:
    def test_empty_solutions(self):
        from jcce.analysis.pareto_stability import compute_edge_stability

        stab = compute_edge_stability([], n_vars=5)
        assert stab.shape == (6, 6)
        assert np.all(stab == 0.0)

    def test_consistent_edges(self):
        """Same edge in all solutions -> stability = 1.0."""
        from jcce.analysis.pareto_stability import compute_edge_stability

        n = 6
        solutions = []
        for _ in range(5):
            A = np.zeros((n, n))
            A[0, 1] = 0.5
            solutions.append({"metrics": {"structure_A_est": A}})

        stab = compute_edge_stability(solutions, n_vars=5, threshold=0.1)
        assert stab[0, 1] == 1.0
        assert stab[1, 0] == 0.0

    def test_partial_stability(self):
        """Edge in 3/5 solutions -> stability = 0.6."""
        from jcce.analysis.pareto_stability import compute_edge_stability

        n = 6
        solutions = []
        for i in range(5):
            A = np.zeros((n, n))
            if i < 3:
                A[2, 3] = 0.5
            solutions.append({"metrics": {"structure_A_est": A}})

        stab = compute_edge_stability(solutions, n_vars=5, threshold=0.1)
        assert abs(stab[2, 3] - 0.6) < 1e-10

    def test_multi_threshold(self):
        """Different thresholds produce different stability maps."""
        from jcce.analysis.pareto_stability import compute_edge_stability_multi_threshold

        n = 6
        solutions = []
        for _ in range(5):
            A = np.zeros((n, n))
            A[0, 1] = 0.2  # Passes threshold 0.1 but not 0.3
            A[2, 3] = 0.5  # Passes all thresholds
            solutions.append({"metrics": {"structure_A_est": A}})

        result = compute_edge_stability_multi_threshold(
            solutions, n_vars=5, thresholds=(0.1, 0.3, 0.5)
        )

        assert result[0.1][0, 1] == 1.0  # 0.2 > 0.1
        assert result[0.3][0, 1] == 0.0  # 0.2 < 0.3
        assert result[0.3][2, 3] == 1.0  # 0.5 > 0.3

    def test_get_stable_edges(self):
        from jcce.analysis.pareto_stability import get_stable_edges

        stab = np.zeros((4, 4))
        stab[0, 1] = 0.9
        stab[2, 3] = 0.5
        stab[1, 2] = 0.85

        edges = get_stable_edges(stab, min_frequency=0.8)
        assert len(edges) == 2  # 0->1 (0.9) and 1->2 (0.85)
        assert edges[0] == (0, 1, 0.9)  # Sorted by frequency desc
        assert edges[1] == (1, 2, 0.85)

    def test_compare_to_ground_truth(self):
        from jcce.analysis.pareto_stability import (
            compare_stability_to_ground_truth,
            compute_edge_stability,
        )

        n = 6
        # All solutions have edge 0->1 and 2->3
        solutions = []
        for _ in range(5):
            A = np.zeros((n, n))
            A[0, 1] = 0.5
            A[2, 3] = 0.5
            solutions.append({"metrics": {"structure_A_est": A}})

        stab = compute_edge_stability(solutions, n_vars=5, threshold=0.1)

        true_graph = np.zeros((n, n))
        true_graph[0, 1] = 1.0
        true_graph[2, 3] = 1.0

        metrics = compare_stability_to_ground_truth(stab, true_graph, min_frequency=0.8)
        assert metrics["precision"] == 1.0  # All stable edges are true
        assert metrics["recall"] == 1.0  # All true edges are stable
        assert metrics["f1"] == 1.0

    def test_sensitivity_analysis(self):
        from jcce.analysis.pareto_stability import stability_sensitivity_analysis

        solutions = _make_solutions(n_solutions=10, n_vars=5)
        result = stability_sensitivity_analysis(
            solutions,
            n_vars=5,
            thresholds=(0.1, 0.3),
            frequencies=(0.5, 0.8),
        )

        assert result["n_stable_edges"].shape == (2, 2)  # 2 thresholds x 2 frequencies
        # Higher threshold + higher frequency -> fewer stable edges
        assert result["n_stable_edges"][1, 1] <= result["n_stable_edges"][0, 0]

    def test_sensitivity_with_ground_truth(self):
        from jcce.analysis.pareto_stability import stability_sensitivity_analysis

        solutions = _make_solutions(n_solutions=10, n_vars=5)
        true_graph = _make_true_graph(n_vars=5)

        result = stability_sensitivity_analysis(
            solutions,
            n_vars=5,
            thresholds=(0.1, 0.3),
            frequencies=(0.5, 0.8),
            true_graph=true_graph,
        )

        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert result["precision"].shape == (2, 2)


# ============================================================================
# Tests: Hypervolume (B.2)
# ============================================================================


class TestHypervolume:
    def test_empty_returns_zero(self):
        from jcce.analysis.hypervolume import compute_hypervolume_2d

        hv = compute_hypervolume_2d(np.zeros((0, 2)), np.array([0.0, 0.0]))
        assert hv == 0.0

    def test_single_point(self):
        from jcce.analysis.hypervolume import compute_hypervolume_2d

        points = np.array([[0.8, 0.6]])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume_2d(points, ref)
        assert abs(hv - 0.8 * 0.6) < 1e-10

    def test_two_non_dominated_points(self):
        from jcce.analysis.hypervolume import compute_hypervolume_2d

        points = np.array([[0.9, 0.3], [0.5, 0.8]])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume_2d(points, ref)
        expected = 0.27 + 0.25  # 0.9*0.3 + 0.5*(0.8-0.3)
        assert abs(hv - expected) < 1e-10

    def test_extract_pareto_front(self):
        from jcce.analysis.hypervolume import extract_pareto_front_2d

        points = np.array(
            [
                [0.9, 0.3],
                [0.5, 0.8],
                [0.4, 0.2],  # Dominated by (0.9, 0.3)
            ]
        )
        pareto = extract_pareto_front_2d(points)
        assert len(pareto) == 2

    def test_anytime_from_solutions(self):
        from jcce.analysis.hypervolume import compute_anytime_hypervolume_from_solutions

        solutions = _make_solutions(n_solutions=5, n_vars=5)
        curve = compute_anytime_hypervolume_from_solutions(solutions)

        assert len(curve["timestamps"]) == 5
        assert len(curve["hypervolumes"]) == 5
        assert len(curve["n_pareto"]) == 5
        # Hypervolume should be non-decreasing
        for i in range(1, len(curve["hypervolumes"])):
            assert curve["hypervolumes"][i] >= curve["hypervolumes"][i - 1]

    def test_compare_anytime_curves(self):
        from jcce.analysis.hypervolume import (
            compare_anytime_curves,
            compute_anytime_hypervolume_from_solutions,
        )

        sols1 = _make_solutions(n_solutions=5, n_vars=5, seed=42)
        sols2 = _make_solutions(n_solutions=5, n_vars=5, seed=99)

        curve1 = compute_anytime_hypervolume_from_solutions(sols1)
        curve2 = compute_anytime_hypervolume_from_solutions(sols2)

        comparison = compare_anytime_curves(
            {"method1": curve1, "method2": curve2},
            target_hv=0.1,
        )

        assert "final_hv" in comparison
        assert "method1" in comparison["final_hv"]
        assert "time_to_target" in comparison

    def test_empty_solutions(self):
        from jcce.analysis.hypervolume import compute_anytime_hypervolume_from_solutions

        curve = compute_anytime_hypervolume_from_solutions([])
        assert len(curve["timestamps"]) == 0


# ============================================================================
# Tests: fANOVA Importance (B.3)
# ============================================================================


class TestFanovaImportance:
    def _make_study(self):
        """Create a minimal study with completed trials."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        rng = np.random.RandomState(42)
        for _ in range(30):
            trial = study.ask()
            proc = trial.suggest_categorical("processor_type", ["elm", "mlp", "gnn"])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            l1 = trial.suggest_float("lambda_1", 0.005, 0.5, log=True)

            bacc = 0.5 + 0.3 * lr * 100 + rng.rand() * 0.1
            sparsity = 0.3 + 0.5 * l1 + rng.rand() * 0.1

            study.tell(trial, [bacc, sparsity])

        return study

    def test_compute_importances(self):
        from jcce.analysis.fanova_importance import compute_param_importances

        study = self._make_study()
        imp = compute_param_importances(study, target_index=0)

        assert isinstance(imp, dict)
        assert len(imp) > 0
        # Importances should sum to ~1
        total = sum(imp.values())
        assert abs(total - 1.0) < 0.01

    def test_both_objectives(self):
        from jcce.analysis.fanova_importance import compute_importances_both_objectives

        study = self._make_study()
        result = compute_importances_both_objectives(study)

        assert "balanced_accuracy" in result
        assert "sparsity" in result
        assert len(result["balanced_accuracy"]) > 0

    def test_get_frozen_params(self):
        from jcce.analysis.fanova_importance import compute_param_importances, get_frozen_params

        study = self._make_study()
        imp = compute_param_importances(study, target_index=0)
        frozen = get_frozen_params(imp, threshold=0.5, study=study)

        # At least some params should be below 50% importance
        assert isinstance(frozen, dict)

    def test_importance_convergence(self):
        from jcce.analysis.fanova_importance import importance_convergence

        # Simulate 3 snapshots with stable importances
        snapshots = [
            {"lr": 0.4, "lambda_1": 0.3, "processor_type": 0.3},
            {"lr": 0.42, "lambda_1": 0.28, "processor_type": 0.3},
            {"lr": 0.41, "lambda_1": 0.29, "processor_type": 0.3},
        ]

        cv = importance_convergence(snapshots, top_k=3)
        assert "lr" in cv
        # CV should be small for stable estimates
        assert cv["lr"] < 0.1


# ============================================================================
# Tests: Pareto Dispersion (B.4)
# ============================================================================


class TestParetoDispersion:
    def test_extract_dags(self):
        from jcce.analysis.pareto_dispersion import extract_pareto_dags

        solutions = _make_solutions(n_solutions=3, n_vars=5)
        dags = extract_pareto_dags(solutions)

        assert len(dags) == 3
        assert dags[0].shape == (6, 6)

    def test_empty_solutions(self):
        from jcce.analysis.pareto_dispersion import extract_pareto_dags

        dags = extract_pareto_dags([])
        assert len(dags) == 0

    def test_hellinger_report_single_dag(self):
        """Single DAG -> diameter=0, coverage=1."""
        from jcce.analysis.pareto_dispersion import pareto_hellinger_report

        n = 6
        A = np.eye(n) * 0  # Zero graph
        solutions = [{"metrics": {"structure_A_est": A}}]

        report = pareto_hellinger_report(solutions)
        assert report["diameter"] == 0.0
        assert report["n_dags"] == 1

    def test_hellinger_report_multiple_dags(self):
        """Multiple DAGs should produce positive diameter."""
        from jcce.analysis.pareto_dispersion import pareto_hellinger_report

        rng = np.random.RandomState(42)
        n = 4  # Small for speed
        solutions = []
        for _ in range(3):
            A = np.abs(rng.randn(n, n)) * 0.3
            np.fill_diagonal(A, 0)
            solutions.append({"metrics": {"structure_A_est": A}})

        report = pareto_hellinger_report(solutions, max_order=1)
        assert report["n_dags"] == 3
        assert report["diameter"] >= 0.0
        assert report["hellinger_matrix"].shape == (3, 3)


# ============================================================================
# Tests: Visualization (B.5)
# ============================================================================


class TestVisualization:
    """Smoke tests: verify figures are created without errors."""

    def test_plot_pareto_front(self):
        from jcce.analysis.visualization import plot_pareto_front

        solutions = _make_solutions(n_solutions=10, n_vars=5)
        fig = plot_pareto_front(solutions)

        assert fig is not None
        assert len(fig.axes) > 0
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt.close(fig)

    def test_plot_edge_stability_heatmap(self):
        from jcce.analysis.visualization import plot_edge_stability_heatmap

        stability = np.random.rand(6, 6)
        np.fill_diagonal(stability, 0)

        fig = plot_edge_stability_heatmap(stability)
        assert fig is not None
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt.close(fig)

    def test_plot_edge_stability_with_names(self):
        from jcce.analysis.visualization import plot_edge_stability_heatmap

        stability = np.random.rand(6, 6)
        names = ["X1", "X2", "X3", "X4", "X5", "Y"]

        fig = plot_edge_stability_heatmap(stability, feature_names=names)
        assert fig is not None
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt.close(fig)

    def test_plot_anytime_hypervolume(self):
        from jcce.analysis.visualization import plot_anytime_hypervolume

        curves = {
            "Method A": {
                "timestamps": np.array([0, 10, 20, 30]),
                "hypervolumes": np.array([0.0, 0.1, 0.2, 0.25]),
            },
            "Method B": {
                "timestamps": np.array([0, 15, 25, 35]),
                "hypervolumes": np.array([0.0, 0.05, 0.15, 0.3]),
            },
        }

        fig = plot_anytime_hypervolume(curves)
        assert fig is not None
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt.close(fig)

    def test_plot_importance_bar(self):
        from jcce.analysis.visualization import plot_importance_bar

        importances = {
            "lr": 0.35,
            "lambda_1": 0.25,
            "processor_type": 0.20,
            "lambda_2": 0.10,
            "hidden_dim": 0.03,
            "other": 0.07,
        }

        fig = plot_importance_bar(importances)
        assert fig is not None
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt.close(fig)

    def test_plot_stability_sensitivity(self):
        from jcce.analysis.visualization import plot_stability_sensitivity

        sensitivity = {
            "thresholds": [0.1, 0.3],
            "frequencies": [0.5, 0.8],
            "n_stable_edges": np.array([[10, 5], [6, 2]]),
        }

        fig = plot_stability_sensitivity(sensitivity)
        assert fig is not None
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt.close(fig)
