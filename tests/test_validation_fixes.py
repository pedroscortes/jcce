"""
Tests for P0 validation fixes.

Covers:
1. DML circularity fix — verify X_effect is used (not X_struct)
2. Fixed-structure CV — verify A is frozen across folds
3. Optuna DML integration — verify sample splitting
4. Convenience wrapper evaluate_fixed_structure_cv
"""

from unittest.mock import MagicMock, patch

import numpy as np

from jcce.validation.unified_cv_evaluation import (
    UnifiedCVResult,
    evaluate_fixed_structure_cv,
    evaluate_pareto_solution_cv,
)

# =============================================================================
# 1. DML Circularity Fix Tests
# =============================================================================


class TestDMLCircularityFix:
    """Verify that DML uses held-out data (X_effect), not search data (X_struct)."""

    def test_sample_split_reproducible(self):
        """Verify that train_test_split with random_state=42 is deterministic."""
        from sklearn.model_selection import train_test_split

        np.random.seed(0)
        X = np.random.randn(200, 5).astype(np.float32)
        Y = (X[:, 0] > 0).astype(int)

        X_s1, X_e1, Y_s1, Y_e1 = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
        X_s2, X_e2, Y_s2, Y_e2 = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

        np.testing.assert_array_equal(X_s1, X_s2)
        np.testing.assert_array_equal(X_e1, X_e2)

    def test_sample_split_sizes(self):
        """Verify 70/30 split proportions."""
        from sklearn.model_selection import train_test_split

        np.random.seed(0)
        X = np.random.randn(1000, 5).astype(np.float32)
        Y = (X[:, 0] > 0).astype(int)

        X_struct, X_effect, Y_struct, Y_effect = train_test_split(
            X, Y, test_size=0.3, stratify=Y, random_state=42
        )

        assert X_struct.shape[0] == 700
        assert X_effect.shape[0] == 300
        assert Y_struct.shape[0] == 700
        assert Y_effect.shape[0] == 300

    def test_split_no_overlap(self):
        """Verify struct and effect sets have no overlapping samples."""
        from sklearn.model_selection import train_test_split

        np.random.seed(0)
        n = 200
        X = np.random.randn(n, 5).astype(np.float32)
        # Add unique ID column for tracking
        X = np.column_stack([np.arange(n), X])
        Y = (X[:, 1] > 0).astype(int)

        X_struct, X_effect, _, _ = train_test_split(
            X, Y, test_size=0.3, stratify=Y, random_state=42
        )

        struct_ids = set(X_struct[:, 0].astype(int))
        effect_ids = set(X_effect[:, 0].astype(int))

        assert len(struct_ids & effect_ids) == 0, "Struct and effect sets overlap!"
        assert len(struct_ids | effect_ids) == n, "Not all samples accounted for"

    def test_rerun_dml_creates_split(self):
        """Verify --rerun-dml mode creates proper sample split."""
        # This is a structural test — verify the code pattern exists
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_ablation_study", "/home/user/phd/repos/jcce/scripts/run_ablation_study.py"
        )
        # Don't actually import (has heavy deps), just read source
        with open("/home/user/phd/repos/jcce/scripts/run_ablation_study.py") as f:
            source = f.read()

        # Verify DML calls use X_effect, not X_struct
        # Check main pipeline DML (P3)
        assert "run_multi_parent_dml(\n                    X=X_effect, Y=Y_effect," in source, (
            "Main pipeline DML should use X_effect"
        )

        # Check all-edges DML (P3b)
        assert "run_all_edges_dml_fn(\n                    X=X_effect, Y=Y_effect," in source, (
            "All-edges DML should use X_effect"
        )

        # Check CV (P2)
        assert "evaluate_pareto_solution_cv(\n                X=X_effect, Y=Y_effect," in source, (
            "CV should use X_effect"
        )

        # Verify --rerun-dml creates split
        assert "X_full, Y_full, ds_config = load_dataset(dataset_name)" in source, (
            "--rerun-dml should load full data first"
        )
        assert "X_struct, X_effect, Y_struct, Y_effect = train_test_split(" in source, (
            "--rerun-dml should create 70/30 split"
        )


# =============================================================================
# 2. Fixed-Structure CV Tests
# =============================================================================


class TestFixedStructureCV:
    """Verify fixed-structure CV freezes A and only retrains processor."""

    def test_freeze_structure_param_exists(self):
        """Verify evaluate_pareto_solution_cv accepts freeze_structure param."""
        import inspect

        sig = inspect.signature(evaluate_pareto_solution_cv)
        assert "freeze_structure" in sig.parameters
        # Default should be False (backward compatible)
        assert sig.parameters["freeze_structure"].default is False

    def test_evaluate_fixed_structure_cv_exists(self):
        """Verify convenience wrapper function exists and is callable."""
        assert callable(evaluate_fixed_structure_cv)

    def test_evaluate_fixed_structure_cv_signature(self):
        """Verify convenience wrapper has expected params."""
        import inspect

        sig = inspect.signature(evaluate_fixed_structure_cv)
        params = set(sig.parameters.keys())
        assert "A_frozen" in params
        assert "X" in params
        assert "Y" in params
        assert "processor_type" in params
        assert "golem_max_iter" in params

    def test_fixed_structure_cv_passes_freeze_flag(self):
        """Verify that evaluate_fixed_structure_cv passes freeze_structure=True."""
        with patch("jcce.validation.unified_cv_evaluation.evaluate_pareto_solution_cv") as mock_cv:
            mock_cv.return_value = MagicMock(spec=UnifiedCVResult)

            evaluate_fixed_structure_cv(
                X=np.random.randn(100, 5).astype(np.float32),
                Y=np.random.randint(0, 2, 100),
                hyperparams={"lambda_1": 0.02, "lambda_2": 0.01, "lambda_class": 1.0, "lr": 0.001},
                processor_type="elm",
                A_frozen=np.zeros((6, 6), dtype=np.float32),
            )

            mock_cv.assert_called_once()
            call_kwargs = mock_cv.call_args[1]
            assert call_kwargs["freeze_structure"] is True
            assert call_kwargs["cold_start"] is False
            assert call_kwargs["use_v7"] is True


# =============================================================================
# 3. Optuna DML Integration Tests
# =============================================================================


class TestOptunaDMLIntegration:
    """Verify Optuna pipeline has DML phase with proper sample splitting."""

    def test_run_posthoc_dml_exists(self):
        """Verify run_posthoc_dml function exists."""
        from jcce.structure_learning.experiment_runner import run_posthoc_dml

        assert callable(run_posthoc_dml)

    def test_run_posthoc_dml_empty_solutions(self):
        """Verify graceful handling of empty solutions list."""
        from jcce.structure_learning.experiment_runner import run_posthoc_dml

        results = run_posthoc_dml(
            enhanced_solutions=[],
            X_effect=np.random.randn(100, 5),
            Y_effect=np.random.randint(0, 2, 100),
        )
        assert results == []

    def test_pipeline_dml_params_exist(self):
        """Verify run_full_optuna_pipeline accepts DML params."""
        import inspect

        from jcce.structure_learning.experiment_runner import run_full_optuna_pipeline

        sig = inspect.signature(run_full_optuna_pipeline)
        params = set(sig.parameters.keys())
        assert "run_dml" in params
        assert "dml_max_solutions" in params
        assert "dml_run_refutation" in params
        assert "feature_names" in params
        assert "known_treatment_idx" in params

    def test_pipeline_dml_default_on(self):
        """Verify DML is on by default (mandatory per consensus)."""
        import inspect

        from jcce.structure_learning.experiment_runner import run_full_optuna_pipeline

        sig = inspect.signature(run_full_optuna_pipeline)
        assert sig.parameters["run_dml"].default is True

    def test_run_posthoc_dml_calls_multi_parent_dml(self):
        """Verify run_posthoc_dml dispatches to multi_parent_dml."""
        from jcce.structure_learning.experiment_runner import run_posthoc_dml

        # Create fake enhanced solution with A matrix
        n_vars = 5
        A_fake = np.zeros((n_vars + 1, n_vars + 1))
        A_fake[0, n_vars] = 0.5  # Edge X0 -> Y

        enhanced_solutions = [
            {
                "metrics": {
                    "structure_A_est": A_fake,
                    "processor_type": "elm",
                }
            }
        ]

        with patch("jcce.validation.multi_parent_dml.run_multi_parent_dml") as mock_dml:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {}
            mock_result.n_parents_discovered = 1
            mock_result.n_parents_significant = 1
            mock_dml.return_value = mock_result

            X_eff = np.random.randn(60, n_vars)
            Y_eff = np.random.randint(0, 2, 60)

            results = run_posthoc_dml(
                enhanced_solutions=enhanced_solutions,
                X_effect=X_eff,
                Y_effect=Y_eff,
            )

            mock_dml.assert_called_once()
            call_kwargs = mock_dml.call_args[1]
            # Verify it uses X_effect, not some other data
            np.testing.assert_array_equal(call_kwargs["X"], X_eff)
            np.testing.assert_array_equal(call_kwargs["Y"], Y_eff)
            assert len(results) == 1
            assert results[0]["n_significant"] == 1


# =============================================================================
# 4. Integration: freeze_A already supported in GOLEM
# =============================================================================


class TestFreezeASupport:
    """Verify that learn_structure has freeze_A param."""

    def test_freeze_a_param_exists(self):
        """Verify freeze_A is a parameter of learn_structure."""
        import inspect

        from jcce.structure_learning.jcce_learner import learn_structure

        sig = inspect.signature(learn_structure)
        assert "freeze_A" in sig.parameters
        assert sig.parameters["freeze_A"].default is False


# =============================================================================
# 5. CV with freeze_structure flag in _evaluate_single_fold
# =============================================================================


class TestEvaluateSingleFoldFreezeStructure:
    """Verify _evaluate_single_fold accepts and uses freeze_structure param."""

    def test_freeze_structure_param_in_fold_function(self):
        """Verify _evaluate_single_fold has freeze_structure param."""
        import inspect

        from jcce.validation.unified_cv_evaluation import _evaluate_single_fold

        sig = inspect.signature(_evaluate_single_fold)
        assert "freeze_structure" in sig.parameters
        assert sig.parameters["freeze_structure"].default is False

    def test_run_posthoc_cv_defaults_to_freeze(self):
        """Pipeline's run_posthoc_cv defaults to freeze_structure=True."""
        import inspect

        from jcce.structure_learning.experiment_runner import run_posthoc_cv

        sig = inspect.signature(run_posthoc_cv)
        assert "freeze_structure" in sig.parameters
        assert sig.parameters["freeze_structure"].default is True

    def test_pipeline_has_structural_validation_param(self):
        """Pipeline has run_structural_validation param (default True)."""
        import inspect

        from jcce.structure_learning.experiment_runner import run_full_optuna_pipeline

        sig = inspect.signature(run_full_optuna_pipeline)
        assert "run_structural_validation" in sig.parameters
        assert sig.parameters["run_structural_validation"].default is True
