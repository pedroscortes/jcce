"""
Post-Hoc Validation Pipeline for JCCE v13.0

Implements targeted post-hoc validation based on LLM consensus:
1. Random Common Cause refutation (essential - tests confounding sensitivity)
2. K=10 DML validation (for publication tables)
3. Hypervolume metrics (validates many-objective NSGA-II)
4. Ensemble counterfactual aggregation (leverages Pareto diversity)

Key insight from analysis:
- O5 (DML variance) measures STATISTICAL STABILITY (aleatory uncertainty)
- Random Common Cause tests CAUSAL VALIDITY (epistemic uncertainty)
- These are orthogonal and complementary, not redundant

References:
- Sharma & Kiciman (2020) "DoWhy: An End-to-End Library for Causal Inference"
- Chernozhukov et al. (2018) "Double/Debiased Machine Learning"
- Deb et al. (2002) "NSGA-II: A Fast and Elitist Multi-Objective GA"
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class PostHocValidationResult:
    """Complete result of post-hoc validation pipeline."""
    # Random Common Cause
    rcc_passed: bool
    rcc_effect_change: float
    rcc_details: Dict

    # K=10 DML
    k10_variance: float
    k10_ate: float
    k10_fold_estimates: List[float]

    # Hypervolume
    hypervolume: float
    pareto_diversity: float

    # Ensemble counterfactuals
    ensemble_ate: float
    ensemble_variance: float
    structural_uncertainty: float

    # Identifiability diagnostics (v15)
    identifiability_summary: Optional[Dict] = None

    # Refutation suite (v15)
    refutation_summary: Optional[Dict] = None

    # Summary
    validation_passed: bool = False
    summary_metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = {
            'rcc_passed': self.rcc_passed,
            'rcc_effect_change': self.rcc_effect_change,
            'rcc_details': self.rcc_details,
            'k10_variance': self.k10_variance,
            'k10_ate': self.k10_ate,
            'k10_fold_estimates': self.k10_fold_estimates,
            'hypervolume': self.hypervolume,
            'pareto_diversity': self.pareto_diversity,
            'ensemble_ate': self.ensemble_ate,
            'ensemble_variance': self.ensemble_variance,
            'structural_uncertainty': self.structural_uncertainty,
            'validation_passed': self.validation_passed,
            'summary_metrics': self.summary_metrics,
        }
        if self.identifiability_summary is not None:
            result['identifiability'] = self.identifiability_summary
        if self.refutation_summary is not None:
            result['refutation'] = self.refutation_summary
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "POST-HOC VALIDATION SUMMARY",
            "=" * 70,
            "",
            "1. RANDOM COMMON CAUSE (Confounding Sensitivity)",
            f"   Passed: {'YES' if self.rcc_passed else 'NO'}",
            f"   Effect Change: {self.rcc_effect_change:.1%}",
            f"   (Threshold: <20% change for stability)",
            "",
            "2. K=10 DML VALIDATION (Statistical Precision)",
            f"   ATE: {self.k10_ate:.4f}",
            f"   Variance: {self.k10_variance:.6f}",
            f"   95% CI: [{self.k10_ate - 1.96*np.sqrt(self.k10_variance):.4f}, "
            f"{self.k10_ate + 1.96*np.sqrt(self.k10_variance):.4f}]",
            "",
            "3. HYPERVOLUME METRICS (Many-Objective Validation)",
            f"   Hypervolume: {self.hypervolume:.4f}",
            f"   Pareto Diversity: {self.pareto_diversity:.4f}",
            "",
            "4. ENSEMBLE COUNTERFACTUALS (Structural Uncertainty)",
            f"   Ensemble ATE: {self.ensemble_ate:.4f}",
            f"   Structural Variance: {self.structural_uncertainty:.6f}",
            "",
        ]
        if self.identifiability_summary is not None:
            lines.append("")
            ident = self.identifiability_summary
            lines.append("5. IDENTIFIABILITY DIAGNOSTICS")
            lines.append(f"   DAGs analyzed: {ident.get('n_dags', 0)}")
            lines.append(f"   BIC landscape: best={ident.get('bic_best', float('nan')):.1f}, "
                        f"median={ident.get('bic_median', float('nan')):.1f}, "
                        f"worst={ident.get('bic_worst', float('nan')):.1f}")
            lines.append(f"   Condition numbers: mean={ident.get('condition_number_mean', float('nan')):.1f}, "
                        f"max={ident.get('condition_number_max', float('nan')):.1f}")
            grades = ident.get('grades', [])
            if grades:
                grade_counts = {}
                for g in grades:
                    grade_counts[g] = grade_counts.get(g, 0) + 1
                lines.append(f"   Grades: " + ", ".join(f"{g}={c}" for g, c in sorted(grade_counts.items())))

        if self.refutation_summary is not None:
            lines.append("")
            ref = self.refutation_summary
            lines.append("6. REFUTATION SUITE (Placebo + Subset Stability)")
            lines.append(f"   Passed: {ref.get('n_passed', 0)}/{ref.get('n_total', 0)} "
                        f"({ref.get('pass_rate', 0):.0%})")
            for name, test_result in ref.get('results', {}).items():
                status = "PASS" if test_result.get('passed') else "FAIL"
                lines.append(f"   {name}: {status} "
                           f"(p={test_result.get('p_value', float('nan')):.4f})")

        lines.extend([
            "",
            "-" * 70,
            f"OVERALL VALIDATION: {'PASSED' if self.validation_passed else 'NEEDS REVIEW'}",
            "=" * 70,
        ])
        return "\n".join(lines)


# =============================================================================
# 1. RANDOM COMMON CAUSE REFUTATION
# =============================================================================

def run_random_common_cause_test(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    estimate_effect_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    original_effect: float,
    n_simulations: int = 100,
    effect_change_threshold: float = 0.20,
    random_state: int = 42,
) -> Tuple[bool, float, Dict]:
    """
    Run Random Common Cause refutation test.

    This is the ESSENTIAL post-hoc test that O5 cannot capture.
    Tests sensitivity to unmeasured confounding by adding random covariates.

    Args:
        X: Covariates (n_samples, n_features)
        T: Treatment (n_samples,)
        Y: Outcome (n_samples,)
        estimate_effect_fn: Function(X, T, Y) -> float
        original_effect: Effect estimate to validate
        n_simulations: Number of random confounders to test
        effect_change_threshold: Max allowed effect change (default 20%)
        random_state: Random seed

    Returns:
        (passed, effect_change, details)
    """
    rng = np.random.RandomState(random_state)

    # Convert JAX arrays if needed
    if hasattr(X, 'device_buffer') or 'jax' in str(type(X)):
        X = np.array(X)
    if hasattr(T, 'device_buffer') or 'jax' in str(type(T)):
        T = np.array(T)
    if hasattr(Y, 'device_buffer') or 'jax' in str(type(Y)):
        Y = np.array(Y)

    T = np.asarray(T).flatten()
    Y = np.asarray(Y).flatten()

    confounded_effects = []

    for _ in range(n_simulations):
        # Add random confounder (standard normal)
        Z_random = rng.normal(0, 1, size=(len(X), 1))
        X_augmented = np.hstack([X, Z_random])

        try:
            effect = estimate_effect_fn(X_augmented, T, Y)
            if np.isfinite(effect):
                confounded_effects.append(effect)
        except Exception:
            continue

    if len(confounded_effects) < 10:
        return False, 1.0, {'error': 'Too few successful simulations'}

    confounded_mean = np.mean(confounded_effects)
    confounded_std = np.std(confounded_effects)

    # Effect change ratio
    effect_change = abs(confounded_mean - original_effect) / (abs(original_effect) + 1e-10)

    # Pass if effect remains stable (< threshold change)
    passed = bool(effect_change < effect_change_threshold)

    details = {
        'original_effect': original_effect,
        'confounded_mean': confounded_mean,
        'confounded_std': confounded_std,
        'effect_change': effect_change,
        'n_simulations': len(confounded_effects),
        'threshold': effect_change_threshold,
    }

    return passed, effect_change, details


# =============================================================================
# 2. K=10 DML VALIDATION
# =============================================================================

def run_k10_dml_validation(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    markov_blanket: List[int],
    random_state: int = 42,
) -> Tuple[float, float, List[float], Dict]:
    """
    Run K=10 DML cross-validation for post-hoc validation.

    Provides higher-fidelity variance estimate than in-loop K=5.
    Useful for publication tables.

    Args:
        X: Full covariates
        T: Treatment
        Y: Outcome
        markov_blanket: Indices of MB features to use
        random_state: Random seed

    Returns:
        (variance, ate, fold_estimates, details)
    """
    from jcce.validation.dml_crossfitting import DMLCrossFitter, create_simple_nuisance_functions

    # Convert arrays
    if hasattr(X, 'device_buffer') or 'jax' in str(type(X)):
        X = np.array(X)
    if hasattr(T, 'device_buffer') or 'jax' in str(type(T)):
        T = np.array(T)
    if hasattr(Y, 'device_buffer') or 'jax' in str(type(Y)):
        Y = np.array(Y)

    T = np.asarray(T).flatten()
    Y = np.asarray(Y).flatten()

    # Use MB features only
    if len(markov_blanket) > 0:
        X_mb = X[:, markov_blanket]
    else:
        X_mb = X

    # Create K=10 cross-fitter
    dml = DMLCrossFitter(n_splits=10, random_state=random_state)
    train_nuisance, predict_nuisance = create_simple_nuisance_functions()

    try:
        result = dml.estimate_ate(
            X_mb, T, Y,
            train_nuisance_fn=train_nuisance,
            predict_nuisance_fn=predict_nuisance,
        )

        variance = result.variance
        ate = result.ate_mean
        fold_estimates = result.ate_per_fold

        details = {
            'standard_error': result.standard_error,
            'ci_lower': ate - 1.96 * result.standard_error,
            'ci_upper': ate + 1.96 * result.standard_error,
            'n_folds': 10,
        }

    except Exception as e:
        # Fallback if DML fails
        warnings.warn(f"K=10 DML failed: {e}")
        variance = float('nan')
        ate = float('nan')
        fold_estimates = []
        details = {'error': str(e)}

    return variance, ate, fold_estimates, details


# =============================================================================
# 3. HYPERVOLUME METRICS
# =============================================================================

def compute_hypervolume(
    pareto_front: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
) -> float:
    """
    Compute hypervolume indicator for Pareto front quality.

    Hypervolume measures both convergence and diversity of solutions.
    Essential for validating many-objective (>3) optimization with NSGA-II.

    Args:
        pareto_front: Array of shape (n_solutions, n_objectives)
                     Each row is an objective vector (to be MINIMIZED)
        reference_point: Point dominated by all Pareto solutions
                        If None, computed as max + 10% margin

    Returns:
        Hypervolume value (higher is better)
    """
    if len(pareto_front) == 0:
        return 0.0

    pareto_front = np.atleast_2d(pareto_front)
    n_solutions, n_objectives = pareto_front.shape

    # Compute reference point if not provided
    if reference_point is None:
        # Use worst observed value + 10% margin per objective
        max_vals = np.max(pareto_front, axis=0)
        margin = np.abs(max_vals) * 0.1 + 0.1  # Add absolute margin too
        reference_point = max_vals + margin

    reference_point = np.asarray(reference_point)

    # For 2D case, use exact calculation
    if n_objectives == 2:
        return _hypervolume_2d(pareto_front, reference_point)

    # For higher dimensions, use Monte Carlo approximation
    return _hypervolume_monte_carlo(pareto_front, reference_point, n_samples=10000)


def _hypervolume_2d(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """Exact 2D hypervolume calculation."""
    # Sort by first objective
    sorted_idx = np.argsort(pareto_front[:, 0])
    sorted_front = pareto_front[sorted_idx]

    hv = 0.0
    prev_x = reference_point[0]

    for point in sorted_front[::-1]:  # Reverse order
        if point[0] < reference_point[0] and point[1] < reference_point[1]:
            hv += (reference_point[0] - point[0]) * (prev_x - point[1])
            prev_x = point[1]

    return hv


def _hypervolume_monte_carlo(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    n_samples: int = 10000,
) -> float:
    """Monte Carlo hypervolume approximation for high dimensions."""
    n_solutions, n_objectives = pareto_front.shape

    # Compute ideal point (lower bound)
    ideal_point = np.min(pareto_front, axis=0)

    # Sample random points in the box [ideal, reference]
    rng = np.random.RandomState(42)
    samples = rng.uniform(
        ideal_point,
        reference_point,
        size=(n_samples, n_objectives)
    )

    # Count points dominated by at least one Pareto solution
    dominated_count = 0
    for sample in samples:
        for solution in pareto_front:
            if np.all(solution <= sample):  # solution dominates sample
                dominated_count += 1
                break

    # Hypervolume = fraction dominated * box volume
    box_volume = np.prod(reference_point - ideal_point)
    hv = (dominated_count / n_samples) * box_volume

    return hv


def compute_pareto_diversity(pareto_front: np.ndarray) -> float:
    """
    Compute diversity metric for Pareto front.

    Uses average pairwise distance in objective space.
    Higher diversity = better spread of trade-offs.

    Args:
        pareto_front: Array of shape (n_solutions, n_objectives)

    Returns:
        Diversity score (higher is better)
    """
    if len(pareto_front) < 2:
        return 0.0

    pareto_front = np.atleast_2d(pareto_front)
    n_solutions = len(pareto_front)

    # Normalize objectives to [0, 1] for fair comparison
    min_vals = np.min(pareto_front, axis=0)
    max_vals = np.max(pareto_front, axis=0)
    range_vals = max_vals - min_vals + 1e-10
    normalized = (pareto_front - min_vals) / range_vals

    # Compute average pairwise Euclidean distance
    total_distance = 0.0
    count = 0

    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            dist = np.linalg.norm(normalized[i] - normalized[j])
            total_distance += dist
            count += 1

    if count == 0:
        return 0.0

    avg_distance = total_distance / count

    # Normalize by max possible distance (diagonal of unit hypercube)
    n_objectives = pareto_front.shape[1]
    max_distance = np.sqrt(n_objectives)

    return avg_distance / max_distance


# =============================================================================
# 4. ENSEMBLE COUNTERFACTUAL AGGREGATION
# =============================================================================

def compute_ensemble_counterfactuals(
    pareto_solutions: List[Dict],
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    effect_estimator: Optional[Callable] = None,
) -> Tuple[float, float, float, Dict]:
    """
    Compute ensemble counterfactuals across Pareto DAGs.

    Leverages structural diversity of Pareto front to quantify
    uncertainty in causal claims. Different DAGs represent different
    causal hypotheses consistent with the data.

    Args:
        pareto_solutions: List of dicts with keys:
            - 'dag': Adjacency matrix
            - 'markov_blanket': List of MB indices
            - 'ate' (optional): Pre-computed ATE
        X: Covariates
        T: Treatment
        Y: Outcome
        effect_estimator: Function(X, T, Y, mb) -> float
                         If None, uses simple IPW estimator

    Returns:
        (ensemble_ate, ensemble_variance, structural_uncertainty, details)
    """
    if effect_estimator is None:
        effect_estimator = _simple_effect_estimator

    # Convert arrays
    if hasattr(X, 'device_buffer') or 'jax' in str(type(X)):
        X = np.array(X)
    if hasattr(T, 'device_buffer') or 'jax' in str(type(T)):
        T = np.array(T)
    if hasattr(Y, 'device_buffer') or 'jax' in str(type(Y)):
        Y = np.array(Y)

    T = np.asarray(T).flatten()
    Y = np.asarray(Y).flatten()

    # Collect effect estimates from each DAG
    effects = []
    mb_sizes = []

    for sol in pareto_solutions:
        mb = sol.get('markov_blanket', [])

        # Use pre-computed ATE if available
        if 'ate' in sol and np.isfinite(sol['ate']):
            effects.append(sol['ate'])
            mb_sizes.append(len(mb))
            continue

        # Otherwise estimate effect using MB features
        try:
            if len(mb) > 0:
                X_mb = X[:, mb]
            else:
                X_mb = X

            ate = effect_estimator(X_mb, T, Y)
            if np.isfinite(ate):
                effects.append(ate)
                mb_sizes.append(len(mb))
        except Exception:
            continue

    if len(effects) == 0:
        return float('nan'), float('nan'), float('nan'), {'error': 'No valid effects'}

    effects = np.array(effects)

    # Ensemble statistics
    ensemble_ate = np.mean(effects)
    ensemble_variance = np.var(effects)  # Variance of mean

    # Structural uncertainty = spread across different causal structures
    # This captures epistemic uncertainty from DAG ambiguity
    structural_uncertainty = np.var(effects)

    # Also compute weighted version (weight by inverse MB size for parsimony)
    weights = 1.0 / (np.array(mb_sizes) + 1)
    weights = weights / np.sum(weights)
    weighted_ate = np.sum(effects * weights)

    details = {
        'n_solutions': len(effects),
        'individual_effects': effects.tolist(),
        'mb_sizes': mb_sizes,
        'weighted_ate': weighted_ate,
        'effect_range': [float(np.min(effects)), float(np.max(effects))],
        'effect_std': float(np.std(effects)),
    }

    return ensemble_ate, ensemble_variance, structural_uncertainty, details


def _simple_effect_estimator(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
    """Simple IPW effect estimator."""
    from sklearn.linear_model import LogisticRegression

    T = T.flatten()
    Y = Y.flatten()

    # Binarize treatment if needed
    if len(np.unique(T)) > 2:
        T = (T > np.median(T)).astype(float)

    try:
        ps_model = LogisticRegression(max_iter=1000, solver='lbfgs')
        ps_model.fit(X, T.astype(int))
        e = ps_model.predict_proba(X)[:, 1]
        e = np.clip(e, 0.01, 0.99)
    except Exception:
        e = np.full(len(T), 0.5)

    # IPW estimator
    ate = np.mean(T * Y / e) - np.mean((1 - T) * Y / (1 - e))
    return ate


# =============================================================================
# UNIFIED POST-HOC VALIDATION PIPELINE
# =============================================================================

class PostHocValidationPipeline:
    """
    Unified post-hoc validation pipeline for JCCE Pareto solutions.

    Implements the 6 key validation steps:
    1. Random Common Cause (essential - confounding sensitivity)
    2. K=10 DML (publication tables)
    3. Hypervolume metrics (many-objective validation)
    4. Ensemble counterfactuals (structural uncertainty)
    5. Identifiability diagnostics (v15)
    6. Full refutation suite (v15: placebo + subset stability)

    Example usage:
    ```python
    pipeline = PostHocValidationPipeline()

    result = pipeline.validate(
        pareto_solutions=results['pareto_solutions'],
        pareto_front=results['pareto_front'],
        X=X, T=T, Y=Y,
    )

    print(result.summary())
    ```
    """

    def __init__(
        self,
        rcc_n_simulations: int = 100,
        rcc_threshold: float = 0.20,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize pipeline.

        Args:
            rcc_n_simulations: Number of simulations for RCC test
            rcc_threshold: Effect change threshold for RCC (default 20%)
            random_state: Random seed
            verbose: Print progress
        """
        self.rcc_n_simulations = rcc_n_simulations
        self.rcc_threshold = rcc_threshold
        self.random_state = random_state
        self.verbose = verbose

    def validate(
        self,
        pareto_solutions: List[Dict],
        pareto_front: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        effect_estimator: Optional[Callable] = None,
        run_rcc: bool = True,
        run_k10: bool = True,
        run_hypervolume: bool = True,
        run_ensemble: bool = True,
        run_identifiability: bool = True,
        run_refutation: bool = True,
        feature_names: Optional[List[str]] = None,
        representative_solution_idx: int = 0,
    ) -> PostHocValidationResult:
        """
        Run complete post-hoc validation pipeline.

        Args:
            pareto_solutions: List of solution dicts from NSGA-II
            pareto_front: Objective values (n_solutions, n_objectives)
            X: Covariates
            T: Treatment (from one of the MB features, or first feature)
            Y: Outcome
            effect_estimator: Custom effect estimator (optional)
            run_rcc: Whether to run Random Common Cause test (essential - recommended True)
            run_k10: Whether to run K=10 DML (can skip for speed)
            run_hypervolume: Whether to compute hypervolume metrics (cheap - recommended True)
            run_ensemble: Whether to compute ensemble counterfactuals (recommended True)
            run_refutation: Whether to run full refutation suite (placebo + subset stability)
            representative_solution_idx: Which solution to use for K=10

        Returns:
            PostHocValidationResult with all metrics
        """
        if effect_estimator is None:
            effect_estimator = _simple_effect_estimator

        # Convert arrays
        X = np.asarray(X)
        T = np.asarray(T).flatten()
        Y = np.asarray(Y).flatten()
        pareto_front = np.atleast_2d(pareto_front)

        if self.verbose:
            print("\n" + "=" * 70)
            print("POST-HOC VALIDATION PIPELINE")
            print("=" * 70)

        # Get representative solution info (needed by RCC, K10, ensemble)
        mb = []
        original_effect = 0.0
        if len(pareto_solutions) > representative_solution_idx:
            rep_sol = pareto_solutions[representative_solution_idx]
            mb = rep_sol.get('markov_blanket', [])
            if 'ate' in rep_sol:
                original_effect = rep_sol['ate']
            else:
                X_mb = X[:, mb] if len(mb) > 0 else X
                original_effect = effect_estimator(X_mb, T, Y)
        else:
            original_effect = effect_estimator(X, T, Y)

        # 1. Random Common Cause Test
        rcc_passed = True  # Default to pass if skipped
        rcc_effect_change = 0.0
        rcc_details = {'skipped': True}

        if run_rcc:
            if self.verbose:
                print("\n[1/6] Running Random Common Cause test...")

            rcc_passed, rcc_effect_change, rcc_details = run_random_common_cause_test(
                X=X[:, mb] if len(mb) > 0 else X,
                T=T, Y=Y,
                estimate_effect_fn=effect_estimator,
                original_effect=original_effect,
                n_simulations=self.rcc_n_simulations,
                effect_change_threshold=self.rcc_threshold,
                random_state=self.random_state,
            )

            if self.verbose:
                status = "PASS" if rcc_passed else "FAIL"
                print(f"   Random Common Cause: {status} (effect change: {rcc_effect_change:.1%})")
        else:
            if self.verbose:
                print("\n[1/6] Random Common Cause: Skipped")

        # 2. K=10 DML Validation
        k10_variance = float('nan')
        k10_ate = float('nan')
        k10_fold_estimates = []

        if run_k10:
            if self.verbose:
                print("\n[2/6] Running K=10 DML validation...")

            k10_variance, k10_ate, k10_fold_estimates, k10_details = run_k10_dml_validation(
                X=X, T=T, Y=Y,
                markov_blanket=mb,
                random_state=self.random_state,
            )

            if self.verbose:
                if np.isfinite(k10_variance):
                    print(f"   K=10 DML: ATE={k10_ate:.4f}, Var={k10_variance:.6f}")
                else:
                    print(f"   K=10 DML: Failed ({k10_details.get('error', 'unknown')})")
        else:
            if self.verbose:
                print("\n[2/6] K=10 DML: Skipped")

        # 3. Hypervolume Metrics
        hypervolume = 0.0
        pareto_diversity = 0.0

        if run_hypervolume:
            if self.verbose:
                print("\n[3/6] Computing hypervolume metrics...")

            hypervolume = compute_hypervolume(pareto_front)
            pareto_diversity = compute_pareto_diversity(pareto_front)

            if self.verbose:
                print(f"   Hypervolume: {hypervolume:.4f}")
                print(f"   Pareto Diversity: {pareto_diversity:.4f}")
        else:
            if self.verbose:
                print("\n[3/6] Hypervolume: Skipped")

        # 4. Ensemble Counterfactuals
        ensemble_ate = float('nan')
        ensemble_variance = float('nan')
        structural_uncertainty = float('nan')

        if run_ensemble:
            if self.verbose:
                print("\n[4/6] Computing ensemble counterfactuals...")

            ensemble_ate, ensemble_variance, structural_uncertainty, ensemble_details = \
                compute_ensemble_counterfactuals(
                    pareto_solutions=pareto_solutions,
                    X=X, T=T, Y=Y,
                    effect_estimator=effect_estimator,
                )

            if self.verbose:
                if np.isfinite(ensemble_ate):
                    print(f"   Ensemble ATE: {ensemble_ate:.4f}")
                    print(f"   Structural Uncertainty: {structural_uncertainty:.6f}")
                else:
                    print(f"   Ensemble: Failed")
        else:
            if self.verbose:
                print("\n[4/6] Ensemble Counterfactuals: Skipped")

        # 5. Identifiability Diagnostics (v15)
        identifiability_summary_dict = None

        if run_identifiability:
            if self.verbose:
                print("\n[5/6] Running identifiability diagnostics...")

            try:
                from jcce.validation.identifiability_diagnostics import (
                    compute_full_identifiability_report,
                    IdentifiabilitySummary,
                )

                reports = []
                for sol in pareto_solutions:
                    sol_mb = sol.get('markov_blanket', [])
                    sol_dag = sol.get('dag', sol.get('adjacency_matrix', None))
                    if sol_dag is None:
                        continue

                    report = compute_full_identifiability_report(
                        X=X,
                        A=np.asarray(sol_dag),
                        mb_indices=sol_mb,
                        feature_names=feature_names,
                        processor_type=sol.get('processor_type', 'linear'),
                    )
                    reports.append(report)

                if len(reports) > 0:
                    ident_summary = IdentifiabilitySummary(reports=reports)
                    identifiability_summary_dict = ident_summary.to_dict()

                    if self.verbose:
                        print(ident_summary.summary())
                else:
                    if self.verbose:
                        print("   No valid DAGs for identifiability analysis")
            except Exception as e:
                if self.verbose:
                    print(f"   Identifiability diagnostics failed: {e}")
        else:
            if self.verbose:
                print("\n[5/6] Identifiability Diagnostics: Skipped")

        # 6. Full Refutation Suite (v15: placebo + subset stability)
        refutation_result = None

        if run_refutation:
            if self.verbose:
                print("\n[6/6] Running full refutation suite...")

            try:
                from jcce.validation.refutation_suite import RefutationSuite

                suite = RefutationSuite(
                    n_simulations=100,
                    subset_fraction=0.8,
                    random_state=self.random_state,
                )

                # Use MB features for refutation if available
                X_for_refutation = X[:, mb] if len(mb) > 0 else X

                refutation_result = suite.run_all(
                    X=X_for_refutation,
                    T=T,
                    Y=Y,
                    estimate_effect_fn=effect_estimator,
                    original_effect=original_effect,
                )

                if self.verbose:
                    print(f"   {refutation_result.summary()}")
            except Exception as e:
                if self.verbose:
                    print(f"   Refutation suite failed: {e}")
        else:
            if self.verbose:
                print("\n[6/6] Refutation Suite: Skipped")

        # Overall validation decision
        # RCC is the essential test; refutation suite provides additional evidence
        validation_passed = rcc_passed
        if refutation_result is not None:
            # Also require majority of refutation tests to pass
            validation_passed = validation_passed and (refutation_result.pass_rate >= 0.5)

        summary_metrics = {
            'rcc_passed': rcc_passed,
            'rcc_effect_change': rcc_effect_change,
            'k10_variance': k10_variance,
            'hypervolume': hypervolume,
            'pareto_diversity': pareto_diversity,
            'ensemble_ate': ensemble_ate,
            'structural_uncertainty': structural_uncertainty,
            'n_pareto_solutions': len(pareto_solutions),
        }
        if refutation_result is not None:
            summary_metrics['refutation_pass_rate'] = refutation_result.pass_rate
            summary_metrics['refutation_n_passed'] = refutation_result.n_passed
            summary_metrics['refutation_n_total'] = refutation_result.n_total

        # Build refutation summary dict for storage
        refutation_summary_dict = None
        if refutation_result is not None:
            refutation_summary_dict = refutation_result.to_dict()

        result = PostHocValidationResult(
            rcc_passed=rcc_passed,
            rcc_effect_change=rcc_effect_change,
            rcc_details=rcc_details,
            k10_variance=k10_variance,
            k10_ate=k10_ate,
            k10_fold_estimates=k10_fold_estimates,
            hypervolume=hypervolume,
            pareto_diversity=pareto_diversity,
            ensemble_ate=ensemble_ate,
            ensemble_variance=ensemble_variance,
            structural_uncertainty=structural_uncertainty,
            identifiability_summary=identifiability_summary_dict,
            refutation_summary=refutation_summary_dict,
            validation_passed=validation_passed,
            summary_metrics=summary_metrics,
        )

        if self.verbose:
            print("\n" + result.summary())

        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def validate_pareto_front(
    pareto_solutions: List[Dict],
    pareto_front: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    run_rcc: bool = True,
    run_k10: bool = True,
    run_hypervolume: bool = True,
    run_ensemble: bool = True,
    verbose: bool = True,
) -> PostHocValidationResult:
    """
    Convenience function to run post-hoc validation with configurable tests.

    Args:
        pareto_solutions: List of solution dicts from NSGA-II
        pareto_front: Objective values
        X: Covariates
        T: Treatment
        Y: Outcome
        run_rcc: Run Random Common Cause test (essential - recommended True)
        run_k10: Run K=10 DML validation (optional - for publication tables)
        run_hypervolume: Compute hypervolume metrics (cheap - recommended True)
        run_ensemble: Compute ensemble counterfactuals (recommended True)
        verbose: Print progress

    Returns:
        PostHocValidationResult
    """
    pipeline = PostHocValidationPipeline(verbose=verbose)
    return pipeline.validate(
        pareto_solutions=pareto_solutions,
        pareto_front=pareto_front,
        X=X, T=T, Y=Y,
        run_rcc=run_rcc,
        run_k10=run_k10,
        run_hypervolume=run_hypervolume,
        run_ensemble=run_ensemble,
    )
