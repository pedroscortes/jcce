"""
Refutation Test Suite for JCCE

DoWhy-style refutation tests for validating causal claims.
Applied post-hoc to Pareto-optimal solutions.

Tests included:
1. Placebo Treatment: Replace treatment with noise, effect should vanish
2. Random Common Cause: Add random confounder, effect should remain stable
3. Data Subset: Test stability across random subsets
4. Dummy Outcome: Replace outcome with noise, effect should vanish

References:
- Sharma & Kiciman (2020) "DoWhy: An End-to-End Library for Causal Inference"
- Cinelli & Hazlett (2020) "Making Sense of Sensitivity"
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import stats


@dataclass
class RefutationResult:
    """Result of a single refutation test."""

    test_name: str
    passed: bool
    p_value: float
    original_effect: float
    refuted_effect: float
    effect_ratio: float
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "p_value": self.p_value,
            "original_effect": self.original_effect,
            "refuted_effect": self.refuted_effect,
            "effect_ratio": self.effect_ratio,
            "details": self.details,
        }


@dataclass
class RefutationSuiteResult:
    """Aggregated result of all refutation tests."""

    results: Dict[str, RefutationResult]
    n_passed: int = field(init=False)
    n_total: int = field(init=False)
    pass_rate: float = field(init=False)
    corrected_results: Optional[Dict[str, bool]] = None

    def __post_init__(self):
        self.n_total = len(self.results)
        self.n_passed = sum(1 for r in self.results.values() if r.passed)
        self.pass_rate = self.n_passed / self.n_total if self.n_total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "pass_rate": self.pass_rate,
            "corrected_results": self.corrected_results,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Refutation Test Summary: {self.n_passed}/{self.n_total} passed ({self.pass_rate:.1%})",
            "-" * 60,
        ]
        for name, result in self.results.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(
                f"  {name}: {status} (p={result.p_value:.4f}, "
                f"effect_ratio={result.effect_ratio:.2f})"
            )
        return "\n".join(lines)


class RefutationSuite:
    """
    Suite of refutation tests for causal claims.

    These tests probe whether the estimated causal effect is robust
    to various perturbations that should either:
    - Destroy the effect (placebo, dummy outcome)
    - Leave the effect unchanged (random common cause, subsets)

    Example usage:
    ```python
    suite = RefutationSuite(n_simulations=100)

    def estimate_effect(X, T, Y):
        # Your effect estimation logic
        return ate_estimate

    original_effect = estimate_effect(X, T, Y)
    results = suite.run_all(X, T, Y, estimate_effect, original_effect)

    print(results.summary())
    print(f"Pass rate: {results.pass_rate:.1%}")
    ```
    """

    def __init__(
        self,
        n_simulations: int = 100,
        subset_fraction: float = 0.8,
        random_state: int = 42,
        effect_change_threshold: float = 0.2,
        placebo_threshold: float = 0.1,
    ):
        """
        Initialize refutation suite.

        Args:
            n_simulations: Number of Monte Carlo simulations per test
            subset_fraction: Fraction of data to use in subset test
            random_state: Random seed
            effect_change_threshold: Max allowed effect change for stability tests
            placebo_threshold: Max allowed effect for placebo/dummy tests
        """
        self.n_simulations = n_simulations
        self.subset_fraction = subset_fraction
        self.random_state = random_state
        self.effect_change_threshold = effect_change_threshold
        self.placebo_threshold = placebo_threshold
        self.rng = np.random.RandomState(random_state)

    def run_all(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        estimate_effect_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        original_effect: float,
        tests: Optional[List[str]] = None,
        apply_correction: bool = True,
    ) -> RefutationSuiteResult:
        """
        Run all (or selected) refutation tests.

        Args:
            X: Covariates
            T: Treatment
            Y: Outcome
            estimate_effect_fn: Function to estimate causal effect
                Signature: (X, T, Y) -> float
            original_effect: Original effect estimate to test
            tests: List of test names to run (default: all)
            apply_correction: Whether to apply Benjamini-Hochberg correction

        Returns:
            RefutationSuiteResult with all test results
        """
        # Handle JAX arrays
        if hasattr(X, "device_buffer") or str(type(X)).find("jax") >= 0:
            X = np.array(X)
        if hasattr(T, "device_buffer") or str(type(T)).find("jax") >= 0:
            T = np.array(T)
        if hasattr(Y, "device_buffer") or str(type(Y)).find("jax") >= 0:
            Y = np.array(Y)

        T = T.flatten()
        Y = Y.flatten()

        available_tests = {
            "placebo_treatment": self.placebo_treatment_test,
            "random_common_cause": self.random_common_cause_test,
            "data_subset": self.data_subset_test,
            "dummy_outcome": self.dummy_outcome_test,
        }

        if tests is None:
            tests = list(available_tests.keys())

        results = {}
        for test_name in tests:
            if test_name not in available_tests:
                warnings.warn(f"Unknown test: {test_name}")
                continue

            try:
                result = available_tests[test_name](X, T, Y, estimate_effect_fn, original_effect)
                results[test_name] = result
            except Exception as e:
                warnings.warn(f"Test {test_name} failed: {e}")
                results[test_name] = RefutationResult(
                    test_name=test_name,
                    passed=False,
                    p_value=1.0,
                    original_effect=original_effect,
                    refuted_effect=float("nan"),
                    effect_ratio=float("nan"),
                    details={"error": str(e)},
                )

        # Apply multiple testing correction
        corrected_results = None
        if apply_correction and len(results) > 1:
            p_values = [r.p_value for r in results.values()]
            rejected = self.benjamini_hochberg(p_values)
            corrected_results = {name: rej for name, rej in zip(results.keys(), rejected)}

        return RefutationSuiteResult(results=results, corrected_results=corrected_results)

    def placebo_treatment_test(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        estimate_effect_fn: Callable,
        original_effect: float,
    ) -> RefutationResult:
        """
        Placebo treatment test.

        Replace treatment with random permutation. The effect should vanish
        since the placebo treatment has no causal relationship with the outcome.

        A robust causal claim should show that:
        - Original effect is significantly different from placebo effects
        - Placebo effects are centered near zero
        """
        placebo_effects = []

        for _ in range(self.n_simulations):
            # Random permutation breaks any T-Y relationship
            T_placebo = self.rng.permutation(T)

            try:
                effect = estimate_effect_fn(X, T_placebo, Y)
                if np.isfinite(effect):
                    placebo_effects.append(effect)
            except:
                continue

        if len(placebo_effects) < 10:
            return RefutationResult(
                test_name="placebo_treatment",
                passed=False,
                p_value=1.0,
                original_effect=original_effect,
                refuted_effect=float("nan"),
                effect_ratio=float("nan"),
                details={"error": "Too few successful simulations"},
            )

        placebo_mean = np.mean(placebo_effects)
        placebo_std = np.std(placebo_effects)

        # Test: Is original effect significantly different from placebo distribution?
        if placebo_std > 1e-10:
            z_score = (original_effect - placebo_mean) / placebo_std
            # Two-sided test: we want original to be different from placebo
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            # No variance in placebo - just check if they're different
            p_value = 0.0 if abs(original_effect - placebo_mean) > 0.01 else 1.0

        # Pass if:
        # 1. Original effect is significantly different from placebo (p < 0.05)
        # 2. Placebo effects are near zero
        placebo_near_zero = abs(placebo_mean) < self.placebo_threshold * abs(
            original_effect + 1e-10
        )
        significantly_different = p_value < 0.05

        passed = significantly_different and placebo_near_zero

        effect_ratio = abs(original_effect) / (abs(placebo_mean) + 1e-10)

        return RefutationResult(
            test_name="placebo_treatment",
            passed=passed,
            p_value=p_value,
            original_effect=original_effect,
            refuted_effect=placebo_mean,
            effect_ratio=effect_ratio,
            details={
                "placebo_mean": placebo_mean,
                "placebo_std": placebo_std,
                "z_score": z_score if placebo_std > 1e-10 else None,
                "n_simulations": len(placebo_effects),
                "placebo_near_zero": placebo_near_zero,
            },
        )

    def random_common_cause_test(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        estimate_effect_fn: Callable,
        original_effect: float,
    ) -> RefutationResult:
        """
        Random common cause test.

        Add a random variable as an additional covariate. The effect estimate
        should remain stable since a random variable cannot be a true confounder.

        Tests whether the estimator is robust to irrelevant covariates.
        """
        confounded_effects = []

        for _ in range(self.n_simulations):
            # Add random confounder
            Z_random = self.rng.normal(0, 1, size=(len(X), 1))
            X_augmented = np.hstack([X, Z_random])

            try:
                effect = estimate_effect_fn(X_augmented, T, Y)
                if np.isfinite(effect):
                    confounded_effects.append(effect)
            except:
                continue

        if len(confounded_effects) < 10:
            return RefutationResult(
                test_name="random_common_cause",
                passed=False,
                p_value=1.0,
                original_effect=original_effect,
                refuted_effect=float("nan"),
                effect_ratio=float("nan"),
                details={"error": "Too few successful simulations"},
            )

        confounded_mean = np.mean(confounded_effects)
        confounded_std = np.std(confounded_effects)

        # Effect change ratio
        effect_change = abs(confounded_mean - original_effect) / (abs(original_effect) + 1e-10)

        # Pass if effect remains stable (< threshold change)
        passed = effect_change < self.effect_change_threshold

        # P-value: is the effect significantly different after adding random confounder?
        # We want p > 0.05 (no significant change = good)
        if confounded_std > 1e-10:
            z_score = (original_effect - confounded_mean) / confounded_std
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            # No variance: if effect unchanged, high p-value; if changed, low
            p_value = 1.0 if abs(original_effect - confounded_mean) < 1e-10 else 0.0

        return RefutationResult(
            test_name="random_common_cause",
            passed=passed,
            p_value=p_value,
            original_effect=original_effect,
            refuted_effect=confounded_mean,
            effect_ratio=1 - effect_change,
            details={
                "confounded_mean": confounded_mean,
                "confounded_std": confounded_std,
                "effect_change": effect_change,
                "n_simulations": len(confounded_effects),
            },
        )

    def data_subset_test(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        estimate_effect_fn: Callable,
        original_effect: float,
    ) -> RefutationResult:
        """
        Data subset test.

        Estimate effect on random subsets of data. A robust causal effect
        should be consistent across different subsamples (low variance).

        High variance indicates the effect may be driven by specific data points.
        """
        subset_effects = []
        n_subset = int(len(X) * self.subset_fraction)

        for _ in range(self.n_simulations):
            idx = self.rng.choice(len(X), size=n_subset, replace=False)

            try:
                effect = estimate_effect_fn(X[idx], T[idx], Y[idx])
                if np.isfinite(effect):
                    subset_effects.append(effect)
            except:
                continue

        if len(subset_effects) < 10:
            return RefutationResult(
                test_name="data_subset",
                passed=False,
                p_value=1.0,
                original_effect=original_effect,
                refuted_effect=float("nan"),
                effect_ratio=float("nan"),
                details={"error": "Too few successful simulations"},
            )

        subset_mean = np.mean(subset_effects)
        subset_std = np.std(subset_effects)

        # Coefficient of variation
        cv = subset_std / (abs(subset_mean) + 1e-10)

        # Pass if low variance (CV < 0.5)
        passed = cv < 0.5

        # Also check if mean is close to original
        mean_deviation = abs(subset_mean - original_effect) / (abs(original_effect) + 1e-10)

        return RefutationResult(
            test_name="data_subset",
            passed=passed,
            p_value=0.95 if passed else 0.01,  # Heuristic p-value
            original_effect=original_effect,
            refuted_effect=subset_mean,
            effect_ratio=1 - cv,
            details={
                "subset_mean": subset_mean,
                "subset_std": subset_std,
                "coefficient_of_variation": cv,
                "mean_deviation": mean_deviation,
                "n_simulations": len(subset_effects),
            },
        )

    def dummy_outcome_test(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        estimate_effect_fn: Callable,
        original_effect: float,
    ) -> RefutationResult:
        """
        Dummy outcome test.

        Replace outcome with random noise. The effect should vanish since
        there's no real relationship between treatment and the dummy outcome.

        Tests whether the estimator can distinguish real effects from noise.
        """
        dummy_effects = []

        for _ in range(self.n_simulations):
            # Random outcome with same distribution as Y
            Y_dummy = self.rng.normal(np.mean(Y), np.std(Y), size=len(Y))

            try:
                effect = estimate_effect_fn(X, T, Y_dummy)
                if np.isfinite(effect):
                    dummy_effects.append(effect)
            except:
                continue

        if len(dummy_effects) < 10:
            return RefutationResult(
                test_name="dummy_outcome",
                passed=False,
                p_value=1.0,
                original_effect=original_effect,
                refuted_effect=float("nan"),
                effect_ratio=float("nan"),
                details={"error": "Too few successful simulations"},
            )

        dummy_mean = np.mean(dummy_effects)
        dummy_std = np.std(dummy_effects)

        # Pass if dummy effects are near zero relative to original effect
        # Use effect ratio (per DoWhy), not absolute threshold
        passed = abs(dummy_mean) < self.placebo_threshold * (abs(original_effect) + 1e-10)

        # P-value: is dummy effect significantly different from zero?
        # We want dummy effect near zero → p > 0.05
        if dummy_std > 1e-10:
            z_score = dummy_mean / dummy_std
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            p_value = 1.0 if abs(dummy_mean) < 1e-10 else 0.0

        effect_ratio = abs(original_effect) / (abs(dummy_mean) + 1e-10)

        return RefutationResult(
            test_name="dummy_outcome",
            passed=passed,
            p_value=p_value,
            original_effect=original_effect,
            refuted_effect=dummy_mean,
            effect_ratio=effect_ratio,
            details={
                "dummy_mean": dummy_mean,
                "dummy_std": dummy_std,
                "n_simulations": len(dummy_effects),
            },
        )

    @staticmethod
    def benjamini_hochberg(p_values: List[float], fdr: float = 0.05) -> List[bool]:
        """
        Benjamini-Hochberg FDR correction for multiple testing.

        Args:
            p_values: List of p-values from multiple tests
            fdr: False discovery rate threshold

        Returns:
            List of booleans indicating which hypotheses to reject
        """
        n = len(p_values)
        if n == 0:
            return []

        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]

        # BH thresholds
        thresholds = [(i + 1) / n * fdr for i in range(n)]

        # Find largest k where p_k <= (k/n) * fdr
        reject_sorted = sorted_p <= thresholds

        if not any(reject_sorted):
            return [False] * n

        max_reject_idx = np.max(np.where(reject_sorted)[0])

        # All hypotheses up to max_reject_idx are rejected
        reject_sorted_final = np.zeros(n, dtype=bool)
        reject_sorted_final[: max_reject_idx + 1] = True

        # Map back to original order
        reject = [False] * n
        for i, idx in enumerate(sorted_idx):
            reject[idx] = reject_sorted_final[i]

        return reject


# =============================================================================
# Utility Functions
# =============================================================================


def create_simple_effect_estimator():
    """
    Create a simple effect estimator for testing refutation suite.

    Uses difference-in-means with propensity score weighting.
    """
    from sklearn.linear_model import LogisticRegression

    def estimate_effect(X, T, Y):
        """Simple IPW estimator."""
        T = T.flatten()
        Y = Y.flatten()

        # Binarize treatment if needed
        if len(np.unique(T)) > 2:
            T = (T > np.median(T)).astype(float)

        # Fit propensity model
        try:
            ps_model = LogisticRegression(max_iter=1000)
            ps_model.fit(X, T.astype(int))
            e = ps_model.predict_proba(X)[:, 1]
            e = np.clip(e, 0.01, 0.99)
        except:
            e = np.full(len(T), 0.5)

        # IPW estimator
        ate = np.mean(T * Y / e) - np.mean((1 - T) * Y / (1 - e))
        return ate

    return estimate_effect
