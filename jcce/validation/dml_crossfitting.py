"""
DML Cross-Fitting for JCCE

Double Machine Learning with K-fold cross-fitting for robust causal effect estimation.
Integrates with DragonNet/AIPW for nuisance function estimation.

Key benefits:
- Reduces regularization bias in neural network-based effect estimation
- Provides variance estimates for effect robustness assessment
- Can be used as NSGA-II objective (minimize effect variance)

References:
- Chernozhukov et al. (2018) "Double/Debiased ML for Treatment and Structural Parameters"
- Shi et al. (2019) "Adapting Neural Networks for the Estimation of Treatment Effects"
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class DMLDiagnostics:
    """Diagnostics for DML estimation quality.

    Reports propensity overlap, covariate balance, and nuisance fit quality.
    All metrics are reporting-only (do not affect estimation).

    References:
        - Overlap: Crump et al. (2009) "Dealing with limited overlap"
        - Balance: Austin (2009) "Using the SMD to compare covariates"
        - Convergence: Chernozhukov et al. (2018) Assumption 3.1
    """

    # Propensity overlap
    propensity_min: float = 0.0
    propensity_max: float = 1.0
    propensity_mean: float = 0.5
    pct_clipped: float = 0.0  # Fraction clipped by clip_propensity bounds

    # Covariate balance (standardized mean difference)
    mean_smd: float = 0.0  # Mean |SMD| across covariates
    max_smd: float = 0.0  # Max |SMD| across covariates
    n_imbalanced: int = 0  # Covariates with |SMD| > 0.1

    # Nuisance fit quality (out-of-fold)
    outcome_r2_mean: float = 0.0  # Mean R² of outcome models across folds
    propensity_logloss_mean: float = 0.0  # Mean log-loss of propensity across folds

    def to_dict(self) -> Dict:
        return {
            "propensity_min": self.propensity_min,
            "propensity_max": self.propensity_max,
            "propensity_mean": self.propensity_mean,
            "pct_clipped": self.pct_clipped,
            "mean_smd": self.mean_smd,
            "max_smd": self.max_smd,
            "n_imbalanced": self.n_imbalanced,
            "outcome_r2_mean": self.outcome_r2_mean,
            "propensity_logloss_mean": self.propensity_logloss_mean,
        }

    def summary(self) -> str:
        lines = [
            "DML Diagnostics:",
            f"  Propensity: min={self.propensity_min:.3f} max={self.propensity_max:.3f} "
            f"mean={self.propensity_mean:.3f} clipped={self.pct_clipped:.1%}",
            f"  Balance: mean|SMD|={self.mean_smd:.3f} max|SMD|={self.max_smd:.3f} "
            f"imbalanced={self.n_imbalanced}",
            f"  Nuisance: outcome_R²={self.outcome_r2_mean:.3f} "
            f"propensity_logloss={self.propensity_logloss_mean:.3f}",
        ]
        return "\n".join(lines)


@dataclass
class DMLResult:
    """Result of DML cross-fitting estimation."""

    ate_mean: float
    ate_std: float
    ate_per_fold: List[float]
    n_folds: int
    treatment_idx: Optional[int] = None
    diagnostics: Optional[DMLDiagnostics] = None

    @property
    def variance(self) -> float:
        """Variance of ATE estimates across folds."""
        return float(np.var(self.ate_per_fold))

    @property
    def standard_error(self) -> float:
        """Influence-function standard error (Chernozhukov et al. 2018).

        ate_std is now set to the influence-function SE directly:
        SE = sqrt(Var(ψ_i) / n) where ψ_i are individual AIPW scores.
        """
        return self.ate_std

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute confidence interval for ATE using normal approximation."""
        z_crit = stats.norm.ppf(1 - alpha / 2)
        margin = z_crit * self.standard_error
        return (self.ate_mean - margin, self.ate_mean + margin)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = {
            "ate_mean": self.ate_mean,
            "ate_std": self.ate_std,
            "ate_variance": self.variance,
            "ate_per_fold": self.ate_per_fold,
            "n_folds": self.n_folds,
            "treatment_idx": self.treatment_idx,
            "ci_95": self.confidence_interval(0.05),
        }
        if self.diagnostics is not None:
            d["diagnostics"] = self.diagnostics.to_dict()
        return d


@dataclass
class DMLMultiTreatmentResult:
    """Result of DML cross-fitting for multiple treatments (MB features)."""

    results: Dict[int, DMLResult]  # treatment_idx -> DMLResult
    total_variance: float = field(init=False)
    mean_variance: float = field(init=False)

    def __post_init__(self):
        variances = [r.variance for r in self.results.values()]
        self.total_variance = sum(variances)
        self.mean_variance = np.mean(variances) if variances else 0.0

    def to_dict(self) -> Dict:
        return {
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "total_variance": self.total_variance,
            "mean_variance": self.mean_variance,
            "n_treatments": len(self.results),
        }


class DMLCrossFitter:
    """
    Double Machine Learning with K-fold cross-fitting.

    This class implements DML for robust causal effect estimation,
    designed to integrate with JCCE's DragonNet architecture.

    The key idea is to use cross-fitting to avoid overfitting bias:
    1. Split data into K folds
    2. For each fold k, train nuisance models on D \\ D_k
    3. Compute AIPW scores on D_k using out-of-sample predictions
    4. Aggregate scores across all folds

    This produces effect estimates that are:
    - Less biased due to regularization
    - Come with valid variance estimates
    - Can be used to assess robustness (high variance = less reliable)

    Example usage:
    ```python
    dml = DMLCrossFitter(n_splits=5)

    # For a single treatment
    result = dml.estimate_ate(X, T, Y, train_fn, predict_fn)
    print(f"ATE: {result.ate_mean:.4f} +/- {result.ate_std:.4f}")
    print(f"Variance: {result.variance:.6f}")

    # For multiple treatments (MB features)
    mb_result = dml.estimate_ate_multi_treatment(
        X, Y, treatment_indices=[0, 2, 5],
        train_fn, predict_fn
    )
    print(f"Total variance: {mb_result.total_variance:.6f}")
    ```
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: int = 42,
        stratify: bool = True,
        clip_propensity: Tuple[float, float] = (0.01, 0.99),
    ):
        """
        Initialize DML cross-fitter.

        Args:
            n_splits: Number of folds for cross-fitting (K)
            n_repeats: Number of times to repeat cross-fitting (for stability)
            random_state: Random seed for reproducibility
            stratify: Whether to use stratified K-fold (recommended for classification)
            clip_propensity: Min/max bounds for propensity scores to avoid division issues
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.stratify = stratify
        self.clip_propensity = clip_propensity

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        train_nuisance_fn: Callable,
        predict_nuisance_fn: Callable,
        treatment_idx: Optional[int] = None,
    ) -> DMLResult:
        """
        Estimate Average Treatment Effect using cross-fitted AIPW.

        Args:
            X: Covariates (n_samples, n_features)
            T: Treatment indicator (n_samples,), should be binary or continuous
            Y: Outcome (n_samples,)
            train_nuisance_fn: Function to train nuisance models
                Signature: (X_train, T_train, Y_train) -> model
            predict_nuisance_fn: Function to predict with nuisance models
                Signature: (model, X) -> (e_hat, mu0_hat, mu1_hat)
                Where:
                    e_hat: propensity scores P(T=1|X)
                    mu0_hat: E[Y|X, T=0]
                    mu1_hat: E[Y|X, T=1]
            treatment_idx: Optional index of treatment variable (for logging)

        Returns:
            DMLResult with ATE estimate and variance
        """
        # Handle JAX arrays
        if hasattr(X, "device_buffer") or str(type(X)).find("jax") >= 0:
            X = np.array(X)
        if hasattr(T, "device_buffer") or str(type(T)).find("jax") >= 0:
            T = np.array(T)
        if hasattr(Y, "device_buffer") or str(type(Y)).find("jax") >= 0:
            Y = np.array(Y)

        # Ensure proper shapes
        T = T.flatten()
        Y = Y.flatten()

        ate_estimates = []
        # Collect all individual AIPW scores for influence-function SE
        # (Chernozhukov et al. 2018: Var(ATE) = (1/n) * Var(ψ_i))
        all_aipw_scores = np.full(len(Y), np.nan)

        # Diagnostic accumulators
        all_e_raw = []  # Raw propensity scores (before clipping)
        outcome_r2s = []  # Per-fold outcome R²
        propensity_loglosses = []  # Per-fold propensity log-loss

        for repeat in range(self.n_repeats):
            seed = self.random_state + repeat

            if self.stratify:
                # For stratified, we need discrete Y
                Y_discrete = (Y > np.median(Y)).astype(int) if Y.dtype == float else Y
                kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
                split_iterator = kfold.split(X, Y_discrete)
            else:
                kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
                split_iterator = kfold.split(X)

            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
                try:
                    # Train nuisance models on training folds
                    model = train_nuisance_fn(X[train_idx], T[train_idx], Y[train_idx])

                    # Predict on validation fold (out-of-sample)
                    e_hat, mu0_hat, mu1_hat = predict_nuisance_fn(model, X[val_idx])

                    # Ensure proper shapes
                    e_hat = np.array(e_hat).flatten()
                    mu0_hat = np.array(mu0_hat).flatten()
                    mu1_hat = np.array(mu1_hat).flatten()

                    # Collect raw propensity before clipping
                    all_e_raw.append(e_hat.copy())

                    # Nuisance fit diagnostics (out-of-fold)
                    T_val = T[val_idx]
                    Y_val = Y[val_idx]
                    T_binary_val = (T_val > 0.5).astype(int)

                    # Outcome R²: weighted average of mu0/mu1 predictions
                    mu_hat = np.where(T_binary_val, mu1_hat, mu0_hat)
                    ss_res = np.sum((Y_val - mu_hat) ** 2)
                    ss_tot = np.sum((Y_val - np.mean(Y_val)) ** 2)
                    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
                    outcome_r2s.append(float(r2))

                    # Propensity log-loss
                    e_safe = np.clip(e_hat, 1e-7, 1 - 1e-7)
                    logloss = -np.mean(
                        T_binary_val * np.log(e_safe) + (1 - T_binary_val) * np.log(1 - e_safe)
                    )
                    propensity_loglosses.append(float(logloss))

                    # Clip propensity to avoid extreme weights
                    e_hat = np.clip(e_hat, self.clip_propensity[0], self.clip_propensity[1])

                    # AIPW score for each sample
                    # ψ = μ₁(X) - μ₀(X) + T(Y - μ₁(X))/e(X) - (1-T)(Y - μ₀(X))/(1-e(X))
                    scores = (
                        mu1_hat
                        - mu0_hat
                        + T_val * (Y_val - mu1_hat) / e_hat
                        - (1 - T_val) * (Y_val - mu0_hat) / (1 - e_hat)
                    )

                    # Handle any remaining NaN/Inf
                    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

                    fold_ate = float(np.mean(scores))
                    fold_scores.append(fold_ate)

                    # Store per-sample scores (last repeat overwrites — fine for SE)
                    all_aipw_scores[val_idx] = scores

                except Exception as e:
                    warnings.warn(f"Fold {fold_idx} failed: {e}")
                    continue

            if fold_scores:
                # Average across folds for this repeat
                ate_estimates.append(np.mean(fold_scores))

        if not ate_estimates:
            # All folds failed - return degenerate result
            return DMLResult(
                ate_mean=0.0,
                ate_std=float("inf"),
                ate_per_fold=[],
                n_folds=0,
                treatment_idx=treatment_idx,
            )

        # Influence-function SE: sqrt(Var(ψ_i) / n)
        # This is the proper DML variance (Chernozhukov et al. 2018, Thm 3.1)
        valid_scores = all_aipw_scores[~np.isnan(all_aipw_scores)]
        if len(valid_scores) > 1:
            influence_se = float(np.std(valid_scores) / np.sqrt(len(valid_scores)))
        else:
            influence_se = float(np.std(ate_estimates))

        # Build diagnostics
        diagnostics = self._compute_diagnostics(X, T, all_e_raw, outcome_r2s, propensity_loglosses)

        return DMLResult(
            ate_mean=float(np.mean(ate_estimates)),
            ate_std=influence_se,
            ate_per_fold=ate_estimates,
            n_folds=len(ate_estimates),
            treatment_idx=treatment_idx,
            diagnostics=diagnostics,
        )

    def _compute_diagnostics(
        self,
        X: np.ndarray,
        T: np.ndarray,
        all_e_raw: List[np.ndarray],
        outcome_r2s: List[float],
        propensity_loglosses: List[float],
    ) -> DMLDiagnostics:
        """Compute reporting-only diagnostics from cross-fitting results."""
        # Propensity overlap
        if all_e_raw:
            e_all = np.concatenate(all_e_raw)
            pct_clipped = float(
                np.mean((e_all < self.clip_propensity[0]) | (e_all > self.clip_propensity[1]))
            )
            prop_min = float(np.min(e_all))
            prop_max = float(np.max(e_all))
            prop_mean = float(np.mean(e_all))
        else:
            prop_min, prop_max, prop_mean, pct_clipped = 0.0, 1.0, 0.5, 0.0

        # Covariate balance: standardized mean difference (SMD)
        # |SMD| > 0.1 is conventional imbalance threshold (Austin 2009)
        T_binary = (T > 0.5).astype(bool)
        treated = X[T_binary]
        control = X[~T_binary]
        if len(treated) > 1 and len(control) > 1:
            mean_t = np.mean(treated, axis=0)
            mean_c = np.mean(control, axis=0)
            var_t = np.var(treated, axis=0)
            var_c = np.var(control, axis=0)
            pooled_std = np.sqrt((var_t + var_c) / 2.0 + 1e-10)
            smds = np.abs(mean_t - mean_c) / pooled_std
            mean_smd = float(np.mean(smds))
            max_smd = float(np.max(smds))
            n_imbalanced = int(np.sum(smds > 0.1))
        else:
            mean_smd, max_smd, n_imbalanced = 0.0, 0.0, 0

        return DMLDiagnostics(
            propensity_min=prop_min,
            propensity_max=prop_max,
            propensity_mean=prop_mean,
            pct_clipped=pct_clipped,
            mean_smd=mean_smd,
            max_smd=max_smd,
            n_imbalanced=n_imbalanced,
            outcome_r2_mean=float(np.mean(outcome_r2s)) if outcome_r2s else 0.0,
            propensity_logloss_mean=float(np.mean(propensity_loglosses))
            if propensity_loglosses
            else 0.0,
        )

    def estimate_ate_multi_treatment(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        treatment_indices: List[int],
        train_nuisance_fn: Callable,
        predict_nuisance_fn: Callable,
    ) -> DMLMultiTreatmentResult:
        """
        Estimate ATE for multiple treatment variables (e.g., all MB features).

        This is the main method for JCCE integration - it estimates effects
        for each feature in the Markov Blanket and returns the total variance
        which can be used as an NSGA-II objective.

        Args:
            X: Covariates (n_samples, n_features)
            Y: Outcome (n_samples,)
            treatment_indices: List of feature indices to treat as treatments
            train_nuisance_fn: Function to train nuisance models
            predict_nuisance_fn: Function to predict with nuisance models

        Returns:
            DMLMultiTreatmentResult with per-treatment results and total variance
        """
        results = {}

        for t_idx in treatment_indices:
            # Extract treatment variable
            T = X[:, t_idx]

            # Binarize if continuous (above/below median)
            if len(np.unique(T)) > 10:
                T_binary = (T > np.median(T)).astype(float)
            else:
                T_binary = T

            # Create modified X without the treatment column for covariates
            # (optional - some implementations keep all features)
            X_covariates = X  # Keep all features as covariates

            result = self.estimate_ate(
                X_covariates,
                T_binary,
                Y,
                train_nuisance_fn,
                predict_nuisance_fn,
                treatment_idx=t_idx,
            )
            results[t_idx] = result

        return DMLMultiTreatmentResult(results=results)

    def compute_effect_cv_objective(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        markov_blanket: List[int],
        train_nuisance_fn: Callable,
        predict_nuisance_fn: Callable,
        aggregation: str = "sum",
    ) -> float:
        """
        Compute the DML-based effect CV variance objective for NSGA-II.

        This is the new O5 objective: L_effect_cv = Σ Var(τ_t) for t in MB

        Lower values indicate more robust/stable effect estimates.

        Args:
            X: Covariates
            Y: Outcome
            markov_blanket: List of feature indices in the Markov Blanket
            train_nuisance_fn: Nuisance model training function
            predict_nuisance_fn: Nuisance model prediction function
            aggregation: How to aggregate variances ('sum', 'mean', 'max')

        Returns:
            Effect CV variance objective value (lower = better)
        """
        if not markov_blanket:
            return float("inf")  # No MB = bad solution

        result = self.estimate_ate_multi_treatment(
            X, Y, markov_blanket, train_nuisance_fn, predict_nuisance_fn
        )

        if aggregation == "sum":
            return result.total_variance
        elif aggregation == "mean":
            return result.mean_variance
        elif aggregation == "max":
            return max(r.variance for r in result.results.values())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def get_progressive_k(generation: int, max_generation: int = 40) -> int:
    """
    Get K value for progressive cross-fitting strategy.

    Early generations use smaller K (faster), later generations use larger K (more accurate).

    Args:
        generation: Current NSGA-II generation
        max_generation: Maximum number of generations

    Returns:
        K value for cross-fitting
    """
    progress = generation / max_generation

    if progress <= 0.25:
        return 2  # Exploration phase: K=2
    elif progress <= 0.625:
        return 3  # Mid phase: K=3
    else:
        return 5  # Exploitation phase: K=5


# =============================================================================
# Simple Nuisance Model Implementations (for testing)
# =============================================================================


def create_simple_nuisance_functions():
    """
    Create simple nuisance model functions for testing.
    Uses logistic regression for propensity and linear regression for outcomes.

    Returns:
        train_fn, predict_fn tuple
    """
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV

    def train_fn(X_train, T_train, Y_train):
        """Train propensity and outcome models."""
        # Propensity model: P(T=1|X) with CV-tuned regularization
        # LogisticRegressionCV selects best C via cross-validation,
        # ensuring convergence rate > 1/2 per Chernozhukov (2018) Thm 3.1
        T_binary = (
            (T_train > np.median(T_train)).astype(int)
            if len(np.unique(T_train)) > 2
            else T_train.astype(int)
        )

        propensity_model = LogisticRegressionCV(
            max_iter=1000, solver="lbfgs", cv=3, scoring="neg_log_loss"
        )
        try:
            propensity_model.fit(X_train, T_binary)
        except:
            # Fallback if fitting fails
            propensity_model = None

        # Outcome models: E[Y|X, T=0] and E[Y|X, T=1]
        # RidgeCV auto-selects alpha via efficient LOOCV (no extra cost)
        outcome_model_0 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        outcome_model_1 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

        T_binary = T_binary.astype(bool)

        if np.sum(~T_binary) > 0:
            outcome_model_0.fit(X_train[~T_binary], Y_train[~T_binary])
        if np.sum(T_binary) > 0:
            outcome_model_1.fit(X_train[T_binary], Y_train[T_binary])

        return {
            "propensity": propensity_model,
            "outcome_0": outcome_model_0,
            "outcome_1": outcome_model_1,
            "T_median": np.median(T_train) if len(np.unique(T_train)) > 2 else 0.5,
        }

    def predict_fn(model, X):
        """Predict propensity and potential outcomes."""
        n = len(X)

        # Propensity
        if model["propensity"] is not None:
            try:
                e_hat = model["propensity"].predict_proba(X)[:, 1]
            except:
                e_hat = np.full(n, 0.5)
        else:
            e_hat = np.full(n, 0.5)

        # Outcomes
        try:
            mu0_hat = model["outcome_0"].predict(X)
        except:
            mu0_hat = np.zeros(n)

        try:
            mu1_hat = model["outcome_1"].predict(X)
        except:
            mu1_hat = np.zeros(n)

        return e_hat, mu0_hat, mu1_hat

    return train_fn, predict_fn


def create_gbm_nuisance_functions(
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
):
    """
    Create gradient boosting nuisance model functions for DML.

    Nonlinear nuisance models can satisfy the convergence rate requirement
    (Chernozhukov 2018, Assumption 3.1: rate > n^{-1/4}) in settings where
    linear models are misspecified.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Shrinkage factor

    Returns:
        train_fn, predict_fn tuple
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    def train_fn(X_train, T_train, Y_train):
        T_binary = (
            (T_train > np.median(T_train)).astype(int)
            if len(np.unique(T_train)) > 2
            else T_train.astype(int)
        )

        propensity_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
        try:
            if len(np.unique(T_binary)) > 1:
                propensity_model.fit(X_train, T_binary)
            else:
                propensity_model = None
        except Exception:
            propensity_model = None

        outcome_model_0 = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
        outcome_model_1 = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )

        T_binary = T_binary.astype(bool)
        if np.sum(~T_binary) > 0:
            outcome_model_0.fit(X_train[~T_binary], Y_train[~T_binary])
        if np.sum(T_binary) > 0:
            outcome_model_1.fit(X_train[T_binary], Y_train[T_binary])

        return {
            "propensity": propensity_model,
            "outcome_0": outcome_model_0,
            "outcome_1": outcome_model_1,
            "T_median": np.median(T_train) if len(np.unique(T_train)) > 2 else 0.5,
        }

    def predict_fn(model, X):
        n = len(X)

        if model["propensity"] is not None:
            try:
                e_hat = model["propensity"].predict_proba(X)[:, 1]
            except Exception:
                e_hat = np.full(n, 0.5)
        else:
            e_hat = np.full(n, 0.5)

        try:
            mu0_hat = model["outcome_0"].predict(X)
        except Exception:
            mu0_hat = np.zeros(n)

        try:
            mu1_hat = model["outcome_1"].predict(X)
        except Exception:
            mu1_hat = np.zeros(n)

        return e_hat, mu0_hat, mu1_hat

    return train_fn, predict_fn


def create_nuisance_functions(method: str = "linear", **kwargs):
    """
    Factory for nuisance model functions.

    Args:
        method: 'linear' for RidgeCV/LogisticRegressionCV,
                'gbm' for GradientBoosting
        **kwargs: Passed to the underlying factory

    Returns:
        train_fn, predict_fn tuple
    """
    if method == "linear":
        return create_simple_nuisance_functions()
    elif method == "gbm":
        return create_gbm_nuisance_functions(**kwargs)
    else:
        raise ValueError(f"Unknown nuisance method: {method}. Use 'linear' or 'gbm'.")
