"""
Causal Effect Estimation Module for JCCE.

This module implements the TARNet/DragonNet-style effect estimation heads
for estimating Average Treatment Effect (ATE) and Conditional Average
Treatment Effect (CATE) jointly with causal structure learning.

Architecture:
- Y(0) Head: Predicts potential outcome under no treatment
- Y(1) Head: Predicts potential outcome under treatment
- Propensity Head: Predicts treatment probability P(T=1|X,L)

Key insight: By using the learned causal structure A and latent confounders L,
we can estimate causal effects more accurately than standard methods.

References:
- Shalit et al. (2017): "Estimating individual treatment effect..."
- Shi et al. (2019): DragonNet
- 6-LLM Consensus (2024): JCCE Effect Estimation Design

v6.0: Initial implementation following CAUSAL_EFFECT_ESTIMATION_IMPLEMENTATION_PLAN.md
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple

import jax.numpy as jnp
from flax import linen as nn

# ============================================================================
# Effect Heads Module
# ============================================================================


class EffectHeads(nn.Module):
    """
    Effect estimation heads for CATE prediction.

    Implements TARNet-style dual heads for potential outcomes plus
    a propensity score head for targeted regularization.

    Attributes:
        head_hidden_dim: Hidden dimension for effect heads (default: 32)
        dropout_rate: Dropout rate for regularization (default: 0.1)
    """

    head_hidden_dim: int = 32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, augmented_rep: jnp.ndarray, training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through effect heads.

        Args:
            augmented_rep: (batch_size, hidden_dim + latent_dim)
                          Concatenation of encoder output and latent scores
            training: Whether in training mode (enables dropout)

        Returns:
            y0: (batch_size, 1) - Predicted potential outcome under T=0
            y1: (batch_size, 1) - Predicted potential outcome under T=1
            propensity: (batch_size, 1) - Predicted P(T=1|X,L)
        """
        # Y(0) head - potential outcome without treatment
        y0 = nn.Dense(self.head_hidden_dim, name="y0_dense1")(augmented_rep)
        y0 = nn.relu(y0)
        y0 = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(y0)
        y0 = nn.Dense(1, name="y0_dense2")(y0)

        # Y(1) head - potential outcome with treatment
        y1 = nn.Dense(self.head_hidden_dim, name="y1_dense1")(augmented_rep)
        y1 = nn.relu(y1)
        y1 = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(y1)
        y1 = nn.Dense(1, name="y1_dense2")(y1)

        # Propensity head - P(T=1|X,L)
        # Simpler architecture (single linear layer + sigmoid)
        propensity_logit = nn.Dense(1, name="propensity_dense")(augmented_rep)
        propensity = nn.sigmoid(propensity_logit)

        return y0, y1, propensity


# ============================================================================
# Effect Computation Functions
# ============================================================================


def compute_cate(y0: jnp.ndarray, y1: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Conditional Average Treatment Effect (CATE).

    CATE(x) = E[Y(1) - Y(0) | X=x]

    Args:
        y0: Predicted potential outcome under T=0
        y1: Predicted potential outcome under T=1

    Returns:
        cate: Individual treatment effects (tau)
    """
    return y1 - y0


def compute_ate(cate: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Average Treatment Effect (ATE).

    ATE = E[Y(1) - Y(0)] = E[CATE(X)]

    Args:
        cate: Individual treatment effects

    Returns:
        ate: Population average treatment effect
    """
    return jnp.mean(cate)


def compute_att(cate: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Average Treatment Effect on the Treated (ATT).

    ATT = E[Y(1) - Y(0) | T=1]

    Args:
        cate: Individual treatment effects
        T: Treatment assignment

    Returns:
        att: Average effect among treated
    """
    treated_mask = T > 0.5
    n_treated = jnp.sum(treated_mask)
    att = jnp.where(n_treated > 0, jnp.sum(cate * treated_mask) / n_treated, 0.0)
    return att


# ============================================================================
# Loss Function Components
# ============================================================================


def compute_factual_outcome_loss(
    Y: jnp.ndarray, T: jnp.ndarray, y0: jnp.ndarray, y1: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute factual outcome loss.

    Key insight: We only observe the factual outcome (Y|T), not counterfactual.
    - If T=1, we observe Y(1), so loss on y1
    - If T=0, we observe Y(0), so loss on y0

    Args:
        Y: Observed outcomes (n,)
        T: Treatment assignments (n,)
        y0: Predicted Y(0) (n, 1) or (n,)
        y1: Predicted Y(1) (n, 1) or (n,)

    Returns:
        L_outcome: Mean squared error on factual outcomes
    """
    # Ensure proper shapes
    y0_flat = y0.squeeze()
    y1_flat = y1.squeeze()

    # Select factual prediction based on treatment
    y_factual = T * y1_flat + (1 - T) * y0_flat

    # MSE loss
    L_outcome = jnp.mean((y_factual - Y) ** 2)

    return L_outcome


def compute_propensity_loss(
    T: jnp.ndarray, propensity: jnp.ndarray, epsilon: float = 1e-7
) -> jnp.ndarray:
    """
    Compute propensity score loss (Binary Cross-Entropy).

    Args:
        T: True treatment assignments (n,)
        propensity: Predicted P(T=1|X,L) (n, 1) or (n,)
        epsilon: Small constant for numerical stability

    Returns:
        L_propensity: BCE loss for propensity prediction
    """
    # Ensure proper shapes and clipping
    p = jnp.clip(propensity.squeeze(), epsilon, 1 - epsilon)

    # Binary cross-entropy
    L_propensity = -jnp.mean(T * jnp.log(p) + (1 - T) * jnp.log(1 - p))

    return L_propensity


def compute_targeted_regularization(
    Y: jnp.ndarray,
    T: jnp.ndarray,
    y0: jnp.ndarray,
    y1: jnp.ndarray,
    propensity: jnp.ndarray,
    epsilon_param: float = 0.0,
    clip_coef: float = 10.0,
) -> jnp.ndarray:
    """
    Compute targeted regularization loss (DragonNet style).

    This improves asymptotic efficiency of ATE estimation by adding
    a targeted residual term based on propensity scores.

    Args:
        Y: Observed outcomes (n,)
        T: Treatment assignments (n,)
        y0: Predicted Y(0)
        y1: Predicted Y(1)
        propensity: Predicted P(T=1|X,L)
        epsilon_param: Trainable fluctuation parameter (default: 0)
        clip_coef: Max absolute value for IPW coefficient

    Returns:
        L_targeted: Targeted regularization loss
    """
    # Flatten predictions
    y0_flat = y0.squeeze()
    y1_flat = y1.squeeze()
    p = jnp.clip(propensity.squeeze(), 1e-4, 1 - 1e-4)

    # CATE prediction
    tau_pred = y1_flat - y0_flat

    # Clever covariate (IPW-style)
    # h(x,t) = t/p(x) - (1-t)/(1-p(x))
    h_coef = T / p - (1 - T) / (1 - p)
    h_coef = jnp.clip(h_coef, -clip_coef, clip_coef)  # Stability

    # Targeted prediction
    # y_targeted = y0 + T * tau + epsilon * h
    y_targeted = y0_flat + T * tau_pred + epsilon_param * h_coef

    # MSE loss
    L_targeted = jnp.mean((Y - y_targeted) ** 2)

    return L_targeted


def compute_effect_losses(
    Y: jnp.ndarray,
    T: jnp.ndarray,
    y0: jnp.ndarray,
    y1: jnp.ndarray,
    propensity: jnp.ndarray,
    epsilon_param: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """
    Compute all effect estimation loss components.

    Args:
        Y: Observed outcomes (n,)
        T: Treatment assignments (n,)
        y0: Predicted Y(0)
        y1: Predicted Y(1)
        propensity: Predicted P(T=1|X,L)
        epsilon_param: Trainable fluctuation parameter

    Returns:
        Dictionary with:
        - L_outcome: Factual outcome loss
        - L_propensity: Propensity score loss
        - L_targeted: Targeted regularization loss
        - ATE: Average treatment effect estimate
        - CATE_std: Standard deviation of individual effects
    """
    L_outcome = compute_factual_outcome_loss(Y, T, y0, y1)
    L_propensity = compute_propensity_loss(T, propensity)
    L_targeted = compute_targeted_regularization(Y, T, y0, y1, propensity, epsilon_param)

    # Effect statistics
    cate = compute_cate(y0.squeeze(), y1.squeeze())
    ate = compute_ate(cate)
    cate_std = jnp.std(cate)

    return {
        "L_outcome": L_outcome,
        "L_propensity": L_propensity,
        "L_targeted": L_targeted,
        "ATE": ate,
        "CATE_std": cate_std,
        "CATE": cate,
    }


# ============================================================================
# Treatment Selection Protocol
# ============================================================================


def select_treatment_candidates(
    A_learned: jnp.ndarray, Y_idx: int, threshold: float = 0.1, max_treatments: int = 5
) -> list:
    """
    Select candidate treatment variables from learned causal structure.

    Protocol:
    1. Use parents of Y (non-zero entries in A[:, Y_idx]) as candidates
    2. Filter by threshold on edge weight
    3. Limit to top-k by absolute weight

    Args:
        A_learned: Learned adjacency matrix (d, d)
        Y_idx: Index of target variable
        threshold: Minimum |A[i, Y_idx]| to consider
        max_treatments: Maximum number of treatments

    Returns:
        List of candidate treatment variable indices
    """
    # Get parent weights for Y
    parent_weights = jnp.abs(A_learned[:, Y_idx])

    # Filter by threshold and exclude Y itself
    candidate_mask = (parent_weights > threshold) & (jnp.arange(len(parent_weights)) != Y_idx)
    candidate_indices = jnp.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return []

    # Sort by weight (descending) and take top-k
    candidate_weights = parent_weights[candidate_indices]
    sorted_order = jnp.argsort(-candidate_weights)  # Descending
    top_k = min(max_treatments, len(candidate_indices))
    top_indices = candidate_indices[sorted_order[:top_k]]

    return [int(idx) for idx in top_indices]


def binarize_treatment(X: jnp.ndarray, treatment_idx: int, method: str = "median") -> jnp.ndarray:
    """
    Binarize a continuous treatment variable.

    Args:
        X: Input features (n_samples, d_features)
        treatment_idx: Index of treatment feature
        method: 'median', 'mean', or 'threshold:<value>'

    Returns:
        T: Binary treatment vector (n_samples,)
    """
    treatment_values = X[:, treatment_idx]

    if method == "median":
        threshold = jnp.median(treatment_values)
    elif method == "mean":
        threshold = jnp.mean(treatment_values)
    elif method.startswith("threshold:"):
        threshold = float(method.split(":")[1])
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    T = (treatment_values > threshold).astype(jnp.float32)

    return T


# ============================================================================
# Bow-Free Constraint (L matrix integration)
# ============================================================================


def compute_bow_free_penalty(A: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    Compute bow-free penalty to prevent simultaneous direct edge and confounding.

    A "bow" structure occurs when:
    - A[i,j] != 0 (direct edge from i to j)
    - AND L[:,i] and L[:,j] share strong latent factors (confounding)

    This violates identifiability (can't separate direct from confounded).

    Args:
        A: Adjacency matrix (d, d)
        U: Sample latent scores (n, k)
        V: Variable loadings (d, k)

    Returns:
        bow_penalty: Scalar penalty value
    """
    n_samples = U.shape[0]

    # Reconstruct L confounding matrix
    L = U @ V.T  # (n, d)

    # Compute confounding correlation matrix
    # Omega[i,j] = correlation of L[:,i] and L[:,j]
    Omega = (L.T @ L) / n_samples  # (d, d)

    # Normalize to correlation-like scale
    diag = jnp.sqrt(jnp.diag(Omega) + 1e-8)
    Omega_normalized = Omega / (diag[:, None] * diag[None, :] + 1e-8)

    # Bow penalty: penalize when both A[i,j] and Omega[i,j] are large
    bow_penalty = jnp.sum(jnp.abs(A) * jnp.abs(Omega_normalized))

    return bow_penalty


# ============================================================================
# Training Phase Protocol
# ============================================================================


@dataclass
class TrainingPhase:
    """Training phase configuration."""

    name: str
    effect_weight_scale: float
    structure_lr_scale: float
    confounders_lr_scale: float


def get_training_phase(current_step: int, total_steps: int) -> TrainingPhase:
    """
    Determine training phase based on progress.

    Phase 1 (Warmup): 0-30% of steps
        - Learn A, U, V with classification + structure losses
        - Effect heads frozen (lambda=0)

    Phase 2 (Effects): 30-70% of steps
        - Introduce effect heads with reduced weights
        - Structure learning LR reduced

    Phase 3 (Joint): 70-100% of steps
        - All losses at full strength
        - Fine-tune everything together

    Args:
        current_step: Current training iteration
        total_steps: Total training iterations

    Returns:
        TrainingPhase with configuration
    """
    progress = current_step / total_steps if total_steps > 0 else 1.0

    if progress < 0.30:
        return TrainingPhase(
            name="warmup", effect_weight_scale=0.0, structure_lr_scale=1.0, confounders_lr_scale=1.0
        )
    elif progress < 0.70:
        return TrainingPhase(
            name="effects",
            effect_weight_scale=0.5,
            structure_lr_scale=0.1,
            confounders_lr_scale=0.5,
        )
    else:
        return TrainingPhase(
            name="joint", effect_weight_scale=1.0, structure_lr_scale=0.1, confounders_lr_scale=0.5
        )


# ============================================================================
# Adaptive Curriculum
# ============================================================================


@dataclass
class AdaptiveCurriculumState:
    """
    Tracks convergence signals for adaptive phase transitions (Option D).

    Instead of fixed phase thresholds (40%, 80%), phase transitions are
    triggered by convergence signals:
    - Phase 1 → 2: Structure stabilized (h(A) < threshold or recon plateau)
    - Phase 2 → 3: Classification stabilized (class plateau or accuracy > threshold)

    This adapts to different datasets - easy problems transition faster,
    hard problems get more structure learning time.
    """

    current_phase: int = 1  # 1=structure, 2=classification, 3=effects
    phase_start_iter: int = 0

    # Loss histories for plateau detection
    recon_history: list = None
    class_history: list = None
    effect_history: list = None
    h_A_history: list = None

    # Convergence thresholds
    plateau_window: int = 10
    plateau_threshold: float = 0.05  # 5% relative variance = plateau
    min_phase_iters: int = 10  # Minimum iterations per phase
    max_phase1_ratio: float = 0.5  # Don't spend more than 50% in phase 1
    max_phase2_ratio: float = 0.8  # Transition to phase 3 by 80% at latest

    # Convergence thresholds for phase transitions
    h_A_threshold: float = 0.1  # Structure converged when h(A) < this
    accuracy_threshold: float = 0.70  # Classification "good enough" when acc > this

    def __post_init__(self):
        """Initialize history lists if None."""
        if self.recon_history is None:
            self.recon_history = []
        if self.class_history is None:
            self.class_history = []
        if self.effect_history is None:
            self.effect_history = []
        if self.h_A_history is None:
            self.h_A_history = []


def detect_plateau(history: list, window: int, threshold: float) -> bool:
    """
    Detect if loss has plateaued (low relative variance over window).

    Args:
        history: List of loss values
        window: Number of recent values to consider
        threshold: Maximum relative variance to be considered a plateau

    Returns:
        True if loss has plateaued
    """
    if len(history) < window:
        return False

    recent = history[-window:]
    mean_val = sum(recent) / len(recent)

    if mean_val < 1e-6:
        return True  # Near-zero is converged

    # Compute relative variance (coefficient of variation)
    variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)
    std_dev = variance**0.5
    relative_var = std_dev / (abs(mean_val) + 1e-8)

    return relative_var < threshold


def update_adaptive_curriculum(
    state: AdaptiveCurriculumState,
    current_iter: int,
    max_iter: int,
    recon_loss: float,
    class_loss: float,
    effect_loss: float,
    h_A: float,
    accuracy: float,
    verbose: bool = False,
    n_vars: int = 12,
) -> tuple:
    """
    Update curriculum state and return phase weights.

    This implements the adaptive curriculum logic:
    - Phase 1 (Structure): Focus on learning A, w_effect = 0
    - Phase 2 (Classification): Balance structure and classification, w_effect = 0.3
    - Phase 3 (Effects): Focus on effects, w_effect = 1.0

    Transitions are triggered by convergence signals, not fixed iteration counts.

    Args:
        state: Current curriculum state
        current_iter: Current training iteration
        max_iter: Maximum training iterations
        recon_loss: Current reconstruction loss
        class_loss: Current classification loss
        effect_loss: Current effect estimation loss
        h_A: Current DAG constraint value
        accuracy: Current classification accuracy
        verbose: Print phase transitions

    Returns:
        (updated_state, (w_recon, w_class, w_effect))
    """
    # Update histories
    state.recon_history.append(float(recon_loss))
    state.class_history.append(float(class_loss))
    state.effect_history.append(float(effect_loss))
    state.h_A_history.append(float(h_A))

    iters_in_phase = current_iter - state.phase_start_iter
    progress = current_iter / max_iter if max_iter > 0 else 1.0

    # Phase 1 → 2: Structure → Classification
    if state.current_phase == 1:
        # Require h(A) < 0.3 for transition, even if plateau detected
        # This prevents premature transition before structure converges
        # Scale h_A threshold with dimension: h(A)=0.3 is fine for d=12 but too lenient
        # for d=30 where it still indicates significant cyclicity
        h_A_limit = max(0.1, 0.3 / max(1, n_vars / 12))
        h_A_reasonable = (h_A >= 0.0) and (h_A < h_A_limit)  # Must be valid (>=0) and converged

        # Convergence signals (h(A) must be non-negative — negative means DAGMA domain violation)
        structure_converged = (
            (h_A >= 0.0 and h_A < state.h_A_threshold)  # Strict convergence (valid h)
            or (
                detect_plateau(state.recon_history, state.plateau_window, state.plateau_threshold)
                and h_A_reasonable
            )  # Plateau only counts if h(A) is reasonable
        )

        # Forced transition if we've spent too long in phase 1
        forced_transition = progress >= state.max_phase1_ratio

        # Transition if converged (and min iters met) OR forced
        if (structure_converged and iters_in_phase >= state.min_phase_iters) or forced_transition:
            state.current_phase = 2
            state.phase_start_iter = current_iter
            if verbose:
                if forced_transition:
                    reason = "forced"
                elif h_A < state.h_A_threshold:
                    reason = f"h(A)={h_A:.4f}"
                else:
                    reason = f"plateau+h(A)={h_A:.4f}"
                print(f"  [Curriculum] Phase 1→2 at iter {current_iter} ({reason})")

    # Phase 2 → 3: Classification → Effects
    elif state.current_phase == 2:
        # Convergence signals
        classification_converged = accuracy > state.accuracy_threshold or detect_plateau(
            state.class_history, state.plateau_window, state.plateau_threshold
        )

        # Don't transition if structure has degraded significantly
        structure_stable = h_A < 0.5

        # Forced transition if we've spent too long in phase 1+2
        forced_transition = progress >= state.max_phase2_ratio

        # Transition if converged (and min iters met) OR forced
        if (
            classification_converged
            and structure_stable
            and iters_in_phase >= state.min_phase_iters
        ) or forced_transition:
            state.current_phase = 3
            state.phase_start_iter = current_iter
            if verbose:
                reason = "forced" if forced_transition else f"acc={accuracy:.2%}"
                print(f"  [Curriculum] Phase 2→3 at iter {current_iter} ({reason})")

    # Return weights based on current phase
    if state.current_phase == 1:
        return state, (
            1.0,
            0.5,
            0.0,
        )  # Focus on structure (w_class=0.5 enables classification gradients through A)
    elif state.current_phase == 2:
        return state, (0.3, 1.0, 0.3)  # Focus on classification, introduce effects
    else:
        # Maintain structure weight at 0.5 to prevent h(A) degradation
        # Previous (0.1, 0.5, 1.0) allowed DAG constraint to drift
        return state, (0.5, 0.3, 1.0)  # Focus on effects while maintaining structure


def get_phase_weights_fixed(progress: float, phase_splits: tuple = (0.4, 0.8)) -> tuple:
    """
    Get phase weights using fixed thresholds (fallback).

    This is the original curriculum logic for comparison/fallback.

    Args:
        progress: Training progress (0.0 to 1.0)
        phase_splits: (phase1_end, phase2_end) as fractions of total iterations.
            Default (0.4, 0.8) = 40% structure, 40% classification, 20% effects.
            Use (0.333, 0.667) for equal thirds.

    Returns:
        (w_recon, w_class, w_effect)
    """
    p1, p2 = phase_splits
    if progress < p1:
        return (
            1.0,
            0.5,
            0.0,
        )  # Phase 1: Structure (w_class=0.5 enables classification gradients through A)
    elif progress < p2:
        return (0.3, 1.0, 0.3)  # Phase 2: Classification
    else:
        # Maintain structure weight at 0.5 to prevent h(A) degradation
        return (0.5, 0.3, 1.0)  # Phase 3: Effects (maintain structure)


# ============================================================================
# Valid Adjustment Set Computation (6-LLM Consensus Fix)
# ============================================================================


def compute_valid_adjustment_sets(
    A_weighted: jnp.ndarray,
    Y_idx: int,
    threshold: float = 0.05,
) -> dict:
    """
    Compute valid adjustment sets using the backdoor criterion.

    This is the KEY FIX identified by 6 LLMs for the effect direction problem.

    For each potential treatment T:
      Valid adjustment set = Parents(T) - Descendants(T) - {T, Y}

    IMPORTANT: For exogenous variables (no parents), the valid adjustment set
    is EMPTY. This is CORRECT - we should use an unadjusted diff-in-means
    estimator for root nodes (like Genetics in LUCAS).

    Args:
        A_weighted: Weighted adjacency matrix (d, d) where A[i,j] = edge i→j
        Y_idx: Index of outcome variable
        threshold: Edge weight threshold for binarization (default 0.05)
                   v11.4: Lowered from 0.15 because X→X edges are weaker than X→Y
                   (X→Y edges ~0.1-0.2, X→X edges ~0.01-0.05 due to sparsity)

    Returns:
        Dictionary mapping treatment_idx → set of valid covariate indices

    References:
        - Pearl (2009): Causality - Backdoor criterion
        - 6-LLM Consensus (2024): JCCE Effect Estimation Fix
    """
    import numpy as np  # Use numpy for graph algorithms

    # Convert to numpy for easier manipulation
    A_np = np.array(A_weighted)
    n_vars = A_np.shape[0]

    # Binarize with threshold (consensus: 0.1-0.3)
    A_binary = (np.abs(A_np) > threshold).astype(int)

    adjustment_sets = {}

    for treatment_idx in range(n_vars):
        if treatment_idx == Y_idx:
            adjustment_sets[treatment_idx] = set()
            continue

        # Find parents of treatment: nodes j where A[j, treatment_idx] = 1
        parents_T = set(np.where(A_binary[:, treatment_idx] > 0)[0])

        # Find descendants of treatment using BFS/reachability
        # A descendant is reachable following directed edges from T
        descendants_T = _get_descendants(A_binary, treatment_idx)

        # Valid adjustment = Parents(T) - Descendants(T) - {T, Y}
        valid_set = parents_T - descendants_T - {treatment_idx, Y_idx}

        adjustment_sets[treatment_idx] = valid_set

    return adjustment_sets


def _get_descendants(A_binary: "np.ndarray", node: int) -> set:
    """
    Get all descendants of a node using BFS on the adjacency matrix.

    A[i,j] = 1 means i → j, so descendants are reachable by following
    outgoing edges from the node.

    Args:
        A_binary: Binary adjacency matrix
        node: Source node index

    Returns:
        Set of descendant node indices
    """
    import numpy as np

    n = A_binary.shape[0]
    descendants = set()
    queue = []

    # Start with direct children
    children = np.where(A_binary[node, :] > 0)[0]
    for c in children:
        if c != node:  # Avoid self-loops
            queue.append(c)

    # BFS to find all descendants
    visited = {node}
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        descendants.add(current)

        # Add children of current
        children = np.where(A_binary[current, :] > 0)[0]
        for c in children:
            if c not in visited:
                queue.append(c)

    return descendants


def get_covariate_indices(
    adjustment_set: set,
    treatment_idx: int,
    Y_idx: int,
    n_features: int,
    min_covariates: int = 0,
) -> list:
    """
    Convert adjustment set to list of covariate indices.

    For exogenous variables (empty adjustment set), returns empty list.
    This is CORRECT - diff-in-means is the right estimator.

    Args:
        adjustment_set: Set of valid covariate indices
        treatment_idx: Treatment variable index
        Y_idx: Outcome variable index
        n_features: Total number of features
        min_covariates: Minimum number of covariates (for fallback)

    Returns:
        List of covariate indices to use in effect estimation
    """
    # The valid adjustment set is already filtered correctly
    covariate_indices = sorted(list(adjustment_set))

    # If empty and min_covariates > 0, fall back to using nothing
    # This is actually CORRECT for exogenous variables!
    # Empty means "no confounding to adjust for"

    return covariate_indices


# ============================================================================
# Default Lambda Values
# ============================================================================

DEFAULT_EFFECT_LAMBDAS = {
    # Effect estimation losses
    "outcome": 1.0,  # Factual outcome MSE
    "propensity": 0.1,  # Propensity BCE (lower to prevent domination)
    "targeted": 1.0,  # Targeted regularization
    # Latent confounder losses (from L extension)
    "nuclear": 0.01,  # Nuclear norm (low-rank)
    "bow": 0.1,  # Bow-free penalty
}


# ============================================================================
# Utility Functions
# ============================================================================


class EffectEstimates(NamedTuple):
    """Container for effect estimation results."""

    ATE: float
    CATE: jnp.ndarray
    CATE_std: float
    y0: jnp.ndarray
    y1: jnp.ndarray
    propensity: jnp.ndarray


def estimate_effects_from_outputs(
    y0: jnp.ndarray, y1: jnp.ndarray, propensity: jnp.ndarray
) -> EffectEstimates:
    """
    Compute effect estimates from model outputs.

    Args:
        y0: Predicted potential outcomes under T=0
        y1: Predicted potential outcomes under T=1
        propensity: Predicted treatment probabilities

    Returns:
        EffectEstimates named tuple
    """
    cate = compute_cate(y0.squeeze(), y1.squeeze())
    ate = float(compute_ate(cate))
    cate_std = float(jnp.std(cate))

    return EffectEstimates(
        ATE=ate, CATE=cate, CATE_std=cate_std, y0=y0, y1=y1, propensity=propensity
    )
