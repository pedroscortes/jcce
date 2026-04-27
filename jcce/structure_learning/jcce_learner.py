"""
JCCE Learner: Structure learning with processor-agnostic neural causal discovery.

This module provides learn_structure() for GOLEM-based causal structure learning
with any processor adapter (MLP, Transformer, Mamba, ELM, GNN).

Usage:
    from jcce.structure_learning.jcce_learner import learn_structure, create_processor

    processor = create_processor('mamba', key, d_model=128, ...)

    A_est, processor, params, metrics = learn_structure(
        data, Y, Y_idx, processor, key, ...
    )
"""

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jax import random

from .effect_estimation import (
    AdaptiveCurriculumState,
    compute_valid_adjustment_sets,
    get_phase_weights_fixed,
    update_adaptive_curriculum,
)

# GPS-DragonNet with variable type detection for proper causal effects
from .gps_dragonnet import (
    compute_xx_effect_unified,  # Auto-detects treatment type
    )
from .processor_adapters import (
    ELMAdapter,
    GNNAdapter,
    MambaAdapter,
    MLPAdapter,
    TransformerAdapter,
)

from jcce.counterfactual.sparsity import ste_hard_parents

# ============================================================================
# Helper functions for joint processor param optimization
# ============================================================================
# Processor params contain non-array metadata (tree_def, shapes, n_inputs, etc.)
# that optax can't handle. These functions extract/merge trainable arrays.


def extract_trainable_params(processor_params):
    """
    Extract only trainable JAX arrays from processor params.

    Filters out metadata like tree_def, shapes, n_inputs, _weights_solved, etc.
    that would cause optax initialization to fail.

    Args:
        processor_params: List of param dicts (one per variable)

    Returns:
        List of filtered param dicts containing only trainable arrays
    """

    def is_trainable(x):
        # Only float arrays are trainable (not integers, booleans, objects).
        # Also accepts numpy arrays (from artifact storage conversion).
        import numpy as np

        return isinstance(x, (jnp.ndarray, np.ndarray)) and x.dtype in [
            jnp.float32,
            jnp.float64,
            jnp.float16,
            jnp.bfloat16,
            np.float32,
            np.float64,
            np.float16,
        ]

    def extract_from_dict(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                # Recursively handle nested dicts (like Flax params)
                nested = extract_from_dict(v)
                if nested:  # Only add if not empty
                    result[k] = nested
            elif is_trainable(v):
                result[k] = v
            # Skip non-trainable: tree_def, shapes, n_inputs, _weights_solved, etc.
        return result

    return [extract_from_dict(p) for p in processor_params]


def merge_trained_params(original_params, trained_params):
    """
    Merge trained arrays back into original params structure.

    Combines trained arrays with original metadata (tree_def, shapes, etc.)
    so params can be used for forward passes.

    Args:
        original_params: List of original param dicts with metadata
        trained_params: List of trained param dicts (arrays only)

    Returns:
        List of merged param dicts ready for forward pass
    """

    def merge_dicts(orig, trained):
        result = orig.copy()
        for k, v in trained.items():
            if isinstance(v, dict) and k in result and isinstance(result[k], dict):
                result[k] = merge_dicts(result[k], v)
            else:
                result[k] = v
        return result

    return [merge_dicts(orig, trained) for orig, trained in zip(original_params, trained_params)]


# ============================================================================
# MEMORY OPTIMIZATION: Processor-Specific Batch Sizes
# ============================================================================
# JAX compiles functions based on input shapes. Using a fixed batch size ensures:
# 1. Compilation happens once with small shape
# 2. All subsequent batches reuse the compiled function
# 3. Avoids OOM during compilation on large datasets (16k+ samples)
# 4. For datasets smaller than batch size, uses full dataset
#
# Processor-specific batch sizes based on memory requirements:
# - ELM: No backprop through processor → lowest memory → 2048
# - MLP: Backprop + large architectures → same as Transformer → 256
# - Transformer: Attention mechanism (O(n²)) → high memory → 256
# - GNN: Graph operations → moderate memory → 512
# - Mamba: State-space model → very high memory → 256

PROCESSOR_BATCH_SIZES = {
    "elm": 512,  # Reduced from 2048 → better for multi-GPU + large datasets
    "mlp": 256,  # 4× reduction from 1024 → MLP Large needs same as Transformer
    "transformer": 256,  # 4× reduction → handles attention matrices
    "gnn": 512,  # Graph ops = moderate memory (2× reduction)
    "mamba": 256,  # 4× reduction → state-space is memory-hungry
}


def get_batch_size(processor_type: str) -> int:
    """Get processor-specific batch size for memory efficiency."""
    return PROCESSOR_BATCH_SIZES.get(processor_type.lower(), 1024)


# ============================================================================
# Processor Factory
# ============================================================================


def create_processor(processor_type: str, key: random.PRNGKey, **kwargs):
    """
    Factory function to create processor adapters.

    Args:
        processor_type: One of ['mlp', 'transformer', 'mamba', 'elm', 'gnn']
        key: JAX random key
        **kwargs: Processor-specific configuration

    Returns:
        processor: Processor adapter instance

    Examples:
        >>> processor = create_processor('mlp', key, hidden_dim=64, n_layers=2)
        >>> processor = create_processor('mamba', key, d_model=128, d_state=16)
    """
    processor_type = processor_type.lower()

    if processor_type == "mlp":
        return MLPAdapter(
            hidden_dim=kwargs.get("hidden_dim", 64),
            n_layers=kwargs.get("n_layers", 2),
            activation=kwargs.get("activation", "relu"),
            key=key,
        )

    elif processor_type == "transformer":
        return TransformerAdapter(
            d_model=kwargs.get("d_model", 128),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            d_ff=kwargs.get("d_ff", 512),
            key=key,
        )

    elif processor_type == "mamba":
        return MambaAdapter(
            d_model=kwargs.get("d_model", 128),
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand=kwargs.get("expand", 2),
            key=key,
            n_features=kwargs.get("n_features", None),  # Auto-scale
        )

    elif processor_type == "elm":
        return ELMAdapter(
            hidden_dim=kwargs.get("hidden_dim", 32),
            n_hidden_nodes=kwargs.get("n_hidden_nodes", 128),
            activation=kwargs.get("activation", "tanh"),
            key=key,
        )

    elif processor_type == "dag_transformer":
        from jcce.structure_learning.processor_adapters import DAGAttentionAdapter

        return DAGAttentionAdapter(
            d_model=kwargs.get("d_model", 64),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            d_ff=kwargs.get("d_ff", 256),
            temperature=kwargs.get("temperature", 5.0),
            key=key,
        )

    elif processor_type == "causal_mamba":
        from jcce.structure_learning.processor_adapters import CausalMambaAdapter

        return CausalMambaAdapter(
            d_model=kwargs.get("d_model", 128),
            d_state=kwargs.get("d_state", 16),
            d_conv=kwargs.get("d_conv", 4),
            expand=kwargs.get("expand", 2),
            key=key,
            n_features=kwargs.get("n_features", None),
        )

    elif processor_type == "gnn":
        return GNNAdapter(
            hidden_dim=kwargs.get("hidden_dim", 64),
            n_layers=kwargs.get("n_layers", 2),
            gnn_type=kwargs.get("gnn_type", "gcn"),
            sage_aggregation=kwargs.get("sage_aggregation", "mean"),
            key=key,
        )

    else:
        raise ValueError(
            f"Unknown processor type: {processor_type}. "
            f"Choose from: mlp, transformer, mamba, elm, gnn"
        )


# ============================================================================
# GOLEM-Unified Forward Model (Generic)
# ============================================================================


# MEMORY OPTIMIZATION: No gradient checkpointing (for now)
# Fixed batch-size compilation (FIXED_BATCH_SIZE=4096) is the primary memory optimization
# Checkpointing can be added later if memory usage is still high (>20GB)
def golem_unified_forward(
    X: jnp.ndarray, A: jnp.ndarray, processor, processor_params: list, threshold: float = 1e-6
) -> jnp.ndarray:
    """
    Reconstruct data using processor-based SCM.

    For each variable j:
        X_j_recon = processor(X[:, parents_j], processor_params[j])

    Special handling for GNN:
        - Uses learned adjacency matrix A for message passing
        - Masks target column to avoid circular dependency
        - Normalizes adjacency for stable gradients

    Args:
        X: (n_samples, n_vars) observed data
        A: (n_vars, n_vars) adjacency matrix (A[i,j] > 0 means edge i → j)
        processor: Processor adapter (MLP/Transformer/Mamba/ELM/GNN)
        processor_params: List of parameter dicts, one per variable
        threshold: Edge weight threshold

    Returns:
        X_recon: (n_samples, n_vars) reconstructed data
    """
    n_samples, n_vars = X.shape
    X_recon = jnp.zeros_like(X)

    # Detect structure-aware processors (need A passed to forward).
    # GNN: receives row-softmax-normalized A with target row/col zeroed.
    # DAG-Attention / CausalMamba: receive raw A (soft mask / topo reorder
    # are computed inside the adapter).
    proc_name = processor.__class__.__name__
    is_gnn = proc_name == "GNNAdapter"
    is_dag_attn = proc_name == "DAGAttentionAdapter"
    is_causal_mamba = proc_name == "CausalMambaAdapter"

    for j in range(n_vars):
        # Soft weighting by adjacency matrix (allows gradient flow)
        weights = jnp.abs(A[:, j])  # (n_vars,)
        X_weighted = X * weights[jnp.newaxis, :]  # (n_samples, n_vars)

        # Forward through processor
        if is_gnn:
            # GNN-SPECIFIC: Pass learned adjacency matrix for message passing
            # Mask target column (can't use X_j to predict X_j!)
            A_gnn = A.at[:, j].set(0.0)  # Zero out column j
            A_gnn = A_gnn.at[j, :].set(0.0)  # Zero out row j (symmetry)

            # Normalize adjacency for stable gradients (row-wise softmax)
            # Add small epsilon to avoid division by zero
            A_normalized = jax.nn.softmax(A_gnn + 1e-8, axis=0)

            # Pass adjacency to GNN
            X_j_recon = processor.forward(X_weighted, processor_params[j], A=A_normalized)
        elif is_dag_attn:
            # DAG-Attention: raw A drives the soft attention mask inside the adapter.
            X_j_recon = processor.forward(X_weighted, processor_params[j], A=A)
        elif is_causal_mamba:
            # CausalMamba: raw A drives the (jit-safe pure_callback) topological
            # sort inside the adapter; SSM scans variables in causal order.
            X_j_recon = processor.forward(X_weighted, processor_params[j], A=A)
        else:
            # Standard processors (MLP, Transformer, Mamba, ELM)
            X_j_recon = processor.forward(X_weighted, processor_params[j])

        X_recon = X_recon.at[:, j].set(X_j_recon)

    return X_recon


# ============================================================================
# Helper Functions: Focused Structure Learning
# ============================================================================


def get_y_neighborhood(
    A: jnp.ndarray, Y_idx: int, radius: int = 2, threshold: float = 1e-6
) -> jnp.ndarray:
    """
    Get variables within 'radius' hops of Y in the causal graph.

    This focuses structure learning on Y's local neighborhood instead of
    the full graph, providing significant speedup for classification tasks.

    Args:
        A: (n_vars, n_vars) adjacency matrix
        Y_idx: Index of target variable Y
        radius: How many hops away to include (default: 2)
        threshold: Edge weight threshold

    Returns:
        Array of variable indices in Y's neighborhood
    """
    n_vars = A.shape[0]
    relevant = set([Y_idx])

    for _ in range(radius):
        new_vars = set()
        for i in relevant:
            # Add parents (incoming edges to i)
            parents = jnp.where(jnp.abs(A[:, i]) > threshold)[0]
            new_vars.update(parents.tolist())
            # Add children (outgoing edges from i)
            children = jnp.where(jnp.abs(A[i, :]) > threshold)[0]
            new_vars.update(children.tolist())
        relevant.update(new_vars)

    return jnp.array(sorted(list(relevant)))


def extract_markov_blanket(A: jnp.ndarray, Y_idx: int, threshold: float = 1e-6) -> jnp.ndarray:
    """
    Extract Markov Blanket of Y from adjacency matrix.

    MB(Y) = Parents(Y) ∪ Children(Y) ∪ Spouses(Y)
    where Spouses(Y) = Parents of Children of Y

    Args:
        A: (n_vars, n_vars) adjacency matrix (A[i,j] > 0 means edge i → j)
        Y_idx: Index of target variable Y
        threshold: Edge weight threshold

    Returns:
        Array of variable indices in Y's Markov Blanket (excluding Y itself)
    """
    # Parents of Y: A[:, Y_idx] > threshold
    parents = jnp.where(jnp.abs(A[:, Y_idx]) > threshold)[0]

    # Children of Y: A[Y_idx, :] > threshold
    children = jnp.where(jnp.abs(A[Y_idx, :]) > threshold)[0]

    # Spouses: Parents of children (excluding Y)
    spouses = []
    for child in children:
        child_parents = jnp.where(jnp.abs(A[:, child]) > threshold)[0]
        spouses.extend(child_parents.tolist())

    # Combine and remove Y itself
    mb = set(parents.tolist() + children.tolist() + spouses)
    mb.discard(Y_idx)

    return jnp.array(sorted(list(mb)))


# ============================================================================
# GOLEM-Unified Loss Function (Generic)
# ============================================================================


# MEMORY OPTIMIZATION: Gradient checkpointing for DAG constraint
# Reduces memory by 40-50% by recomputing forward pass during backward
@jax.checkpoint
def compute_dag_constraint_checkpointed(A: jnp.ndarray) -> float:
    """
    Compute DAG constraint with gradient checkpointing for memory efficiency.

    Memory savings: ~40-50% reduction in backward pass memory
    Compute cost: ~20-30% increase (recomputes forward during backward)

    Args:
        A: Adjacency matrix (n_vars, n_vars)

    Returns:
        h_A: DAG constraint value (0 if DAG, >0 if cyclic)
    """
    A_squared = A * A
    h_A = jnp.trace(jsp.linalg.expm(A_squared)) - A.shape[0]
    return h_A


@jax.checkpoint
def dag_constraint(A: jnp.ndarray, s: float = 1.0) -> float:
    """
    DAGMA DAG constraint using log-determinant formulation (Bello et al., NeurIPS 2022).

    MEMORY EFFICIENT: O(d²) memory vs O(d³) for matrix exponential.

    Formula: h(W) = -log det(sI - W⊙W) + d·log(s)
    where W⊙W is element-wise (Hadamard) square.

    h(W) = 0 iff W is a DAG.

    Uses adaptive s: s = max(s_default, λ_max(W⊙W) + 0.1) to ensure
    M = sI - W⊙W is positive definite (required precondition for DAGMA).
    Without this, PC warm-start with dense A_init can cause λ_max > s,
    making M non-positive-definite and h(A) negative (rewarding cyclicity).

    When sI - W⊙W is still not positive definite (numerical edge cases),
    falls back to a smooth spectral-based penalty.

    Args:
        A: Adjacency matrix (n_vars, n_vars)
        s: Minimum scale parameter (default 1.0)

    Returns:
        h_A: DAG constraint value (0 if DAG, >0 if cyclic)
    """
    d = A.shape[0]

    # Element-wise square (matches DAGMA paper: W⊙W)
    A_sq = A * A

    # Adaptive s: ensure s > λ_max(W⊙W) so M is positive definite
    # This is critical when A_init from PC warm-start has large entries
    max_eig = jnp.max(jnp.linalg.eigvalsh(A_sq))
    s_adaptive = jnp.maximum(s, max_eig + 0.1)

    # Compute M = sI - A⊙A
    M = s_adaptive * jnp.eye(d) - A_sq

    # Compute log determinant
    sign, logdet = jnp.linalg.slogdet(M)

    # DAGMA value when in M-matrix domain (sign > 0)
    h_dagma = -logdet + d * jnp.log(s_adaptive)

    # Smooth fallback when outside M-matrix domain (numerical edge cases):
    # Use tr(A⊙A)/s which provides gradient signal to reduce edge weights.
    h_fallback = jnp.trace(A_sq) / s_adaptive + d * 1.0

    h_A = jnp.where(sign > 0, h_dagma, h_fallback)

    return h_A


@partial(jax.jit, static_argnums=(1,))
def compute_dag_constraint_spectral(
    A: jnp.ndarray, num_iterations: int = 10, epsilon: float = 1e-6
) -> float:
    """
    Spectral radius DAG constraint via power iteration.

    Complexity: O(d² × num_iterations) vs O(d³) for DAGMA/matrix exp
    10× speedup for large graphs (d > 40)

    Theory: A graph is a DAG iff ρ(|A|) < 1
    We penalize: h(A) = ReLU(ρ(|A|) - (1 - ε))

    Args:
        A: (d, d) adjacency matrix
        num_iterations: Power iteration steps (10-15 sufficient)
        epsilon: Margin below 1 for constraint satisfaction

    Returns:
        h_A: Constraint value (0 if DAG, >0 if cyclic)
    """
    d = A.shape[0]
    A_abs = jnp.abs(A)

    # Initialize random vector (using deterministic init for reproducibility)
    v = jnp.ones(d) / jnp.sqrt(d)

    # Power iteration for dominant eigenvalue
    def power_step(v, _):
        v_new = A_abs @ v
        v_norm = jnp.linalg.norm(v_new) + 1e-10
        v_new = v_new / v_norm
        return v_new, v_norm

    v_final, norms = jax.lax.scan(power_step, v, None, length=num_iterations)

    # Rayleigh quotient estimate of spectral radius
    lambda_max = jnp.dot(v_final, A_abs @ v_final)

    # Constraint: penalize when λ_max ≥ 1 - ε
    h_A = jax.nn.relu(lambda_max - (1.0 - epsilon))

    return h_A


def hybrid_dag_constraint(
    A: jnp.ndarray, iteration: int, max_iter: int, use_exact_final: bool = True
) -> float:
    """
    Hybrid approach: Fast spectral during optimization, exact DAGMA for final.

    Expected speedup: 10× during iterations, exact guarantee at end.

    Args:
        A: Adjacency matrix
        iteration: Current iteration (0-indexed)
        max_iter: Maximum iterations
        use_exact_final: Whether to use exact DAGMA for final iteration

    Returns:
        h_A: DAG constraint value
    """
    if iteration < max_iter - 1 or not use_exact_final:
        # Fast spectral constraint (O(d²))
        return compute_dag_constraint_spectral(A, num_iterations=10)
    else:
        # Exact DAGMA for final validation (O(d³))
        return dag_constraint(A, s=1.0)


@jax.jit
def dynamic_pruning(
    A: jnp.ndarray,
    grads: jnp.ndarray,
    iteration: int,
    max_iter: int,
    prune_start: float = 0.15,
    prune_end_percentile: float = 30.0,
) -> jnp.ndarray:
    """
    Progressive pruning based on gradient and magnitude significance.

    Uses OR mask: keep edges that are either learning OR stable.
    Expected: 30-50% memory reduction, faster convergence.

    Args:
        A: Current adjacency matrix
        grads: Gradients w.r.t. A
        iteration: Current iteration (0-indexed)
        max_iter: Maximum iterations
        prune_start: Progress fraction at which to start pruning (default: 15%)
        prune_end_percentile: Final percentile cutoff (default: 30%)

    Returns:
        A_pruned: Pruned adjacency matrix
    """
    d = A.shape[0]
    progress = iteration / max_iter

    # Don't prune in early iterations (allow exploration)
    should_prune = progress >= prune_start

    # Calculate thresholds that increase over time
    # Early: keep 90% of edges, Late: keep 70% of edges
    percentile = 90.0 - (90.0 - prune_end_percentile) * jnp.clip(
        (progress - prune_start) / (1.0 - prune_start), 0.0, 1.0
    )

    # Gradient significance threshold
    grad_abs = jnp.abs(grads)
    grad_threshold = jnp.percentile(grad_abs.flatten(), percentile)

    # Magnitude significance threshold
    A_abs = jnp.abs(A)
    mag_threshold = jnp.percentile(A_abs.flatten(), percentile)

    # OR mask: keep edges that are either learning OR established
    # This preserves both active learning edges AND stable important edges
    mask = (grad_abs > grad_threshold) | (A_abs > mag_threshold)

    # Apply mask conditionally
    A_pruned = jnp.where(should_prune, A * mask.astype(jnp.float32), A)

    # Ensure no self-loops
    A_pruned = A_pruned * (1 - jnp.eye(d))

    return A_pruned


def get_adaptive_config(n_vars: int, processor_type: str) -> dict:
    """
    Get configuration adapted to dataset size and processor type.

    Based on empirical analysis:
    - Heart Disease (d=14): 107 min, 0 OOMs
    - Breast Cancer (d=31): 327 min, some OOMs
    - TEP (d=53): 800+ min, many OOMs

    Args:
        n_vars: Number of variables (features + target)
        processor_type: Processor type ('mlp', 'transformer', 'elm', 'gnn', 'mamba')

    Returns:
        Configuration dictionary with optimized parameters
    """
    # Base configuration
    config = {
        "population_size": 20,
        "generations": 15,
        "golem_iterations": 20,
        "batch_size": 256,
        "genomes_per_gpu": 2,
        "patience": 15,
        "use_spectral_constraint": False,
        "enable_pruning": False,
    }

    # Processor-specific memory factors
    processor_memory_factor = {
        "elm": 0.5,  # Fastest, lowest memory
        "mlp": 1.0,  # Baseline
        "gnn": 1.5,  # Higher due to message passing
        "transformer": 2.0,  # Attention maps are expensive
        "mamba": 1.8,  # State-space overhead
    }

    mem_factor = processor_memory_factor.get(processor_type.lower(), 1.0)
    effective_d = n_vars * mem_factor

    # Small datasets (d < 20)
    if effective_d < 20:
        config.update(
            {
                "golem_iterations": 15,  # Converges faster
                "patience": 10,
            }
        )

    # Medium datasets (20 <= d < 35)
    elif effective_d < 35:
        config.update(
            {
                "batch_size": 192,
                "use_spectral_constraint": True,  # Start using spectral
            }
        )

    # Large datasets (35 <= d < 50)
    elif effective_d < 50:
        config.update(
            {
                "population_size": 18,
                "golem_iterations": 18,
                "batch_size": 128,
                "genomes_per_gpu": 1,  # Sequential to avoid OOM
                "use_spectral_constraint": True,
                "enable_pruning": True,
            }
        )

    # Very large datasets (d >= 50)
    else:
        config.update(
            {
                "population_size": 15,
                "golem_iterations": 15,
                "batch_size": 64,
                "genomes_per_gpu": 1,
                "patience": 10,
                "use_spectral_constraint": True,
                "enable_pruning": True,
            }
        )

    return config


def compute_dag_constraint_auto(
    A: jnp.ndarray,
    use_spectral: bool = False,
    iteration: int = 0,
    max_iter: int = 100,
    use_exact_final: bool = True,
) -> float:
    """
    Automatically select DAG constraint based on configuration (v4.1).

    Args:
        A: Adjacency matrix (n_vars, n_vars)
        use_spectral: If True, use spectral constraint (O(d²)) during optimization
        iteration: Current iteration (for hybrid mode)
        max_iter: Maximum iterations (for hybrid mode)
        use_exact_final: If True and near end, use exact DAGMA for validation

    Returns:
        h_A: DAG constraint value
    """
    if use_spectral:
        # Use hybrid: spectral during optimization, exact at end
        return hybrid_dag_constraint(A, iteration, max_iter, use_exact_final)
    else:
        # Default: DAGMA (O(d²) memory, O(d³) computation)
        return dag_constraint(A, s=1.0)


@jax.checkpoint
def compute_dag_constraint_poly(A: jnp.ndarray, m: int = 10) -> float:
    """
    Polynomial approximation to DAG constraint.

    Faster than matrix exponential, but less accurate.
    Uses: h(A) = trace((I + A²/d)^m) - d

    Memory: O(d²) (similar to DAGMA)
    Speed: Faster than exp (no Padé approximation)
    Accuracy: Good for m ≥ d (use m=52 for 52 features)

    Args:
        A: Adjacency matrix (n_vars, n_vars)
        m: Polynomial degree (higher = more accurate, default 10)

    Returns:
        h_A: DAG constraint value
    """
    d = A.shape[0]
    A_squared = A * A

    # Compute (I + A²/d)^m via repeated squaring
    M = jnp.eye(d) + A_squared / d
    M_power = M
    for _ in range(m - 1):
        M_power = M_power @ M

    h_A = jnp.trace(M_power) - d
    return h_A


def golem_unified_loss(
    X: jnp.ndarray,
    A: jnp.ndarray,
    processor,
    processor_params: list,
    lambda_1: float = 0.1,
    lambda_2: float = 0.01,
    threshold: float = 1e-6,
) -> Tuple[float, float]:
    """
    GOLEM-Unified loss function.

    Loss = Reconstruction + Sparsity + DAG_constraint

    Args:
        X: (n_samples, n_vars) data
        A: (n_vars, n_vars) adjacency matrix
        processor: Processor adapter
        processor_params: Processor parameters for each variable
        lambda_1: Sparsity penalty weight
        lambda_2: DAG constraint penalty weight
        threshold: Edge threshold

    Returns:
        total_loss: Total loss
        h_A: DAG constraint value (should be ~0 for valid DAG)
    """
    # 1. Reconstruction loss (negative log-likelihood)
    X_recon = golem_unified_forward(X, A, processor, processor_params, threshold)
    reconstruction_loss = jnp.sum((X - X_recon) ** 2) / X.shape[0]

    # 2. Sparsity penalty (L1 on adjacency matrix)
    sparsity_loss = lambda_1 * jnp.sum(jnp.abs(A))

    # 3. DAG constraint (acyclicity)
    # h(A) = trace(exp(A ⊙ A)) - n
    A_squared = A * A
    h_A = jnp.trace(jsp.linalg.expm(A_squared)) - A.shape[0]
    dag_loss = lambda_2 * h_A

    # Total loss
    total_loss = reconstruction_loss + sparsity_loss + dag_loss

    return total_loss, h_A


# ============================================================================
# Multi-Task Loss Function (Structure + Classification)
# ============================================================================


def golem_unified_multitask_loss(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    A: jnp.ndarray,
    processor,
    processor_params: list,
    Y_idx: int,
    lambda_1: float = 0.1,
    lambda_2: float = 0.01,
    lambda_class: float = 1.0,
    threshold: float = 1e-6,
    focus_on_mb: bool = True,
    # Optimization parameters
    use_spectral_constraint: bool = False,
    iteration: int = 0,
    max_iter: int = 100,
) -> Tuple[float, float, float, float]:
    """
    v4.0 Multi-Task Loss: Joint Structure Learning + Classification.
    v4.1 Enhancement: Optional spectral DAG constraint for O(d²) performance.

    This is the KEY INNOVATION for v4.0: We optimize structure learning AND
    classification in a SINGLE unified loss function. The processor learns to:
    1. Reconstruct variables from their parents (structure learning)
    2. Predict Y from its Markov Blanket (classification)

    Loss = Reconstruction + Classification + Sparsity + DAG_constraint

    Args:
        X: (n_samples, n_vars) data (includes Y at Y_idx)
        Y: (n_samples,) target labels for classification
        A: (n_vars, n_vars) adjacency matrix
        processor: Processor adapter
        processor_params: Processor parameters for each variable
        Y_idx: Index of Y in X
        lambda_1: Sparsity penalty weight
        lambda_2: DAG constraint penalty weight
        lambda_class: Classification loss weight
        threshold: Edge threshold
        focus_on_mb: If True, only reconstruct Y's neighborhood (speedup)
        use_spectral_constraint: Use O(d²) spectral constraint (v4.1)
        iteration: Current iteration (for hybrid mode)
        max_iter: Maximum iterations (for hybrid mode)

    Returns:
        total_loss: Total loss
        reconstruction_loss: Reconstruction component
        classification_loss: Classification component
        h_A: DAG constraint value
    """
    n_samples, n_vars = X.shape

    # 1. Reconstruction loss (structure learning)
    if focus_on_mb:
        # SPEEDUP: Only reconstruct Y's Markov Blanket + Y itself
        mb_indices = extract_markov_blanket(A, Y_idx, threshold)
        # Add Y itself to reconstruction targets
        recon_vars = jnp.concatenate([mb_indices, jnp.array([Y_idx])])

        # Reconstruct only MB variables
        X_recon_mb = jnp.zeros((n_samples, len(recon_vars)))
        for i, j in enumerate(recon_vars):
            weights = jnp.abs(A[:, j])
            X_weighted = X * weights[jnp.newaxis, :]

            # Forward through processor
            if processor.__class__.__name__ == "GNNAdapter":
                A_normalized = A / (jnp.sum(jnp.abs(A), axis=0, keepdims=True) + 1e-8)
                X_j_recon = processor.forward(X_weighted, processor_params[j], A=A_normalized)
            else:
                X_j_recon = processor.forward(X_weighted, processor_params[j])

            X_recon_mb = X_recon_mb.at[:, i].set(X_j_recon)

        # Compare only reconstructed variables
        X_target = X[:, recon_vars]
        reconstruction_loss = jnp.sum((X_target - X_recon_mb) ** 2) / n_samples
    else:
        # Full graph reconstruction (v3.0 style)
        X_recon = golem_unified_forward(X, A, processor, processor_params, threshold)
        reconstruction_loss = jnp.sum((X - X_recon) ** 2) / n_samples

    # 2. Classification loss (NEW!)
    # Use learned processor f_Y to predict Y from its Markov Blanket
    mb_indices = extract_markov_blanket(A, Y_idx, threshold)

    if len(mb_indices) == 0:
        # Fallback: use all variables except Y
        mb_indices = jnp.array([i for i in range(n_vars) if i != Y_idx])

    # Get MB features with weights from A
    weights = jnp.abs(A[:, Y_idx])
    X_weighted = X * weights[jnp.newaxis, :]

    # Predict Y using learned processor
    if processor.__class__.__name__ == "GNNAdapter":
        A_normalized = A / (jnp.sum(jnp.abs(A), axis=0, keepdims=True) + 1e-8)
        Y_pred_continuous = processor.forward(X_weighted, processor_params[Y_idx], A=A_normalized)
    else:
        Y_pred_continuous = processor.forward(X_weighted, processor_params[Y_idx])

    # Classification loss: MSE between predicted and true Y
    # (For binary classification, Y is 0/1, so MSE works well)
    Y_continuous = Y.astype(jnp.float32)
    classification_loss = jnp.sum((Y_continuous - Y_pred_continuous) ** 2) / n_samples

    # 3. Sparsity penalty (L1 on adjacency matrix)
    sparsity_loss = lambda_1 * jnp.sum(jnp.abs(A))

    # 4. DAG constraint (acyclicity)
    # Use spectral constraint (O(d^2)) or DAGMA (default)
    # Spectral is faster but uses hybrid mode for accurate final validation
    h_A = compute_dag_constraint_auto(
        A,
        use_spectral=use_spectral_constraint,
        iteration=iteration,
        max_iter=max_iter,
        use_exact_final=True,
    )
    dag_loss = lambda_2 * h_A

    # Total loss: weighted combination
    total_loss = reconstruction_loss + lambda_class * classification_loss + sparsity_loss + dag_loss

    return total_loss, reconstruction_loss, classification_loss, h_A


# ============================================================================
# Joint Optimization of A + Processor Params + Uncertainty
# ============================================================================


def _learn_structure_legacy(
    data: jnp.ndarray,
    Y: jnp.ndarray,
    Y_idx: int,
    processor,
    key: random.PRNGKey,
    processor_type: str = "mlp",
    lambda_1: float = 0.02,
    lambda_2_init: float = 0.01,
    lambda_2_max: float = 1e10,
    lambda_class: float = 1.0,
    lr: float = 0.001,
    max_iter: int = 100,
    patience: int = 15,
    verbose: bool = False,
    A_init: Optional[jnp.ndarray] = None,
    # Optimization parameters
    use_spectral_constraint: bool = False,
    enable_pruning: bool = False,
    # Regularization parameters
    use_validation_split: bool = True,
    validation_ratio: float = 0.2,
    weight_decay: float = 1e-4,
    # Latent confounder parameters
    use_latent_confounders: bool = False,
    latent_rank_k: int = 5,
    lambda_L: float = 0.05,
    lambda_bow: float = 0.1,
    warm_start_L_iters: int = 20,
    task: str = "classification",  # 'classification' or 'regression'
) -> Tuple[jnp.ndarray, Any, list, Dict[str, Any]]:
    """
    v4.0 PROPER: Joint optimization of A + processor parameters + uncertainty weights.
    v4.1 Enhancement: Optional spectral DAG constraint and dynamic pruning.

    THIS IS THE TRUE END-TO-END LEARNING APPROACH.

    Key Innovation (based on research synthesis):
    - Optimizes BOTH adjacency matrix A AND processor parameters θ jointly
    - Uses uncertainty weighting (Kendall et al. 2018) for automatic loss balancing
    - Multi-task loss: MSE for features, BCE/MSE for target Y
    - Enables "Supervised Causal Discovery" hypothesis
    - v4.1: Spectral DAG constraint for O(d²) performance on large datasets
    - v4.1: Dynamic pruning to remove weak edges during optimization

    Args:
        data: (n_samples, n_vars) observed data (includes Y at Y_idx)
        Y: (n_samples,) target labels (classification) or values (regression)
        Y_idx: Index of Y in data
        processor: Processor adapter instance
        key: JAX random key
        processor_type: Processor type
        lambda_1: Sparsity penalty
        lambda_2_init: Initial DAG constraint penalty
        lambda_2_max: Maximum DAG constraint penalty
        lambda_class: Classification loss weight (relative to reconstruction)
        lr: Learning rate
        max_iter: Maximum iterations
        patience: Early stopping patience
        verbose: Print progress
        A_init: Optional initial adjacency matrix for warm-start (None = random init)
        use_spectral_constraint: Use O(d²) spectral constraint (v4.1)
        enable_pruning: Enable dynamic pruning during optimization (v4.1)
        use_validation_split: Use validation set for early stopping (v5.1, 6-LLM consensus)
        validation_ratio: Fraction of data to use for validation (default 0.2)
        weight_decay: L2 regularization on A matrix (default 1e-4)
        use_latent_confounders: Enable low-rank L for latent confounders (v6.0)
        latent_rank_k: Rank of L matrix (number of latent factors, default 5)
        lambda_L: Nuclear norm penalty weight on L (default 0.05)
        lambda_bow: Bow-free penalty weight (default 0.1)
        warm_start_L_iters: Iterations without L before warm-starting (default 20)

    Returns:
        A_est: (n_vars, n_vars) estimated adjacency matrix
        processor: Processor adapter (same instance)
        processor_params: List of trained parameters for each variable
        metrics: Training metrics dict
    """
    n_samples_total, n_vars = data.shape

    # Create validation split for proper early stopping
    if use_validation_split:
        key, split_key = random.split(key)
        n_val = int(n_samples_total * validation_ratio)
        n_train = n_samples_total - n_val

        # Shuffle indices
        perm = random.permutation(split_key, n_samples_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        data_train = data[train_idx]
        Y_train = Y[train_idx]
        data_val = data[val_idx]
        Y_val = Y[val_idx]

        n_samples = n_train  # For training
    else:
        data_train = data
        Y_train = Y
        data_val = None
        Y_val = None
        n_samples = n_samples_total

    # Memory optimization: processor-specific batch size
    FIXED_BATCH_SIZE = get_batch_size(processor_type)

    if n_samples <= FIXED_BATCH_SIZE:
        effective_batch_size = n_samples
        use_batching = False
        num_batches_per_iter = 1
    else:
        effective_batch_size = FIXED_BATCH_SIZE
        use_batching = True
        num_batches_per_iter = math.ceil(n_samples / FIXED_BATCH_SIZE)

    # Batching works with L via batch indices
    # U is sliced by batch indices: U_batch = U[batch_indices, :]
    # L_batch = U_batch @ V.T has shape (batch_size, n_vars)

    if verbose:
        processor_name = processor.__class__.__name__.replace("Adapter", "")
        print(f"\n{'=' * 60}")
        print("v5.1: Joint Optimization (A + θ + Uncertainty + Validation)")
        print(f"{'=' * 60}")
        if use_validation_split:
            print(f"Data: {n_samples_total} total → {n_samples} train / {n_val} val")
        else:
            print(f"Data: {n_samples} samples, {n_vars} variables")
        print(f"Target: Variable {Y_idx} (Y)")
        if weight_decay > 0:
            print(f"Weight decay: {weight_decay:.0e}")
        if use_batching:
            print(f"Batch size: {effective_batch_size} (FIXED)")
            print(f"Batches per epoch: {num_batches_per_iter}")
        else:
            print(f"Batch size: {effective_batch_size} (full dataset)")
        print(f"Processor: {processor_name}")
        print(f"Lambda_1 (sparsity): {lambda_1}")
        print(f"Lambda_2 (DAG): {lambda_2_init} -> {lambda_2_max}")
        print(f"Lambda_class (classification): {lambda_class}")
        print(f"Early stopping patience: {patience}")
        if use_spectral_constraint:
            print("v4.1: Spectral DAG constraint ENABLED (O(d²))")
        else:
            print("DAG Constraint: Matrix Exponential (standard)")
        if enable_pruning:
            print("v4.1: Dynamic pruning ENABLED")
        if use_latent_confounders:
            print(f"v6.0: Latent confounders ENABLED (k={latent_rank_k})")
            print(f"  λ_L (nuclear): {lambda_L}, λ_bow (bow-free): {lambda_bow}")
            print(f"  Warm-start: {warm_start_L_iters} iters before L")
        print("INNOVATION: Optimizing A AND processor params jointly!")

    # Initialize adjacency matrix A
    if A_init is not None:
        # Warm-start: use provided initialization
        A = jnp.array(A_init)
        if verbose:
            print(f"Warm-start: Using provided A_init (shape {A.shape})")
    else:
        # Cold start: random initialization
        key, init_key = random.split(key)
        A = (
            random.normal(init_key, (n_vars, n_vars)) * 0.1
        )  # Larger init for better gradient signal

        # Correlation warm-start for A[:,Y_idx]
        # Initialize edges to Y based on correlation with Y to break symmetry
        # This gives classification gradient a head start
        corr_with_Y = jnp.corrcoef(data.T)[Y_idx, :]  # Correlation of each feature with Y
        corr_with_Y = jnp.abs(corr_with_Y) * 0.5  # Scale to [0, 0.5] range
        corr_with_Y = corr_with_Y.at[Y_idx].set(0.0)  # No self-loop
        A = A.at[:, Y_idx].set(corr_with_Y)
        if verbose:
            n_corr_edges = int(jnp.sum(corr_with_Y > 0.05))
            print(f"Correlation warm-start: {n_corr_edges} features correlated with Y (|r|>0.1)")

    # Initialize processor parameters for each variable
    processor_params = []
    for j in range(n_vars):
        params = processor.init_params(n_vars)
        processor_params.append(params)

    # Initialize uncertainty weights (log_var_class removed — BCE fixed scale, curriculum + lambda_class suffice)
    log_var_recon = jnp.array(0.0)

    # Extract trainable params from processor_params
    # This filters out non-array metadata (tree_def, shapes, n_inputs, etc.)
    # that would cause optax initialization to fail
    trainable_proc_params = extract_trainable_params(processor_params)

    # Initialize latent confounder factors U, V
    # L = U @ V.T where U:(n_train*k), V:(n_vars*k)
    if use_latent_confounders:
        key, u_key, v_key = random.split(key, 3)
        # Small random initialization - will be overwritten by warm-start
        U = random.normal(u_key, (n_samples, latent_rank_k)) * 0.01
        V = random.normal(v_key, (n_vars, latent_rank_k)) * 0.01
    else:
        U = None
        V = None

    # Pack optimizable parameters (A + uncertainty weights + processor params + optional L)
    # Includes trainable processor params for joint optimization
    all_params = {
        "A": A,
        "log_var_recon": log_var_recon,
        "processor_params": trainable_proc_params,
    }
    if use_latent_confounders:
        all_params["U"] = U
        all_params["V"] = V

    # Augmented Lagrangian parameter
    lambda_2 = lambda_2_init

    # Initialize pruning mask
    pruning_mask = jnp.ones((n_vars, n_vars))

    # Helper function to compute residuals for warm-start
    def compute_residuals_for_warmstart(params, data_batch):
        """Compute residuals X_j - f(X * |A[:,j]|) for warm-starting L."""
        A_curr = params["A"]
        # Merge trained params with metadata for forward passes
        proc_params = merge_trained_params(processor_params, params["processor_params"])
        n_samp, n_v = data_batch.shape
        residuals = jnp.zeros((n_samp, n_v))

        for j in range(n_v):
            weights = jnp.abs(A_curr[:, j])
            # Mask out self to maintain causal correctness
            weights = weights.at[j].set(0.0)
            weight_sum = jnp.sum(weights)
            fallback_weights = jnp.ones(n_v).at[j].set(0.0)
            # Lower threshold to let A[:,j] grow before fallback
            weights = jnp.where(weight_sum > 0.01, weights, fallback_weights)
            X_weighted = data_batch * weights[jnp.newaxis, :]

            if processor.__class__.__name__ == "GNNAdapter":
                A_normalized = A_curr / (jnp.sum(jnp.abs(A_curr), axis=0, keepdims=True) + 1e-8)
                predicted = processor.forward(X_weighted, proc_params[j], A=A_normalized)
            else:
                predicted = processor.forward(X_weighted, proc_params[j])

            residuals = residuals.at[:, j].set(data_batch[:, j] - predicted.flatten())

        return residuals

    # Track warm-start phase
    L_active = not use_latent_confounders  # If L not used, always "active" (no warm-start needed)
    if use_latent_confounders:
        L_active = False  # Start with L inactive for warm-start

    # Define joint loss function
    def loss_fn(
        params,
        batch_data,
        batch_Y,
        lambda_2_current,
        current_iter,
        use_L_in_loss=True,
        batch_indices=None,
        rng_key=None,
    ):
        """
        Joint loss with uncertainty weighting.

        Args:
            use_L_in_loss: False for validation, True for training.
            batch_indices: For proper U slicing when batching with L.
            rng_key: For dropout during training (None disables dropout).

        Optimizes:
        - A (adjacency matrix)
        - log_var_recon (reconstruction uncertainty weight)
        - processor_params (jointly optimized with A)
        """
        training = rng_key is not None
        A_curr = params["A"]
        # Apply pruning mask if enabled
        if enable_pruning:
            A_curr = A_curr * pruning_mask
        # Merge trained params with metadata for forward passes
        proc_params = merge_trained_params(processor_params, params["processor_params"])
        log_var_r = params["log_var_recon"]

        # Compute L for latent confounders (only for training)
        # For validation, L is not used because it's sample-specific to training data
        # With batching: U_batch = U[batch_indices, :], L_batch = U_batch @ V.T
        if use_latent_confounders and use_L_in_loss:
            U_curr = params["U"]
            V_curr = params["V"]
            if batch_indices is not None:
                # Batching: slice U by batch indices
                U_batch = U_curr[batch_indices, :]  # (batch_size, k)
                L_batch = U_batch @ V_curr.T  # (batch_size, n_vars)
            else:
                # Full batch: use all of U
                L_batch = U_curr @ V_curr.T  # (n_samples, n_vars)
        else:
            L_batch = None

        n_batch, n_v = batch_data.shape

        # Compute reconstruction and classification losses
        total_recon_loss = 0.0
        classification_loss = 0.0

        for j in range(n_v):
            # Soft weighting by adjacency (parent selection)
            weights = jnp.abs(A_curr[:, j])

            # Mask out variable j when predicting j (prevents self-regression)
            # Even though A[j,j]=0 is enforced, the fallback below can override this.
            # A variable cannot cause itself (fundamental DAG property).
            weights = weights.at[j].set(0.0)

            # Different handling for Y vs other variables
            # For Y (classification): NO fallback - force processor to use A-weighted features
            #   This prevents the processor from learning feature importance internally
            #   and forces it to rely on A[:,Y] for feature selection.
            #   Correlation warm-start ensures A[:,Y] starts non-zero.
            # For other variables (reconstruction): Keep fallback to ensure stable training
            if j == Y_idx:
                # No fallback for Y - processor MUST use A-weighted features
                # Add small floor (0.01) to prevent complete zeroing
                weights = jnp.maximum(weights, 0.01)
            else:
                # Fallback for reconstruction variables
                weight_sum = jnp.sum(weights)
                fallback_weights = jnp.ones(n_v).at[j].set(0.0)
                weights = jnp.where(weight_sum > 0.01, weights, fallback_weights)

            X_weighted = batch_data * weights[jnp.newaxis, :]

            # Forward through processor
            # Pass training mode and rng_key to enable dropout for MLP/Transformer
            _proc_name = processor.__class__.__name__
            if _proc_name == "GNNAdapter":
                # GNN needs adjacency matrix (row-normalized)
                A_normalized = A_curr / (jnp.sum(jnp.abs(A_curr), axis=0, keepdims=True) + 1e-8)
                direct_effect = processor.forward(X_weighted, proc_params[j], A=A_normalized)
            elif _proc_name == "DAGAttentionAdapter":
                # DAG-Attention: raw A drives the soft mask; dropout via training/rng_key
                direct_effect = processor.forward(
                    X_weighted, proc_params[j], A=A_curr, training=training, rng_key=rng_key
                )
            elif _proc_name == "CausalMambaAdapter":
                # CausalMamba: raw A drives the jit-safe topological sort
                # (pure_callback inside the adapter).
                direct_effect = processor.forward(
                    X_weighted, proc_params[j], A=A_curr, training=training, rng_key=rng_key
                )
            elif _proc_name in ("MLPAdapter", "TransformerAdapter"):
                # Enable dropout during training
                direct_effect = processor.forward(
                    X_weighted, proc_params[j], training=training, rng_key=rng_key
                )
            else:
                # ELM, Mamba don't have dropout
                direct_effect = processor.forward(X_weighted, proc_params[j])

            # Add confounded effect from L
            if use_latent_confounders and L_batch is not None:
                # L_batch[:, j] is the confounded effect for variable j
                # L_batch has shape (batch_size, n_vars) - already indexed by batch
                confounded_effect = L_batch[:, j]
                output = direct_effect + confounded_effect
            else:
                output = direct_effect

            if j == Y_idx:
                # Task-dependent loss for Y
                if task == "regression":
                    # Regression: MSE (no sigmoid)
                    classification_loss = jnp.mean((output - batch_Y) ** 2)
                else:
                    # Classification: Binary Cross-Entropy
                    Y_pred_logits = output
                    Y_pred_prob = jax.nn.sigmoid(Y_pred_logits)
                    eps = 1e-7
                    bce = -jnp.mean(
                        batch_Y * jnp.log(Y_pred_prob + eps)
                        + (1 - batch_Y) * jnp.log(1 - Y_pred_prob + eps)
                    )
                    classification_loss = bce
            else:
                # Reconstruction: Mean Squared Error
                mse = jnp.mean((batch_data[:, j] - output) ** 2)
                total_recon_loss += mse

        # Average reconstruction loss across non-Y variables
        total_recon_loss = total_recon_loss / (n_v - 1)

        # Sparsity penalty — adaptive Y-column penalty (v4 is always classification)
        class_ratio = jnp.clip(classification_loss / 0.6931, 0.0, 1.0)
        y_sparsity_scale = jnp.clip(1.0 - class_ratio, 0.3, 1.0)  # Floor at 0.3
        sparsity_mask = jnp.ones((n_v, n_v)).at[:, Y_idx].set(y_sparsity_scale)
        sparsity_loss = lambda_1 * jnp.sum(jnp.abs(A_curr) * sparsity_mask)

        # DAG constraint (acyclicity)
        # Use spectral during training if enabled, else DAGMA (log-det)
        if use_spectral_constraint:
            h_A = compute_dag_constraint_auto(
                A_curr,
                use_spectral=True,
                iteration=current_iter,
                max_iter=max_iter,
                use_exact_final=True,
            )
        else:
            h_A = dag_constraint(A_curr)
        dag_loss = lambda_2_current * h_A

        # Uncertainty weighting (Kendall et al. 2018) — log_var_class removed
        precision_recon = jnp.exp(-log_var_r)

        # Combine structural losses (reconstruction + sparsity + DAG)
        structural_loss = total_recon_loss + sparsity_loss + dag_loss

        # Weight decay on A matrix (L2 regularization)
        if weight_decay > 0:
            weight_decay_loss = weight_decay * jnp.sum(A_curr**2)
            structural_loss = structural_loss + weight_decay_loss

        # Latent confounder penalties
        if use_latent_confounders:
            U_curr = params["U"]
            V_curr = params["V"]

            # Nuclear norm via factored representation: ||L||_* ≈ 0.5*(||U||²_F + ||V||²_F)
            nuclear_norm_loss = 0.5 * (jnp.sum(U_curr**2) + jnp.sum(V_curr**2))
            structural_loss = structural_loss + lambda_L * nuclear_norm_loss

            # Bow-free penalty: ||A ⊙ Ω||₁ where Ω = L.T @ L / n
            # This penalizes pairs with both direct edge AND confounding
            L_curr = U_curr @ V_curr.T
            Omega = (L_curr.T @ L_curr) / n_batch  # (n_vars × n_vars) confounding correlation
            bow_free_loss = jnp.sum(jnp.abs(A_curr) * jnp.abs(Omega))
            structural_loss = structural_loss + lambda_bow * bow_free_loss

        # Apply uncertainty weighting
        weighted_structural = 0.5 * precision_recon * structural_loss + 0.5 * log_var_r
        weighted_classification = lambda_class * classification_loss

        total_loss = weighted_structural + weighted_classification

        return total_loss, (h_A, total_recon_loss, classification_loss)

    # Optimizer for all parameters (A + processor_params + uncertainty weights)
    # Gradient clipping prevents exploding gradients (Mamba/GNN)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))
    opt_state = optimizer.init(all_params)

    # Early stopping
    best_loss = float("inf")
    patience_counter = 0
    best_params = all_params.copy()  # Initialize with current params

    # Training loop
    for iter in range(max_iter):
        # Warm-start transition for latent confounders
        # After warm_start_L_iters, compute residuals and initialize U, V via SVD
        if use_latent_confounders and not L_active and iter == warm_start_L_iters:
            if verbose:
                print("\nWarm-start: Computing residuals and initializing L factors...")

            # Compute residuals: X_j - f(X * |A[:,j]|)
            residuals = compute_residuals_for_warmstart(all_params, data_train)

            # Truncated SVD to initialize U, V
            # L = U @ V.T, where L ≈ residuals
            U_svd, S_svd, Vt_svd = jnp.linalg.svd(residuals, full_matrices=False)

            # Take top-k components with scaling to prevent destabilization
            k = min(latent_rank_k, len(S_svd))
            # U_new = U_svd[:, :k] * sqrt(S[:k])
            # V_new = Vt_svd[:k, :].T * sqrt(S[:k])
            # Scale factor 0.1 for gentle initialization (shrink-perturb style)
            scale_factor = 0.1
            sqrt_S = jnp.sqrt(S_svd[:k]) * scale_factor
            U_new = U_svd[:, :k] * sqrt_S[jnp.newaxis, :]
            V_new = Vt_svd[:k, :].T * sqrt_S[jnp.newaxis, :]

            # Update all_params with warm-started U, V
            all_params["U"] = U_new
            all_params["V"] = V_new

            # Reinitialize optimizer state with new params
            opt_state = optimizer.init(all_params)
            L_active = True

            if verbose:
                L_init = U_new @ V_new.T
                L_init_norm = float(jnp.sum(jnp.linalg.svd(L_init, compute_uv=False)))
                print(f"L initialized: shape={L_init.shape}, ||L||_*={L_init_norm:.4f}")

        epoch_loss = 0.0
        epoch_h_A = 0.0
        epoch_recon_loss = 0.0
        epoch_class_loss = 0.0

        if use_batching:
            # Shuffle training data
            key, shuffle_key = random.split(key)
            perm = random.permutation(shuffle_key, n_samples)
            data_shuffled = data_train[perm]
            Y_shuffled = Y_train[perm]

            # Process all batches
            for batch_idx in range(num_batches_per_iter):
                start_idx = batch_idx * effective_batch_size
                end_idx = min(start_idx + effective_batch_size, n_samples)

                batch_data_raw = data_shuffled[start_idx:end_idx]
                batch_Y_raw = Y_shuffled[start_idx:end_idx]
                actual_batch_size = batch_data_raw.shape[0]

                # Get batch indices for U slicing (original indices before shuffling)
                batch_indices = perm[start_idx:end_idx]

                # Pad last batch if needed
                if actual_batch_size < effective_batch_size:
                    padding_data = jnp.zeros((effective_batch_size - actual_batch_size, n_vars))
                    padding_Y = jnp.zeros(effective_batch_size - actual_batch_size)
                    batch_data = jnp.concatenate([batch_data_raw, padding_data], axis=0)
                    batch_Y = jnp.concatenate([batch_Y_raw, padding_Y], axis=0)
                    # Pad batch_indices too (use index 0 for padding, will be masked)
                    padding_indices = jnp.zeros(
                        effective_batch_size - actual_batch_size, dtype=jnp.int32
                    )
                    batch_indices = jnp.concatenate([batch_indices, padding_indices])
                else:
                    batch_data = batch_data_raw
                    batch_Y = batch_Y_raw

                # Compute loss and gradients for ALL parameters
                # Generate dropout key for this batch
                key, dropout_key = random.split(key)
                (loss_val, (h_A, recon_loss, class_loss)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(
                    all_params,
                    batch_data,
                    batch_Y,
                    lambda_2,
                    iter,
                    L_active,
                    batch_indices,
                    dropout_key,
                )

                # Update ALL parameters (A + processor_params + uncertainty weights)
                updates, opt_state = optimizer.update(grads, opt_state)
                all_params = optax.apply_updates(all_params, updates)

                epoch_loss += loss_val * actual_batch_size
                epoch_h_A += h_A
                epoch_recon_loss += recon_loss * actual_batch_size
                epoch_class_loss += class_loss * actual_batch_size

            # Average
            epoch_loss = epoch_loss / n_samples
            epoch_h_A = epoch_h_A / num_batches_per_iter
            epoch_recon_loss = epoch_recon_loss / n_samples
            epoch_class_loss = epoch_class_loss / n_samples

        else:
            # Full batch on training data
            # Generate dropout key for this iteration
            key, dropout_key = random.split(key)
            (epoch_loss, (epoch_h_A, epoch_recon_loss, epoch_class_loss)), grads = (
                jax.value_and_grad(loss_fn, has_aux=True)(
                    all_params, data_train, Y_train, lambda_2, iter, L_active, None, dropout_key
                )
            )

            # Update ALL parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            all_params = optax.apply_updates(all_params, updates)

        # Compute validation loss for early stopping
        # L is not used for validation (sample-specific to training)
        # rng_key=None disables dropout for validation
        if use_validation_split and data_val is not None:
            val_loss, (_, _, _) = loss_fn(
                all_params,
                data_val,
                Y_val,
                lambda_2,
                iter,
                use_L_in_loss=False,
                batch_indices=None,
                rng_key=None,
            )
            early_stop_loss = float(val_loss)
        else:
            early_stop_loss = float(epoch_loss)

        # Dynamic pruning - update mask periodically
        if enable_pruning and iter > 0 and iter % 10 == 0:
            A_curr = all_params["A"]
            # Compute gradient for pruning decision
            grads_A = grads["A"]
            pruning_mask = dynamic_pruning(A_curr, grads_A, iter, max_iter)

        # Increase DAG penalty if constraint not satisfied
        if epoch_h_A > 0.25:
            lambda_2 = min(lambda_2 * 10, lambda_2_max)

        # Early stopping based on validation loss (or train loss if no validation)
        if early_stop_loss < best_loss:
            best_loss = early_stop_loss
            patience_counter = 0
            # Save best parameters
            best_params = {k: v.copy() if hasattr(v, "copy") else v for k, v in all_params.items()}
        else:
            patience_counter += 1

        if verbose and iter % 10 == 0:
            sigma_r = jnp.exp(0.5 * all_params["log_var_recon"])
            pruning_info = (
                f", pruned={int((1 - pruning_mask.mean()) * 100)}%" if enable_pruning else ""
            )
            spectral_info = " [spectral]" if use_spectral_constraint else ""
            val_info = f", val_loss={early_stop_loss:.3f}" if use_validation_split else ""
            print(
                f"Iter {iter}: train_loss={epoch_loss:.3f} "
                f"(recon={epoch_recon_loss:.3f}, class={epoch_class_loss:.3f}){val_info}, "
                f"h(A)={epoch_h_A:.3f}{spectral_info}, λ2={lambda_2:.2e}, "
                f"σ_r={sigma_r:.3f}{pruning_info}",
                flush=True,
            )

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(
                    f"\nEarly stopping at iteration {iter} (patience={patience}, best_val_loss={best_loss:.4f})"
                )
            # Restore best parameters
            all_params = best_params
            break

    # Extract final parameters
    A_final = all_params["A"]
    # Merge trained params with metadata to get full processor params
    processor_params_final = merge_trained_params(processor_params, all_params["processor_params"])

    # solve_output_weights is DISABLED: end-to-end joint training means
    # processor params are already optimized during GOLEM, no post-hoc step needed.

    if verbose:
        n_selected = int(jnp.sum(jnp.abs(A_final[:, Y_idx]) > 0.05))
        print(f"\nUnified framework: A-weighted features: {n_selected} edges to Y")
        print(f"  weight_max={float(jnp.max(jnp.abs(A_final[:, Y_idx]))):.3f}")
        print("  (No post-hoc logistic regression - using jointly trained params)")

    if verbose:
        print(
            f"\nLearned A matrix (max={jnp.max(jnp.abs(A_final)):.3f}, "
            f"mean={jnp.mean(jnp.abs(A_final)):.3f})"
        )
        print(f"Final uncertainty: σ_recon={jnp.exp(0.5 * all_params['log_var_recon']):.3f}")
        print("Processor params jointly optimized with A")

    # Threshold to get binary DAG (absolute threshold, not relative)
    threshold = 0.05
    A_binary = (jnp.abs(A_final) > threshold).astype(jnp.float32)

    # Compute classification accuracy using the trained processor
    # This is the CORRECT accuracy - from the processor's Y prediction
    #
    # NOTE: The processor is trained with BCE loss using sigmoid(output)
    # So the output is LOGITS, and we must apply sigmoid to get probabilities!
    #
    # Compute accuracy on training data only
    # L is sample-specific to training samples; avoids test leakage
    data_for_acc = data_train
    Y_for_acc = Y_train

    # ALWAYS use A-weighted features for accuracy (JCCE core design)
    # X_w = X * |A[:, Y_idx]| - soft parent selection
    # A selects causally relevant features
    weights_Y = jnp.abs(A_final[:, Y_idx])
    weights_Y = jnp.maximum(weights_Y, 0.01)  # Raw weights with floor (no L1-norm)
    X_for_pred = data_for_acc * weights_Y[jnp.newaxis, :]

    Y_pred_logits = processor.forward(X_for_pred, processor_params_final[Y_idx])

    # Add L contribution to prediction if enabled
    if use_latent_confounders:
        L_final = all_params["U"] @ all_params["V"].T
        Y_pred_logits = Y_pred_logits + L_final[:, Y_idx]
    else:
        L_final = None

    # Task-dependent post-training metrics
    Y_flat = Y_for_acc.flatten()

    if task == "regression":
        # Regression: raw predictions, no sigmoid
        Y_pred = Y_pred_logits.flatten()
        Y_pred_prob = Y_pred  # For consistency in return dict
        Y_pred_binary = Y_pred  # Not used for regression

        # Regression metrics
        ss_res = float(jnp.sum((Y_flat - Y_pred) ** 2))
        ss_tot = float(jnp.sum((Y_flat - jnp.mean(Y_flat)) ** 2))
        r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
        rmse = float(jnp.sqrt(jnp.mean((Y_flat - Y_pred) ** 2)))
        mae = float(jnp.mean(jnp.abs(Y_flat - Y_pred)))
        # Use R² as the primary quality metric (analogous to balanced_accuracy)
        classification_accuracy = max(r2, 0.0)  # Clip negative R² to 0 for fitness
        balanced_accuracy = classification_accuracy
        precision = 0.0
        recall = 0.0
        specificity = 0.0
        f1_score = 0.0
        auc_roc = 0.0

        if verbose:
            n_edges = int(jnp.sum(A_binary))
            print(f"Threshold: {threshold:.4f} (absolute)")
            print(f"Final: {n_edges} edges, h(A)={epoch_h_A:.3f}")
            print(f"Regression R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            if use_latent_confounders and L_final is not None:
                L_singular_values = jnp.linalg.svd(L_final, compute_uv=False)
                effective_rank = int(jnp.sum(L_singular_values > 0.01 * L_singular_values[0]))
                nuclear_norm = float(jnp.sum(L_singular_values))
                print(f"Latent L: effective_rank={effective_rank}, ||L||_*={nuclear_norm:.4f}")
            print(f"{'=' * 60}\n")
    else:
        # Classification: sigmoid + threshold + confusion matrix metrics
        Y_pred_prob = jax.nn.sigmoid(Y_pred_logits)
        Y_pred_binary = (Y_pred_prob > 0.5).astype(jnp.float32).flatten()
        classification_accuracy = float(jnp.mean(Y_pred_binary == Y_flat))
        r2 = 0.0
        rmse = 0.0
        mae = 0.0

        if verbose:
            print("\n[DEBUG] Accuracy computation:")
            print(f"  weight_sum={float(weight_sum):.4f}, using_uniform={float(weight_sum) < 0.01}")
            print(
                f"  Y_pred_logits: min={float(jnp.min(Y_pred_logits)):.4f}, max={float(jnp.max(Y_pred_logits)):.4f}, mean={float(jnp.mean(Y_pred_logits)):.4f}"
            )
            print(
                f"  Y_pred_prob: min={float(jnp.min(Y_pred_prob)):.4f}, max={float(jnp.max(Y_pred_prob)):.4f}, mean={float(jnp.mean(Y_pred_prob)):.4f}"
            )
            print(
                f"  Y_pred_binary: sum={float(jnp.sum(Y_pred_binary)):.0f}/{len(Y_pred_binary)} ({100 * float(jnp.mean(Y_pred_binary)):.1f}%)"
            )
            print(
                f"  Y_true: sum={float(jnp.sum(Y_flat)):.0f}/{len(Y_flat)} ({100 * float(jnp.mean(Y_flat)):.1f}%)"
            )

        if verbose:
            n_edges = int(jnp.sum(A_binary))
            print(f"Threshold: {threshold:.4f} (absolute)")
            print(f"Final: {n_edges} edges, h(A)={epoch_h_A:.3f}")
            print(f"Classification accuracy (processor): {classification_accuracy:.4f}")
            if use_latent_confounders and L_final is not None:
                L_singular_values = jnp.linalg.svd(L_final, compute_uv=False)
                effective_rank = int(jnp.sum(L_singular_values > 0.01 * L_singular_values[0]))
                nuclear_norm = float(jnp.sum(L_singular_values))
                print(f"Latent L: effective_rank={effective_rank}, ||L||_*={nuclear_norm:.4f}")
            print(f"{'=' * 60}\n")

        # Classification metrics: confusion matrix
        Y_pred_flat = Y_pred_binary.flatten()
        Y_true_flat = Y_flat.flatten()
        tp = float(jnp.sum((Y_pred_flat == 1) & (Y_true_flat == 1)))
        tn = float(jnp.sum((Y_pred_flat == 0) & (Y_true_flat == 0)))
        fp = float(jnp.sum((Y_pred_flat == 1) & (Y_true_flat == 0)))
        fn = float(jnp.sum((Y_pred_flat == 0) & (Y_true_flat == 1)))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        balanced_accuracy = (recall + specificity) / 2

        # AUC-ROC
        Y_probs_flat = Y_pred_prob.flatten()
        sorted_indices = jnp.argsort(Y_probs_flat)[::-1]
        Y_true_sorted = Y_true_flat[sorted_indices]
        n_pos = float(jnp.sum(Y_true_flat == 1))
        n_neg = float(jnp.sum(Y_true_flat == 0))
        if n_pos > 0 and n_neg > 0:
            tpr_cumsum = jnp.cumsum(Y_true_sorted) / n_pos
            fpr_cumsum = jnp.cumsum(1 - Y_true_sorted) / n_neg
            auc_roc = float(
                jnp.sum((fpr_cumsum[1:] - fpr_cumsum[:-1]) * (tpr_cumsum[1:] + tpr_cumsum[:-1]) / 2)
            )
        else:
            auc_roc = 0.5

    # Markov Blanket: features with non-zero weight for Y
    # MB(Y) = Parents(Y) ∪ Children(Y) ∪ Spouses(Y)
    # From A_binary: Parents(Y) = {i : A[i,Y] = 1}, Children(Y) = {j : A[Y,j] = 1}
    # Spouses are parents of children, more complex - for now just use direct connections
    parents_Y = [int(i) for i in range(n_vars) if i != Y_idx and float(A_binary[i, Y_idx]) > 0]
    children_Y = [int(j) for j in range(n_vars) if j != Y_idx and float(A_binary[Y_idx, j]) > 0]
    # Markov blanket = parents ∪ children (simplified)
    markov_blanket = sorted(set(parents_Y + children_Y))

    # Sparsity as percentage
    max_edges = n_vars * (n_vars - 1)  # Maximum possible edges (no self-loops)
    sparsity = 1.0 - (float(jnp.sum(A_binary)) / max_edges) if max_edges > 0 else 1.0

    metrics = {
        # Structure metrics
        "n_edges": int(jnp.sum(A_binary)),
        "sparsity": sparsity,
        "final_h_A": float(epoch_h_A),
        "markov_blanket": markov_blanket,
        "markov_blanket_size": len(markov_blanket),
        # Loss metrics
        "final_loss": float(epoch_loss),
        "final_recon_loss": float(epoch_recon_loss),
        "final_class_loss": float(epoch_class_loss),
        # Classification metrics (v6.1.2) — also populated for regression (R² maps to balanced_accuracy)
        "classification_accuracy": classification_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "auc_roc": auc_roc,
        # Regression metrics (0.0 for classification)
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "task": task,
        # Training info
        "iterations": iter + 1,
        "early_stopped": patience_counter >= patience,
        "final_sigma_recon": float(jnp.exp(0.5 * all_params["log_var_recon"])),
        "final_sigma_class": 1.0,  # Sentinel — log_var_class removed, curriculum + lambda_class suffice
        # Adjacency matrix (for DAG visualization)
        "A_binary": A_binary.tolist() if hasattr(A_binary, "tolist") else A_binary,
        "A_weights": A_final.tolist() if hasattr(A_final, "tolist") else A_final,
    }

    # Add confusion matrix only for classification
    if task == "classification":
        Y_pred_flat = Y_pred_binary.flatten()
        Y_true_flat = Y_flat.flatten()
        tp_val = float(jnp.sum((Y_pred_flat == 1) & (Y_true_flat == 1)))
        tn_val = float(jnp.sum((Y_pred_flat == 0) & (Y_true_flat == 0)))
        fp_val = float(jnp.sum((Y_pred_flat == 1) & (Y_true_flat == 0)))
        fn_val = float(jnp.sum((Y_pred_flat == 0) & (Y_true_flat == 1)))
        metrics["confusion_matrix"] = {
            "tp": int(tp_val),
            "tn": int(tn_val),
            "fp": int(fp_val),
            "fn": int(fn_val),
        }

    # Add L metrics if latent confounders enabled
    if use_latent_confounders and L_final is not None:
        L_singular_values = jnp.linalg.svd(L_final, compute_uv=False)
        metrics["L_effective_rank"] = int(jnp.sum(L_singular_values > 0.01 * L_singular_values[0]))
        metrics["L_nuclear_norm"] = float(jnp.sum(L_singular_values))
        metrics["L_max"] = float(jnp.max(jnp.abs(L_final)))
        metrics["L_mean"] = float(jnp.mean(jnp.abs(L_final)))

    # Return L as well when enabled
    if use_latent_confounders:
        return A_binary, processor, processor_params_final, metrics, L_final
    else:
        return A_binary, processor, processor_params_final, metrics


# ============================================================================
# Effect Estimation Integration
# ============================================================================


def learn_with_effects(
    data: jnp.ndarray,
    Y: jnp.ndarray,
    Y_idx: int,
    processor,
    key: random.PRNGKey,
    # Treatment specification
    treatment_idx: int = None,
    T: jnp.ndarray = None,
    # Effect estimation config
    enable_effects: bool = True,
    latent_dim: int = 4,
    head_hidden_dim: int = 32,
    # Lambda weights for effect losses
    lambda_outcome: float = 1.0,
    lambda_propensity: float = 0.1,
    lambda_targeted: float = 1.0,
    # Standard GOLEM parameters
    processor_type: str = "elm",
    lambda_1: float = 0.02,
    lambda_2_init: float = 0.01,
    lambda_2_max: float = 1e10,
    lambda_class: float = 1.0,
    lr: float = 0.001,
    max_iter: int = 100,
    patience: int = 15,
    verbose: bool = False,
    A_init: Optional[jnp.ndarray] = None,
    use_spectral_constraint: bool = False,
    enable_pruning: bool = False,
    use_validation_split: bool = True,
    validation_ratio: float = 0.2,
    weight_decay: float = 1e-4,
    # L parameters
    use_latent_confounders: bool = True,  # Required for effect estimation
    latent_rank_k: int = 5,
    lambda_L: float = 0.05,
    lambda_bow: float = 0.1,
    warm_start_L_iters: int = 20,
) -> Tuple[jnp.ndarray, Any, list, Dict[str, Any]]:
    """
    v6.0: GOLEM with integrated causal effect estimation.

    Extends _learn_structure_legacy with:
    - Effect heads (Y0, Y1, propensity) via EffectAdapterWrapper
    - Factual outcome loss
    - Propensity loss
    - Targeted regularization (DragonNet-style)
    - 3-phase training protocol

    Args:
        data: (n_samples, n_vars) observed data
        Y: (n_samples,) target labels
        Y_idx: Index of Y in data
        processor: Processor adapter (will be wrapped with effect heads)
        key: JAX random key
        treatment_idx: Index of treatment variable (if None, uses first parent of Y)
        T: Binary treatment vector (if None, will binarize treatment_idx)
        enable_effects: Whether to enable effect estimation
        latent_dim: Dimension for effect head latent input
        head_hidden_dim: Hidden dim for effect heads
        lambda_outcome: Weight for factual outcome loss
        lambda_propensity: Weight for propensity loss
        lambda_targeted: Weight for targeted regularization
        ... (other params same as v4_joint)

    Returns:
        A_est: Estimated adjacency matrix
        processor: Effect-wrapped processor
        processor_params: Trained parameters
        metrics: Dict including effect estimation metrics (ATE, etc.)
    """
    from jcce.structure_learning.effect_estimation import (
        binarize_treatment,
        compute_effect_losses,
        get_training_phase,
    )
    from jcce.structure_learning.processor_adapters import EffectAdapterWrapper

    n_samples_total, n_vars = data.shape

    # Wrap processor with effect heads if enabled
    # NOTE: latent_dim for effect heads should match latent_rank_k for U, V
    effect_latent_dim = latent_rank_k if use_latent_confounders else latent_dim
    if enable_effects:
        key, wrap_key = random.split(key)
        effect_processor = EffectAdapterWrapper(
            base_adapter=processor,
            latent_dim=effect_latent_dim,  # Must match U's rank!
            head_hidden_dim=head_hidden_dim,
            enable_effects=True,
            key=wrap_key,
        )
    else:
        effect_processor = processor

    # Create validation split
    if use_validation_split:
        key, split_key = random.split(key)
        n_val = int(n_samples_total * validation_ratio)
        n_train = n_samples_total - n_val

        perm = random.permutation(split_key, n_samples_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        data_train = data[train_idx]
        Y_train = Y[train_idx]
        data_val = data[val_idx]
        Y_val = Y[val_idx]

        n_samples = n_train
    else:
        data_train = data
        Y_train = Y
        data_val = None
        Y_val = None
        n_samples = n_samples_total
        train_idx = jnp.arange(n_samples_total)

    # Initialize processor parameters for each variable
    key, init_key = random.split(key)
    processor_params = []
    for j in range(n_vars):
        key, param_key = random.split(key)
        if enable_effects:
            params_j = effect_processor.init_params(n_vars)
        else:
            params_j = processor.init_params(n_vars)
        processor_params.append(params_j)

    # Initialize A matrix
    key, a_key = random.split(key)
    if A_init is not None:
        A = A_init.copy()
    else:
        A = random.normal(a_key, (n_vars, n_vars)) * 0.01
        A = A.at[jnp.diag_indices(n_vars)].set(0)

        # Correlation warm-start for A[:,Y_idx]
        corr_with_Y = jnp.corrcoef(data_train.T)[Y_idx, :]
        corr_with_Y = jnp.abs(corr_with_Y) * 0.5
        corr_with_Y = corr_with_Y.at[Y_idx].set(0.0)
        A = A.at[:, Y_idx].set(corr_with_Y)
        if verbose:
            n_corr_edges = int(jnp.sum(corr_with_Y > 0.05))
            print(f"Correlation warm-start: {n_corr_edges} features correlated with Y")

    # Initialize uncertainty parameters (log_var_class removed — see v7 comment)
    log_var_recon = jnp.array(0.0)

    # Initialize U, V for latent confounders
    if use_latent_confounders:
        key, uv_key = random.split(key)
        k = latent_rank_k
        U = random.normal(uv_key, (n_samples, k)) * 0.01
        V = random.normal(random.split(uv_key)[0], (n_vars, k)) * 0.01
    else:
        U = None
        V = None

    # Handle treatment
    if enable_effects:
        if T is None:
            # Use treatment_idx or find from A
            if treatment_idx is None:
                # Will select after A is learned (use first parent for now)
                treatment_idx = 0 if Y_idx != 0 else 1
            T_full = binarize_treatment(data, treatment_idx, method="median")
            T_train = T_full[train_idx] if use_validation_split else T_full
        else:
            T_train = T[train_idx] if use_validation_split else T
    else:
        T_train = None

    # Extract trainable params for joint optimization
    trainable_proc_params = extract_trainable_params(processor_params)

    # Package all parameters
    all_params = {
        "A": A,
        "log_var_recon": log_var_recon,
        "processor_params": trainable_proc_params,
    }
    if use_latent_confounders:
        all_params["U"] = U
        all_params["V"] = V

    # Dynamic lambda_2 growth
    lambda_2 = lambda_2_init
    lambda_2_growth = 2.0

    # Batching setup
    effective_batch_size = min(128, n_samples)
    use_batching = n_samples > effective_batch_size
    num_batches_per_iter = max(1, n_samples // effective_batch_size)

    # Pruning mask
    if enable_pruning:
        pruning_mask = jnp.ones((n_vars, n_vars))
    else:
        pruning_mask = None

    # Track L activation
    L_active = not use_latent_confounders
    if use_latent_confounders:
        L_active = False

    # =========================================================================
    # Loss function with effect estimation
    # =========================================================================
    def loss_fn(
        params,
        batch_data,
        batch_Y,
        batch_T,
        lambda_2_current,
        current_iter,
        use_L_in_loss=True,
        batch_indices=None,
        rng_key=None,
    ):
        """
        Extended loss function with effect estimation.
        """
        A_curr = params["A"]
        if enable_pruning and pruning_mask is not None:
            A_curr = A_curr * pruning_mask

        # Merge trained params with metadata for joint optimization
        proc_params = merge_trained_params(processor_params, params["processor_params"])
        log_var_r = params["log_var_recon"]

        # Get training phase for effect weight scaling
        phase = get_training_phase(current_iter, max_iter)
        effect_scale = phase.effect_weight_scale

        # Compute L for latent confounders
        if use_latent_confounders and use_L_in_loss:
            U_curr = params["U"]
            V_curr = params["V"]
            if batch_indices is not None:
                U_batch = U_curr[batch_indices, :]
                L_batch = U_batch @ V_curr.T
            else:
                U_batch = U_curr
                L_batch = U_curr @ V_curr.T
        else:
            U_batch = None
            L_batch = None

        n_batch, n_v = batch_data.shape

        # Standard GOLEM losses
        total_recon_loss = 0.0
        classification_loss = 0.0

        training = rng_key is not None

        for j in range(n_v):
            weights = jnp.abs(A_curr[:, j])
            # Mask out self to maintain causal correctness
            weights = weights.at[j].set(0.0)

            # Different handling for Y vs other variables
            # For Y: NO fallback - force processor to use A-weighted features
            # For other variables: Keep fallback for stable reconstruction
            if j == Y_idx:
                weights = jnp.maximum(weights, 0.01)
            else:
                weight_sum = jnp.sum(weights)
                fallback_weights = jnp.ones(n_v).at[j].set(0.0)
                weights = jnp.where(weight_sum > 0.01, weights, fallback_weights)

            X_weighted = batch_data * weights[jnp.newaxis, :]

            # Pass training and rng_key for dropout
            if processor.__class__.__name__ == "GNNAdapter":
                A_normalized = A_curr / (jnp.sum(jnp.abs(A_curr), axis=0, keepdims=True) + 1e-8)
                direct_effect = processor.forward(X_weighted, proc_params[j], A=A_normalized)
            elif processor.__class__.__name__ in ("MLPAdapter", "TransformerAdapter"):
                direct_effect = processor.forward(
                    X_weighted, proc_params[j], training=training, rng_key=rng_key
                )
            else:
                direct_effect = processor.forward(X_weighted, proc_params[j])

            if use_latent_confounders and L_batch is not None:
                confounded_effect = L_batch[:, j]
                output = direct_effect + confounded_effect
            else:
                output = direct_effect

            if j == Y_idx:
                Y_pred_logits = output
                Y_pred_prob = jax.nn.sigmoid(Y_pred_logits)
                eps = 1e-7
                bce = -jnp.mean(
                    batch_Y * jnp.log(Y_pred_prob + eps)
                    + (1 - batch_Y) * jnp.log(1 - Y_pred_prob + eps)
                )
                classification_loss = bce
            else:
                mse = jnp.mean((batch_data[:, j] - output) ** 2)
                total_recon_loss += mse

        total_recon_loss = total_recon_loss / (n_v - 1)

        # Sparsity and DAG losses — adaptive Y-column penalty (v6 is always classification)
        class_ratio = jnp.clip(classification_loss / 0.6931, 0.0, 1.0)
        y_sparsity_scale = jnp.clip(1.0 - class_ratio, 0.3, 1.0)  # Floor at 0.3
        sparsity_mask = jnp.ones((n_v, n_v)).at[:, Y_idx].set(y_sparsity_scale)
        sparsity_loss = lambda_1 * jnp.sum(jnp.abs(A_curr) * sparsity_mask)

        if use_spectral_constraint:
            h_A = compute_dag_constraint_auto(
                A_curr,
                use_spectral=True,
                iteration=current_iter,
                max_iter=max_iter,
                use_exact_final=True,
            )
        else:
            h_A = dag_constraint(A_curr)

        dag_loss = lambda_2_current * h_A

        structural_loss = total_recon_loss + sparsity_loss + dag_loss

        if weight_decay > 0:
            structural_loss = structural_loss + weight_decay * jnp.sum(A_curr**2)

        # L regularization
        if use_latent_confounders:
            U_curr = params["U"]
            V_curr = params["V"]
            nuclear_norm_loss = 0.5 * (jnp.sum(U_curr**2) + jnp.sum(V_curr**2))
            structural_loss = structural_loss + lambda_L * nuclear_norm_loss

            L_curr = U_curr @ V_curr.T
            Omega = (L_curr.T @ L_curr) / n_batch
            bow_free_loss = jnp.sum(jnp.abs(A_curr) * jnp.abs(Omega))
            structural_loss = structural_loss + lambda_bow * bow_free_loss

        # =================================================================
        # Effect estimation losses (NEW)
        # =================================================================
        effect_loss = 0.0
        effect_metrics = {}

        if enable_effects and batch_T is not None and effect_scale > 0:
            # Get representation for Y (using weights from A)
            weights_Y = jnp.abs(A_curr[:, Y_idx])
            # Mask out Y itself to maintain causal correctness
            weights_Y = weights_Y.at[Y_idx].set(0.0)
            weights_Y = jnp.maximum(weights_Y, 0.01)  # Raw weights with floor (no L1-norm)
            X_weighted_Y = batch_data * weights_Y[jnp.newaxis, :]

            # Forward with effect heads
            if U_batch is not None:
                if processor.__class__.__name__ == "GNNAdapter":
                    A_norm = A_curr / (jnp.sum(jnp.abs(A_curr), axis=0, keepdims=True) + 1e-8)
                    effect_outputs = effect_processor.forward_with_effects(
                        X_weighted_Y,
                        proc_params[Y_idx],
                        U_batch,
                        batch_T,
                        training=True,
                        rng_key=rng_key,
                        A=A_norm,
                    )
                else:
                    effect_outputs = effect_processor.forward_with_effects(
                        X_weighted_Y,
                        proc_params[Y_idx],
                        U_batch,
                        batch_T,
                        training=True,
                        rng_key=rng_key,
                    )

                y0 = effect_outputs["y0"]
                y1 = effect_outputs["y1"]
                propensity = effect_outputs["propensity"]

                # Compute effect losses
                losses = compute_effect_losses(batch_Y, batch_T, y0, y1, propensity)

                L_outcome = losses["L_outcome"]
                L_propensity = losses["L_propensity"]
                L_targeted = losses["L_targeted"]

                # Scaled effect loss
                effect_loss = effect_scale * (
                    lambda_outcome * L_outcome
                    + lambda_propensity * L_propensity
                    + lambda_targeted * L_targeted
                )

                # NOTE: Can't use float() inside JAX-traced function
                # Store as JAX arrays, convert to float outside
                effect_metrics = {
                    "L_outcome": L_outcome,
                    "L_propensity": L_propensity,
                    "L_targeted": L_targeted,
                    "ATE": losses["ATE"],
                    "CATE_std": losses["CATE_std"],
                }

        # Uncertainty weighting — log_var_class removed
        precision_recon = jnp.exp(-log_var_r)

        weighted_structural = 0.5 * precision_recon * structural_loss + 0.5 * log_var_r
        weighted_classification = lambda_class * classification_loss

        total_loss = weighted_structural + weighted_classification + effect_loss

        return total_loss, (h_A, total_recon_loss, classification_loss, effect_metrics)

    # Optimizer with gradient clipping (prevents exploding gradients in Mamba/GNN)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))
    opt_state = optimizer.init(all_params)

    # Early stopping
    best_loss = float("inf")
    patience_counter = 0
    best_params = all_params.copy()

    # Training loop
    epoch_h_A = 0.0
    epoch_recon_loss = 0.0
    epoch_class_loss = 0.0
    epoch_effect_metrics = {}

    for iter in range(max_iter):
        # L warm-start
        if use_latent_confounders and not L_active and iter == warm_start_L_iters:
            if verbose:
                print("Warm-start: Initializing L factors via SVD...")
            L_active = True

        epoch_loss = 0.0

        if use_batching:
            key, shuffle_key = random.split(key)
            perm = random.permutation(shuffle_key, n_samples)
            data_shuffled = data_train[perm]
            Y_shuffled = Y_train[perm]
            T_shuffled = T_train[perm] if T_train is not None else None

            for batch_idx in range(num_batches_per_iter):
                start_idx = batch_idx * effective_batch_size
                end_idx = min(start_idx + effective_batch_size, n_samples)

                batch_data = data_shuffled[start_idx:end_idx]
                batch_Y = Y_shuffled[start_idx:end_idx]
                batch_T = T_shuffled[start_idx:end_idx] if T_shuffled is not None else None
                batch_indices = perm[start_idx:end_idx]

                key, rng_key = random.split(key)

                (loss_val, (h_A, recon_loss, class_loss, eff_metrics)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(
                    all_params,
                    batch_data,
                    batch_Y,
                    batch_T,
                    lambda_2,
                    iter,
                    L_active,
                    batch_indices,
                    rng_key,
                )

                updates, opt_state = optimizer.update(grads, opt_state, all_params)
                all_params = optax.apply_updates(all_params, updates)
                all_params["A"] = all_params["A"].at[jnp.diag_indices(n_vars)].set(0)

                epoch_loss += loss_val
                epoch_h_A = h_A
                epoch_recon_loss = recon_loss
                epoch_class_loss = class_loss
                epoch_effect_metrics = eff_metrics

            epoch_loss /= num_batches_per_iter
        else:
            key, rng_key = random.split(key)
            batch_indices = jnp.arange(n_samples)

            (loss_val, (h_A, recon_loss, class_loss, eff_metrics)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(
                all_params,
                data_train,
                Y_train,
                T_train,
                lambda_2,
                iter,
                L_active,
                batch_indices,
                rng_key,
            )

            updates, opt_state = optimizer.update(grads, opt_state, all_params)
            all_params = optax.apply_updates(all_params, updates)
            all_params["A"] = all_params["A"].at[jnp.diag_indices(n_vars)].set(0)

            epoch_loss = loss_val
            epoch_h_A = h_A
            epoch_recon_loss = recon_loss
            epoch_class_loss = class_loss
            epoch_effect_metrics = eff_metrics

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_params = {k: v.copy() if hasattr(v, "copy") else v for k, v in all_params.items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Grow lambda_2
        if epoch_h_A > 1e-6:
            lambda_2 = min(lambda_2 * lambda_2_growth, lambda_2_max)

        if verbose and iter % 10 == 0:
            ate_val = float(epoch_effect_metrics.get("ATE", 0)) if epoch_effect_metrics else 0
            ate_str = f", ATE={ate_val:.4f}" if epoch_effect_metrics else ""
            print(
                f"Iter {iter}: loss={float(epoch_loss):.4f}, h(A)={float(epoch_h_A):.4f}, "
                f"recon={float(epoch_recon_loss):.4f}, class={float(epoch_class_loss):.4f}{ate_str}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at iteration {iter}")
            all_params = best_params
            break

    # Extract results
    A_final = all_params["A"]
    threshold = 0.05
    A_binary = (jnp.abs(A_final) > threshold).astype(jnp.float32)

    # Compute classification metrics
    # Merge trained params with metadata to get full processor params
    processor_params_final = merge_trained_params(processor_params, all_params["processor_params"])

    # Use A-weighted features for prediction
    weights_Y = jnp.abs(A_final[:, Y_idx])
    weights_Y = jnp.maximum(weights_Y, 0.01)  # Raw weights with floor (no L1-norm)
    X_for_pred = data_train * weights_Y[jnp.newaxis, :]

    # Get predictions from processor
    proc_params_Y = processor_params_final[Y_idx]

    Y_pred_logits = processor.forward(X_for_pred, proc_params_Y)

    Y_pred_prob = jax.nn.sigmoid(Y_pred_logits)
    Y_pred_binary = (Y_pred_prob > 0.5).astype(jnp.float32)

    # Compute classification metrics
    Y_pred_flat = Y_pred_binary.flatten()
    Y_true_flat = Y_train.flatten()
    tp = float(jnp.sum((Y_pred_flat == 1) & (Y_true_flat == 1)))
    tn = float(jnp.sum((Y_pred_flat == 0) & (Y_true_flat == 0)))
    fp = float(jnp.sum((Y_pred_flat == 1) & (Y_true_flat == 0)))
    fn = float(jnp.sum((Y_pred_flat == 0) & (Y_true_flat == 1)))

    classification_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    balanced_accuracy = (recall + specificity) / 2

    # Markov blanket
    parents_Y = [int(i) for i in range(n_vars) if i != Y_idx and float(A_binary[i, Y_idx]) > 0]
    children_Y = [int(j) for j in range(n_vars) if j != Y_idx and float(A_binary[Y_idx, j]) > 0]
    markov_blanket = sorted(set(parents_Y + children_Y))

    # Compute final metrics
    metrics = {
        "n_edges": int(jnp.sum(A_binary)),
        "final_loss": float(epoch_loss),
        "final_h_A": float(epoch_h_A),
        "iterations": iter + 1,
        # Classification metrics
        "classification_accuracy": classification_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        # Structure
        "markov_blanket": markov_blanket,
        "markov_blanket_size": len(markov_blanket),
        # DAG structure (can be used to build graph visualization)
        "A_binary": A_binary.tolist() if hasattr(A_binary, "tolist") else A_binary,
        "A_weights": A_final.tolist() if hasattr(A_final, "tolist") else A_final,
    }

    # Add effect metrics (convert JAX arrays to float)
    if enable_effects and epoch_effect_metrics:
        metrics.update(
            {
                "ATE": float(epoch_effect_metrics.get("ATE", 0.0)),
                "CATE_std": float(epoch_effect_metrics.get("CATE_std", 0.0)),
                "L_outcome": float(epoch_effect_metrics.get("L_outcome", 0.0)),
                "L_propensity": float(epoch_effect_metrics.get("L_propensity", 0.0)),
            }
        )

    if use_latent_confounders:
        L_final = all_params["U"] @ all_params["V"].T
        metrics["L_nuclear_norm"] = float(jnp.sum(jnp.linalg.svd(L_final, compute_uv=False)))

    if verbose:
        print(f"\nFinal: {metrics['n_edges']} edges")
        print(f"Classification: Acc={classification_accuracy:.2%}, BAcc={balanced_accuracy:.2%}")
        print(f"MB={markov_blanket}")
        if enable_effects:
            print(f"Effect estimates: ATE={metrics.get('ATE', 0):.4f}")

    return A_binary, effect_processor, processor_params_final, metrics


# ============================================================================
# Stage 1: Multi-Edge Effect Estimation (v7.0)
# ============================================================================


def compute_ate_for_single_treatment(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    T: jnp.ndarray,
    L: jnp.ndarray,
    treatment_idx: int,
    Y_idx: int = None,
    include_L: bool = True,
    key: random.PRNGKey = None,
    n_hidden: int = 64,
    n_steps: int = 200,
    lr: float = 0.01,
) -> dict:
    """
    Compute ATE for a single treatment using TARNet-style estimation.

    This is a standalone effect estimator that:
    1. Takes pre-computed latent confounders L
    2. Builds input features excluding treatment AND Y (+ L if include_L)
    3. Trains simple TARNet heads (Y0, Y1, propensity)
    4. Returns ATE and uncertainty

    Args:
        X: Features (n_samples, n_features) - may include Y column
        Y: Outcome (n_samples,)
        T: Treatment indicator (n_samples,) - binary or continuous
        L: Latent confounders (n_samples, n_latent) or None
        treatment_idx: Index of treatment feature in X
        Y_idx: Index of Y in X (to exclude from covariates). If None, assumes Y not in X.
        include_L: Whether to include L as covariates (Path 1)
        key: JAX random key
        n_hidden: Hidden dimension for effect heads
        n_steps: Training steps for effect heads
        lr: Learning rate

    Returns:
        dict with 'ATE', 'ATE_std', 'propensity_mean', 'feature_type'
    """
    if key is None:
        key = random.PRNGKey(0)

    n_samples, n_features = X.shape

    # Detect feature type
    n_unique = len(jnp.unique(T))
    feature_type = "binary" if n_unique <= 10 else "continuous"

    # Build input: X without treatment column AND without Y column + L
    # CRITICAL: Y must not be a covariate when predicting Y!
    if Y_idx is not None:
        mask = (jnp.arange(n_features) != treatment_idx) & (jnp.arange(n_features) != Y_idx)
    else:
        mask = jnp.arange(n_features) != treatment_idx
    X_no_treat = X[:, mask]

    if include_L and L is not None:
        effect_input = jnp.concatenate([X_no_treat, L], axis=1)
    else:
        effect_input = X_no_treat

    input_dim = effect_input.shape[1]

    # For binary treatment: standard TARNet
    # For continuous: simple regression approach (VCNet in Stage 2)
    if feature_type == "binary":
        return _compute_ate_binary(effect_input, Y, T, key, input_dim, n_hidden, n_steps, lr)
    else:
        return _compute_ate_continuous(effect_input, Y, T, key, input_dim, n_hidden, n_steps, lr)


def _compute_ate_binary(effect_input, Y, T, key, input_dim, n_hidden, n_steps, lr):
    """TARNet for binary treatment."""
    n_samples = effect_input.shape[0]

    # Initialize parameters for 3 heads: Y(0), Y(1), propensity
    key, k1, k2, k3, k4, k5, k6 = random.split(key, 7)

    # Y(0) head
    W0_h = random.normal(k1, (input_dim, n_hidden)) * 0.1
    b0_h = jnp.zeros(n_hidden)
    W0_o = random.normal(k2, (n_hidden, 1)) * 0.1
    b0_o = jnp.zeros(1)

    # Y(1) head
    W1_h = random.normal(k3, (input_dim, n_hidden)) * 0.1
    b1_h = jnp.zeros(n_hidden)
    W1_o = random.normal(k4, (n_hidden, 1)) * 0.1
    b1_o = jnp.zeros(1)

    # Propensity head
    Wp_h = random.normal(k5, (input_dim, n_hidden)) * 0.1
    bp_h = jnp.zeros(n_hidden)
    Wp_o = random.normal(k6, (n_hidden, 1)) * 0.1
    bp_o = jnp.zeros(1)

    params = {
        "W0_h": W0_h,
        "b0_h": b0_h,
        "W0_o": W0_o,
        "b0_o": b0_o,
        "W1_h": W1_h,
        "b1_h": b1_h,
        "W1_o": W1_o,
        "b1_o": b1_o,
        "Wp_h": Wp_h,
        "bp_h": bp_h,
        "Wp_o": Wp_o,
        "bp_o": bp_o,
    }

    def forward(params, X, T_val=None):
        # Y(0) prediction
        h0 = jax.nn.relu(X @ params["W0_h"] + params["b0_h"])
        y0 = jax.nn.sigmoid(h0 @ params["W0_o"] + params["b0_o"]).squeeze()

        # Y(1) prediction
        h1 = jax.nn.relu(X @ params["W1_h"] + params["b1_h"])
        y1 = jax.nn.sigmoid(h1 @ params["W1_o"] + params["b1_o"]).squeeze()

        # Propensity
        hp = jax.nn.relu(X @ params["Wp_h"] + params["bp_h"])
        prop = jax.nn.sigmoid(hp @ params["Wp_o"] + params["bp_o"]).squeeze()

        return y0, y1, prop

    def loss_fn(params, X, Y, T):
        y0, y1, prop = forward(params, X)

        # Factual outcome loss
        y_pred = jnp.where(T > 0.5, y1, y0)
        eps = 1e-7
        outcome_loss = -jnp.mean(Y * jnp.log(y_pred + eps) + (1 - Y) * jnp.log(1 - y_pred + eps))

        # Propensity loss
        prop_loss = -jnp.mean(T * jnp.log(prop + eps) + (1 - T) * jnp.log(1 - prop + eps))

        return outcome_loss + 0.5 * prop_loss

    # Training loop
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, X, Y, T):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, Y, T)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(n_steps):
        params, opt_state, _ = train_step(params, opt_state, effect_input, Y, T)

    # Compute ATE
    y0_final, y1_final, prop_final = forward(params, effect_input)
    ite = y1_final - y0_final  # Individual treatment effects
    ate = float(jnp.mean(ite))
    ate_std = float(jnp.std(ite))

    return {
        "ATE": ate,
        "ATE_std": ate_std,
        "CATE_std": ate_std,
        "propensity_mean": float(jnp.mean(prop_final)),
        "feature_type": "binary",
        "n_treated": int(jnp.sum(T > 0.5)),
        "n_control": int(jnp.sum(T <= 0.5)),
    }


def _compute_ate_continuous(effect_input, Y, T, key, input_dim, n_hidden, n_steps, lr):
    """
    VCNet (Varying Coefficient Network) for continuous treatment effect estimation.

    Stage 2 Implementation:
    - Feature encoder: Φ(X) → latent representation
    - Treatment encoder: ψ(T) → treatment embedding
    - Varying coefficients: β(T) = f(ψ(T))
    - Outcome prediction: Y(T) = sigmoid(β(T) · Φ(X))
    - ADRF (Average Dose-Response Function): E[Y(t)] for t ∈ [0, 1]

    Key advantages over simple regression:
    - Smooth dose-response curves
    - Treatment-covariate interaction modeling
    - Interpretable varying coefficients
    """
    n_samples = effect_input.shape[0]
    latent_dim = min(32, n_hidden // 2)  # Latent representation dimension
    treatment_embed_dim = 16  # Treatment embedding dimension

    # Normalize treatment to [0, 1]
    T_min, T_max = float(jnp.min(T)), float(jnp.max(T))
    T_norm = (T - T_min) / (T_max - T_min + 1e-8)

    # =========================================================================
    # Initialize VCNet Parameters
    # =========================================================================
    key, k1, k2, k3, k4, k5, k6 = random.split(key, 7)

    # Feature encoder: X → Φ(X)
    W_phi_1 = random.normal(k1, (input_dim, n_hidden)) * 0.1
    b_phi_1 = jnp.zeros(n_hidden)
    W_phi_2 = random.normal(k2, (n_hidden, latent_dim)) * 0.1
    b_phi_2 = jnp.zeros(latent_dim)

    # Treatment encoder: T → ψ(T)
    W_psi_1 = random.normal(k3, (1, treatment_embed_dim)) * 0.1
    b_psi_1 = jnp.zeros(treatment_embed_dim)
    W_psi_2 = random.normal(k4, (treatment_embed_dim, treatment_embed_dim)) * 0.1
    b_psi_2 = jnp.zeros(treatment_embed_dim)

    # Varying coefficient generator: ψ(T) → β(T)
    W_beta = random.normal(k5, (treatment_embed_dim, latent_dim)) * 0.1
    b_beta = jnp.zeros(latent_dim)

    # GPS (Generalized Propensity Score) head for continuous treatment
    W_gps = random.normal(k6, (input_dim, 2)) * 0.1  # mu, log_sigma
    b_gps = jnp.zeros(2)

    params = {
        # Feature encoder
        "W_phi_1": W_phi_1,
        "b_phi_1": b_phi_1,
        "W_phi_2": W_phi_2,
        "b_phi_2": b_phi_2,
        # Treatment encoder
        "W_psi_1": W_psi_1,
        "b_psi_1": b_psi_1,
        "W_psi_2": W_psi_2,
        "b_psi_2": b_psi_2,
        # Varying coefficients
        "W_beta": W_beta,
        "b_beta": b_beta,
        # GPS
        "W_gps": W_gps,
        "b_gps": b_gps,
    }

    # =========================================================================
    # VCNet Forward Pass
    # =========================================================================
    def feature_encoder(params, X):
        """Encode features: X → Φ(X)"""
        h = jax.nn.relu(X @ params["W_phi_1"] + params["b_phi_1"])
        phi = jax.nn.tanh(h @ params["W_phi_2"] + params["b_phi_2"])  # Bounded
        return phi

    def treatment_encoder(params, T_val):
        """Encode treatment: T → ψ(T)"""
        T_input = T_val.reshape(-1, 1)
        h = jax.nn.elu(T_input @ params["W_psi_1"] + params["b_psi_1"])
        psi = jax.nn.elu(h @ params["W_psi_2"] + params["b_psi_2"])
        return psi

    def varying_coefficients(params, psi):
        """Generate varying coefficients: ψ(T) → β(T)"""
        beta = jax.nn.tanh(psi @ params["W_beta"] + params["b_beta"])  # Bounded [-1, 1]
        return beta

    def vcnet_forward(params, X, T_val):
        """Full VCNet forward: Y(T) = sigmoid(β(T) · Φ(X))"""
        phi = feature_encoder(params, X)  # (n, latent_dim)
        psi = treatment_encoder(params, T_val)  # (n, treatment_embed_dim)
        beta = varying_coefficients(params, psi)  # (n, latent_dim)

        # Outcome = dot product of varying coefficients and features
        logits = jnp.sum(beta * phi, axis=-1)  # (n,)
        y_pred = jax.nn.sigmoid(logits)
        return y_pred

    def gps_forward(params, X, T_val):
        """Generalized Propensity Score: P(T|X) assuming Gaussian"""
        gps_out = X @ params["W_gps"] + params["b_gps"]
        mu = gps_out[:, 0]
        log_sigma = jnp.clip(gps_out[:, 1], -3, 3)  # Stability
        sigma = jnp.exp(log_sigma)

        # Gaussian log-likelihood
        log_prob = -0.5 * ((T_val - mu) / sigma) ** 2 - log_sigma - 0.5 * jnp.log(2 * jnp.pi)
        return log_prob, mu, sigma

    # =========================================================================
    # Loss Function
    # =========================================================================
    def loss_fn(params, X, Y, T_val):
        # Outcome loss (BCE)
        y_pred = vcnet_forward(params, X, T_val)
        eps = 1e-7
        outcome_loss = -jnp.mean(Y * jnp.log(y_pred + eps) + (1 - Y) * jnp.log(1 - y_pred + eps))

        # GPS loss (negative log-likelihood)
        gps_log_prob, _, _ = gps_forward(params, X, T_val)
        gps_loss = -jnp.mean(gps_log_prob)

        # Targeted regularization: encourage smooth dose-response
        # Sample random treatment values and penalize large second derivatives
        key_reg = random.PRNGKey(0)
        T_samples = random.uniform(key_reg, (100,))
        T_samples_expanded = jnp.tile(T_samples, (X.shape[0], 1)).T  # (100, n)

        # Compute predictions at multiple doses for a subset of samples
        subset_idx = jnp.arange(min(50, X.shape[0]))
        X_subset = X[subset_idx]
        y_at_doses = jnp.array(
            [vcnet_forward(params, X_subset, jnp.full(len(subset_idx), t)) for t in T_samples[:10]]
        )  # (10, subset_size)

        # Smoothness: penalize variance across doses (encourages smooth curves)
        smoothness_loss = jnp.mean(jnp.var(y_at_doses, axis=0))

        return outcome_loss + 0.3 * gps_loss + 0.1 * smoothness_loss

    # =========================================================================
    # Training
    # =========================================================================
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, X, Y, T_val):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, Y, T_val)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(n_steps):
        params, opt_state, _ = train_step(params, opt_state, effect_input, Y, T_norm)

    # =========================================================================
    # Estimate ADRF and ATE
    # =========================================================================
    n_dose_points = 20
    T_grid = jnp.linspace(0, 1, n_dose_points)

    # Compute ADRF: E[Y(t)] for each dose level
    adrf = []
    for t in T_grid:
        T_batch = jnp.full(n_samples, t)
        y_at_t = vcnet_forward(params, effect_input, T_batch)
        adrf.append(float(jnp.mean(y_at_t)))
    adrf = jnp.array(adrf)

    # ATE = ADRF(T=1) - ADRF(T=0)
    y_high = vcnet_forward(params, effect_input, jnp.ones(n_samples))
    y_low = vcnet_forward(params, effect_input, jnp.zeros(n_samples))
    ite = y_high - y_low
    ate = float(jnp.mean(ite))
    ate_std = float(jnp.std(ite))

    # Compute GPS statistics
    _, gps_mu, gps_sigma = gps_forward(params, effect_input, T_norm)

    return {
        "ATE": ate,
        "ATE_std": ate_std,
        "CATE_std": ate_std,
        "feature_type": "continuous",
        "T_range": [T_min, T_max],
        # VCNet-specific outputs
        "ADRF": [float(x) for x in adrf],  # Average Dose-Response Function
        "ADRF_doses": [float(x) for x in T_grid * (T_max - T_min) + T_min],  # Original scale
        "GPS_mu_mean": float(jnp.mean(gps_mu)),
        "GPS_sigma_mean": float(jnp.mean(gps_sigma)),
    }


# ============================================================================
# Stage 3: Mediation Analysis (NDE/NIE)
# ============================================================================


def identify_mediation_paths(A_binary: jnp.ndarray, Y_idx: int) -> list:
    """
    Identify all mediation paths T → M → Y in the learned DAG.

    A mediation path exists when:
    1. A[T, M] > 0  (T causes M)
    2. A[M, Y] > 0  (M causes Y)
    3. A[T, Y] > 0  (T also directly causes Y) - optional but typical

    Args:
        A_binary: Thresholded adjacency matrix (n_vars, n_vars)
        Y_idx: Index of outcome variable

    Returns:
        List of tuples: [(treatment_idx, mediator_idx, has_direct_effect), ...]
    """
    n_vars = A_binary.shape[0]
    mediation_paths = []

    # Find all variables that directly affect Y (potential treatments)
    parents_of_Y = [i for i in range(n_vars) if i != Y_idx and float(A_binary[i, Y_idx]) > 0]

    for t_idx in parents_of_Y:
        # Find children of T that are also parents of Y (mediators)
        children_of_T = [
            j for j in range(n_vars) if j != t_idx and j != Y_idx and float(A_binary[t_idx, j]) > 0
        ]

        for m_idx in children_of_T:
            # Check if M → Y exists
            if float(A_binary[m_idx, Y_idx]) > 0:
                # Found mediation path: T → M → Y
                has_direct = float(A_binary[t_idx, Y_idx]) > 0
                mediation_paths.append((t_idx, m_idx, has_direct))

    return mediation_paths


def compute_mediation_effects(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    treatment_idx: int,
    mediator_idx: int,
    L: jnp.ndarray = None,
    include_L: bool = True,
    key: random.PRNGKey = None,
    n_hidden: int = 64,
    n_steps: int = 200,
    lr: float = 0.01,
) -> dict:
    """
    Compute Natural Direct Effect (NDE) and Natural Indirect Effect (NIE).

    Mediation formulas (Pearl):
    - NDE = E[Y(1, M(0)) - Y(0, M(0))]  # Direct effect (not through M)
    - NIE = E[Y(0, M(1)) - Y(0, M(0))]  # Indirect effect (through M only)
    - TE = NDE + NIE (under no interaction assumption)

    Implementation:
    1. Fit mediator model: M = g(T, X, L)
    2. Fit outcome model: Y = f(T, M, X, L)
    3. Counterfactual simulation for NDE/NIE

    Args:
        X: Features (n_samples, n_features)
        Y: Outcome (n_samples,)
        treatment_idx: Index of treatment variable in X
        mediator_idx: Index of mediator variable in X
        L: Latent confounders (n_samples, n_latent) or None
        include_L: Whether to include L as covariates
        key: JAX random key
        n_hidden: Hidden dimension
        n_steps: Training steps
        lr: Learning rate

    Returns:
        dict with 'NDE', 'NIE', 'total_effect', 'proportion_mediated'
    """
    if key is None:
        key = random.PRNGKey(0)

    n_samples, n_features = X.shape

    # Extract treatment and mediator
    T = X[:, treatment_idx]
    M = X[:, mediator_idx]

    # Build covariate matrix (excluding T and M)
    mask = jnp.array([i != treatment_idx and i != mediator_idx for i in range(n_features)])
    X_cov = X[:, mask]

    if include_L and L is not None:
        covariates = jnp.concatenate([X_cov, L], axis=1)
    else:
        covariates = X_cov

    cov_dim = covariates.shape[1]

    # Detect mediator type BEFORE JIT (can't use jnp.unique inside traced functions)
    mediator_is_binary = len(jnp.unique(M)) <= 10

    # =========================================================================
    # Step 1: Fit Mediator Model M = g(T, X, L)
    # =========================================================================
    key, k1, k2 = random.split(key, 3)

    # Simple MLP for mediator prediction
    Wm_h = random.normal(k1, (cov_dim + 1, n_hidden)) * 0.1  # +1 for T
    bm_h = jnp.zeros(n_hidden)
    Wm_o = random.normal(k2, (n_hidden, 1)) * 0.1
    bm_o = jnp.zeros(1)

    mediator_params = {"Wm_h": Wm_h, "bm_h": bm_h, "Wm_o": Wm_o, "bm_o": bm_o}

    def mediator_forward(params, X_cov, T_val):
        X_input = jnp.concatenate([X_cov, T_val.reshape(-1, 1)], axis=1)
        h = jax.nn.relu(X_input @ params["Wm_h"] + params["bm_h"])
        m_pred = jax.nn.sigmoid(h @ params["Wm_o"] + params["bm_o"]).squeeze()
        return m_pred

    def mediator_loss_binary(params, X_cov, T_val, M_true):
        m_pred = mediator_forward(params, X_cov, T_val)
        eps = 1e-7
        return -jnp.mean(M_true * jnp.log(m_pred + eps) + (1 - M_true) * jnp.log(1 - m_pred + eps))

    def mediator_loss_continuous(params, X_cov, T_val, M_true):
        m_pred = mediator_forward(params, X_cov, T_val)
        return jnp.mean((m_pred - M_true) ** 2)

    # Select loss function based on mediator type (detected before JIT)
    mediator_loss = mediator_loss_binary if mediator_is_binary else mediator_loss_continuous

    # Train mediator model
    optimizer_m = optax.adam(lr)
    opt_state_m = optimizer_m.init(mediator_params)

    @jax.jit
    def train_mediator_step(params, opt_state, X_cov, T, M):
        loss, grads = jax.value_and_grad(mediator_loss)(params, X_cov, T, M)
        updates, opt_state = optimizer_m.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(n_steps):
        mediator_params, opt_state_m, _ = train_mediator_step(
            mediator_params, opt_state_m, covariates, T, M
        )

    # =========================================================================
    # Step 2: Fit Outcome Model Y = f(T, M, X, L)
    # =========================================================================
    key, k3, k4 = random.split(key, 3)

    # MLP for outcome prediction
    Wo_h = random.normal(k3, (cov_dim + 2, n_hidden)) * 0.1  # +2 for T and M
    bo_h = jnp.zeros(n_hidden)
    Wo_o = random.normal(k4, (n_hidden, 1)) * 0.1
    bo_o = jnp.zeros(1)

    outcome_params = {"Wo_h": Wo_h, "bo_h": bo_h, "Wo_o": Wo_o, "bo_o": bo_o}

    def outcome_forward(params, X_cov, T_val, M_val):
        X_input = jnp.concatenate([X_cov, T_val.reshape(-1, 1), M_val.reshape(-1, 1)], axis=1)
        h = jax.nn.relu(X_input @ params["Wo_h"] + params["bo_h"])
        y_pred = jax.nn.sigmoid(h @ params["Wo_o"] + params["bo_o"]).squeeze()
        return y_pred

    def outcome_loss(params, X_cov, T_val, M_val, Y_true):
        y_pred = outcome_forward(params, X_cov, T_val, M_val)
        eps = 1e-7
        return -jnp.mean(Y_true * jnp.log(y_pred + eps) + (1 - Y_true) * jnp.log(1 - y_pred + eps))

    # Train outcome model
    optimizer_o = optax.adam(lr)
    opt_state_o = optimizer_o.init(outcome_params)

    @jax.jit
    def train_outcome_step(params, opt_state, X_cov, T, M, Y):
        loss, grads = jax.value_and_grad(outcome_loss)(params, X_cov, T, M, Y)
        updates, opt_state = optimizer_o.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for _ in range(n_steps):
        outcome_params, opt_state_o, _ = train_outcome_step(
            outcome_params, opt_state_o, covariates, T, M, Y
        )

    # =========================================================================
    # Step 3: Counterfactual Simulation
    # =========================================================================
    T_0 = jnp.zeros(n_samples)
    T_1 = jnp.ones(n_samples)

    # M(0): Mediator value under T=0
    M_0 = mediator_forward(mediator_params, covariates, T_0)

    # M(1): Mediator value under T=1
    M_1 = mediator_forward(mediator_params, covariates, T_1)

    # Counterfactual outcomes
    Y_00 = outcome_forward(outcome_params, covariates, T_0, M_0)  # Y(0, M(0))
    Y_10 = outcome_forward(outcome_params, covariates, T_1, M_0)  # Y(1, M(0))
    Y_01 = outcome_forward(outcome_params, covariates, T_0, M_1)  # Y(0, M(1))
    Y_11 = outcome_forward(outcome_params, covariates, T_1, M_1)  # Y(1, M(1))

    # NDE = E[Y(1, M(0)) - Y(0, M(0))]
    nde = float(jnp.mean(Y_10 - Y_00))

    # NIE = E[Y(0, M(1)) - Y(0, M(0))]
    nie = float(jnp.mean(Y_01 - Y_00))

    # Total Effect = E[Y(1, M(1)) - Y(0, M(0))]
    te = float(jnp.mean(Y_11 - Y_00))

    # Alternative TE via path decomposition (should be close to NDE + NIE)
    te_decomposed = nde + nie

    # Proportion mediated (indirect / total)
    prop_mediated = nie / te if abs(te) > 1e-6 else 0.0

    return {
        "NDE": nde,  # Natural Direct Effect
        "NIE": nie,  # Natural Indirect Effect
        "total_effect": te,
        "total_effect_decomposed": te_decomposed,
        "proportion_mediated": prop_mediated,
        "treatment_idx": treatment_idx,
        "mediator_idx": mediator_idx,
        "M_0_mean": float(jnp.mean(M_0)),
        "M_1_mean": float(jnp.mean(M_1)),
    }


def learn_with_multi_effects(
    data: jnp.ndarray,
    Y: jnp.ndarray,
    Y_idx: int,
    processor,
    key: random.PRNGKey,
    processor_type: str = "transformer",
    # Latent confounder settings
    use_latent_confounders: bool = True,
    latent_rank_k: int = 3,
    lambda_L: float = 0.05,
    lambda_bow: float = 0.1,
    warm_start_L_iters: int = 20,
    # Effect estimation settings
    include_L_in_effects: bool = True,  # Path 1: L as covariates
    effect_n_hidden: int = 64,
    effect_n_steps: int = 200,
    effect_lr: float = 0.01,
    # Mediation analysis settings (Stage 3)
    compute_mediation: bool = True,  # Compute NDE/NIE for mediation paths
    # All edges settings (Stage 4)
    compute_all_edges: bool = False,  # Compute effects for ALL edges (Xi→Xj)
    # GOLEM settings
    lambda_1: float = 0.02,
    lambda_class: float = 1.0,
    max_iter: int = 50,
    lr: float = 0.001,
    verbose: bool = True,
) -> tuple:
    """
    Full causal effect estimation pipeline (Stages 1-4).

    This function:
    1. Runs GOLEM training to learn DAG structure A and latent confounders L
    2. Extracts Markov Blanket MB(Y) from learned A
    3. For each feature in MB(Y), computes causal effect to Y (Stage 1)
    4. Computes mediation effects for T→M→Y paths (Stage 3)
    5. Optionally computes effects for ALL edges in DAG (Stage 4)

    Args:
        data: Input features (n_samples, n_features)
        Y: Outcome variable (n_samples,)
        Y_idx: Index of outcome in data
        processor: Processor adapter (Transformer, MLP, etc.)
        key: JAX random key
        processor_type: Type of processor ('transformer', 'mlp', etc.)
        use_latent_confounders: Whether to learn latent confounders L
        latent_rank_k: Rank of latent confounder matrix
        lambda_L: Nuclear norm regularization for L
        lambda_bow: Bow-free constraint weight
        warm_start_L_iters: Iterations before activating L
        include_L_in_effects: Whether to include L in effect estimation (Path 1)
        effect_n_hidden: Hidden dim for effect estimation heads
        effect_n_steps: Training steps for effect estimation
        effect_lr: Learning rate for effect estimation
        compute_mediation: Compute NDE/NIE for mediation paths (Stage 3)
        compute_all_edges: Compute effects for ALL edges Xi→Xj (Stage 4)
        lambda_1: Sparsity regularization
        lambda_class: Classification loss weight
        max_iter: Maximum GOLEM iterations
        lr: GOLEM learning rate
        verbose: Print progress

    Returns:
        A_binary: Thresholded DAG adjacency matrix
        processor: Trained processor
        params: Final parameters
        metrics: Dict containing:
            - Classification metrics (accuracy, balanced_accuracy, etc.)
            - markov_blanket: List of MB feature indices
            - causal_effects: Dict mapping 'Xi→Y' to effect estimates (Stage 1)
            - mediation: Dict mapping 'Xi→Xj→Y' to NDE/NIE (Stage 3)
            - feature_effects: Dict mapping 'Xi→Xj' to effects (Stage 4)
            - L_info: Latent confounder information
        L: Latent confounder matrix (if use_latent_confounders=True)
    """
    if verbose:
        print("=" * 60)
        print("JCCE v7.0: Multi-Edge Causal Effect Estimation")
        print("=" * 60)
        print(
            f"Settings: include_L={include_L_in_effects}, latent_k={latent_rank_k}, "
            f"mediation={compute_mediation}, all_edges={compute_all_edges}"
        )

    # =========================================================================
    # Phase 1: GOLEM Training (DAG + Latent Confounders)
    # =========================================================================
    if verbose:
        print("\n--- Phase 1: GOLEM Training ---")

    key, subkey = random.split(key)

    if use_latent_confounders:
        A_binary, processor, params, base_metrics, L = _learn_structure_legacy(
            data=data,
            Y=Y,
            Y_idx=Y_idx,
            processor=processor,
            key=subkey,
            processor_type=processor_type,
            use_latent_confounders=True,
            latent_rank_k=latent_rank_k,
            lambda_L=lambda_L,
            lambda_bow=lambda_bow,
            warm_start_L_iters=warm_start_L_iters,
            lambda_1=lambda_1,
            lambda_class=lambda_class,
            max_iter=max_iter,
            lr=lr,
            verbose=verbose,
        )
    else:
        A_binary, processor, params, base_metrics = _learn_structure_legacy(
            data=data,
            Y=Y,
            Y_idx=Y_idx,
            processor=processor,
            key=subkey,
            processor_type=processor_type,
            use_latent_confounders=False,
            lambda_1=lambda_1,
            lambda_class=lambda_class,
            max_iter=max_iter,
            lr=lr,
            verbose=verbose,
        )
        L = None

    # Extract Markov Blanket
    markov_blanket = base_metrics.get("markov_blanket", [])

    if verbose:
        print(
            f"\nPhase 1 complete: MB={markov_blanket}, BAcc={base_metrics['balanced_accuracy']:.2%}"
        )

    # =========================================================================
    # Phase 2: Multi-Edge Effect Estimation
    # =========================================================================
    # IMPORTANT: L is computed only for training samples (80% by default)
    # We need to use matching subsets of data for effect estimation
    n_samples = data.shape[0]
    if L is not None:
        n_train = L.shape[0]
        # Use only training portion of data to match L
        data_for_effects = data[:n_train]
        Y_for_effects = Y[:n_train]
    else:
        data_for_effects = data
        Y_for_effects = Y
        n_train = n_samples

    if verbose:
        print("\n--- Phase 2: Multi-Edge Effect Estimation ---")
        print(f"Computing effects for {len(markov_blanket)} MB features → Y")
        if include_L_in_effects and L is not None:
            print(f"Including L ({L.shape[1]} latent dims) as covariates")
            print(f"Using {n_train}/{n_samples} samples (training subset matching L)")

    causal_effects = {}

    for i, feature_idx in enumerate(markov_blanket):
        key, subkey = random.split(key)

        # Get treatment variable (from matching subset)
        T = data_for_effects[:, feature_idx]

        if verbose:
            n_unique = len(jnp.unique(T))
            ftype = "binary" if n_unique <= 10 else "continuous"
            print(
                f"  [{i + 1}/{len(markov_blanket)}] X{feature_idx}→Y ({ftype}, {n_unique} unique)...",
                end=" ",
            )

        # Compute effect (using training subset that matches L)
        # CRITICAL: Pass Y_idx to exclude Y from covariates!
        effect = compute_ate_for_single_treatment(
            X=data_for_effects,
            Y=Y_for_effects,
            T=T,
            L=L,
            treatment_idx=feature_idx,
            Y_idx=Y_idx,  # Exclude Y from covariates
            include_L=include_L_in_effects,
            key=subkey,
            n_hidden=effect_n_hidden,
            n_steps=effect_n_steps,
            lr=effect_lr,
        )

        causal_effects[f"X{feature_idx}→Y"] = effect

        if verbose:
            print(f"ATE={effect['ATE']:.4f}")

    # =========================================================================
    # Phase 3: Mediation Analysis (Stage 3)
    # =========================================================================
    mediation_results = {}

    if compute_mediation:
        # Convert A_binary to numpy for path finding
        A_binary_np = jnp.array(A_binary)

        # Find mediation paths T → M → Y
        mediation_paths = identify_mediation_paths(A_binary_np, Y_idx)

        if verbose:
            print("\n--- Phase 3: Mediation Analysis ---")
            print(f"Found {len(mediation_paths)} mediation paths in DAG")

        if len(mediation_paths) > 0:
            for i, (t_idx, m_idx, has_direct) in enumerate(mediation_paths):
                key, subkey = random.split(key)

                if verbose:
                    path_str = f"X{t_idx}→X{m_idx}→Y"
                    direct_str = " (+ direct X{t_idx}→Y)" if has_direct else ""
                    print(f"  [{i + 1}/{len(mediation_paths)}] {path_str}{direct_str}...", end=" ")

                # Compute NDE/NIE for this mediation path
                med_effect = compute_mediation_effects(
                    X=data_for_effects,
                    Y=Y_for_effects,
                    treatment_idx=t_idx,
                    mediator_idx=m_idx,
                    L=L,
                    include_L=include_L_in_effects,
                    key=subkey,
                    n_hidden=effect_n_hidden,
                    n_steps=effect_n_steps,
                    lr=effect_lr,
                )

                path_key = f"X{t_idx}→X{m_idx}→Y"
                mediation_results[path_key] = med_effect

                if verbose:
                    print(
                        f"NDE={med_effect['NDE']:.4f}, NIE={med_effect['NIE']:.4f}, "
                        f"%Med={med_effect['proportion_mediated'] * 100:.1f}%"
                    )
        else:
            if verbose:
                print("  No mediation paths found (no T→M→Y structure in DAG)")

    # =========================================================================
    # Phase 4: All Edge Effects (Stage 4)
    # =========================================================================
    feature_effects = {}

    if compute_all_edges:
        # Convert A_binary to numpy for edge iteration
        A_binary_np = jnp.array(A_binary)
        n_vars = A_binary_np.shape[0]

        # Find all edges (excluding edges to Y, which are already computed)
        all_edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                if A_binary_np[i, j] > 0 and j != Y_idx:
                    # Edge i → j exists and j is not Y
                    all_edges.append((i, j))

        if verbose:
            print("\n--- Phase 4: All Edge Effects ---")
            print(f"Computing effects for {len(all_edges)} feature→feature edges")

        if len(all_edges) > 0:
            for idx, (src, dst) in enumerate(all_edges):
                key, subkey = random.split(key)

                # Treatment = source feature, Outcome = destination feature
                T_edge = data_for_effects[:, src]
                Y_edge = data_for_effects[:, dst]

                # Detect types
                n_unique_t = len(jnp.unique(T_edge))
                n_unique_y = len(jnp.unique(Y_edge))
                t_type = "binary" if n_unique_t <= 10 else "continuous"
                y_type = "binary" if n_unique_y <= 10 else "continuous"

                if verbose:
                    print(
                        f"  [{idx + 1}/{len(all_edges)}] X{src}→X{dst} ({t_type} T, {y_type} Y)...",
                        end=" ",
                    )

                # Build effect input (exclude both src and dst features)
                mask = jnp.array([k != src and k != dst for k in range(n_vars)])
                X_effect = data_for_effects[:, mask]

                if include_L_in_effects and L is not None:
                    effect_input = jnp.concatenate([X_effect, L], axis=1)
                else:
                    effect_input = X_effect

                input_dim = effect_input.shape[1]

                # Compute effect based on treatment type
                if t_type == "binary":
                    # Use TARNet for binary treatment
                    effect = _compute_ate_binary(
                        effect_input,
                        Y_edge,
                        T_edge,
                        subkey,
                        input_dim,
                        effect_n_hidden,
                        effect_n_steps,
                        effect_lr,
                    )
                else:
                    # Use VCNet for continuous treatment
                    effect = _compute_ate_continuous(
                        effect_input,
                        Y_edge,
                        T_edge,
                        subkey,
                        input_dim,
                        effect_n_hidden,
                        effect_n_steps,
                        effect_lr,
                    )

                # Add metadata
                effect["source_idx"] = src
                effect["target_idx"] = dst
                effect["target_type"] = y_type

                edge_key = f"X{src}→X{dst}"
                feature_effects[edge_key] = effect

                if verbose:
                    print(f"ATE={effect['ATE']:.4f}")
        else:
            if verbose:
                print("  No feature→feature edges found (all edges go to Y)")

    # =========================================================================
    # Compile Final Metrics
    # =========================================================================
    metrics = {
        **base_metrics,
        "causal_effects": causal_effects,
        "mediation": mediation_results,
        "feature_effects": feature_effects,
        "include_L_in_effects": include_L_in_effects,
        "effect_estimation": {
            "n_effects_computed": len(causal_effects),
            "n_mediation_paths": len(mediation_results),
            "n_feature_effects": len(feature_effects),
            "n_hidden": effect_n_hidden,
            "n_steps": effect_n_steps,
        },
    }

    if L is not None:
        metrics["L_info"] = {
            "shape": list(L.shape),
            "rank": latent_rank_k,
            "nuclear_norm": float(jnp.sum(jnp.linalg.svd(L, compute_uv=False))),
        }

    if verbose:
        print("\n" + "=" * 60)
        print("Multi-Edge Effect Estimation Complete")
        print("=" * 60)
        print(
            f"Classification: Acc={metrics['classification_accuracy']:.2%}, BAcc={metrics['balanced_accuracy']:.2%}"
        )
        print(f"Markov Blanket: {markov_blanket}")
        print("\nCausal Effects (MB → Y):")
        for edge, eff in causal_effects.items():
            print(
                f"  {edge}: ATE={eff['ATE']:.4f} ± {eff.get('ATE_std', 0):.4f} ({eff['feature_type']})"
            )

        if mediation_results:
            print("\nMediation Analysis:")
            for path, med in mediation_results.items():
                print(f"  {path}:")
                print(f"    Total Effect:    {med['total_effect']:.4f}")
                print(f"    Direct (NDE):    {med['NDE']:.4f}")
                print(f"    Indirect (NIE):  {med['NIE']:.4f}")
                print(f"    % Mediated:      {med['proportion_mediated'] * 100:.1f}%")

        if feature_effects:
            print("\nFeature→Feature Effects (Stage 4):")
            for edge, eff in feature_effects.items():
                print(
                    f"  {edge}: ATE={eff['ATE']:.4f} ± {eff.get('ATE_std', 0):.4f} "
                    f"({eff['feature_type']}→{eff['target_type']})"
                )

    if use_latent_confounders:
        return A_binary, processor, params, metrics, L
    else:
        return A_binary, processor, params, metrics


# ============================================================================
# Stage 5: DAG Evaluation Metrics
# ============================================================================


def compute_dag_metrics(
    A_learned: jnp.ndarray, A_true: jnp.ndarray, threshold: float = 0.05
) -> dict:
    """
    Compute DAG structure evaluation metrics.

    Args:
        A_learned: Learned adjacency matrix (can be weighted or binary)
        A_true: Ground truth adjacency matrix (binary)
        threshold: Threshold to binarize A_learned if weighted

    Returns:
        dict with: SHD, TPR, FPR, FDR, precision, recall, F1,
                   n_true_edges, n_learned_edges, n_correct_edges
    """
    # Binarize learned matrix if needed
    A_learned = jnp.array(A_learned)
    A_true = jnp.array(A_true)

    if A_learned.max() > 1.0 or (A_learned > 0).sum() != (A_learned == 1).sum():
        A_pred = (jnp.abs(A_learned) > threshold).astype(jnp.float32)
    else:
        A_pred = A_learned

    # Remove diagonal (no self-loops)
    n = A_pred.shape[0]
    mask = 1 - jnp.eye(n)
    A_pred = A_pred * mask
    A_true = A_true * mask

    # True/False Positives/Negatives
    TP = jnp.sum((A_pred == 1) & (A_true == 1))  # Correct edges
    FP = jnp.sum((A_pred == 1) & (A_true == 0))  # Spurious edges
    FN = jnp.sum((A_pred == 0) & (A_true == 1))  # Missing edges
    TN = jnp.sum((A_pred == 0) & (A_true == 0))  # Correct non-edges

    # Structural Hamming Distance = FP + FN (edge errors)
    SHD = int(FP + FN)

    # Rates
    TPR = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0  # Recall/Sensitivity
    FPR = float(FP / (FP + TN)) if (FP + TN) > 0 else 0.0  # False Positive Rate
    FDR = float(FP / (FP + TP)) if (FP + TP) > 0 else 0.0  # False Discovery Rate

    # Precision, Recall, F1
    precision = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    recall = TPR
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "SHD": SHD,
        "TPR": TPR,
        "FPR": FPR,
        "FDR": FDR,
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
        "n_true_edges": int(jnp.sum(A_true)),
        "n_learned_edges": int(jnp.sum(A_pred)),
        "n_correct_edges": int(TP),
    }


def compute_markov_blanket_metrics(mb_learned: list, mb_true: list) -> dict:
    """
    Compute Markov Blanket evaluation metrics.

    Args:
        mb_learned: List of learned MB feature indices
        mb_true: List of true MB feature indices

    Returns:
        dict with: precision, recall, F1, jaccard, n_correct, n_extra, n_missing
    """
    mb_learned_set = set(mb_learned)
    mb_true_set = set(mb_true)

    correct = mb_learned_set & mb_true_set
    extra = mb_learned_set - mb_true_set  # False positives
    missing = mb_true_set - mb_learned_set  # False negatives

    precision = len(correct) / len(mb_learned_set) if mb_learned_set else 0.0
    recall = len(correct) / len(mb_true_set) if mb_true_set else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Jaccard similarity
    union = mb_learned_set | mb_true_set
    jaccard = len(correct) / len(union) if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "jaccard": jaccard,
        "n_correct": len(correct),
        "n_extra": len(extra),
        "n_missing": len(missing),
        "correct_features": sorted(list(correct)),
        "extra_features": sorted(list(extra)),
        "missing_features": sorted(list(missing)),
    }


def get_true_markov_blanket(A_true: jnp.ndarray, Y_idx: int) -> list:
    """
    Compute true Markov Blanket from ground truth DAG.

    MB(Y) = Parents(Y) ∪ Children(Y) ∪ Parents_of_Children(Y) - {Y}

    Args:
        A_true: Ground truth adjacency matrix (A[i,j]=1 means i→j)
        Y_idx: Index of target variable

    Returns:
        List of MB feature indices
    """
    A = jnp.array(A_true)
    n = A.shape[0]

    mb = set()

    # Parents of Y: nodes i where A[i, Y] = 1
    parents = set(int(i) for i in range(n) if A[i, Y_idx] > 0)
    mb.update(parents)

    # Children of Y: nodes j where A[Y, j] = 1
    children = set(int(j) for j in range(n) if A[Y_idx, j] > 0)
    mb.update(children)

    # Parents of children (spouses)
    for child in children:
        parents_of_child = set(int(i) for i in range(n) if A[i, child] > 0 and i != Y_idx)
        mb.update(parents_of_child)

    # Remove Y itself
    mb.discard(Y_idx)

    return sorted(list(mb))


def evaluate_against_ground_truth(
    A_learned: jnp.ndarray,
    A_true: jnp.ndarray,
    Y_idx: int,
    mb_learned: list = None,
    threshold: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Comprehensive evaluation of learned DAG against ground truth.

    Args:
        A_learned: Learned adjacency matrix
        A_true: Ground truth adjacency matrix
        Y_idx: Index of target variable
        mb_learned: Learned Markov Blanket (optional, computed from A if None)
        threshold: Threshold for binarizing A_learned
        verbose: Print results

    Returns:
        dict with DAG metrics and MB metrics
    """
    # Compute DAG metrics
    dag_metrics = compute_dag_metrics(A_learned, A_true, threshold)

    # Compute true MB
    mb_true = get_true_markov_blanket(A_true, Y_idx)

    # Compute learned MB if not provided
    if mb_learned is None:
        A_binary = (jnp.abs(jnp.array(A_learned)) > threshold).astype(jnp.float32)
        mb_learned = [
            int(i) for i in range(A_binary.shape[0]) if A_binary[i, Y_idx] > 0 and i != Y_idx
        ]

    # Compute MB metrics
    mb_metrics = compute_markov_blanket_metrics(mb_learned, mb_true)

    if verbose:
        print("\n" + "=" * 60)
        print("DAG Evaluation Against Ground Truth")
        print("=" * 60)
        print("\nStructural Metrics:")
        print(f"  SHD (Structural Hamming Distance): {dag_metrics['SHD']}")
        print(
            f"  Edges: {dag_metrics['n_learned_edges']} learned / {dag_metrics['n_true_edges']} true / {dag_metrics['n_correct_edges']} correct"
        )
        print(f"  Precision: {dag_metrics['precision']:.3f}")
        print(f"  Recall (TPR): {dag_metrics['recall']:.3f}")
        print(f"  F1 Score: {dag_metrics['F1']:.3f}")
        print(f"  FPR: {dag_metrics['FPR']:.4f}, FDR: {dag_metrics['FDR']:.3f}")

        print("\nMarkov Blanket Metrics:")
        print(f"  True MB: {mb_true}")
        print(f"  Learned MB: {list(mb_learned)}")
        print(f"  Precision: {mb_metrics['precision']:.3f}")
        print(f"  Recall: {mb_metrics['recall']:.3f}")
        print(f"  F1 Score: {mb_metrics['F1']:.3f}")
        print(f"  Jaccard: {mb_metrics['jaccard']:.3f}")
        if mb_metrics["missing_features"]:
            print(f"  Missing: {mb_metrics['missing_features']}")
        if mb_metrics["extra_features"]:
            print(f"  Extra: {mb_metrics['extra_features']}")

    return {
        "dag": dag_metrics,
        "markov_blanket": mb_metrics,
        "mb_true": mb_true,
        "mb_learned": list(mb_learned),
    }


# ============================================================================
# Unified Causal Discovery with Bi-directed Edges & Amortized Effects
# ============================================================================
# Key innovations:
# 1. A_confound: Explicit bi-directed edge matrix for latent confounders
# 2. Amortized effect network: Single network handles ALL treatments
# 3. Unified loss: Structure + Classification + Effect in one training loop
# ============================================================================


def init_amortized_effect_params(
    key: random.PRNGKey,
    n_features: int,
    effect_hidden_dim: int = 64,
    effect_embed_dim: int = 16,
) -> Dict[str, jnp.ndarray]:
    """
    Initialize amortized effect network parameters.

    Architecture:
        [covariates, treatment_embedding] -> MLP -> [y0, y1, propensity]

    The same network handles ALL treatments via conditioning on treatment embedding.
    """
    keys = random.split(key, 6)

    # Treatment embedding: each feature can be a treatment
    treatment_embed = random.normal(keys[0], (n_features, effect_embed_dim)) * 0.1

    # Input dim: covariates (n_features - 2 for T and Y) + treatment embedding
    effect_input_dim = n_features - 2 + effect_embed_dim

    # Shared hidden layer
    shared_w1 = random.normal(keys[1], (effect_input_dim, effect_hidden_dim)) * 0.1
    shared_b1 = jnp.zeros(effect_hidden_dim)

    # Y(0) head
    y0_w = random.normal(keys[2], (effect_hidden_dim, 1)) * 0.1
    y0_b = jnp.zeros(1)

    # Y(1) head
    y1_w = random.normal(keys[3], (effect_hidden_dim, 1)) * 0.1
    y1_b = jnp.zeros(1)

    # Propensity head
    prop_w = random.normal(keys[4], (effect_hidden_dim, 1)) * 0.1
    prop_b = jnp.zeros(1)

    return {
        "treatment_embed": treatment_embed,
        "shared_w1": shared_w1,
        "shared_b1": shared_b1,
        "y0_w": y0_w,
        "y0_b": y0_b,
        "y1_w": y1_w,
        "y1_b": y1_b,
        "prop_w": prop_w,
        "prop_b": prop_b,
    }


def amortized_effect_forward(
    X: jnp.ndarray,
    treatment_idx: int,
    Y_idx: int,
    effect_params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Amortized effect prediction for a single treatment.

    Args:
        X: (batch, n_features) full data
        treatment_idx: Which feature is the treatment
        Y_idx: Index of outcome variable
        effect_params: Effect network parameters

    Returns:
        y0: (batch,) predicted outcome under T=0
        y1: (batch,) predicted outcome under T=1
        propensity: (batch,) predicted P(T=1|X)
    """
    n_features = X.shape[1]
    batch_size = X.shape[0]

    # Build covariate mask: exclude treatment and Y
    covariate_indices = []
    for i in range(n_features):
        if i != treatment_idx and i != Y_idx:
            covariate_indices.append(i)
    covariate_indices = jnp.array(covariate_indices)

    # Extract covariates
    covariates = X[:, covariate_indices]

    # Get treatment embedding
    t_embed = effect_params["treatment_embed"][treatment_idx]  # (embed_dim,)
    t_embed_broadcast = jnp.tile(t_embed, (batch_size, 1))  # (batch, embed_dim)

    # Concatenate: [covariates, treatment_embedding]
    effect_input = jnp.concatenate([covariates, t_embed_broadcast], axis=1)

    # Shared hidden layer
    hidden = effect_input @ effect_params["shared_w1"] + effect_params["shared_b1"]
    hidden = jax.nn.relu(hidden)

    # Outcome heads
    y0 = (hidden @ effect_params["y0_w"] + effect_params["y0_b"]).squeeze(-1)
    y1 = (hidden @ effect_params["y1_w"] + effect_params["y1_b"]).squeeze(-1)

    # Propensity
    propensity = jax.nn.sigmoid(hidden @ effect_params["prop_w"] + effect_params["prop_b"]).squeeze(
        -1
    )

    return y0, y1, propensity


def compute_amortized_effect_loss(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    treatment_idx: int,
    Y_idx: int,
    effect_params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute effect estimation loss for a single treatment.

    Loss = factual_outcome_MSE + 0.1 * propensity_BCE
    """
    # Extract treatment values
    T = X[:, treatment_idx]

    # Smart binarization: for binary (0/1) data, use threshold at 0.5
    # For continuous data, use median split
    # Bug fix: median split fails when majority is 1 (median=1, T>1 = all False)
    T_range = jnp.max(T) - jnp.min(T)
    use_threshold = T_range <= 1.0  # Binary-like if range is 0 or 1
    T_binary = jnp.where(
        use_threshold,
        (T >= 0.5).astype(jnp.float32),  # Threshold at 0.5 for binary
        (T > jnp.median(T)).astype(jnp.float32),  # Median for continuous
    )

    # Forward pass
    y0, y1, propensity = amortized_effect_forward(X, treatment_idx, Y_idx, effect_params)

    # Factual outcome loss
    y_factual = T_binary * y1 + (1 - T_binary) * y0
    outcome_loss = jnp.mean((y_factual - Y) ** 2)

    # Propensity loss (BCE)
    eps = 1e-7
    prop_clipped = jnp.clip(propensity, eps, 1 - eps)
    propensity_loss = -jnp.mean(
        T_binary * jnp.log(prop_clipped) + (1 - T_binary) * jnp.log(1 - prop_clipped)
    )

    # Total effect loss
    total_loss = outcome_loss + 0.1 * propensity_loss

    # Compute ATE
    ate = jnp.mean(y1 - y0)

    metrics = {
        "ate": ate,
        "outcome_loss": outcome_loss,
        "propensity_loss": propensity_loss,
    }

    return total_loss, metrics


# ============================================================================
# DragonNet Architecture (Shi et al., 2019)
# Deeper network with targeted regularization for better effect estimation
# ============================================================================


def init_dragonnet_params(
    key: random.PRNGKey,
    n_features: int,
    hidden_dims: Tuple[int, int, int] = (128, 128, 64),
    effect_embed_dim: int = 16,
) -> Dict[str, jnp.ndarray]:
    """
    Initialize DragonNet parameters with deeper architecture.

    Architecture:
        [covariates, treatment_embedding]
        -> SharedLayer1 (128) -> ReLU
        -> SharedLayer2 (128) -> ReLU
        -> SharedLayer3 (64) -> ReLU
        -> [y0_head, y1_head, propensity_head]

    DragonNet improvements over TARNet:
    1. Deeper shared representation (3 layers vs 1)
    2. Targeted regularization encourages useful gradients for both tasks
    3. Better propensity estimation for AIPW
    """
    keys = random.split(key, 10)
    h1, h2, h3 = hidden_dims

    # Treatment embedding
    treatment_embed = random.normal(keys[0], (n_features, effect_embed_dim)) * 0.1

    # Input dim: covariates + treatment embedding
    input_dim = n_features - 2 + effect_embed_dim

    # Shared Layer 1
    shared_w1 = random.normal(keys[1], (input_dim, h1)) * jnp.sqrt(2.0 / input_dim)
    shared_b1 = jnp.zeros(h1)

    # Shared Layer 2
    shared_w2 = random.normal(keys[2], (h1, h2)) * jnp.sqrt(2.0 / h1)
    shared_b2 = jnp.zeros(h2)

    # Shared Layer 3
    shared_w3 = random.normal(keys[3], (h2, h3)) * jnp.sqrt(2.0 / h2)
    shared_b3 = jnp.zeros(h3)

    # Y(0) head - separate layers for potential outcomes
    y0_w1 = random.normal(keys[4], (h3, 32)) * jnp.sqrt(2.0 / h3)
    y0_b1 = jnp.zeros(32)
    y0_w2 = random.normal(keys[5], (32, 1)) * 0.1
    y0_b2 = jnp.zeros(1)

    # Y(1) head
    y1_w1 = random.normal(keys[6], (h3, 32)) * jnp.sqrt(2.0 / h3)
    y1_b1 = jnp.zeros(32)
    y1_w2 = random.normal(keys[7], (32, 1)) * 0.1
    y1_b2 = jnp.zeros(1)

    # Propensity head (epsilon in DragonNet)
    prop_w1 = random.normal(keys[8], (h3, 32)) * jnp.sqrt(2.0 / h3)
    prop_b1 = jnp.zeros(32)
    prop_w2 = random.normal(keys[9], (32, 1)) * 0.1
    prop_b2 = jnp.zeros(1)

    return {
        "treatment_embed": treatment_embed,
        # Shared layers
        "shared_w1": shared_w1,
        "shared_b1": shared_b1,
        "shared_w2": shared_w2,
        "shared_b2": shared_b2,
        "shared_w3": shared_w3,
        "shared_b3": shared_b3,
        # Y(0) head
        "y0_w1": y0_w1,
        "y0_b1": y0_b1,
        "y0_w2": y0_w2,
        "y0_b2": y0_b2,
        # Y(1) head
        "y1_w1": y1_w1,
        "y1_b1": y1_b1,
        "y1_w2": y1_w2,
        "y1_b2": y1_b2,
        # Propensity head
        "prop_w1": prop_w1,
        "prop_b1": prop_b1,
        "prop_w2": prop_w2,
        "prop_b2": prop_b2,
    }


def dragonnet_forward(
    X: jnp.ndarray,
    treatment_idx: int,
    Y_idx: int,
    effect_params: Dict[str, jnp.ndarray],
    valid_covariates: Optional[list] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    DragonNet forward pass with deeper shared representation.

    v11 FIX: Uses valid_covariates (from backdoor criterion) to MASK
    invalid covariates (zeros them out). This prevents over-adjustment
    while maintaining consistent dimensions with pre-trained weights.

    For exogenous variables (empty valid_covariates), ALL covariates are
    masked to zero, making the model predict based on treatment embedding
    only (equivalent to diff-in-means estimator).

    Args:
        X: Input data (batch, n_features)
        treatment_idx: Index of treatment variable
        Y_idx: Index of outcome variable
        effect_params: DragonNet parameters
        valid_covariates: List of valid covariate indices from backdoor criterion.
                          If None, falls back to all variables (legacy behavior).
                          If empty list, all covariates are masked (diff-in-means).

    Returns:
        y0: (batch,) predicted E[Y|T=0, X]
        y1: (batch,) predicted E[Y|T=1, X]
        propensity: (batch,) predicted P(T=1|X)
        shared_repr: (batch, h3) shared representation for targeted reg
    """
    n_features = X.shape[1]
    batch_size = X.shape[0]

    # Build standard covariate indices (exclude treatment and Y)
    all_covariate_indices = [i for i in range(n_features) if i != treatment_idx and i != Y_idx]
    covariates = X[:, all_covariate_indices]

    # Apply mask to zero out invalid covariates
    if valid_covariates is not None:
        # Create mask: 1 for valid, 0 for invalid
        mask = jnp.zeros(len(all_covariate_indices))
        valid_set = set(valid_covariates)
        for local_idx, global_idx in enumerate(all_covariate_indices):
            if global_idx in valid_set:
                mask = mask.at[local_idx].set(1.0)
        # Apply mask (zeros out invalid covariates)
        covariates = covariates * mask[jnp.newaxis, :]
        # Note: for empty valid_covariates (exogenous vars), mask is all zeros
        # This means model predicts based on treatment embedding only (diff-in-means)

    # Get treatment embedding
    t_embed = effect_params["treatment_embed"][treatment_idx]
    t_embed_broadcast = jnp.tile(t_embed, (batch_size, 1))

    # Concatenate input
    x = jnp.concatenate([covariates, t_embed_broadcast], axis=1)

    # Shared layers with ReLU
    h = x @ effect_params["shared_w1"] + effect_params["shared_b1"]
    h = jax.nn.relu(h)
    h = h @ effect_params["shared_w2"] + effect_params["shared_b2"]
    h = jax.nn.relu(h)
    h = h @ effect_params["shared_w3"] + effect_params["shared_b3"]
    shared_repr = jax.nn.relu(h)

    # Y(0) head
    y0_h = shared_repr @ effect_params["y0_w1"] + effect_params["y0_b1"]
    y0_h = jax.nn.relu(y0_h)
    y0 = (y0_h @ effect_params["y0_w2"] + effect_params["y0_b2"]).squeeze(-1)

    # Y(1) head
    y1_h = shared_repr @ effect_params["y1_w1"] + effect_params["y1_b1"]
    y1_h = jax.nn.relu(y1_h)
    y1 = (y1_h @ effect_params["y1_w2"] + effect_params["y1_b2"]).squeeze(-1)

    # Propensity head
    prop_h = shared_repr @ effect_params["prop_w1"] + effect_params["prop_b1"]
    prop_h = jax.nn.relu(prop_h)
    propensity = jax.nn.sigmoid(
        prop_h @ effect_params["prop_w2"] + effect_params["prop_b2"]
    ).squeeze(-1)

    return y0, y1, propensity, shared_repr


## ========== Structural DML: Independent Effect Estimation ==========
# Replaces the shared-representation DragonNet with simple JAX-native
# linear models fitted on raw features X[Z] (backdoor adjustment set).
# Key advantage: nuisance models do NOT share parameters with the processor,
# avoiding the representation saturation that makes DragonNet ATEs unreliable.


def init_structural_dml_params(n_features: int, key: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Initialize parameters for the structural DML nuisance models.

    Two independent linear models:
    - Propensity: logistic regression P(T=1|X[Z])
    - Outcome: ridge regression E[Y|X[Z], T]
    """
    k1, k2 = jax.random.split(key)
    return {
        # Propensity model: w_prop @ X[Z] + b_prop -> sigmoid -> P(T=1)
        "w_prop": jax.random.normal(k1, (n_features,)) * 0.01,
        "b_prop": jnp.zeros(1),
        # Outcome model: w_out @ [X[Z], T] + b_out -> E[Y|X,T]
        "w_out": jax.random.normal(k2, (n_features + 1,)) * 0.01,
        "b_out": jnp.zeros(1),
    }


def structural_dml_effect_loss(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    treatment_idx: int,
    Y_idx: int,
    dml_params: Dict[str, jnp.ndarray],
    valid_covariates: Optional[list] = None,
    A_weights: Optional[jnp.ndarray] = None,
    ridge_alpha: float = 0.01,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute effect loss using structural DML with independent linear models.

    Unlike DragonNet which shares Phi(X) with reconstruction/classification,
    this uses simple linear models on raw features X[Z] where Z is the
    backdoor-valid adjustment set from the current DAG.

    The gradient flows to A through the adjustment set selection:
    changing A changes which variables are in Z, which changes the
    features available to the nuisance models, which changes the effect estimate.

    Args:
        X: Feature matrix (n, d) — raw features, NOT processor representation
        Y: Target vector (n,) or (n,1)
        treatment_idx: Index of treatment variable in X
        Y_idx: Index of outcome variable (for exclusion)
        dml_params: Parameters from init_structural_dml_params
        valid_covariates: Backdoor-valid adjustment set indices (from A)
        A_weights: Current adjacency weights (for soft covariate selection)
        ridge_alpha: L2 penalty for outcome model

    Returns:
        (loss, info_dict) where loss is differentiable through dml_params and
        info_dict contains ATE estimate and diagnostics.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    Y_flat = Y.flatten()

    # Extract treatment
    T = X[:, treatment_idx]
    T_range = jnp.max(T) - jnp.min(T)
    T_binary = jnp.where(
        T_range <= 1.0, (T >= 0.5).astype(jnp.float32), (T > jnp.median(T)).astype(jnp.float32)
    )

    # Build covariate matrix X_z from adjustment set
    if valid_covariates is not None and len(valid_covariates) > 0:
        # Hard selection: use only valid covariates
        cov_indices = jnp.array(valid_covariates)
        X_z = X[:, cov_indices]
    elif A_weights is not None:
        # Soft selection: weight features by their adjacency to treatment
        # This is differentiable through A
        weights = jnp.abs(A_weights[:n_features, treatment_idx])
        # Exclude treatment and outcome from covariates
        weights = weights.at[treatment_idx].set(0.0)
        if Y_idx < n_features:
            weights = weights.at[Y_idx].set(0.0)
        # Soft masking: X_z = X * weights (features with zero weight contribute nothing)
        X_z = X * weights[None, :]
    else:
        # Fallback: use all features except treatment and outcome
        mask = jnp.ones(n_features)
        mask = mask.at[treatment_idx].set(0.0)
        if Y_idx < n_features:
            mask = mask.at[Y_idx].set(0.0)
        X_z = X * mask[None, :]

    # ========== Propensity model: P(T=1 | X[Z]) ==========
    # Simple logistic regression with independent parameters
    logit_prop = X_z @ dml_params["w_prop"][: X_z.shape[1]] + dml_params["b_prop"][0]
    propensity = jax.nn.sigmoid(logit_prop)
    eps = 1e-4
    propensity = jnp.clip(propensity, eps, 1 - eps)

    # Propensity loss (BCE)
    prop_loss = -jnp.mean(
        T_binary * jnp.log(propensity + eps) + (1 - T_binary) * jnp.log(1 - propensity + eps)
    )

    # ========== Outcome model: E[Y | X[Z], T] ==========
    # Ridge regression: [X_z, T] -> Y
    X_zt = jnp.concatenate([X_z, T_binary[:, None]], axis=1)
    y_pred = X_zt @ dml_params["w_out"][: X_zt.shape[1]] + dml_params["b_out"][0]

    # Potential outcomes under T=1 and T=0
    X_z1 = jnp.concatenate([X_z, jnp.ones((n_samples, 1))], axis=1)
    X_z0 = jnp.concatenate([X_z, jnp.zeros((n_samples, 1))], axis=1)
    mu_1 = X_z1 @ dml_params["w_out"][: X_z1.shape[1]] + dml_params["b_out"][0]
    mu_0 = X_z0 @ dml_params["w_out"][: X_z0.shape[1]] + dml_params["b_out"][0]

    # Outcome loss (MSE + ridge)
    outcome_loss = jnp.mean((y_pred - Y_flat) ** 2)
    ridge_loss = ridge_alpha * jnp.sum(dml_params["w_out"] ** 2)

    # ========== AIPW doubly robust estimator ==========
    w1 = T_binary / propensity
    w0 = (1 - T_binary) / (1 - propensity)

    pseudo_y1 = mu_1 + w1 * (Y_flat - mu_1)
    pseudo_y0 = mu_0 + w0 * (Y_flat - mu_0)

    ate_aipw = jnp.mean(pseudo_y1 - pseudo_y0)

    # Effect loss: outcome fit + propensity fit + pseudo-outcome variance
    aipw_var_loss = jnp.mean((pseudo_y1 - jnp.mean(pseudo_y1)) ** 2) + jnp.mean(
        (pseudo_y0 - jnp.mean(pseudo_y0)) ** 2
    )

    total_loss = outcome_loss + 0.5 * prop_loss + 0.1 * aipw_var_loss + ridge_loss

    info = {
        "ate_aipw": ate_aipw,
        "ate_simple": jnp.mean(mu_1 - mu_0),
        "propensity_mean": jnp.mean(propensity),
        "outcome_mse": outcome_loss,
        "prop_loss": prop_loss,
    }

    return total_loss, info


def compute_dragonnet_loss(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    treatment_idx: int,
    Y_idx: int,
    effect_params: Dict[str, jnp.ndarray],
    targeted_reg: float = 1.0,
    valid_covariates: Optional[list] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    DragonNet loss with AIPW and targeted regularization.

    Loss = outcome_loss + propensity_loss + targeted_reg * epsilon_loss

    The targeted regularization (epsilon_loss) encourages the shared
    representation to be useful for BOTH outcome and propensity prediction,
    which improves effect estimation.

    v11 FIX: Accepts valid_covariates from backdoor criterion to prevent
    over-adjustment bias (e.g., adjusting for children of treatment).
    """
    # Extract treatment values
    T = X[:, treatment_idx]

    # Smart binarization: for binary (0/1) data, use threshold at 0.5
    # For continuous data, use median split
    # Bug fix: median split fails when majority is 1 (median=1, T>1 = all False)
    T_range = jnp.max(T) - jnp.min(T)
    use_threshold = T_range <= 1.0  # Binary-like if range is 0 or 1
    T_binary = jnp.where(
        use_threshold,
        (T >= 0.5).astype(jnp.float32),  # Threshold at 0.5 for binary
        (T > jnp.median(T)).astype(jnp.float32),  # Median for continuous
    )

    # Forward pass (v11: with valid covariates)
    y0, y1, propensity, shared_repr = dragonnet_forward(
        X, treatment_idx, Y_idx, effect_params, valid_covariates
    )

    # Clip propensity for numerical stability
    eps = 1e-4
    propensity = jnp.clip(propensity, eps, 1 - eps)

    # ========== Factual outcome loss ==========
    y_factual = T_binary * y1 + (1 - T_binary) * y0
    outcome_loss = jnp.mean((y_factual - Y) ** 2)

    # ========== Propensity loss (BCE) ==========
    propensity_loss = -jnp.mean(
        T_binary * jnp.log(propensity + eps) + (1 - T_binary) * jnp.log(1 - propensity + eps)
    )

    # ========== AIPW pseudo-outcome for doubly robust estimation ==========
    # Pseudo-outcome: corrects for propensity weighting
    # If outcome model is correct, AIPW = outcome-based estimate
    # If propensity model is correct, AIPW = IPW estimate
    # If BOTH are correct, AIPW is semiparametrically efficient

    # IPW weights
    w1 = T_binary / propensity
    w0 = (1 - T_binary) / (1 - propensity)

    # AIPW pseudo-outcomes
    pseudo_y1 = y1 + w1 * (Y - y1)  # Augmented Y(1) estimate
    pseudo_y0 = y0 + w0 * (Y - y0)  # Augmented Y(0) estimate

    # AIPW ATE (doubly robust)
    ate_aipw = jnp.mean(pseudo_y1 - pseudo_y0)

    # AIPW loss: variance of pseudo-outcomes
    aipw_loss = jnp.mean((pseudo_y1 - jnp.mean(pseudo_y1)) ** 2) + jnp.mean(
        (pseudo_y0 - jnp.mean(pseudo_y0)) ** 2
    )

    # ========== Targeted regularization (DragonNet epsilon) ==========
    # Encourages shared representation gradients to be aligned
    # for both outcome and propensity prediction
    # This is approximated by minimizing the difference between
    # treated and control representations weighted by propensity
    treated_mask = T_binary > 0.5
    control_mask = ~treated_mask

    n_treated = jnp.sum(treated_mask) + 1e-6
    n_control = jnp.sum(control_mask) + 1e-6

    # Mean representations for treated and control
    mean_repr_treated = jnp.sum(shared_repr * treated_mask[:, None], axis=0) / n_treated
    mean_repr_control = jnp.sum(shared_repr * control_mask[:, None], axis=0) / n_control

    # Targeted regularization: penalize difference in representations
    # This encourages overlap and balanced representations
    epsilon_loss = jnp.mean((mean_repr_treated - mean_repr_control) ** 2)

    # ========== Total loss ==========
    total_loss = (
        outcome_loss + 0.5 * propensity_loss + 0.1 * aipw_loss + targeted_reg * epsilon_loss
    )

    # Simple ATE (for comparison)
    ate_simple = jnp.mean(y1 - y0)

    metrics = {
        "ate": ate_simple,
        "ate_aipw": ate_aipw,
        "outcome_loss": outcome_loss,
        "propensity_loss": propensity_loss,
        "aipw_loss": aipw_loss,
        "epsilon_loss": epsilon_loss,
    }

    return total_loss, metrics


def init_confound_matrix(key: random.PRNGKey, n_vars: int, scale: float = 0.1) -> jnp.ndarray:
    """
    Initialize bi-directed edge matrix A_confound.

    A_confound[i,j] > 0 means Xi <-> Xj (shared latent cause).
    Matrix is symmetric, diagonal is zero.

    Args:
        key: JAX random key
        n_vars: Number of variables
        scale: Initialization scale (default 0.1, larger for better gradient flow)
    """
    A_raw = random.normal(key, (n_vars, n_vars)) * scale
    A_confound = (A_raw + A_raw.T) / 2
    A_confound = A_confound - jnp.diag(jnp.diag(A_confound))
    return A_confound


def symmetrize_confound_matrix(A_confound: jnp.ndarray) -> jnp.ndarray:
    """Ensure A_confound stays symmetric during optimization."""
    A_sym = (A_confound + A_confound.T) / 2
    return A_sym - jnp.diag(jnp.diag(A_sym))


def compute_bow_free_penalty_v7(
    A_direct: jnp.ndarray,
    A_confound: jnp.ndarray,
) -> jnp.ndarray:
    """
    Bow-free constraint: can't have both Xi -> Xj AND Xi <-> Xj.

    Penalty = sum(|A_direct| * |A_confound|)
    """
    return jnp.sum(jnp.abs(A_direct) * jnp.abs(A_confound))


# ============================================================================
# Low-Rank Error Covariance for Latent Confounders (DECOR-style)
# ============================================================================


def init_lowrank_confound(
    key: random.PRNGKey, n_vars: int, rank_k: int = 5, scale: float = 0.05
) -> tuple:
    """
    Initialize low-rank confound factor matrix B and noise variance.

    The error covariance is Ω = B @ B.T + σ²I, where B ∈ R^{d×k}
    captures k latent confounders and σ² is idiosyncratic noise.

    Args:
        key: JAX random key
        n_vars: Number of variables (d)
        rank_k: Number of latent confounders
        scale: Initialization scale

    Returns:
        (B_confound, log_var_confound): B is (d, k), log_var is scalar
    """
    B_confound = random.normal(key, (n_vars, rank_k)) * scale
    log_var_confound = jnp.array(jnp.log(0.1))  # σ² = 0.1 initially
    return B_confound, log_var_confound


def compute_confound_nll(
    residuals: jnp.ndarray, B: jnp.ndarray, log_var_c: jnp.ndarray
) -> jnp.ndarray:
    """
    Gaussian NLL of residuals under Ω = B@B.T + σ²I via Woodbury identity.

    Computes (1/n) tr(R.T Ω⁻¹ R) + log|Ω| efficiently in O(dk² + k³)
    instead of O(d³) by exploiting the low-rank-plus-diagonal structure.

    Woodbury: (B@B.T + σ²I)⁻¹ = σ⁻²I - σ⁻²B(σ²I_k + B.TB)⁻¹B.T
    Matrix determinant lemma: |Ω| = σ^{2(d-k)} |σ²I_k + B.TB|

    Args:
        residuals: (n_batch, d) residuals R = X - X̂_processor
        B: (d, k) factor loadings
        log_var_c: scalar log(σ²)

    Returns:
        nll: scalar, the negative log-likelihood (up to constant)
    """
    d, k = B.shape
    sigma2 = jnp.maximum(jnp.exp(log_var_c), 1e-6)

    BtB = B.T @ B  # (k, k)
    M = sigma2 * jnp.eye(k) + BtB  # (k, k)
    M_inv = jnp.linalg.inv(M)  # O(k³), k is small

    # Efficient quadratic form: tr(R.T Ω⁻¹ R)
    # Woodbury: Ω⁻¹ = σ⁻²I - σ⁻²B M⁻¹ B.T  (where M = σ²I_k + B.TB)
    # So tr(R.T Ω⁻¹ R) = σ⁻² tr(RtR) - σ⁻² tr(B.T RtR B M⁻¹)
    s2_inv = 1.0 / sigma2
    RtR = residuals.T @ residuals  # (d, d)
    BtRtRB = B.T @ RtR @ B  # (k, k)
    quad = s2_inv * jnp.trace(RtR) - s2_inv * jnp.trace(BtRtRB @ M_inv)

    # log|Ω| = (d - k) log(σ²) + log|M|
    _, logdet_M = jnp.linalg.slogdet(M)
    log_det = (d - k) * log_var_c + logdet_M

    n = residuals.shape[0]
    return quad / n + log_det


def learn_structure(
    data: jnp.ndarray,
    Y: jnp.ndarray,
    Y_idx: int,
    processor,
    key: random.PRNGKey,
    processor_type: str = "mlp",
    # Structure learning
    lambda_1: float = 0.02,
    lambda_2_init: float = 0.01,
    lambda_2_max: float = 1e4,  # Capped to prevent loss explosion on large graphs
    lambda_class: float = 1.0,
    lr: float = 0.001,
    max_iter: int = 100,
    patience: int = 25,
    verbose: int = 0,  # 0=minimal, 1=normal, 2=debug
    A_init: Optional[jnp.ndarray] = None,
    proc_params_init: Optional[list] = None,  # Pre-initialized processor params (SPR transfer)
    # L2 regularization to prevent weight saturation
    lambda_L2: float = 0.01,  # L2 penalty on A (prevents weights saturating at ~0.99)
    # Bi-directed edges
    use_confound_matrix: bool = True,
    lambda_confound_sparse: float = 0.02,
    lambda_bow: float = 0.3,  # Prevents spurious direct edges when confounded
    n_latent_confounders: int = 5,  # 0 = legacy A_confound (d*d), >0 = low-rank B(d,k) error covariance
    # Amortized effects
    use_amortized_effects: bool = True,
    effect_hidden_dim: int = 64,
    effect_embed_dim: int = 16,
    lambda_effect: float = 0.5,
    effect_warmup_iter: int = 20,  # Start effects after warmup (curriculum controls phasing)
    max_treatments: int = 10,
    Y_continuous: Optional[jnp.ndarray] = None,  # Continuous Y for effect estimation
    effect_refinement_iters: int = 150,  # Post joint-training refinement
    # DragonNet architecture
    use_dragonnet: bool = True,  # Use DragonNet instead of TARNet for better effect estimation
    targeted_reg: float = 0.1,  # Targeted regularization weight for DragonNet
    # Adaptive curriculum
    use_adaptive_curriculum: bool = True,  # Convergence-based phase transitions
    curriculum_phase_splits: tuple = (
        0.4,
        0.8,
    ),  # Fixed curriculum split points (phase1_end, phase2_end)
    use_pcgrad: bool = False,  # PCGrad gradient surgery for conflicting objectives (Yu et al., 2020)
    use_structural_dml: bool = False,  # Replace DragonNet with independent linear models on X[Z]
    # Latent confounder parameters
    use_latent_confounders: bool = False,
    latent_rank_k: int = 5,
    lambda_L: float = 0.05,
    warm_start_L_iters: int = 20,
    # Validation and constraints
    use_validation_split: bool = True,
    validation_ratio: float = 0.2,
    weight_decay: float = 1e-4,
    use_spectral_constraint: bool = False,
    enable_pruning: bool = False,
    # Identifiability regularizer
    lambda_ident: float = 0.01,  # D-optimality regularizer weight (active Phase 2-3 only)
    ident_threshold: float = 0.3,  # Threshold for MB membership from A matrix
    # Task type
    task: str = "classification",  # 'classification' or 'regression'
    # Ablation overrides
    enforce_outcome_sink: bool = True,  # Set A[Y_idx,:]=0 after each update (disable for A6 ablation)
    freeze_A: bool = False,  # If True, don't update A_direct (only train processor, A7 ablation)
    # Phase 2: Hardware utilization
    use_bf16: bool = False,  # Cast forward pass operands to bfloat16 (master params stay fp32)
    # Phase 3: Optuna integration
    iteration_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    # PC structure constraint: only enable when A_init comes from PC algorithm
    use_pc_constraint: bool = False,
    # Phase 2: DAG-Attention bidirectional signal
    # When > 0 and processor is DAGAttentionAdapter, adds
    # KL(empirical_attention || sigmoid(A * temperature)) to the total loss.
    # This is the bidirectional learning signal: attention pulls A toward
    # observed attention patterns; A pulls attention toward causal edges.
    # Default 0.0 keeps prior behavior (forward-only soft mask).
    lambda_consistency: float = 0.0,
    # Phase 2: DAG-Attention temperature annealing for the consistency loss.
    # Tuple (init, final) cosine-anneals the temperature used in the consistency
    # target sigmoid(A * temp) from init at iter=0 to final at iter=max_iter.
    # Low initial temp → soft target (graded gradient), high final temp →
    # sharp target (concentrates on real edges). Per skills/dag_attention.md
    # the recommended schedule is (1.0, 20.0). When None, uses the adapter's
    # fixed self.temperature for both soft mask and consistency target.
    temperature_consistency_schedule: Optional[Tuple[float, float]] = None,
    # Phase 2: adaptive engagement of consistency_loss. When True,
    # lambda_consistency is interpreted as the MAX value (engaged once
    # collapse is detected); the current multiplier is 0 until A starts
    # collapsing. Engagement triggers when max|A| drops below
    # consistency_collapse_threshold AND iter < consistency_max_engage_iter_frac
    # * max_iter. One-shot (no oscillation): once engaged, stays engaged.
    # Closes the regime-dependence gap: only fires when needed.
    adaptive_consistency: bool = False,
    consistency_collapse_threshold: float = 0.1,
    consistency_max_engage_iter_frac: float = 0.3,
    # AAP sparsity-aware training: when True (and lambda_aap > 0 and
    # processor is DAGAttentionAdapter), the AAP cascade routes parent
    # weights through a straight-through hard mask. The forward pass sees
    # strict 0/1 parent gates so dense attention cannot bypass structural
    # edges; the backward pass keeps identity gradients w.r.t. |A| so
    # structure learning continues to grow/shrink edges. Default False
    # preserves exact prior behavior; set True together with lambda_aap > 0
    # to engage the Sprint-1 sparsity-aware AAP path.
    aap_enforce_hard_parents: bool = False,
    aap_edge_threshold: float = 0.05,
    # Phase 2: AAP (Abduction-Action-Prediction) cascade loss for
    # DAG-Attention. When > 0 and processor is DAGAttentionAdapter, adds
    # MSE(observed X, model_predict_using_predicted_parents) to the total loss.
    # K=1 cascade: model's per-variable predictions feed back as parents in a
    # second forward pass. This creates a self-consistency gradient distinct
    # from per-variable reconstruction (which uses observed parents directly).
    # Default 0.0 keeps prior behavior. Full Pearl AAP with intervention is
    # in jcce.counterfactual.aap.AAPCounterfactual for inference; this loss
    # is a training-time approximation that flows gradient through the
    # SCM cascade.
    lambda_aap: float = 0.0,
    # Phase 2: CausalMamba Sinkhorn temperature annealing schedule.
    # Tuple (init, final) cosine-anneals the temperature passed to the
    # CausalMambaAdapter's Sinkhorn soft sort from init at iter=0 to final
    # at iter=max_iter. Per skills/topological_mamba.md the recommended
    # schedule is (1.0, 0.1) — high temp early (soft permutation, smoother
    # gradient back to A) → low temp late (sharp permutation, stable order).
    # When None, the adapter uses its fixed self.sinkhorn_temperature
    # (default 0.1). Requires CausalMambaAdapter.forward to accept a
    # `temperature` kwarg — coordinated with causal-mamba branch.
    causal_mamba_temperature_schedule: Optional[Tuple[float, float]] = None,
    # Phase 2: edge-only consistency loss variant (Sprint 6a — NEGATIVE result).
    # When True (and lambda_consistency > 0), the consistency target is
    # restricted to A's existing edges (|A| > consistency_edge_threshold) plus
    # the diagonal. Soft-masked via sigmoid((|A| - threshold) * 50) so
    # gradient still flows back to A through the mask. Empirically validated
    # NEGATIVE: reduces over-saturation (37 -> 13 edges on Heart Disease) but
    # BAcc damage persists (-0.378 vs vanilla -0.379). Preserved as opt-in
    # flag for future researchers; default off.
    consistency_edge_only: bool = False,
    consistency_edge_threshold: float = 0.05,
) -> Tuple[jnp.ndarray, Any, list, Dict[str, Any]]:
    """
    v7.0: Unified Causal Discovery with Bi-directed Edges & Amortized Effects.

    Key innovations:
    1. A_confound: Explicit bi-directed edge matrix for latent confounders
    2. Amortized effect network: Single network handles ALL treatments
    3. Unified loss: Structure + Classification/Regression + Effect in one training loop
    4. Effect loss as NSGA-II fitness objective (works without ground truth)

    Args:
        data: (n_samples, n_vars) observed data (includes Y at Y_idx)
        Y: (n_samples,) target labels (classification) or values (regression)
        Y_idx: Index of Y in data
        processor: Processor adapter instance
        key: JAX random key
        processor_type: Processor type string

        # Structure learning
        lambda_1: Sparsity penalty on A_direct
        lambda_2_init: Initial DAG constraint penalty
        lambda_2_max: Maximum DAG constraint penalty
        lambda_class: Classification loss weight
        lr: Learning rate
        max_iter: Maximum iterations
        patience: Early stopping patience
        verbose: Verbosity level (0=minimal, 1=normal, 2=debug)
        A_init: Optional initial adjacency matrix
        lambda_L2: L2 regularization weight on A_direct to prevent weight saturation

        # Bi-directed edges
        use_confound_matrix: Enable A_confound for bi-directed edges
        lambda_confound_sparse: Sparsity penalty on A_confound
        lambda_bow: Bow-free penalty weight

        # Amortized effects
        use_amortized_effects: Enable amortized effect estimation
        effect_hidden_dim: Hidden dimension for effect network
        effect_embed_dim: Treatment embedding dimension
        lambda_effect: Effect loss weight
        effect_warmup_iter: Iterations before enabling effect loss
        max_treatments: Maximum treatments to evaluate per iteration

        # Latent confounders (optional, can be used with A_confound)
        use_latent_confounders: Enable L = UV^T latent matrix
        latent_rank_k: Rank of L matrix
        lambda_L: Nuclear norm penalty on L
        warm_start_L_iters: Iterations before warm-starting L

    Returns:
        A_direct: (n_vars, n_vars) directed edge matrix (DAG)
        processor: Processor adapter
        processor_params: Trained parameters
        metrics: Dict with all metrics including:
            - A_confound: Bi-directed edge matrix
            - effect_params: Amortized effect network params
            - causal_effects: Dict of ATE for each feature->Y
            - effect_loss: Effect estimation loss (for NSGA-II fitness)
    """
    n_samples_total, n_vars = data.shape

    # Use continuous Y for effect estimation if provided
    Y_effect = Y_continuous if Y_continuous is not None else Y

    # Validation split
    if use_validation_split:
        key, split_key = random.split(key)
        n_val = int(n_samples_total * validation_ratio)
        n_train = n_samples_total - n_val
        perm = random.permutation(split_key, n_samples_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        data_train = data[train_idx]
        Y_train = Y[train_idx]
        Y_effect_train = Y_effect[train_idx]
        data_val = data[val_idx]
        Y_val = Y[val_idx]
        Y_effect_val = Y_effect[val_idx]
        n_samples = n_train
    else:
        data_train = data
        Y_train = Y
        Y_effect_train = Y_effect
        data_val = None
        Y_val = None
        Y_effect_val = None
        n_samples = n_samples_total

    # Batch size
    FIXED_BATCH_SIZE = get_batch_size(processor_type)
    effective_batch_size = min(FIXED_BATCH_SIZE, n_samples)
    use_batching = n_samples > FIXED_BATCH_SIZE

    if verbose >= 1:
        import sys

        print(f"\n{'=' * 60}")
        print("v7.0: Unified Discovery + Bi-directed Edges + Amortized Effects")
        print(f"{'=' * 60}")
        sys.stdout.flush()
        print(f"Data: {n_samples} train samples, {n_vars} variables")
        print(f"Processor: {processor_type}")
        sys.stdout.flush()
        if verbose >= 2:
            if use_confound_matrix:
                print(
                    f"Bi-directed edges: lambda_confound={lambda_confound_sparse}, lambda_bow={lambda_bow}"
                )
            if use_amortized_effects:
                print(f"Amortized effects: warmup={effect_warmup_iter}, lambda={lambda_effect}")
            if use_dragonnet:
                print("Using DragonNet architecture (3 shared layers + AIPW)")
            sys.stdout.flush()

    # ==================== Initialize Parameters ====================

    # A_direct (standard DAG adjacency) - includes Y as last variable
    # NOTE: data has only X features (n_vars columns), Y is passed separately
    # Y_idx = n_vars indicates Y's position in the combined [X|Y] space
    # So we need n_total = n_vars + 1 for the full variable space including Y
    n_total = n_vars + 1  # X features + Y position
    if A_init is not None:
        A_direct = jnp.array(A_init)
        # PC warm-start only covers X->X edges (upper-left n_vars*n_vars block)
        # The Y column (column Y_idx) is all zeros because PC doesn't know about Y
        # Add correlation warm-start for X→Y edges to give model a starting point
        data_with_Y = jnp.column_stack([data_train, Y_train.reshape(-1)])
        corr_matrix = jnp.corrcoef(data_with_Y.T)
        corr_with_Y = jnp.abs(corr_matrix[Y_idx, :]) * 0.5
        corr_with_Y = corr_with_Y.at[Y_idx].set(0.0)
        A_direct = A_direct.at[:, Y_idx].set(corr_with_Y)
        # OUTCOME SINK CONSTRAINT - Y has no outgoing edges
        if enforce_outcome_sink:
            A_direct = A_direct.at[Y_idx, :].set(0.0)
    else:
        key, init_key = random.split(key)
        A_direct = random.normal(init_key, (n_total, n_total)) * 0.1
        # Correlation warm-start for A[:,Y_idx] using X-Y correlations
        # NOTE: data_train has only X features, need to stack Y for correlation
        data_with_Y = jnp.column_stack([data_train, Y_train.reshape(-1)])
        corr_matrix = jnp.corrcoef(data_with_Y.T)
        corr_with_Y = jnp.abs(corr_matrix[Y_idx, :]) * 0.5
        corr_with_Y = corr_with_Y.at[Y_idx].set(0.0)
        A_direct = A_direct.at[:, Y_idx].set(corr_with_Y)
        # OUTCOME SINK CONSTRAINT - Y has no outgoing edges from start
        if enforce_outcome_sink:
            A_direct = A_direct.at[Y_idx, :].set(0.0)

    # A_confound (bi-directed edges) or B_confound (low-rank error covariance)
    B_confound, log_var_confound = None, None
    A_confound = None
    if use_confound_matrix:
        key, confound_key = random.split(key)
        if n_latent_confounders > 0:
            B_confound, log_var_confound = init_lowrank_confound(
                confound_key, n_total, rank_k=n_latent_confounders
            )
        else:
            A_confound = init_confound_matrix(confound_key, n_total)

    # Processor parameters - one for each variable including Y
    # Each processor takes X features (n_vars) as input, outputs scalar for variable j
    if proc_params_init is not None and len(proc_params_init) == n_total:
        processor_params = list(proc_params_init)
    else:
        processor_params = []
        for j in range(n_total):
            params = processor.init_params(n_vars)  # n_vars inputs (X only), not n_total
            processor_params.append(params)

    # Effect network parameters
    if use_amortized_effects:
        key, effect_key = random.split(key)
        if use_structural_dml:
            # Structural DML: independent linear models on raw features
            effect_params = init_structural_dml_params(n_vars, effect_key)
            if verbose >= 1:
                print(
                    f"Structural DML: independent linear propensity + outcome models (d={n_vars})"
                )
        elif use_dragonnet:
            # Use DragonNet with deeper architecture
            effect_params = init_dragonnet_params(
                effect_key, n_total, hidden_dims=(128, 128, 64), effect_embed_dim=effect_embed_dim
            )
        else:
            # Original TARNet
            effect_params = init_amortized_effect_params(
                effect_key, n_total, effect_hidden_dim, effect_embed_dim
            )
    else:
        effect_params = None

    # Latent confounder factors (optional)
    if use_latent_confounders:
        key, uv_key = random.split(key)
        U = random.normal(uv_key, (n_samples, latent_rank_k)) * 0.01
        V = random.normal(random.split(uv_key)[0], (n_vars, latent_rank_k)) * 0.01
    else:
        U = None
        V = None

    # log_var_recon fixed at 0 (precision=1), NOT trainable. When learnable, the
    # optimizer exploits it to amplify reconstruction gradient at the expense of
    # classification (same exploit as log_var_class).
    log_var_recon = jnp.array(0.0)  # kept in params dict for backward compat, but frozen

    # Extract trainable params
    trainable_proc_params = extract_trainable_params(processor_params)

    # Pack all parameters
    all_params = {
        "A_direct": A_direct,
        "log_var_recon": log_var_recon,
        "processor_params": trainable_proc_params,
    }
    if use_confound_matrix:
        if B_confound is not None:
            all_params["B_confound"] = B_confound
            all_params["log_var_confound"] = log_var_confound
        elif A_confound is not None:
            all_params["A_confound"] = A_confound
    if use_amortized_effects:
        all_params["effect_params"] = effect_params
    if use_latent_confounders:
        all_params["U"] = U
        all_params["V"] = V

    # Save frozen copy of A_direct for freeze_A ablation
    A_direct_frozen = jnp.array(A_direct) if freeze_A else None

    # ==================== Define Loss Function ====================

    # Store A_init as constant for PC structure constraint (captured in closure)
    # Only apply PC constraint when A_init comes from actual PC algorithm output,
    # not from warm-start cache/SPR/adjacency prior (which would lock in dense garbage)
    A_init_const = A_init if (A_init is not None and use_pc_constraint) else None
    if verbose >= 2 and A_init is not None:
        pc_edges = int(jnp.sum(jnp.abs(A_init) > 0.5))
        pc_status = "with PC constraint" if use_pc_constraint else "no PC constraint"
        print(f"  [PC Constraint] A_init provided with {pc_edges} edges ({pc_status})")

    def loss_fn(
        params, batch_data, batch_Y, batch_Y_effect, lambda_2_current, curriculum_weights,
        current_temp_consistency, current_lambda_consistency, current_cm_temperature,
    ):
        """Unified loss: Structure + Classification + Effects.

        batch_Y: Binary Y for classification loss
        batch_Y_effect: Continuous Y for effect estimation (v7.1)

        v10 Curriculum Learning (now external):
        - Weights (w_recon, w_class, w_effect) computed outside loss_fn
        - Allows adaptive curriculum based on convergence signals

        Note: effect_warmup_completed and adjustment_sets are captured via closure.
        When these change, the JIT-compiled train_step must be recreated.
        """
        A_curr = params["A_direct"]
        proc_params = merge_trained_params(processor_params, params["processor_params"])
        log_var_r = params["log_var_recon"]

        n_batch, n_v = batch_data.shape

        # Curriculum weights passed from outside (allows adaptive curriculum)
        w_recon, w_class, w_effect = curriculum_weights

        # Get confound parameters
        A_conf, B_conf, lvc = None, None, None
        if use_confound_matrix:
            if n_latent_confounders > 0:
                B_conf = params["B_confound"]
                lvc = params["log_var_confound"]
            else:
                A_conf = symmetrize_confound_matrix(params["A_confound"])

        # ========== Reconstruction (X features) + Classification (Y) ==========
        total_recon_loss = 0.0
        classification_loss = 0.0
        n_total_vars = A_curr.shape[0]  # n_vars + 1 (includes Y)

        # Reconstruction for X features (vectorized with vmap)
        # Compute edge weights for all variables at once instead of per-variable loop
        self_loop_mask = 1.0 - jnp.eye(n_v)  # (n_v, n_v) — 0 on diagonal
        all_weights = jnp.abs(A_curr[:n_v, :n_v]).T * self_loop_mask  # (n_v, n_v)
        # Fallback to uniform (self-loop-zeroed) when weight sum is near zero
        weight_sums = jnp.sum(all_weights, axis=1, keepdims=True)  # (n_v, 1)
        all_weights = jnp.where(weight_sums > 0.01, all_weights, self_loop_mask)

        # Weighted inputs for all variables: (n_v, n_batch, n_v)
        all_X_weighted = batch_data[jnp.newaxis, :, :] * all_weights[:, jnp.newaxis, :]

        # Prepare processor params for vmap: split JAX arrays (per-var) from metadata (shared)
        _example_p = proc_params[0]
        _meta = {k: v for k, v in _example_p.items() if not isinstance(v, jnp.ndarray)}
        _arr_keys = sorted(k for k, v in _example_p.items() if isinstance(v, jnp.ndarray))
        _stacked = {k: jnp.stack([proc_params[j][k] for j in range(n_v)]) for k in _arr_keys}

        # vmapped forward: process all n_v variables in parallel (1 kernel vs n_v)
        _proc_name_recon = processor.__class__.__name__
        if _proc_name_recon == "GNNAdapter":
            A_norm = A_curr[:n_v, :n_v] / (
                jnp.sum(jnp.abs(A_curr[:n_v, :n_v]), axis=0, keepdims=True) + 1e-8
            )

            def _recon_fwd(X_w_j, arrays_j):
                return processor.forward(X_w_j, {**_meta, **arrays_j}, A=A_norm)
        elif _proc_name_recon == "DAGAttentionAdapter":
            # DAG-Attention: pass raw A; soft mask is applied inside the adapter.
            A_for_struct = A_curr[:n_v, :n_v]

            def _recon_fwd(X_w_j, arrays_j):
                return processor.forward(X_w_j, {**_meta, **arrays_j}, A=A_for_struct)
        elif _proc_name_recon == "CausalMambaAdapter":
            # CausalMamba: pass raw A; jit-safe topological sort runs inside.
            # Optionally pass current Sinkhorn temperature for the schedule.
            A_for_struct = A_curr[:n_v, :n_v]

            if causal_mamba_temperature_schedule is not None:
                def _recon_fwd(X_w_j, arrays_j):
                    return processor.forward(
                        X_w_j, {**_meta, **arrays_j}, A=A_for_struct,
                        temperature=current_cm_temperature,
                    )
            else:
                def _recon_fwd(X_w_j, arrays_j):
                    return processor.forward(X_w_j, {**_meta, **arrays_j}, A=A_for_struct)
        else:

            def _recon_fwd(X_w_j, arrays_j):
                return processor.forward(X_w_j, {**_meta, **arrays_j})

        if use_bf16:
            all_X_weighted_fwd = all_X_weighted.astype(jnp.bfloat16)
            _stacked_fwd = jax.tree.map(
                lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, _stacked
            )
            all_outputs = jax.vmap(_recon_fwd)(all_X_weighted_fwd, _stacked_fwd).astype(jnp.float32)
        else:
            all_outputs = jax.vmap(_recon_fwd)(all_X_weighted, _stacked)  # (n_v, n_batch)

        # Legacy confound: additive contribution to reconstruction
        if A_conf is not None:
            all_conf_weights = jnp.abs(A_conf[:n_v, :n_v]).T * self_loop_mask  # (n_v, n_v)
            all_conf_contrib = all_conf_weights @ batch_data.T  # (n_v, n_batch)
            all_outputs = all_outputs + all_conf_contrib

        # MSE across all variables (used for monitoring + loss in legacy/no-confound paths)
        all_mse = jnp.mean((batch_data.T - all_outputs) ** 2, axis=1)  # (n_v,)
        total_recon_loss = jnp.mean(all_mse)

        # ========== AAP cascade loss (Phase 2 self-consistency signal) ==========
        # K=1 recursive forward: feed the model's per-variable predictions back
        # as parents and re-predict. If A and f together encode a self-consistent
        # SCM, the cascaded predictions match the observed data. The MSE provides
        # a gradient signal distinct from per-variable recon (which uses observed
        # parents) — it pushes A and f toward SCM-level self-consistency.
        # Active only when processor is DAGAttentionAdapter and lambda_aap > 0;
        # otherwise yields jnp.array(0.0) with no extra forward cost.
        aap_loss = jnp.array(0.0)
        if lambda_aap > 0 and _proc_name_recon == "DAGAttentionAdapter":
            X_cascade_input = all_outputs.T  # (n_batch, n_v) — model's first-stage prediction
            if aap_enforce_hard_parents:
                # Sparsity-enforced cascade. The recon path's uniform fallback
                # (line 4804) is intentionally bypassed: when A collapses, the
                # AAP loss should grow strongly (no parents -> cascade outputs
                # near zero, far from observed data), so the gradient signal
                # pushes A back up. With the soft cascade, the fallback masked
                # this signal.
                aap_weights_raw = jnp.abs(A_curr[:n_v, :n_v]).T * self_loop_mask
                aap_weights = ste_hard_parents(aap_weights_raw, aap_edge_threshold)
            else:
                aap_weights = all_weights  # legacy soft cascade (with uniform fallback)
            all_X_weighted_aap = X_cascade_input[jnp.newaxis, :, :] * aap_weights[:, jnp.newaxis, :]
            if use_bf16:
                all_X_weighted_aap_fwd = all_X_weighted_aap.astype(jnp.bfloat16)
                all_outputs_aap = jax.vmap(_recon_fwd)(
                    all_X_weighted_aap_fwd, _stacked_fwd
                ).astype(jnp.float32)
            else:
                all_outputs_aap = jax.vmap(_recon_fwd)(all_X_weighted_aap, _stacked)
            aap_loss = jnp.mean((batch_data.T - all_outputs_aap) ** 2)

        # Low-rank confound: NLL on processor-only residuals under Ω = B@B.T + σ²I
        confound_nll = 0.0
        if B_conf is not None:
            residuals_T = batch_data.T - all_outputs  # (n_v, n_batch)
            confound_nll = compute_confound_nll(residuals_T.T, B_conf[:n_v, :], lvc)

        # ========== Y Reconstruction ==========
        # Y reconstruction provides gradient signal for X->Y edges
        # With stop_gradient on classification, X→Y edges had NO gradient signal
        # This reconstruction gives A[:, Y_idx] gradients from SEM structure learning
        # (Classification still uses stop_gradient to prevent classifier from corrupting structure)
        weights_Y_recon = jnp.abs(A_curr[:n_v, Y_idx])  # X->Y weights WITH gradient
        # Additive baseline instead of hard jnp.where fallback. jnp.where has
        # zero gradient for the unselected branch, so an additive floor always
        # passes gradient through |A| while ensuring the processor sees some
        # input even when A[:,Y] is near zero.
        weights_Y_recon = weights_Y_recon + 1.0 / n_v  # additive uniform baseline
        X_weighted_Y_recon = batch_data * weights_Y_recon[jnp.newaxis, :]

        # No stop_gradient on proc_params here: both MSE and BCE send gradients
        # to both A and proc_params, enabling co-evolution. Gradient audit
        # confirmed cosine=+0.9999 (MSE/BCE aligned on A), so no conflict.
        _pp_Y_recon = proc_params[Y_idx]

        if use_bf16:
            _X_wr_fwd = X_weighted_Y_recon.astype(jnp.bfloat16)
            _pp_Y_fwd = jax.tree.map(
                lambda x: x.astype(jnp.bfloat16)
                if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32
                else x,
                _pp_Y_recon,
            )
        else:
            _X_wr_fwd = X_weighted_Y_recon
            _pp_Y_fwd = _pp_Y_recon

        _proc_name_yrecon = processor.__class__.__name__
        if _proc_name_yrecon == "GNNAdapter":
            A_norm_recon = A_curr[:n_v, :n_v] / (
                jnp.sum(jnp.abs(A_curr[:n_v, :n_v]), axis=0, keepdims=True) + 1e-8
            )
            Y_recon_output = processor.forward(_X_wr_fwd, _pp_Y_fwd, A=A_norm_recon)
        elif _proc_name_yrecon == "DAGAttentionAdapter":
            Y_recon_output = processor.forward(
                _X_wr_fwd, _pp_Y_fwd, A=A_curr[:n_v, :n_v]
            )
        elif _proc_name_yrecon == "CausalMambaAdapter":
            if causal_mamba_temperature_schedule is not None:
                Y_recon_output = processor.forward(
                    _X_wr_fwd, _pp_Y_fwd, A=A_curr[:n_v, :n_v],
                    temperature=current_cm_temperature,
                )
            else:
                Y_recon_output = processor.forward(
                    _X_wr_fwd, _pp_Y_fwd, A=A_curr[:n_v, :n_v]
                )
        else:
            Y_recon_output = processor.forward(_X_wr_fwd, _pp_Y_fwd)

        if use_bf16:
            Y_recon_output = Y_recon_output.astype(jnp.float32)

        # MSE for Y reconstruction: provides 62x stronger gradient to A[:,Y_idx]
        # than BCE (which has ~constant gradient due to sigmoid compression).
        # MSE drives structure learning (A[:,Y_idx]), BCE drives classification (proc_params).
        Y_target = batch_Y_effect if batch_Y_effect is not None else batch_Y.astype(jnp.float32)
        Y_recon_loss = jnp.mean((Y_recon_output - Y_target) ** 2)
        Y_recon_weight = 3.0
        total_recon_loss = (total_recon_loss * n_v + Y_recon_weight * Y_recon_loss) / (
            n_v + Y_recon_weight
        )

        # Classification for Y (j = Y_idx).
        # No stop_gradient on A: gradient audit confirmed MSE and BCE are perfectly
        # aligned on A (cosine=+0.9999). stop_gradient would create a deadlock
        # where neither path can bootstrap when A is small.
        weights_Y = jnp.abs(A_curr[:n_v, Y_idx])  # X->Y weights WITH gradient
        # Additive floor instead of max floor: jnp.maximum(|A|, c) has zero
        # gradient when |A| < c, creating a dead zone. Additive floor preserves
        # gradient for all nonzero A values (d(|A|+e)/dA = sign(A)).
        weights_Y = weights_Y + 0.01
        X_weighted_Y = batch_data * weights_Y[jnp.newaxis, :]

        if use_bf16:
            _X_wY_fwd = X_weighted_Y.astype(jnp.bfloat16)
            _pp_Yc_fwd = jax.tree.map(
                lambda x: x.astype(jnp.bfloat16)
                if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32
                else x,
                proc_params[Y_idx],
            )
        else:
            _X_wY_fwd = X_weighted_Y
            _pp_Yc_fwd = proc_params[Y_idx]

        # skip_centering=True for classification: mean centering forces output to
        # zero-mean, making sigmoid(~0)=0.5 and BCE=ln(2) (dead signal). By skipping
        # centering ONLY for classification, output_proj_b acts as a learnable class
        # prior. X_recon and Y_recon still use centering (prevents constant-output collapse).
        _proc_name_yclass = processor.__class__.__name__
        if _proc_name_yclass == "GNNAdapter":
            A_norm_class = A_curr[:n_v, :n_v] / (
                jnp.sum(jnp.abs(A_curr[:n_v, :n_v]), axis=0, keepdims=True) + 1e-8
            )
            Y_output = processor.forward(_X_wY_fwd, _pp_Yc_fwd, A=A_norm_class, skip_centering=True)
        elif _proc_name_yclass == "DAGAttentionAdapter":
            Y_output = processor.forward(
                _X_wY_fwd, _pp_Yc_fwd, A=A_curr[:n_v, :n_v], skip_centering=True
            )
        elif _proc_name_yclass == "CausalMambaAdapter":
            if causal_mamba_temperature_schedule is not None:
                Y_output = processor.forward(
                    _X_wY_fwd, _pp_Yc_fwd, A=A_curr[:n_v, :n_v], skip_centering=True,
                    temperature=current_cm_temperature,
                )
            else:
                Y_output = processor.forward(
                    _X_wY_fwd, _pp_Yc_fwd, A=A_curr[:n_v, :n_v], skip_centering=True
                )
        else:
            Y_output = processor.forward(_X_wY_fwd, _pp_Yc_fwd, skip_centering=True)

        if use_bf16:
            Y_output = Y_output.astype(jnp.float32)

        # Task-dependent loss for Y
        if task == "regression":
            classification_loss = jnp.mean((Y_output - batch_Y) ** 2)
        else:
            # Clamp logits to [-6, 6] before sigmoid to prevent BCE explosion
            # when processor produces large uncentered outputs with skip_centering=True.
            # At +/-6, sigmoid is ~0.9975, so max per-sample loss is ~6.
            Y_output_clamped = jnp.clip(Y_output, -6.0, 6.0)
            Y_pred = jax.nn.sigmoid(Y_output_clamped)
            eps = 1e-7
            classification_loss = -jnp.mean(
                batch_Y * jnp.log(Y_pred + eps) + (1 - batch_Y) * jnp.log(1 - Y_pred + eps)
            )

        # ========== DAG-Attention Consistency Loss (Phase 2 bidirectional signal) ==========
        # KL(attention || sigmoid(A * temp)) computed on the Y-classification
        # forward pass — pulls attention patterns toward causal edges and A
        # toward observed attention patterns. Active only when processor is
        # DAGAttentionAdapter and lambda_consistency > 0.
        consistency_loss = jnp.array(0.0)
        if lambda_consistency > 0 and _proc_name_yclass == "DAGAttentionAdapter":
            consistency_loss = processor.consistency_loss(
                _X_wY_fwd, _pp_Yc_fwd, A_curr[:n_v, :n_v],
                temperature=current_temp_consistency,
                edge_only=consistency_edge_only,
                edge_threshold=consistency_edge_threshold,
            )

        # ========== Structure Penalties ==========

        # Sparsity on A_direct — adaptive Y-column penalty based on classification progress.
        # When class_loss ≈ ln(2) (stuck), Y edges are fully protected (scale→0);
        # as classification improves, normal sparsity resumes (scale→1).
        if task == "classification":
            class_ratio = jnp.clip(classification_loss / 0.6931, 0.0, 1.0)
        else:
            class_ratio = jnp.clip(classification_loss / (classification_loss + 1.0), 0.0, 1.0)
        # Floor at 0.0: with additive classification floor, the gradient dead zone
        # is gone, so we can fully protect Y edges when classification is stuck.
        # Classification gradient flows through additive floor, giving Y edges
        # proper grow/shrink signals.
        y_sparsity_scale = jnp.clip(1.0 - class_ratio, 0.0, 1.0)
        sparsity_mask = jnp.ones((n_total_vars, n_total_vars)).at[:, Y_idx].set(y_sparsity_scale)
        sparsity_loss = lambda_1 * jnp.sum(jnp.abs(A_curr) * sparsity_mask)

        # DAG constraint (DAGMA log-det: better gradients, O(d^2) memory).
        # When enforce_outcome_sink=True, compute DAG constraint on X-only
        # submatrix (exclude Y row and column). Y is a sink node (no outgoing
        # edges) and cannot participate in cycles, so X->Y edges should be free
        # from acyclicity pressure. Otherwise the optimizer sacrifices X->Y edges
        # for "cheap" acyclicity.
        if enforce_outcome_sink:
            # Remove Y row and column: only X×X subgraph can have cycles
            # Use index array (not jnp.delete which has dynamic shape issues in JIT)
            x_idx = jnp.concatenate([jnp.arange(Y_idx), jnp.arange(Y_idx + 1, n_total_vars)])
            A_dag = A_curr[jnp.ix_(x_idx, x_idx)]
            h_A = dag_constraint(A_dag)
        else:
            h_A = dag_constraint(A_curr)
        dag_loss = lambda_2_current * h_A

        # structural_loss = total_recon_loss (includes Y_recon). confound_nll covers
        # X vars only and is added as a separate penalty to preserve its gradient
        # signal to B_confound without drowning out Y reconstruction.
        structural_loss = total_recon_loss
        if B_conf is not None:
            penalty_loss_confound = w_recon * confound_nll
        else:
            penalty_loss_confound = 0.0

        # Penalties added outside uncertainty weighting
        penalty_loss = sparsity_loss + dag_loss

        # L2 regularization on A (prevents weight saturation at ~0.99)
        if lambda_L2 > 0:
            l2_loss = lambda_L2 * jnp.sum(A_curr**2)
            penalty_loss += l2_loss

        # Legacy weight decay (deprecated, use lambda_L2 instead)
        if weight_decay > 0 and lambda_L2 == 0:
            penalty_loss += weight_decay * jnp.sum(A_curr**2)

        # ========== PC Structure Constraint (v11.2) ==========
        # Penalize deviation from PC warm-start structure to preserve causal edges
        # A_init_const is captured from outer scope (set before loss_fn definition)
        # NOTE: With stop_gradient decoupling (Option D), this is now a soft guide
        pc_loss = 0.0
        if A_init_const is not None:
            # Encourage edges PC found (A_init > 0) to stay strong
            # Penalize when A_curr is small where A_init is 1
            pc_edge_mask = (jnp.abs(A_init_const) > 0.5).astype(jnp.float32)
            # Loss for deviating from PC edges: encourage A_curr to be large where PC found edges
            pc_deviation = jnp.sum(pc_edge_mask * jnp.maximum(0.1 - jnp.abs(A_curr), 0))
            lambda_pc = 2.0  # Moderate PC guidance
            pc_loss = lambda_pc * pc_deviation
            penalty_loss += pc_loss

        # ========== Bi-directed Edge Penalties (v7.0) ==========

        bow_loss = 0.0
        if use_confound_matrix and B_conf is not None:
            # Low-rank path: penalties on B_confound and Ω_offdiag = B@B.T
            # Sparsity on factor loadings (encourages sparse latent structure)
            penalty_loss += lambda_confound_sparse * jnp.sum(jnp.abs(B_conf))

            # Bow-free penalty using Ω_offdiag: can't have both i→j AND i↔j
            Omega_offdiag = B_conf @ B_conf.T
            Omega_offdiag = Omega_offdiag.at[jnp.diag_indices(Omega_offdiag.shape[0])].set(0.0)
            bow_loss = lambda_bow * jnp.sum(
                jnp.abs(A_curr[:n_v, :n_v]) * jnp.abs(Omega_offdiag[:n_v, :n_v])
            )
            penalty_loss += bow_loss

            # Strong penalty on confound edges TO Y (outcome)
            lambda_confound_to_Y = 2.0
            confound_to_Y = jnp.abs(Omega_offdiag[:n_v, Y_idx])
            penalty_loss += lambda_confound_to_Y * jnp.sum(confound_to_Y)

        elif use_confound_matrix and A_conf is not None:
            # Legacy path: A_confound (d×d symmetric)
            # A_conf already computed and symmetrized above
            # Sparsity on A_confound
            penalty_loss += lambda_confound_sparse * jnp.sum(jnp.abs(A_conf))
            # Bow-free penalty: can't have both X_i -> X_j AND X_i <-> X_j
            bow_loss = lambda_bow * compute_bow_free_penalty_v7(A_curr, A_conf)
            penalty_loss += bow_loss

            # Strong penalty on confound edges TO Y (outcome)
            lambda_confound_to_Y = 2.0
            confound_to_Y = jnp.abs(A_conf[:n_v, Y_idx])
            penalty_loss += lambda_confound_to_Y * jnp.sum(confound_to_Y)

        # ========== D-Optimality Regularizer (v15: Identifiability) ==========
        # Differentiable proxy for condition number of MB design matrix.
        # Active only in Phase 2-3 (gated by effect_enabled) when MB is meaningful.
        if lambda_ident > 0:
            # Dynamic MB from current A: features with strong edges to/from Y
            mb_mask_to_Y = jnp.abs(A_curr[:, Y_idx]) > ident_threshold
            mb_mask_from_Y = jnp.abs(A_curr[Y_idx, :]) > ident_threshold
            mb_mask = mb_mask_to_Y | mb_mask_from_Y
            mb_mask = mb_mask.at[Y_idx].set(False)  # Exclude Y itself
            n_mb = jnp.sum(mb_mask)

            # Only apply when MB has >= 2 features (otherwise trivially conditioned)
            def _compute_d_opt(batch_data, mb_mask):
                from jcce.validation.identifiability_diagnostics import d_optimality_penalty

                # Soft selection via mask multiplication (keeps gradients flowing)
                mb_indices = jnp.where(mb_mask, size=n_total_vars)[0]
                X_mb = batch_data[:, mb_indices]
                return d_optimality_penalty(X_mb)

            # Gate: only active when effect estimation is active (Phase 2-3)
            # AND when MB has at least 2 features
            d_opt_active = jnp.where((n_mb >= 2) & effect_warmup_completed, 1.0, 0.0)
            d_opt_loss = jnp.where(
                d_opt_active > 0.5,
                lambda_ident * _compute_d_opt(batch_data, mb_mask),
                0.0,
            )
            penalty_loss = penalty_loss + d_opt_loss

        # ========== Effect Loss (v7.0, after warmup) ==========

        effect_loss = 0.0
        if use_amortized_effects and effect_warmup_completed:
            eff_params = params["effect_params"]

            # Compute effect loss for ALL X features (potential treatments)
            # Weight each treatment's loss by its edge strength to Y
            parent_weights = jnp.abs(A_curr[:n_v, Y_idx])  # Only X->Y weights

            # Normalize weights to sum to 1 (or use uniform if all small)
            weight_sum = jnp.sum(parent_weights)
            normalized_weights = jnp.where(
                weight_sum > 0.01,
                parent_weights / weight_sum,
                jnp.ones(n_v) / n_v,  # Uniform fallback
            )

            # Compute weighted effect loss over all X features as treatments.
            # Uses valid adjustment sets from backdoor criterion (closure variable).
            total_effect_loss = 0.0
            for t_idx in range(n_v):
                if t_idx == Y_idx:
                    continue  # Skip Y as treatment

                # Get valid covariates for this treatment (or None for legacy)
                valid_covs = None
                if adjustment_sets is not None and t_idx in adjustment_sets:
                    valid_covs = list(adjustment_sets[t_idx])

                if use_structural_dml:
                    t_loss, _ = structural_dml_effect_loss(
                        batch_data,
                        batch_Y_effect,
                        t_idx,
                        Y_idx,
                        eff_params,
                        valid_covariates=valid_covs,
                        A_weights=A_curr if valid_covs is None else None,
                    )
                elif use_dragonnet:
                    t_loss, _ = compute_dragonnet_loss(
                        batch_data,
                        batch_Y_effect,
                        t_idx,
                        Y_idx,
                        eff_params,
                        targeted_reg,
                        valid_covariates=valid_covs,
                    )
                else:
                    t_loss, _ = compute_amortized_effect_loss(
                        batch_data, batch_Y_effect, t_idx, Y_idx, eff_params
                    )
                # Weight by edge strength (or uniform if early training)
                total_effect_loss += normalized_weights[t_idx] * t_loss

            effect_loss = total_effect_loss

        # Effect weight: 0 during warmup, then controlled by curriculum
        # (No ramp-up - causes validation loss to increase, breaking early stopping)
        effect_enabled = 1.0 if effect_warmup_completed else 0.0

        # ========== Total Loss with Curriculum Weighting ==========
        # Uncertainty weighting removed: log_var_class and log_var_recon frozen at 0.
        # The optimizer exploited learnable uncertainty weights to suppress
        # classification, causing the class=0.6931 plateau.
        weighted_structural = w_recon * structural_loss
        weighted_class = w_class * lambda_class * classification_loss
        weighted_effect = w_effect * effect_enabled * lambda_effect * effect_loss
        # Use dynamic lambda_consistency: equals static lambda_consistency unless
        # adaptive engagement holds it at 0 until collapse is detected.
        weighted_consistency = current_lambda_consistency * consistency_loss
        weighted_aap = lambda_aap * aap_loss

        total_loss = (
            weighted_structural
            + weighted_class
            + weighted_effect
            + weighted_consistency
            + weighted_aap
            + penalty_loss
            + penalty_loss_confound
        )

        return total_loss, (
            h_A,
            total_recon_loss,
            classification_loss,
            effect_loss,
            bow_loss,
            Y_output,
            consistency_loss,
            aap_loss,
        )

    # ==================== Optimizer ====================

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr))
    opt_state = optimizer.init(all_params)

    lambda_2 = lambda_2_init
    best_loss = float("inf")
    patience_counter = 0
    best_params = all_params.copy()
    gradient_diagnostics = []  # Gradient cosine similarity logs

    # Initialize adaptive curriculum
    if use_adaptive_curriculum:
        curriculum_state = AdaptiveCurriculumState(
            plateau_window=10,
            plateau_threshold=0.05,
            min_phase_iters=max(20, max_iter // 5),  # At least 20% of training per phase
            max_phase1_ratio=0.5,  # Spend up to 50% on structure
            max_phase2_ratio=0.8,  # Transition to effects by 80%
            h_A_threshold=0.05,  # Stricter: require h(A) < 0.05 for phase transition
            accuracy_threshold=0.75,  # Stricter: require 75% accuracy
        )
        curriculum_weights = (
            1.0,
            0.5,
            0.0,
        )  # Start in Phase 1 (w_class=0.5 enables classification gradients through A)
    else:
        curriculum_state = None
        curriculum_weights = get_phase_weights_fixed(0.0, phase_splits=curriculum_phase_splits)

    # Track metrics for curriculum decisions
    last_accuracy = 0.0

    # Cache for valid adjustment sets (computed once after structure stabilizes)
    adjustment_sets = None
    adjustment_sets_computed = False

    # Track previous h(A) for adaptive lambda update
    prev_h_A = float("inf")

    # ==================== Data Pipeline ====================
    # Place training data on device once to avoid repeated host→device transfers.
    # Batch selection is done inside JIT to eliminate Python↔XLA round trips.
    data_train = jax.device_put(data_train)
    Y_train = jax.device_put(Y_train)
    Y_effect_train = jax.device_put(Y_effect_train)
    if data_val is not None:
        data_val = jax.device_put(data_val)
        Y_val = jax.device_put(Y_val)
        Y_effect_val = jax.device_put(Y_effect_val)

    # ==================== JIT-Compiled Train Step ====================
    # Wraps loss+grad+optimizer+constraints into a single JIT-compiled function.
    # Avoids re-tracing the computation graph every iteration (~3-10x speedup).
    # Closure variables (effect_warmup_completed, adjustment_sets, data arrays)
    # are captured at trace time. When they change, make_train_step() is called
    # to create a new JIT function (at most 2-3 recompilations per training run).
    effect_warmup_completed = False

    # Pre-allocate JAX scalars to avoid per-iteration allocation
    lambda_2_jax = jnp.float32(lambda_2)
    curriculum_w_jax = jnp.array(
        [curriculum_weights[0], curriculum_weights[1], curriculum_weights[2]], dtype=jnp.float32
    )

    def _pcgrad_project(grads_a, grads_b):
        """PCGrad: project grads_a onto the normal plane of grads_b when they conflict.
        Operates on the A_direct component only (the shared structural parameter).
        Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020."""
        g_a = grads_a["A_direct"].flatten()
        g_b = grads_b["A_direct"].flatten()
        dot = jnp.sum(g_a * g_b)
        # Only project if conflicting (negative cosine)
        proj = jnp.where(dot < 0, g_a - (dot / (jnp.sum(g_b**2) + 1e-12)) * g_b, g_a)
        return {**grads_a, "A_direct": proj.reshape(grads_a["A_direct"].shape)}

    def make_train_step():
        """Create JIT-compiled training step. Recreated when closure variables change."""

        @jax.jit
        def _step(
            params, opt_state, batch_key, lambda_2_jax, curriculum_w,
            current_temp, current_lambda_consistency_jax, current_cm_temp_jax,
        ):
            # Batch selection inside JIT (avoids 3 Python↔XLA round trips)
            if use_batching:
                batch_idx = random.choice(
                    batch_key, n_samples, shape=(effective_batch_size,), replace=False
                )
                batch_data = data_train[batch_idx]
                batch_Y = Y_train[batch_idx]
                batch_Y_effect = Y_effect_train[batch_idx]
            else:
                batch_data = data_train
                batch_Y = Y_train
                batch_Y_effect = Y_effect_train

            cw = (curriculum_w[0], curriculum_w[1], curriculum_w[2])

            if use_pcgrad:
                # PCGrad: compute per-task gradients and resolve conflicts on A_direct
                # The total loss = structural + class + effect + penalty (DAG + sparsity + L2)
                # PCGrad projects the 3 task gradients; penalty gradient is added back unchanged
                # so h(A) enforcement is never lost.
                def _recon_loss(p):
                    _, aux = loss_fn(
                        p, batch_data, batch_Y, batch_Y_effect, lambda_2_jax, cw, current_temp,
                        current_lambda_consistency_jax, current_cm_temp_jax,
                    )
                    return aux[1]  # total_recon_loss

                def _class_loss(p):
                    _, aux = loss_fn(
                        p, batch_data, batch_Y, batch_Y_effect, lambda_2_jax, cw, current_temp,
                        current_lambda_consistency_jax, current_cm_temp_jax,
                    )
                    return aux[2]  # classification_loss

                def _effect_loss(p):
                    _, aux = loss_fn(
                        p, batch_data, batch_Y, batch_Y_effect, lambda_2_jax, cw, current_temp,
                        current_lambda_consistency_jax, current_cm_temp_jax,
                    )
                    return aux[3]  # effect_loss

                # Get total loss + aux for logging (single forward pass)
                (loss_val, aux), grads_total = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, batch_data, batch_Y, batch_Y_effect, lambda_2_jax, cw, current_temp,
                    current_lambda_consistency_jax, current_cm_temp_jax,
                )
                # Per-task gradients (only for A_direct projection)
                g_recon = jax.grad(_recon_loss)(params)
                g_class = jax.grad(_class_loss)(params)
                g_effect = jax.grad(_effect_loss)(params)
                # Project: class onto recon's normal plane, effect onto class's normal plane
                g_class_proj = _pcgrad_project(g_class, g_recon)
                g_effect_proj = _pcgrad_project(g_effect, g_class_proj)
                # Penalty gradient = total - (recon + class + effect) on A_direct
                # This preserves DAG constraint (h(A)), sparsity (L1), and L2 gradients
                g_penalty_A = grads_total["A_direct"] - (
                    g_recon["A_direct"] + g_class["A_direct"] + g_effect["A_direct"]
                )
                # Combine: projected task gradients + unmodified penalty gradient
                grads = {
                    **grads_total,
                    "A_direct": g_recon["A_direct"]
                    + g_class_proj["A_direct"]
                    + g_effect_proj["A_direct"]
                    + g_penalty_A,
                }
            else:
                (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, batch_data, batch_Y, batch_Y_effect, lambda_2_jax, cw, current_temp,
                    current_lambda_consistency_jax, current_cm_temp_jax,
                )

            # Freeze log_var_recon (its gradient exploits uncertainty weighting)
            grads = {**grads, "log_var_recon": jnp.zeros_like(grads["log_var_recon"])}
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            # Post-update hard constraints (Python bools evaluated at trace time)
            if enforce_outcome_sink:
                new_params["A_direct"] = new_params["A_direct"].at[Y_idx, :].set(0.0)
            if freeze_A:
                new_params["A_direct"] = A_direct_frozen
            if use_confound_matrix and "A_confound" in new_params:
                new_params["A_confound"] = symmetrize_confound_matrix(new_params["A_confound"])
            # Low-rank B_confound: no post-update needed (Ω = B@B.T is symmetric by construction)
            return new_params, new_opt_state, loss_val, aux, batch_Y

        return _step

    train_step = make_train_step()

    # Adaptive consistency engagement state (Python-side; not jit-traced)
    consistency_engaged = False
    consistency_engaged_iter = -1  # -1 = never engaged

    # ==================== Training Loop ====================

    for iter in range(max_iter):
        # JIT recompilation triggers (checked before train_step call)
        if not effect_warmup_completed and iter >= effect_warmup_iter:
            effect_warmup_completed = True
            train_step = make_train_step()
            if verbose >= 2:
                print(f"  [JIT] Recompiled train_step (effect warmup at iter {iter})")

        # Batch key for on-device batch selection inside JIT
        key, batch_key = random.split(key)

        # Temperature schedule for DAG-Attention consistency loss (cosine anneal).
        # When schedule is None, current_temp falls back to processor's default (or 5.0)
        # — equivalent to no scheduling.
        if temperature_consistency_schedule is not None:
            t_init, t_final = temperature_consistency_schedule
            iter_frac = iter / max(max_iter - 1, 1)
            cur_temp = float(t_init + 0.5 * (t_final - t_init) * (1.0 - jnp.cos(jnp.pi * iter_frac)))
        else:
            cur_temp = float(getattr(processor, "temperature", 5.0))
        current_temp_jax = jnp.float32(cur_temp)

        # CausalMamba Sinkhorn temperature schedule (cosine anneal, separate from
        # DAG-Attention's consistency temperature). Always pass as JAX scalar to
        # keep JIT signature stable; the Python check inside loss_fn (trace-time)
        # decides whether to actually pass it as a kwarg to the adapter.
        if causal_mamba_temperature_schedule is not None:
            cm_init, cm_final = causal_mamba_temperature_schedule
            iter_frac_cm = iter / max(max_iter - 1, 1)
            cur_cm_temp = float(
                cm_init + 0.5 * (cm_final - cm_init) * (1.0 - jnp.cos(jnp.pi * iter_frac_cm))
            )
        else:
            cur_cm_temp = 0.0  # sentinel; not used when schedule is None
        current_cm_temp_jax = jnp.float32(cur_cm_temp)

        # Adaptive consistency engagement (Python-side decision, fed as JAX scalar).
        # When adaptive_consistency=False: always lambda_consistency.
        # When adaptive_consistency=True: per-iter dynamic — engage only when A is
        # CURRENTLY collapsed. This self-regulates: anchors A while it's small,
        # disengages once A recovers. Avoids the one-shot "stay engaged" failure
        # mode where a transient dip would lock in over-saturation on healthy datasets.
        # consistency_engaged_iter records the FIRST iter engagement fired (for diagnostic);
        # consistency_engaged is now a per-iter boolean recording the LATEST state.
        if adaptive_consistency and lambda_consistency > 0:
            n_v_check = data_train.shape[1] - 1  # X-only submatrix; Y is at Y_idx
            max_abs_A_now = float(
                jnp.max(jnp.abs(all_params["A_direct"][:n_v_check, :n_v_check]))
            )
            is_currently_collapsed = (
                max_abs_A_now < consistency_collapse_threshold
                and iter < consistency_max_engage_iter_frac * max_iter
            )
            if is_currently_collapsed and consistency_engaged_iter < 0:
                consistency_engaged_iter = iter
                if verbose >= 1:
                    print(
                        f"  [adaptive consistency] first engaged at iter {iter}, "
                        f"max|A|={max_abs_A_now:.4f} < threshold={consistency_collapse_threshold}"
                    )
            consistency_engaged = is_currently_collapsed
        else:
            consistency_engaged = (lambda_consistency > 0)  # static-on path

        if adaptive_consistency:
            cur_lambda_consistency = lambda_consistency if consistency_engaged else 0.0
        else:
            cur_lambda_consistency = lambda_consistency
        current_lambda_consistency_jax = jnp.float32(cur_lambda_consistency)

        # JIT-compiled: batch selection + forward + backward + optimizer update + constraints
        all_params, opt_state, loss_val, aux, batch_Y = train_step(
            all_params, opt_state, batch_key, lambda_2_jax, curriculum_w_jax,
            current_temp_jax, current_lambda_consistency_jax, current_cm_temp_jax,
        )
        h_A, recon_loss, class_loss, effect_loss, bow_loss, Y_logits, consistency_val, aap_val = aux

        # Update adaptive curriculum after each iteration
        if use_adaptive_curriculum and curriculum_state is not None:
            # Use Y_logits from loss computation (avoids redundant forward pass)
            if task == "regression":
                mse_val = float(jnp.mean((Y_logits - batch_Y) ** 2))
                y_var = float(jnp.var(batch_Y)) + 1e-10
                last_accuracy = max(1.0 - mse_val / y_var, 0.0)
            else:
                Y_pred = (jax.nn.sigmoid(Y_logits) > 0.5).astype(jnp.float32)
                last_accuracy = float(jnp.mean(Y_pred == batch_Y))

            old_phase = curriculum_state.current_phase
            curriculum_state, curriculum_weights = update_adaptive_curriculum(
                curriculum_state,
                current_iter=iter,
                max_iter=max_iter,
                recon_loss=float(recon_loss),
                class_loss=float(class_loss),
                effect_loss=float(effect_loss),
                h_A=float(h_A),
                accuracy=last_accuracy,
                verbose=(verbose >= 2),
                n_vars=n_vars,
            )
            # Update cached JAX array when curriculum weights change
            curriculum_w_jax = jnp.array(
                [curriculum_weights[0], curriculum_weights[1], curriculum_weights[2]],
                dtype=jnp.float32,
            )

            # Compute adjustment sets when transitioning to Phase 2
            if curriculum_state.current_phase >= 2 and not adjustment_sets_computed:
                A_curr = all_params["A_direct"]
                adjustment_sets = compute_valid_adjustment_sets(A_curr, Y_idx, threshold=0.05)
                adjustment_sets_computed = True
                train_step = make_train_step()
                if verbose >= 2:
                    print(f"  [JIT] Recompiled train_step (adjustment sets at iter {iter})")
        else:
            # Fixed curriculum
            progress = iter / max_iter if max_iter > 0 else 1.0
            curriculum_weights = get_phase_weights_fixed(
                progress, phase_splits=curriculum_phase_splits
            )
            curriculum_w_jax = jnp.array(
                [curriculum_weights[0], curriculum_weights[1], curriculum_weights[2]],
                dtype=jnp.float32,
            )

            # For fixed curriculum, compute adjustment sets at 40%
            if progress >= 0.4 and not adjustment_sets_computed:
                A_curr = all_params["A_direct"]
                adjustment_sets = compute_valid_adjustment_sets(A_curr, Y_idx, threshold=0.05)
                adjustment_sets_computed = True
                train_step = make_train_step()
                if verbose >= 2:
                    print(f"  [JIT] Recompiled train_step (adjustment sets at iter {iter})")

        # DAG penalty schedule (Python-level, outside JIT)
        h_A_val = float(h_A)
        old_lambda_2 = lambda_2
        if h_A_val > 0.1:
            if iter > 5 and h_A_val >= prev_h_A * 0.99:
                lambda_2 = min(lambda_2 * 2.0, lambda_2_max)
        elif h_A_val > 0.01 and iter > 20:
            # Only increase if h_A is not improving (unconditional increase
            # causes gradient clipping death spiral at local minima).
            if h_A_val >= prev_h_A * 0.99:
                lambda_2 = min(lambda_2 * 1.5, lambda_2_max)
        if lambda_2 != old_lambda_2:
            lambda_2_jax = jnp.float32(lambda_2)
        prev_h_A = h_A_val

        # Early stopping on validation (eager mode, only every 10 iters)
        if use_validation_split and iter % 10 == 0 and iter >= effect_warmup_iter:
            val_loss, _ = loss_fn(
                all_params, data_val, Y_val, Y_effect_val, lambda_2, curriculum_weights,
                cur_temp, cur_lambda_consistency, current_cm_temp_jax,
            )
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best_params = {
                    k: v.copy() if hasattr(v, "copy") else v for k, v in all_params.items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Phase 3: Optuna iteration callback for Hyperband pruning
        if iteration_callback is not None and iter % 10 == 0:
            # Compute proper balanced accuracy (not plain accuracy)
            if task == "regression":
                mse_val = float(jnp.mean((Y_logits - batch_Y) ** 2))
                y_var = float(jnp.var(batch_Y)) + 1e-10
                _cb_acc = max(1.0 - mse_val / y_var, 0.0)
            else:
                Y_pred_cb = (jax.nn.sigmoid(Y_logits) > 0.5).astype(jnp.float32)
                _tp = float(jnp.sum(Y_pred_cb * batch_Y))
                _tn = float(jnp.sum((1 - Y_pred_cb) * (1 - batch_Y)))
                _fp = float(jnp.sum(Y_pred_cb * (1 - batch_Y)))
                _fn = float(jnp.sum((1 - Y_pred_cb) * batch_Y))
                _sens = _tp / (_tp + _fn + 1e-8)
                _spec = _tn / (_tn + _fp + 1e-8)
                _cb_acc = (_sens + _spec) / 2.0
            # Include curriculum phase so pruner can skip Phase 1 trials
            _cb_phase = (
                curriculum_state.current_phase
                if (use_adaptive_curriculum and curriculum_state is not None)
                else (1 if (iter / max_iter) < 0.4 else 2)
            )
            iteration_callback(
                iter,
                {
                    "balanced_accuracy": _cb_acc,
                    "h_A": h_A_val,
                    "loss": float(loss_val),
                    "curriculum_phase": _cb_phase,
                },
            )

        if verbose >= 2 and iter % 20 == 0:
            print(
                f"Iter {iter:3d}: loss={float(loss_val):.4f}, h(A)={h_A_val:.4f}, "
                f"class={float(class_loss):.4f}, effect={float(effect_loss):.4f}"
            )

        # Gradient cosine diagnostic: measure alignment between loss components
        # Computed every 50 iters in eager mode (outside JIT) to detect conflicts
        if iter % 50 == 0 and iter > 0:
            try:
                A_cur = all_params["A_direct"]

                # Compute gradient of each weighted loss component w.r.t. A
                def _recon_only(params, *args):
                    _, aux = loss_fn(params, *args)
                    return aux[1]  # total_recon_loss

                def _class_only(params, *args):
                    _, aux = loss_fn(params, *args)
                    return aux[2]  # classification_loss

                def _effect_only(params, *args):
                    _, aux = loss_fn(params, *args)
                    return aux[3]  # effect_loss

                _loss_args = (
                    data_train, Y_train, Y_effect_train, lambda_2_jax, curriculum_w_jax,
                    current_temp_jax, current_lambda_consistency_jax, current_cm_temp_jax,
                )
                g_r = jax.grad(_recon_only)(all_params, *_loss_args)["A_direct"].flatten()
                g_c = jax.grad(_class_only)(all_params, *_loss_args)["A_direct"].flatten()

                def _cos_sim(a, b):
                    dot = float(jnp.sum(a * b))
                    na = float(jnp.sqrt(jnp.sum(a**2))) + 1e-12
                    nb = float(jnp.sqrt(jnp.sum(b**2))) + 1e-12
                    return dot / (na * nb)

                cos_rc = _cos_sim(g_r, g_c)
                if effect_warmup_completed:
                    g_e = jax.grad(_effect_only)(all_params, *_loss_args)["A_direct"].flatten()
                    cos_re = _cos_sim(g_r, g_e)
                    cos_ce = _cos_sim(g_c, g_e)
                else:
                    cos_re, cos_ce = 0.0, 0.0

                gradient_diagnostics.append(
                    {
                        "iter": iter,
                        "cos_recon_class": cos_rc,
                        "cos_recon_effect": cos_re,
                        "cos_class_effect": cos_ce,
                    }
                )
                if verbose >= 2:
                    eff_str = (
                        f", cos(r,e)={cos_re:.3f}, cos(c,e)={cos_ce:.3f}"
                        if effect_warmup_completed
                        else ""
                    )
                    print(f"  [Gradient] cos(recon,class)={cos_rc:.3f}{eff_str}")
            except Exception as _grad_err:
                if verbose >= 2:
                    print(f"  [Gradient diagnostic failed: {_grad_err}]")

    # Use best params
    if use_validation_split:
        all_params = best_params

    # Track which treatments were actually trained (for consistent evaluation)
    trained_treatments = []

    # ==================== Post-hoc Effect Refinement ====================
    # After structure converges, refine effect network with frozen structure
    if use_amortized_effects and effect_refinement_iters > 0:
        # Create effect-only loss function (structure frozen)
        A_frozen = all_params["A_direct"].copy()

        # Recompute adjustment sets with final frozen structure
        adjustment_sets_final = compute_valid_adjustment_sets(A_frozen, Y_idx, threshold=0.05)

        # Low threshold (0.01) with top-K fallback to capture more treatments
        edge_weights_to_Y = jnp.abs(A_frozen[:n_vars, Y_idx])
        edge_threshold = 0.01
        significant_treatments = [
            t_idx
            for t_idx in range(n_vars)
            if t_idx != Y_idx and float(edge_weights_to_Y[t_idx]) > edge_threshold
        ]

        # Top-K fallback if threshold is too aggressive
        min_treatments = 3
        if len(significant_treatments) < min_treatments:
            sorted_indices = jnp.argsort(-edge_weights_to_Y)  # Descending
            significant_treatments = [
                int(idx) for idx in sorted_indices[: min_treatments + 1] if int(idx) != Y_idx
            ][:min_treatments]

        trained_treatments = significant_treatments.copy()

        def effect_only_loss(effect_params, batch_data, batch_Y_effect):
            """Minimize only effect loss with frozen structure."""
            total_effect_loss = 0.0
            n_treatments = 0
            for t_idx in significant_treatments:
                # Get valid covariates for this treatment
                valid_covs = None
                if t_idx in adjustment_sets_final:
                    valid_covs = list(adjustment_sets_final[t_idx])

                # Use the appropriate effect estimation method
                if use_structural_dml:
                    t_loss, _ = structural_dml_effect_loss(
                        batch_data,
                        batch_Y_effect,
                        t_idx,
                        Y_idx,
                        effect_params,
                        valid_covariates=valid_covs,
                    )
                elif use_dragonnet:
                    t_loss, _ = compute_dragonnet_loss(
                        batch_data,
                        batch_Y_effect,
                        t_idx,
                        Y_idx,
                        effect_params,
                        targeted_reg,
                        valid_covariates=valid_covs,
                    )
                else:
                    t_loss, _ = compute_amortized_effect_loss(
                        batch_data, batch_Y_effect, t_idx, Y_idx, effect_params
                    )
                # Uniform weighting for significant treatments
                total_effect_loss += t_loss
                n_treatments += 1
            return total_effect_loss / max(n_treatments, 1)

        # Separate optimizer for effect refinement (higher LR)
        effect_optimizer = optax.adam(learning_rate=lr * 2)
        effect_opt_state = effect_optimizer.init(all_params["effect_params"])

        @jax.jit
        def effect_refine_step(effect_params, opt_state, batch_data, batch_Y_effect):
            loss, grads = jax.value_and_grad(effect_only_loss)(
                effect_params, batch_data, batch_Y_effect
            )
            updates, new_opt_state = effect_optimizer.update(grads, opt_state, effect_params)
            new_params = optax.apply_updates(effect_params, updates)
            return new_params, new_opt_state, loss

        # Refinement loop
        for refine_iter in range(effect_refinement_iters):
            all_params["effect_params"], effect_opt_state, effect_loss = effect_refine_step(
                all_params["effect_params"], effect_opt_state, data_train, Y_effect_train
            )

    # ==================== Extract Results ====================

    A_final = all_params["A_direct"]

    # Threshold for binary DAG
    # Balanced threshold (1% was too low, 10% too high)
    max_weight = jnp.max(jnp.abs(A_final))
    threshold = max(
        0.05 * max_weight, 0.03
    )  # 5% of max or at least 0.03 (raised: S1 fix makes weights sparser)
    A_binary = (jnp.abs(A_final) > threshold).astype(jnp.float32)

    # Zero diagonal
    A_binary = A_binary - jnp.diag(jnp.diag(A_binary))

    # Merge processor params
    final_proc_params = merge_trained_params(processor_params, all_params["processor_params"])

    # Extract Markov Blanket (v8.0: Full MB with spouses + confound neighbors)
    # MB(Y) = parents(Y) ∪ children(Y) ∪ spouses(Y) ∪ confound_neighbors(Y)
    # where spouses = other parents of Y's children

    # Parents: edges TO Y (X → Y)
    parents = set(int(i) for i in jnp.where(A_binary[:, Y_idx] > 0)[0])

    # Children: edges FROM Y (Y → X)
    children = set(int(i) for i in jnp.where(A_binary[Y_idx, :] > 0)[0])

    # Spouses: co-parents of Y's children (X → Child ← Y)
    spouses = set()
    for child in children:
        if child == Y_idx:
            continue
        # Find all parents of this child (other than Y)
        child_parents = set(int(k) for k in jnp.where(A_binary[:, child] > 0)[0])
        child_parents.discard(Y_idx)
        child_parents.discard(child)
        spouses.update(child_parents)

    # Confound neighbors: bi-directed edges to Y (X ↔ Y via latent)
    # Moderately strict threshold for confound edges
    confound_neighbors = set()
    if use_confound_matrix and "B_confound" in all_params:
        # Low-rank path: compute Ω_offdiag = B@B.T, threshold off-diagonal
        B_conf_final = all_params["B_confound"]
        Omega = B_conf_final @ B_conf_final.T
        Omega_offdiag = Omega.at[jnp.diag_indices(Omega.shape[0])].set(0.0)
        max_conf_weight = jnp.max(jnp.abs(Omega_offdiag))
        conf_threshold = max(0.1 * float(max_conf_weight), 0.05)
        A_conf_binary = (jnp.abs(Omega_offdiag) > conf_threshold).astype(jnp.float32)
        confound_neighbors = set(int(i) for i in jnp.where(A_conf_binary[:, Y_idx] > 0)[0])
        confound_neighbors.discard(Y_idx)
    elif use_confound_matrix and "A_confound" in all_params:
        # Legacy path
        A_conf_final = symmetrize_confound_matrix(all_params["A_confound"])
        max_conf_weight = jnp.max(jnp.abs(A_conf_final))
        conf_threshold = max(0.1 * float(max_conf_weight), 0.05)
        A_conf_binary = (jnp.abs(A_conf_final) > conf_threshold).astype(jnp.float32)
        confound_neighbors = set(int(i) for i in jnp.where(A_conf_binary[:, Y_idx] > 0)[0])
        confound_neighbors.discard(Y_idx)

    # Combine all MB components
    mb = parents | children | spouses | confound_neighbors
    mb.discard(Y_idx)

    # Effect-based MB augmentation
    # Compute effects for ALL X variables and add those with large effects
    # This helps catch true causes that might have weak edge weights
    effect_based_parents = set()
    if use_amortized_effects:
        eff_params = all_params["effect_params"]
        all_effects = {}

        # Compute effects for ALL X variables (not just current MB)
        for t_idx in range(n_vars):
            if t_idx == Y_idx:
                continue
            try:
                if use_structural_dml:
                    _, t_metrics = structural_dml_effect_loss(
                        data,
                        Y_effect,
                        t_idx,
                        Y_idx,
                        eff_params,
                        valid_covariates=None,
                    )
                    ate_value = t_metrics["ate_aipw"]
                elif use_dragonnet:
                    _, t_metrics = compute_dragonnet_loss(
                        data,
                        Y_effect,
                        t_idx,
                        Y_idx,
                        eff_params,
                        targeted_reg,
                        valid_covariates=None,
                    )
                    ate_value = t_metrics.get("ate_aipw", t_metrics["ate"])
                else:
                    _, t_metrics = compute_amortized_effect_loss(
                        data, Y_effect, t_idx, Y_idx, eff_params
                    )
                    ate_value = t_metrics["ate"]
                all_effects[t_idx] = abs(float(ate_value))
            except:
                all_effects[t_idx] = 0.0

        # Select top-k variables by effect magnitude
        # Changed from top-5 to top-3, with higher threshold to reduce false positives
        if all_effects:
            sorted_effects = sorted(all_effects.items(), key=lambda x: x[1], reverse=True)
            effect_values = [e for _, e in sorted_effects]
            if effect_values:
                median_effect = sorted(effect_values)[len(effect_values) // 2]
                # Higher threshold (1.5x median) to reduce over-selection
                effect_threshold = max(median_effect * 1.5, 0.05)

                # Only top-3 by effect magnitude
                # Note: Don't require edge weight - effect estimation catches causes
                # that structure learning misses (e.g., Smoking when Yellow_Fingers dominates)
                for t_idx, effect_mag in sorted_effects[:3]:
                    if effect_mag > effect_threshold:
                        effect_based_parents.add(t_idx)

    # Combine structure-based and effect-based MB
    mb = mb | effect_based_parents
    mb.discard(Y_idx)
    markov_blanket = sorted(list(mb))

    # Store MB components for analysis
    mb_components = {
        "parents": sorted(list(parents - {Y_idx})),
        "children": sorted(list(children - {Y_idx})),
        "spouses": sorted(list(spouses)),
        "confound_neighbors": sorted(list(confound_neighbors)),
        "effect_based": sorted(list(effect_based_parents)),
    }

    # Classification accuracy
    # Use combined A_direct + A_confound weights for classification
    # This ensures features are weighted correctly even if edges are in A_confound
    Y_pred_logits = jnp.zeros(n_samples_total)
    weights = jnp.abs(A_final[:, Y_idx])
    if use_confound_matrix and "A_confound" in all_params:
        # Add confound weights with 0.5 factor (confounding contributes less than direct)
        A_conf_weights = jnp.abs(A_conf_final[:, Y_idx])
        weights = weights + 0.5 * A_conf_weights
    weights = weights.at[Y_idx].set(0.0)
    # Raw |A| with floor — matches training-time weighting (no softmax)
    weights_Y_eval = jnp.maximum(weights[:n_vars], 0.01)
    # Use training data only for in-sample metrics (avoid test leakage)
    X_weighted = data_train * weights_Y_eval[jnp.newaxis, :]
    _proc_name_eval = processor.__class__.__name__
    if _proc_name_eval == "GNNAdapter":
        A_norm = A_final[:n_vars, :n_vars] / (
            jnp.sum(jnp.abs(A_final[:n_vars, :n_vars]), axis=0, keepdims=True) + 1e-8
        )
        Y_pred_logits = processor.forward(X_weighted, final_proc_params[Y_idx], A=A_norm)
    elif _proc_name_eval == "DAGAttentionAdapter":
        Y_pred_logits = processor.forward(
            X_weighted, final_proc_params[Y_idx], A=A_final[:n_vars, :n_vars]
        )
    elif _proc_name_eval == "CausalMambaAdapter":
        Y_pred_logits = processor.forward(
            X_weighted, final_proc_params[Y_idx], A=A_final[:n_vars, :n_vars]
        )
    else:
        Y_pred_logits = processor.forward(X_weighted, final_proc_params[Y_idx])
    Y_flat = Y_train.flatten()

    # Task-dependent post-training metrics
    if task == "regression":
        Y_pred = Y_pred_logits.flatten()
        ss_res = float(jnp.sum((Y_flat - Y_pred) ** 2))
        ss_tot = float(jnp.sum((Y_flat - jnp.mean(Y_flat)) ** 2))
        r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
        rmse = float(jnp.sqrt(jnp.mean((Y_flat - Y_pred) ** 2)))
        mae = float(jnp.mean(jnp.abs(Y_flat - Y_pred)))
        classification_accuracy = max(r2, 0.0)
        balanced_accuracy = classification_accuracy
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        specificity = 0.0
        auc_roc = 0.0
    else:
        r2 = 0.0
        rmse = 0.0
        mae = 0.0
        Y_pred_prob = jax.nn.sigmoid(Y_pred_logits)
        Y_pred_binary = (Y_pred_prob > 0.5).astype(jnp.float32)
        classification_accuracy = float(jnp.mean(Y_pred_binary == Y_flat))

        tp = float(jnp.sum((Y_pred_binary == 1) & (Y_flat == 1)))
        tn = float(jnp.sum((Y_pred_binary == 0) & (Y_flat == 0)))
        fp = float(jnp.sum((Y_pred_binary == 1) & (Y_flat == 0)))
        fn = float(jnp.sum((Y_pred_binary == 0) & (Y_flat == 1)))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        balanced_accuracy = (recall + specificity) / 2

        # ROC AUC
        Y_pred_prob_flat = Y_pred_prob.flatten()
        try:
            sorted_indices = jnp.argsort(-Y_pred_prob_flat)
            Y_sorted = Y_flat[sorted_indices]
            n_pos = float(jnp.sum(Y_flat == 1))
            n_neg = float(jnp.sum(Y_flat == 0))
            if n_pos > 0 and n_neg > 0:
                tpr_cumsum = jnp.cumsum(Y_sorted) / n_pos
                fpr_cumsum = jnp.cumsum(1 - Y_sorted) / n_neg
                auc_roc = float(
                    jnp.sum(
                        (fpr_cumsum[1:] - fpr_cumsum[:-1]) * (tpr_cumsum[1:] + tpr_cumsum[:-1]) / 2
                    )
                )
            else:
                auc_roc = 0.5
        except Exception:
            auc_roc = 0.5

    # Compute final effect estimates for trained treatments only.
    # Uses valid adjustment sets for proper causal effect estimation.
    causal_effects = {}
    final_effect_loss = 0.0
    if use_amortized_effects:
        eff_params = all_params["effect_params"]
        n_effects = 0

        # Compute adjustment sets for final evaluation (use final A matrix)
        adjustment_sets_eval = compute_valid_adjustment_sets(A_final, Y_idx, threshold=0.05)

        # Use trained_treatments if available, otherwise fall back to markov_blanket
        eval_treatments = trained_treatments if trained_treatments else markov_blanket
        for t_idx in eval_treatments:
            if t_idx == Y_idx:
                continue

            # Get valid covariates for this treatment
            valid_covs = None
            if t_idx in adjustment_sets_eval:
                valid_covs = list(adjustment_sets_eval[t_idx])

            if use_structural_dml:
                t_loss, t_metrics = structural_dml_effect_loss(
                    data,
                    Y_effect,
                    t_idx,
                    Y_idx,
                    eff_params,
                    valid_covariates=valid_covs,
                    A_weights=A_direct if valid_covs is None else None,
                )
                ate_value = t_metrics["ate_aipw"]
            elif use_dragonnet:
                t_loss, t_metrics = compute_dragonnet_loss(
                    data,
                    Y_effect,
                    t_idx,
                    Y_idx,
                    eff_params,
                    targeted_reg,
                    valid_covariates=valid_covs,
                )
                # Use AIPW estimate (doubly robust) when available
                ate_value = t_metrics.get("ate_aipw", t_metrics["ate"])
            else:
                t_loss, t_metrics = compute_amortized_effect_loss(
                    data, Y_effect, t_idx, Y_idx, eff_params
                )
                ate_value = t_metrics["ate"]
            causal_effects[f"X{t_idx}->Y"] = float(ate_value)
            final_effect_loss += float(t_loss)
            n_effects += 1
        if n_effects > 0:
            final_effect_loss = final_effect_loss / n_effects

        # Compute X->X effects using GPS-DragonNet (proper continuous treatment handling)
        # GPS-DragonNet trains a fresh model per edge with:
        # - Standardized outcomes (bounded effects in std units)
        # - Generalized Propensity Scores for continuous treatments
        # - AIPW for doubly-robust estimation

        # Count significant edges
        xx_edge_count = 0
        xx_edges_computed = 0
        for i in range(n_vars):
            if i == Y_idx:
                continue
            for j in range(n_vars):
                if j == Y_idx or j == i:
                    continue
                if abs(float(A_final[i, j])) > 0.05:
                    xx_edge_count += 1

        for i in range(n_vars):
            if i == Y_idx:
                continue  # Skip Y as source
            for j in range(n_vars):
                if j == Y_idx or j == i:
                    continue  # Skip Y as target and self-loops
                # Check if significant edge Xi→Xj exists
                edge_weight = float(jnp.abs(A_final[i, j]))
                if edge_weight > 0.05:  # Lower threshold for X→X edges
                    try:
                        # Use unified effect estimation with auto type detection
                        key, effect_key = random.split(key)
                        ate_xx, effect_metrics = compute_xx_effect_unified(
                            data=data,
                            treatment_idx=i,
                            outcome_idx=j,
                            key=effect_key,
                            max_iter=50,  # Fewer iterations for efficiency
                            hidden_dim=32,  # Smaller architecture for X→X
                            verbose=False,
                        )
                        causal_effects[f"X{i}->X{j}"] = float(ate_xx)
                        xx_edges_computed += 1

                    except Exception:
                        causal_effects[f"X{i}->X{j}"] = 0.0  # Default to zero on failure

    # Build metrics
    max_edges = n_vars * (n_vars - 1)
    sparsity = 1.0 - (float(jnp.sum(A_binary)) / max_edges) if max_edges > 0 else 1.0

    metrics = {
        # Structure
        "n_edges": int(jnp.sum(A_binary)),
        "sparsity": sparsity,
        "final_h_A": float(h_A),
        # Adaptive consistency engagement diagnostic: -1 if never engaged
        # (or adaptive_consistency=False); else iter at which it engaged.
        "consistency_engaged_iter": consistency_engaged_iter,
        "markov_blanket": markov_blanket,
        "markov_blanket_size": len(markov_blanket),
        # MB components for detailed analysis
        "mb_components": mb_components,
        "n_parents": len(mb_components["parents"]),
        "n_children": len(mb_components["children"]),
        "n_spouses": len(mb_components["spouses"]),
        "n_confound_neighbors": len(mb_components["confound_neighbors"]),
        # Classification/Regression metrics (v16: task-aware)
        "classification_accuracy": classification_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "balanced_accuracy": balanced_accuracy,
        "auc_roc": auc_roc,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "task": task,
        # Loss components
        "final_loss": float(loss_val),
        "final_recon_loss": float(recon_loss),
        "final_class_loss": float(class_loss),
        # Effect metrics
        "effect_loss": final_effect_loss,  # For NSGA-II fitness!
        "causal_effects": causal_effects,
        "trained_treatments": trained_treatments,
        # Bi-directed edges
        "bow_loss": float(bow_loss),
        # Training
        "iterations": iter + 1,
        # Gradient diagnostics (cosine similarity between loss components)
        "gradient_diagnostics": gradient_diagnostics,
        # Adjacency matrices
        "A_direct": A_binary.tolist(),
        "A_weights": A_final.tolist(),
    }

    # Add A_confound if used
    if use_confound_matrix and "B_confound" in all_params:
        # Low-rank path: compute Ω = B@B.T + σ²I, store Ω_offdiag under backward-compat keys
        B_conf_final = all_params["B_confound"]
        log_var_c = all_params["log_var_confound"]
        sigma2 = float(jnp.maximum(jnp.exp(log_var_c), 1e-6))
        Omega = B_conf_final @ B_conf_final.T + sigma2 * jnp.eye(B_conf_final.shape[0])
        Omega_offdiag = Omega.at[jnp.diag_indices(Omega.shape[0])].set(0.0)
        A_conf_binary = (jnp.abs(Omega_offdiag) > threshold).astype(jnp.float32)
        metrics["A_confound"] = A_conf_binary.tolist()  # backward compat
        metrics["A_confound_weights"] = Omega_offdiag.tolist()  # backward compat
        metrics["n_confound_edges"] = int(jnp.sum(A_conf_binary) / 2)  # Symmetric
        # New low-rank specific keys
        metrics["B_confound"] = B_conf_final.tolist()
        metrics["log_var_confound"] = float(log_var_c)
        metrics["n_latent_confounders"] = B_conf_final.shape[1]
    elif use_confound_matrix and "A_confound" in all_params:
        # Legacy path
        A_conf_final = symmetrize_confound_matrix(all_params["A_confound"])
        A_conf_binary = (jnp.abs(A_conf_final) > threshold).astype(jnp.float32)
        metrics["A_confound"] = A_conf_binary.tolist()
        metrics["A_confound_weights"] = A_conf_final.tolist()
        metrics["n_confound_edges"] = int(jnp.sum(A_conf_binary) / 2)  # Symmetric

    # Add effect params if used
    if use_amortized_effects:
        metrics["effect_params"] = all_params["effect_params"]

    # ========== Bow-Free Post-Hoc Checks (Option A + C) ==========
    # A: Check no variable pair has both strong directed AND bi-directed edges
    # C: Test residual normality (non-Gaussian → cite Wang & Drton 2023 for identifiability)
    if use_confound_matrix and metrics.get("A_confound_weights") is not None:
        A_dir_abs = jnp.abs(A_final)
        Omega_abs = jnp.abs(jnp.array(metrics["A_confound_weights"]))
        # Pairs with both directed and bi-directed edges above threshold
        bow_violations = int(jnp.sum((A_dir_abs > threshold) & (Omega_abs > threshold)))
        metrics["bow_free_violations"] = bow_violations

    # Residual normality test (Option C)
    try:
        import numpy as np
        from scipy.stats import shapiro

        # Compute residuals for first 5 variables (sample for speed)
        n_test = min(5, n_vars)
        normality_pvals = []
        for j in range(n_test):
            weights_j = jnp.abs(A_final[:, j]).at[j].set(0.0)
            X_w = data_train * weights_j[jnp.newaxis, :n_vars]
            pred_j = processor.forward(X_w[:200], final_proc_params[j])
            residuals = np.array(data_train[:200, j] - pred_j)
            if len(np.unique(residuals)) > 3:  # Need variability for Shapiro
                _, p = shapiro(residuals[: min(200, len(residuals))])
                normality_pvals.append(float(p))
        metrics["residual_normality_pvals"] = normality_pvals
        metrics["residuals_non_gaussian"] = (
            all(p < 0.05 for p in normality_pvals) if normality_pvals else False
        )
    except Exception as _res_err:
        if verbose >= 1:
            print(f"  [Residual normality test failed: {_res_err}]")

    if verbose >= 1:
        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"Directed edges: {metrics['n_edges']}, h(A)={h_A:.4f}")
        if use_confound_matrix:
            print(f"Bi-directed edges: {metrics.get('n_confound_edges', 0)}")
        print(f"MB: {markov_blanket} ({len(markov_blanket)} members)")
        print(
            f"Acc: {classification_accuracy:.4f}, BAcc: {balanced_accuracy:.4f}, "
            f"F1: {f1_score:.4f}, AUC: {auc_roc:.4f}, Effect loss: {final_effect_loss:.4f}"
        )
        if verbose >= 2 and causal_effects:
            print(f"Effects: {causal_effects}")

    return A_binary, processor, final_proc_params, metrics


# ============================================================================
# Phase 2.2: ANM Direction Tests for Edge Validation
# ============================================================================


def anm_direction_test(
    X_cause: jnp.ndarray, X_effect: jnp.ndarray, sigma: float = 1.0
) -> Tuple[float, float]:
    """
    Additive Noise Model (ANM) direction test using HSIC.

    ANM assumption: Y = f(X) + noise, where noise ⊥ X
    If X→Y is correct direction:
        - Residual r = Y - f(X) should be independent of X
        - HSIC(r, X) should be low

    Args:
        X_cause: (n_samples,) potential cause variable
        X_effect: (n_samples,) potential effect variable
        sigma: RBF kernel bandwidth

    Returns:
        hsic_forward: HSIC for X→Y direction (lower = more likely correct)
        hsic_backward: HSIC for Y→X direction
    """
    n = X_cause.shape[0]

    # Reshape for regression
    X_c = X_cause.reshape(-1, 1) if X_cause.ndim == 1 else X_cause
    X_e = X_effect.reshape(-1, 1) if X_effect.ndim == 1 else X_effect

    # Forward direction: X → Y
    # Regress Y on X, compute residual
    X_with_intercept = jnp.column_stack([jnp.ones(n), X_c])
    beta_forward = jnp.linalg.lstsq(X_with_intercept, X_effect, rcond=None)[0]
    residual_forward = X_effect - X_with_intercept @ beta_forward

    # HSIC(residual, X) - lower means X→Y is more likely
    hsic_forward = _compute_hsic(residual_forward, X_cause, sigma)

    # Backward direction: Y → X
    X_with_intercept_rev = jnp.column_stack([jnp.ones(n), X_e])
    beta_backward = jnp.linalg.lstsq(X_with_intercept_rev, X_cause, rcond=None)[0]
    residual_backward = X_cause - X_with_intercept_rev @ beta_backward

    # HSIC(residual, Y) - lower means Y→X is more likely
    hsic_backward = _compute_hsic(residual_backward, X_effect, sigma)

    return float(hsic_forward), float(hsic_backward)


def _compute_hsic(X: jnp.ndarray, Y: jnp.ndarray, sigma: float = 1.0) -> float:
    """Compute HSIC between X and Y using RBF kernel."""
    n = X.shape[0]

    # Reshape to 2D
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    # RBF kernel
    def rbf_kernel(A, B):
        A_sq = jnp.sum(A**2, axis=1, keepdims=True)
        B_sq = jnp.sum(B**2, axis=1, keepdims=True)
        sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
        return jnp.exp(-sq_dists / (2 * sigma**2))

    K = rbf_kernel(X, X)
    L = rbf_kernel(Y, Y)

    # Centering
    H = jnp.eye(n) - jnp.ones((n, n)) / n

    # HSIC = trace(KHLH) / n²
    hsic = jnp.trace(K @ H @ L @ H) / (n**2)

    return float(hsic)


def validate_edge_directions(
    data: jnp.ndarray,
    A_directed: jnp.ndarray,
    Y_idx: int,
    threshold: float = 0.1,
    verbose: bool = False,
) -> Dict:
    """
    Validate learned edge directions using ANM tests.

    For each edge Xi → Xj, compare:
        - HSIC for Xi → Xj direction
        - HSIC for Xj → Xi direction

    Args:
        data: (n_samples, n_vars) data matrix (X only, not including Y)
        A_directed: (n_vars+1, n_vars+1) learned adjacency matrix
        Y_idx: Index of outcome variable
        threshold: HSIC difference threshold for confident direction
        verbose: Print results

    Returns:
        dict with:
            - 'valid_edges': List of edges where direction is confirmed
            - 'reversed_edges': List of edges where reverse direction is better
            - 'uncertain_edges': List of edges where direction is uncertain
            - 'direction_scores': Dict of (i,j) -> (hsic_ij, hsic_ji)
    """
    n_vars = data.shape[1]
    n_total = A_directed.shape[0]

    valid_edges = []
    reversed_edges = []
    uncertain_edges = []
    direction_scores = {}

    # Check all non-zero edges (excluding edges to Y)
    for i in range(n_vars):  # Source (cause)
        for j in range(n_vars):  # Target (effect)
            if i == j:
                continue
            if A_directed[i, j] < 0.01:  # No edge
                continue

            # Get data for this edge
            X_i = data[:, i]
            X_j = data[:, j]

            # ANM test
            hsic_forward, hsic_backward = anm_direction_test(X_i, X_j)
            direction_scores[(i, j)] = (hsic_forward, hsic_backward)

            # Compare directions
            diff = hsic_backward - hsic_forward  # Positive = forward is better

            if diff > threshold:
                valid_edges.append((i, j, diff))
            elif diff < -threshold:
                reversed_edges.append((i, j, diff))
            else:
                uncertain_edges.append((i, j, diff))

    # Also check edges to Y
    edges_to_Y = []
    for i in range(n_vars):
        if A_directed[i, Y_idx] > 0.01:
            # For edges to Y, we can only check forward direction (Y is outcome)
            # Use data[:, i] as cause, need Y data which isn't in 'data'
            edges_to_Y.append(i)

    result = {
        "valid_edges": valid_edges,
        "reversed_edges": reversed_edges,
        "uncertain_edges": uncertain_edges,
        "direction_scores": direction_scores,
        "edges_to_Y": edges_to_Y,
        "n_valid": len(valid_edges),
        "n_reversed": len(reversed_edges),
        "n_uncertain": len(uncertain_edges),
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("ANM Direction Validation")
        print(f"{'=' * 60}")
        print(f"Valid edges (correct direction): {len(valid_edges)}")
        print(f"Reversed edges (wrong direction): {len(reversed_edges)}")
        print(f"Uncertain edges: {len(uncertain_edges)}")
        if reversed_edges:
            print(f"WARNING: Edges to consider reversing: {[(i, j) for i, j, _ in reversed_edges]}")

    return result


# ============================================================================
# Phase 2.4: Negative Control Calibration
# ============================================================================


def compute_negative_control_penalty(
    markov_blanket: List[int],
    causal_effects: Dict[str, float],
    negative_control_idx: int,
    verbose: bool = False,
) -> Tuple[float, Dict]:
    """
    Compute penalty for negative control variable appearing in results.

    A negative control is a variable known to have NO causal effect on outcome.
    In LUCAS: Born_an_Even_Day (X6) should not affect Lung_cancer.

    If negative control appears in MB or has non-zero effect, this indicates:
        - Spurious correlations being learned
        - Model is overfitting

    Args:
        markov_blanket: List of variable indices in learned MB
        causal_effects: Dict of 'Xi->Y' -> effect magnitude
        negative_control_idx: Index of negative control variable (e.g., 6 for LUCAS)
        verbose: Print diagnostics

    Returns:
        penalty: Value to subtract from fitness (0 if NC correctly excluded)
        diagnostics: Dict with details
    """
    nc_key = f"X{negative_control_idx}->Y"

    # Check if negative control is in MB
    nc_in_mb = negative_control_idx in markov_blanket

    # Check effect magnitude
    nc_effect = abs(causal_effects.get(nc_key, 0.0))

    # Compute penalty
    # Penalty = 0.2 if in MB + effect magnitude
    penalty = 0.0
    if nc_in_mb:
        penalty += 0.2  # Fixed penalty for being in MB
    penalty += nc_effect  # Add effect magnitude as additional penalty

    diagnostics = {
        "negative_control_idx": negative_control_idx,
        "in_markov_blanket": nc_in_mb,
        "effect_magnitude": nc_effect,
        "penalty": penalty,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("Negative Control Calibration")
        print(f"{'=' * 60}")
        print(f"Negative control: X{negative_control_idx}")
        print(f"In Markov Blanket: {nc_in_mb}")
        print(f"Effect magnitude: {nc_effect:.4f}")
        print(f"Penalty: {penalty:.4f}")
        if nc_in_mb:
            print("WARNING: Negative control in MB - possible overfitting!")

    return penalty, diagnostics


def calibrate_effects_with_negative_control(
    causal_effects: Dict[str, float], negative_control_idx: int, verbose: bool = False
) -> Dict[str, float]:
    """
    Calibrate causal effects using negative control as baseline noise.

    Any effect smaller than the negative control effect is likely noise.

    Args:
        causal_effects: Dict of 'Xi->Y' -> raw effect
        negative_control_idx: Index of negative control variable
        verbose: Print details

    Returns:
        calibrated_effects: Effects with noise baseline subtracted
    """
    nc_key = f"X{negative_control_idx}->Y"
    baseline_noise = abs(causal_effects.get(nc_key, 0.0))

    calibrated = {}
    for key, effect in causal_effects.items():
        if key == nc_key:
            calibrated[key] = 0.0  # Negative control has zero true effect
        else:
            # Subtract baseline, keep sign, floor at 0
            sign = 1 if effect >= 0 else -1
            magnitude = abs(effect) - baseline_noise
            calibrated[key] = sign * max(0.0, magnitude)

    if verbose:
        print(f"\nEffect Calibration (baseline noise = {baseline_noise:.4f}):")
        for key in sorted(causal_effects.keys()):
            raw = causal_effects[key]
            cal = calibrated[key]
            print(f"  {key}: {raw:.4f} -> {cal:.4f}")

    return calibrated


# ============================================================================
# Phase 2.3: Optional PC Algorithm Warm-Start
# ============================================================================


def get_pc_warmstart(
    data: jnp.ndarray, alpha: float = 0.05, max_cond_size: int = 2, verbose: bool = False
) -> jnp.ndarray:
    """
    Run PC algorithm to get initial adjacency matrix for warm-starting GOLEM.

    PC provides a good initial structure based on conditional independence tests,
    which can help GOLEM converge faster and avoid local optima.

    Args:
        data: (n_samples, n_vars) data matrix
        alpha: Significance level for CI tests
        max_cond_size: Maximum conditioning set size (keep small for speed)
        verbose: Print progress

    Returns:
        A_init: (n_vars, n_vars) initial adjacency matrix from PC
    """
    from jax import random

    from jcce.structure_learning.pc import learn_with_pc

    if verbose:
        print(f"\n{'=' * 60}")
        print("PC Algorithm Warm-Start")
        print(f"{'=' * 60}")

    key = random.PRNGKey(0)  # Deterministic for reproducibility

    A_pc = learn_with_pc(
        data=data, key=key, alpha=alpha, max_cond_size=max_cond_size, verbose=verbose
    )

    if verbose:
        n_edges = int(jnp.sum(A_pc))
        print(f"PC warm-start: {n_edges} directed edges")

    return A_pc
