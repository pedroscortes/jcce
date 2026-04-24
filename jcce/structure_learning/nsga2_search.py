"""
NSGA-II for Unified Supervised Causal Discovery (v4.0/v7.0) - REDUCED HYPERPARAMETER SPACE.

Memory-optimized version with reduced processor hyperparameter combinations:
- Total variants: ~32 (vs ~624 in full version)
- Avoids OOM during JAX XLA compilation
- Focuses on most important hyperparameters per processor.

Pure unified approach: GOLEM structure learning + Processor joint optimization.
No two-stage, no separate algorithms - just GOLEM with different processors.

NSGA-II explores:
- Processor type (ELM, GNN, Mamba, MLP, Transformer)
- GOLEM hyperparameters (lambda_1, lambda_2, lambda_class, lr)
- Processor hyperparameters (architecture-specific)
- Effect estimation hyperparameters (hidden_dim, embed_dim, lambda_effect, etc.)

Objectives:
1. Maximize classification accuracy
2. Maximize sparsity (minimize graph density)
3. Minimize acyclicity violation h(A)
4. Minimize effect estimation loss (factual + propensity)

Pareto validation filters out solutions with h(A) > threshold (cycles).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any, Optional, List
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import gc
import time
import ctypes

from jcce.structure_learning.genome import MACliteGenome

# DML cross-fitting (O5) and actionability proxy (O6) for causal validation
from jcce.validation.dml_crossfitting import DMLCrossFitter, get_progressive_k
from jcce.counterfactual.causal_constraints import compute_actionability_proxy, get_ancestors

# Phase 3: Warm-start cache for structure initialization
from jcce.structure_learning.warm_start_cache import ImprovedWarmStartCache


# ============================================================================
# Aggressive Memory Cleanup for Long-Running Experiments
# ============================================================================

# Global counter for JAX cache clearing (only clear every N evaluations)
_EVAL_COUNTER = 0
_JAX_CACHE_CLEAR_INTERVAL = 100  # Only clear JAX caches every 100 evals


def aggressive_memory_cleanup(sleep_time: float = 0.1, clear_jax_cache: bool = False):
    """
    Memory cleanup to prevent OOM during long NSGA-II runs.

    NOTE: JAX cache clearing is now disabled by default because it forces
    expensive recompilation. Only enable when truly necessary (OOM issues).

    This function:
    1. Optionally clears JAX's compilation caches (disabled by default)
    2. Runs Python's garbage collector for all generations
    3. Calls malloc_trim on Linux to return memory to OS
    4. Optionally sleeps to allow OS memory reclamation
    """
    global _EVAL_COUNTER

    # 1. Only clear JAX caches periodically (every N evals) to avoid
    # forcing expensive recompilation on every evaluation
    _EVAL_COUNTER += 1
    if clear_jax_cache or (_EVAL_COUNTER % _JAX_CACHE_CLEAR_INTERVAL == 0):
        try:
            jax.clear_caches()
        except Exception:
            pass

    # 2. Aggressive garbage collection (all generations)
    gc.collect(0)  # Young generation
    gc.collect(1)  # Middle generation
    gc.collect(2)  # Old generation

    # 3. On Linux, call malloc_trim to return freed memory to OS
    # This is critical because Python's memory allocator doesn't always
    # return freed memory to the OS
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass  # Not on Linux or libc not available

    # 4. Small sleep to allow OS to reclaim memory (reduced from default)
    if sleep_time > 0:
        time.sleep(sleep_time)


# ============================================================================
# Soft F1 Computation for MB Optimization
# ============================================================================

def compute_soft_mb_f1(
    predicted_mb: List[int],
    true_mb: List[int],
    n_vars: int,
    mb_weights: Optional[np.ndarray] = None,
    beta: float = 10.0
) -> Tuple[float, float, float]:
    """
    Compute soft (differentiable-friendly) F1 score for Markov Blanket.

    If mb_weights is provided, uses continuous weights for soft F1.
    Otherwise, uses binary predicted_mb for hard F1.

    Args:
        predicted_mb: List of predicted MB variable indices
        true_mb: List of ground truth MB variable indices
        n_vars: Total number of variables (excluding Y)
        mb_weights: Optional continuous weights for each variable
        beta: Temperature for sigmoid (higher = sharper)

    Returns:
        (f1, precision, recall) tuple
    """
    true_set = set(true_mb)

    if mb_weights is not None:
        # Soft F1 using continuous weights
        # sigmoid to create "soft inclusion" probability
        soft_pred = 1.0 / (1.0 + np.exp(-beta * (mb_weights - 0.5)))

        # Create true mask
        true_mask = np.zeros(n_vars)
        for idx in true_mb:
            if 0 <= idx < n_vars:
                true_mask[idx] = 1.0

        # Soft TP, FP, FN
        tp = np.sum(soft_pred * true_mask)
        fp = np.sum(soft_pred * (1 - true_mask))
        fn = np.sum((1 - soft_pred) * true_mask)

        # Soft precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        # Hard F1 using binary sets
        pred_set = set(predicted_mb)

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(f1), float(precision), float(recall)


# ============================================================================
# Unified Genome (No Structure Algorithm Selection)
# ============================================================================

PROCESSOR_TYPES = ['elm', 'gnn', 'mlp', 'transformer', 'mamba']

# GOLEM hyperparameters
GOLEM_LAMBDA_1 = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]  # Sparsity (extended for high-d datasets like breast cancer d=30)
GOLEM_LAMBDA_2 = [0.001, 0.01, 0.1, 1.0]  # DAG constraint init
GOLEM_LAMBDA_CLASS = [0.1, 0.5, 1.0, 2.0, 5.0]  # Classification weight
GOLEM_LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01]

# Effect estimation hyperparameters
EFFECT_HIDDEN_DIMS = [64, 128, 256]  # Effect network hidden dimension (increased for complex datasets)
EFFECT_EMBED_DIMS = [16, 32, 64]  # Treatment embedding dimension (increased for better treatment representation)
LAMBDA_EFFECTS = [5.0, 10.0, 20.0, 50.0]
EFFECT_WARMUP_ITERS = [10, 20, 30, 50]  # Warmup before effect training
LAMBDA_CONFOUND_SPARSE = [0.01, 0.05, 0.1, 0.5]  # A_confound sparsity (v11: increased 10x for sparser bi-directed edges)
LAMBDA_BOW_V7 = [0.1, 0.3, 0.5, 1.0]
EFFECT_REFINEMENT_ITERS = [0, 30, 50, 100]  # Post-hoc effect refinement iterations

# Processor-specific configs (reuse from search_space.py)
from jcce.structure_learning.search_space import (
    MAMBA_D_MODELS, MAMBA_D_STATES, MAMBA_D_CONVS, MAMBA_EXPANDS,
    TRANSFORMER_D_MODELS, TRANSFORMER_N_HEADS, TRANSFORMER_N_LAYERS, TRANSFORMER_D_FFS,
    GNN_HIDDEN_DIMS, GNN_N_LAYERS, GNN_TYPES, GNN_AGGREGATIONS,
    ELM_HIDDEN_DIMS, ELM_N_HIDDEN_NODES, ELM_ACTIVATIONS,
    MLP_HIDDEN_DIMS, MLP_N_LAYERS, MLP_ACTIVATIONS,
)



def genome_to_unified_config(genome: MACliteGenome, use_v7: bool = False, n_vars: int = 12) -> Dict[str, Any]:
    """
    Convert genome to unified configuration (processor + GOLEM hyperparams).

    Returns:
        config dict with:
            - processor_type: str
            - processor_config: dict (architecture-specific)
            - lambda_1, lambda_2, lambda_class, lr: float
            - effect_hidden_dim, effect_embed_dim, lambda_effect, etc.
    """
    # Processor type
    processor_type = PROCESSOR_TYPES[genome.processor_type_idx]

    # GOLEM hyperparams — scale lambda_1 with dimensionality (quadratic)
    # Possible edges grow as d², so sparsity penalty must scale accordingly
    lambda_1_base = GOLEM_LAMBDA_1[genome.sl_lambda_1_idx]
    dim_scale = max(1.0, (n_vars / 12.0) ** 2)
    lambda_1 = lambda_1_base * dim_scale
    lambda_2 = GOLEM_LAMBDA_2[genome.sl_lambda_2_idx]
    lr = GOLEM_LRS[genome.sl_lr_idx]
    # Use processor_epochs_idx as lambda_class_idx (reuse genome field)
    lambda_class_idx = genome.processor_epochs_idx % len(GOLEM_LAMBDA_CLASS)
    lambda_class = GOLEM_LAMBDA_CLASS[lambda_class_idx]

    # Processor config (architecture-specific)
    if processor_type == 'elm':
        processor_config = {
            'hidden_dim': ELM_HIDDEN_DIMS[genome.elm_hidden_dim_idx],
            'n_hidden_nodes': ELM_N_HIDDEN_NODES[genome.elm_n_hidden_nodes_idx],
            'activation': ELM_ACTIVATIONS[genome.elm_activation_idx],
        }
    elif processor_type == 'gnn':
        processor_config = {
            'hidden_dim': GNN_HIDDEN_DIMS[genome.gnn_hidden_dim_idx],
            'n_layers': GNN_N_LAYERS[genome.gnn_n_layers_idx],
            'gnn_type': GNN_TYPES[genome.gnn_type_idx],
            'sage_aggregation': GNN_AGGREGATIONS[genome.gnn_aggregation_idx],
        }
    elif processor_type == 'mamba':
        processor_config = {
            'd_model': MAMBA_D_MODELS[genome.mamba_d_model_idx],
            'd_state': MAMBA_D_STATES[genome.mamba_d_state_idx],
            'd_conv': MAMBA_D_CONVS[genome.mamba_d_conv_idx],
            'expand': MAMBA_EXPANDS[genome.mamba_expand_idx],
        }
    elif processor_type == 'mlp':
        # Use Transformer indices for MLP (we don't have dedicated MLP genes)
        processor_config = {
            'hidden_dim': MLP_HIDDEN_DIMS[genome.transformer_d_model_idx % len(MLP_HIDDEN_DIMS)],
            'n_layers': MLP_N_LAYERS[genome.transformer_n_layers_idx % len(MLP_N_LAYERS)],
            'activation': MLP_ACTIVATIONS[genome.transformer_n_heads_idx % len(MLP_ACTIVATIONS)],
        }
    elif processor_type == 'transformer':
        processor_config = {
            'd_model': TRANSFORMER_D_MODELS[genome.transformer_d_model_idx],
            'n_heads': TRANSFORMER_N_HEADS[genome.transformer_n_heads_idx],
            'n_layers': TRANSFORMER_N_LAYERS[genome.transformer_n_layers_idx],
            'd_ff': TRANSFORMER_D_FFS[genome.transformer_d_ff_idx],
        }
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")

    config = {
        'processor_type': processor_type,
        'processor_config': processor_config,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'lambda_class': lambda_class,
        'lr': lr,
    }

    # Add effect estimation parameters if available
    if use_v7:
        # Extract v7 gene indices with defaults for backward compatibility
        effect_hidden_dim_idx = getattr(genome, 'effect_hidden_dim_idx', 1)  # default: 64
        effect_embed_dim_idx = getattr(genome, 'effect_embed_dim_idx', 1)  # default: 16
        lambda_effect_idx = getattr(genome, 'lambda_effect_idx', 1)  # default: 10.0
        effect_warmup_iter_idx = getattr(genome, 'effect_warmup_iter_idx', 1)  # default: 20
        lambda_confound_sparse_idx = getattr(genome, 'lambda_confound_sparse_idx', 1)  # default: 0.05
        lambda_bow_v7_idx = getattr(genome, 'lambda_bow_v7_idx', 1)  # default: 0.3
        effect_refinement_iters_idx = getattr(genome, 'effect_refinement_iters_idx', 2)  # default: 50

        config['effect_hidden_dim'] = EFFECT_HIDDEN_DIMS[effect_hidden_dim_idx % len(EFFECT_HIDDEN_DIMS)]
        config['effect_embed_dim'] = EFFECT_EMBED_DIMS[effect_embed_dim_idx % len(EFFECT_EMBED_DIMS)]
        config['lambda_effect'] = LAMBDA_EFFECTS[lambda_effect_idx % len(LAMBDA_EFFECTS)]
        config['effect_warmup_iter'] = EFFECT_WARMUP_ITERS[effect_warmup_iter_idx % len(EFFECT_WARMUP_ITERS)]
        config['lambda_confound_sparse'] = LAMBDA_CONFOUND_SPARSE[lambda_confound_sparse_idx % len(LAMBDA_CONFOUND_SPARSE)]
        config['lambda_bow_v7'] = LAMBDA_BOW_V7[lambda_bow_v7_idx % len(LAMBDA_BOW_V7)]
        config['effect_refinement_iters'] = EFFECT_REFINEMENT_ITERS[effect_refinement_iters_idx % len(EFFECT_REFINEMENT_ITERS)]

    return config


# ============================================================================
# NSGA-II Problem Definition
# ============================================================================

class UnifiedSCDProblem(Problem):
    """
    Multi-objective optimization for Unified Supervised Causal Discovery.

    Objectives (to minimize):
    1. -accuracy (negate to maximize)
    2. -sparsity (negate to maximize)
    3. h(A) (minimize acyclicity violation)
    4. effect_loss (minimize effect estimation error)
    """

    def __init__(
        self,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        n_vars: int,
        true_graph: Optional[jnp.ndarray] = None,
        true_mb: Optional[List[int]] = None,
        stage1_max_iter: int = 50,
        jax_key_seed: int = 0,
        verbose: bool = False,
        use_v7: bool = False,  # Enable effect estimation + bi-directed edges
        use_pc_warmstart: bool = False,  # Optional PC algorithm warm-start
        negative_control_idx: Optional[int] = None,  # Negative control for calibration
        # Causal validation objectives
        use_dml_objective: bool = False,  # O5: DML effect variance
        use_actionability_objective: bool = False,  # O6: Actionability proxy
        classifier_for_actionability: Optional[Any] = None,  # Pre-trained classifier for O6
        # Identifiability constraint
        use_condition_constraint: bool = True,  # kappa(MB) < condition_threshold
        condition_threshold: float = 100.0,  # Standard ill-conditioning threshold
        # Task type
        task: str = 'classification',  # 'classification' or 'regression'
        # GOLEM overrides for ablation studies
        golem_overrides: Optional[Dict[str, Any]] = None,
        # Phase 3: Warm-start cache
        enable_warm_start: bool = False,
        warm_start_prob: float = 0.3,
    ):
        """
        Args:
            X_train: Training features (n_samples, n_features)
            Y_train: Training labels/values (n_samples,)
            n_vars: Number of variables (for A_topology size)
            true_graph: Optional ground truth graph for evaluation
            true_mb: Optional ground truth Markov Blanket indices for F1 optimization (v8.0)
            stage1_max_iter: Max iterations for GOLEM optimization
            jax_key_seed: Seed for JAX random key
            verbose: Print evaluation details
            use_v7: Enable v7.0 features (effect estimation, bi-directed edges, 4 objectives)
            use_pc_warmstart: Run PC algorithm once to get initial structure (v9.1)
            negative_control_idx: Index of known non-causal variable for calibration (v9.1)
            use_dml_objective: Enable O5 (DML effect variance) objective (v13.0)
            use_actionability_objective: Enable O6 (actionability proxy) objective (v13.0)
            classifier_for_actionability: Pre-trained classifier for O6 margin computation
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_vars = n_vars
        self.true_graph = true_graph
        self.true_mb = true_mb
        self.stage1_max_iter = stage1_max_iter
        self.jax_key = random.PRNGKey(jax_key_seed)
        self.verbose = verbose
        self.use_v7 = use_v7
        self.use_pc_warmstart = use_pc_warmstart
        self.negative_control_idx = negative_control_idx
        self.pc_init = None  # Cached PC result

        # Causal validation objectives
        self.use_dml_objective = use_dml_objective
        self.use_actionability_objective = use_actionability_objective
        self.classifier_for_actionability = classifier_for_actionability
        self.dml_random_state = jax_key_seed

        # Identifiability constraint
        self.use_condition_constraint = use_condition_constraint
        self.condition_threshold = condition_threshold

        self.task = task
        self.golem_overrides = golem_overrides or {}

        # Phase 3: Warm-start cache
        self.enable_warm_start = enable_warm_start
        self.warm_start_prob = warm_start_prob
        self.warm_start_cache = ImprovedWarmStartCache(max_size=50) if enable_warm_start else None

        # Inter-processor migration (Island Model)
        # Track best structure per processor type for cross-pollination
        self.structure_pool = {}  # processor_type -> {'A': best_A, 'fitness': score, 'gen': generation}
        self.current_generation = 0
        self.migration_interval = 2  # Migrate every N generations
        self.migration_enabled = True  # Can be disabled for ablation

        # Store full evaluation metrics for enhanced results
        # Maps genome hash -> full metrics dict (including A_est, A_confound, etc.)
        self.evaluation_cache = {}  # genome_hash -> metrics dict

        # Genome encoding (simplified - no A_topology for now)
        # Base: 20 hyperparameter indices
        # +6 effect estimation indices = 26 total
        # [processor_type_idx, lambda_1_idx, lambda_2_idx, lr_idx, lambda_class_idx,
        #  mamba_d_model, mamba_d_state, mamba_d_conv, mamba_expand,
        #  transformer_d_model, transformer_n_heads, transformer_n_layers, transformer_d_ff,
        #  gnn_hidden_dim, gnn_n_layers, gnn_type, gnn_aggregation,
        #  elm_hidden_dim, elm_n_hidden_nodes, elm_activation,
        #  (v7) effect_hidden_dim, effect_embed_dim, lambda_effect, effect_warmup_iter,
        #  (v7) lambda_confound_sparse, lambda_bow_v7]

        n_vars_genome = 27 if use_v7 else 20

        # Bounds (all integer indices)
        xl = np.zeros(n_vars_genome, dtype=int)
        xu_base = [
            len(PROCESSOR_TYPES) - 1,      # 0: processor_type_idx
            len(GOLEM_LAMBDA_1) - 1,       # 1: lambda_1_idx
            len(GOLEM_LAMBDA_2) - 1,       # 2: lambda_2_idx
            len(GOLEM_LRS) - 1,            # 3: lr_idx
            len(GOLEM_LAMBDA_CLASS) - 1,   # 4: lambda_class_idx
            len(MAMBA_D_MODELS) - 1,       # 5
            len(MAMBA_D_STATES) - 1,       # 6
            len(MAMBA_D_CONVS) - 1,        # 7
            len(MAMBA_EXPANDS) - 1,        # 8
            len(TRANSFORMER_D_MODELS) - 1, # 9
            len(TRANSFORMER_N_HEADS) - 1,  # 10
            len(TRANSFORMER_N_LAYERS) - 1, # 11
            len(TRANSFORMER_D_FFS) - 1,    # 12
            len(GNN_HIDDEN_DIMS) - 1,      # 13
            len(GNN_N_LAYERS) - 1,         # 14
            len(GNN_TYPES) - 1,            # 15
            len(GNN_AGGREGATIONS) - 1,     # 16
            len(ELM_HIDDEN_DIMS) - 1,      # 17
            len(ELM_N_HIDDEN_NODES) - 1,   # 18
            len(ELM_ACTIVATIONS) - 1,      # 19
        ]

        if use_v7:
            xu_base.extend([
                len(EFFECT_HIDDEN_DIMS) - 1,    # 20: effect_hidden_dim_idx
                len(EFFECT_EMBED_DIMS) - 1,     # 21: effect_embed_dim_idx
                len(LAMBDA_EFFECTS) - 1,        # 22: lambda_effect_idx
                len(EFFECT_WARMUP_ITERS) - 1,   # 23: effect_warmup_iter_idx
                len(LAMBDA_CONFOUND_SPARSE) - 1, # 24: lambda_confound_sparse_idx
                len(LAMBDA_BOW_V7) - 1,         # 25: lambda_bow_v7_idx
                len(EFFECT_REFINEMENT_ITERS) - 1, # 26: effect_refinement_iters_idx (v7.2)
            ])

        xu = np.array(xu_base, dtype=int)

        # 2 objectives: balanced accuracy and sparsity (both maximized, negated for minimization).
        # h(A) is a constraint (validity condition), not an objective.
        n_obj = 2

        # Constraints:
        # C1: h(A) < 0.1 (acyclicity)
        # C2: kappa(MB) < threshold (condition number)
        n_constraints = 1  # h(A) constraint always active
        if use_condition_constraint:
            n_constraints += 1

        super().__init__(
            n_var=n_vars_genome,
            n_obj=n_obj,
            n_constr=n_constraints,
            xl=xl,
            xu=xu,
            type_var=int,
        )

    def _get_migrated_structure(self, processor_type: str) -> Optional[jnp.ndarray]:
        """
        v11: Get best structure from another processor type for cross-pollination.

        Migration strategy:
        - Fast processors (ELM, MLP) scout first
        - Their best structures warm-start slow processors (Transformer, Mamba)

        Returns A_init matrix or None if no suitable donor found.
        """
        if not self.migration_enabled or not self.structure_pool:
            return None

        # Only migrate every N generations
        if self.current_generation % self.migration_interval != 0:
            return None

        # Define processor speed tiers (fast -> slow)
        speed_order = ['elm', 'mlp', 'gnn', 'transformer', 'mamba']

        # Find best donor (faster processor with good structure)
        best_donor = None
        best_fitness = -float('inf')

        for donor_type, info in self.structure_pool.items():
            if donor_type == processor_type:
                continue  # Don't self-migrate

            # Prefer faster processors as donors
            try:
                donor_speed = speed_order.index(donor_type)
                current_speed = speed_order.index(processor_type)
            except ValueError:
                continue

            # Only migrate from faster to slower processors
            if donor_speed < current_speed and info['fitness'] > best_fitness:
                best_fitness = info['fitness']
                best_donor = info

        if best_donor is not None:
            return best_donor['A']
        return None

    def _update_structure_pool(self, processor_type: str, A_est: jnp.ndarray,
                               fitness: float, generation: int):
        """
        v11: Update structure pool with best structure per processor type.
        """
        if not self.migration_enabled:
            return

        if processor_type not in self.structure_pool:
            self.structure_pool[processor_type] = {
                'A': A_est, 'fitness': fitness, 'gen': generation
            }
        elif fitness > self.structure_pool[processor_type]['fitness']:
            self.structure_pool[processor_type] = {
                'A': A_est, 'fitness': fitness, 'gen': generation
            }

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate population of genomes.

        Args:
            X: (pop_size, n_vars_genome) decision variables
            out: Dictionary to store objectives

        v8.0: Added MB F1 as optional 5th objective when true_mb is provided.
        """
        pop_size = X.shape[0]

        n_obj = 2  # balanced accuracy + sparsity
        F = np.zeros((pop_size, n_obj))

        # Run PC algorithm once for warm-start (cached)
        if self.use_pc_warmstart and self.pc_init is None:
            from jcce.structure_learning.jcce_learner import get_pc_warmstart
            if self.verbose:
                print("Running PC algorithm for warm-start...")
            pc_result = get_pc_warmstart(
                self.X_train,
                alpha=0.05,
                max_cond_size=2,
                verbose=self.verbose
            )
            # PC returns n_vars x n_vars, but v7 needs (n_vars+1) x (n_vars+1)
            # Expand by adding zero row/column for Y (outcome sink constraint)
            n_vars = pc_result.shape[0]
            n_total = n_vars + 1
            self.pc_init = jnp.zeros((n_total, n_total))
            self.pc_init = self.pc_init.at[:n_vars, :n_vars].set(pc_result)
            # Y has no outgoing edges (outcome sink)
            if self.verbose:
                n_pc_edges = int(jnp.sum(pc_result))
                print(f"PC warm-start complete: {n_pc_edges} edges (expanded to {n_total}x{n_total})\n")

        for i in range(pop_size):
            aggressive_memory_cleanup(sleep_time=0)

            try:
                # Decode genome
                decision_vars = X[i, :]
                genome = self._decision_to_genome(decision_vars)

                config = genome_to_unified_config(genome, use_v7=self.use_v7, n_vars=self.n_vars)
                processor_type = config['processor_type']

                # Phase 3: Warm-start priority: (1) cache.get → (2) cache.sample → (3) migration → (4) PC → (5) random
                A_init_to_use = None
                if self.warm_start_cache is not None and np.random.rand() < self.warm_start_prob:
                    A_init_to_use = self.warm_start_cache.get(config)
                    if A_init_to_use is None:
                        A_init_to_use = self.warm_start_cache.sample()
                    if A_init_to_use is not None:
                        A_init_to_use = jnp.array(A_init_to_use)

                # Fall back to inter-processor migration, then PC warm-start
                if A_init_to_use is None:
                    A_init_to_use = self._get_migrated_structure(processor_type)
                if A_init_to_use is None:
                    A_init_to_use = self.pc_init  # PC warm-start fallback

                # Evaluate using unified fitness function
                metrics = evaluate_genome_unified(
                    genome,
                    self.X_train,
                    self.Y_train,
                    self.n_vars,
                    true_graph=self.true_graph,
                    max_iter=self.stage1_max_iter,
                    key=self.jax_key,
                    verbose=self.verbose,
                    use_v7=self.use_v7,
                    A_init=A_init_to_use,
                    task=self.task,
                    golem_overrides=self.golem_overrides,
                )

                # Extract objectives
                accuracy = metrics['classification_accuracy']
                sparsity = metrics['mb_sparsity']
                h_A = metrics.get('structure_h_A', 0.0)

                # 2 objectives: balanced accuracy + sparsity (h(A) is a constraint)
                balanced_acc = metrics.get('classification_balanced_accuracy',
                                           metrics.get('balanced_accuracy', accuracy))
                F[i, 0] = -balanced_acc  # Negate to maximize
                F[i, 1] = -sparsity     # Negate to maximize

                # Compute MB F1 for reporting (not an objective)
                if self.true_mb is not None:
                    predicted_mb = metrics.get('markov_blanket', [])
                    mb_f1, mb_prec, mb_rec = compute_soft_mb_f1(
                        predicted_mb, self.true_mb, self.n_vars
                    )
                    metrics['mb_f1'] = mb_f1
                    metrics['mb_precision'] = mb_prec
                    metrics['mb_recall'] = mb_rec

                # Negative control penalty (applied to accuracy objective)
                if self.negative_control_idx is not None:
                    from jcce.structure_learning.jcce_learner import compute_negative_control_penalty
                    predicted_mb = metrics.get('markov_blanket', [])
                    causal_effects = metrics.get('causal_effects', {})
                    nc_penalty, nc_diag = compute_negative_control_penalty(
                        predicted_mb, causal_effects, self.negative_control_idx, verbose=False
                    )
                    # Apply penalty to accuracy objective (reduce fitness if NC detected)
                    F[i, 0] += nc_penalty  # Makes accuracy worse (more negative = better, so add penalty)
                    metrics['nc_penalty'] = nc_penalty
                    metrics['nc_in_mb'] = nc_diag['in_markov_blanket']

                # Update structure pool for inter-processor migration
                if 'structure_A_est' in metrics:
                    # Compute fitness for migration (weighted combination)
                    migration_fitness = 0.5 * accuracy + 0.3 * sparsity - 0.2 * min(h_A, 1.0)
                    self._update_structure_pool(
                        processor_type, jnp.array(metrics['structure_A_est']),
                        migration_fitness, self.current_generation
                    )

                    # Phase 3: Populate warm-start cache
                    if self.warm_start_cache is not None:
                        self.warm_start_cache.add(
                            config, metrics['structure_A_est'],
                            accuracy, self.current_generation, processor_type
                        )

                # Store full metrics for enhanced results (genome tuple as key)
                genome_key = tuple(X[i, :].tolist())
                self.evaluation_cache[genome_key] = {
                    **metrics,
                    'genome': X[i, :].copy(),
                }

                if self.verbose:
                    base_msg = (f"  Eval {i+1}/{pop_size}: {processor_type:12} "
                                f"bacc={balanced_acc:.3f} spar={sparsity:.3f} h(A)={h_A:.4f}")

                    if self.use_v7:
                        effect_loss = metrics.get('effect_loss', 1.0)
                        base_msg += f" eff={effect_loss:.4f}"

                    if self.true_mb is not None:
                        mb_f1 = metrics.get('mb_f1', 0.0)
                        base_msg += f" F1={mb_f1:.3f}"

                    print(base_msg)

                # Cleanup after evaluation (Mamba's jax.lax.scan accumulates memory)
                aggressive_memory_cleanup(sleep_time=0.01)

            except Exception as e:
                print(f"  Evaluation {i+1}/{pop_size} failed: {e}")
                F[i, 0] = 0.0    # Worst balanced accuracy (negated, so 0 = worst)
                F[i, 1] = 0.0    # Worst sparsity (negated, so 0 = worst)
                aggressive_memory_cleanup(sleep_time=0.05)

        self.current_generation += 1

        # Phase 3: Advance warm-start cache generation
        if self.warm_start_cache is not None:
            self.warm_start_cache.next_generation()

        # Log migration status periodically
        if self.verbose and self.migration_enabled and len(self.structure_pool) > 0:
            if self.current_generation % self.migration_interval == 0:
                pool_summary = ", ".join([f"{p}:{info['fitness']:.3f}" for p, info in self.structure_pool.items()])
                print(f"  [Migration pool: {pool_summary}]")

        # Cleanup at end of each generation
        aggressive_memory_cleanup(sleep_time=0.05)

        out["F"] = F

        # Constraints
        # C1: h(A) < 0.1 (acyclicity — reject cyclic solutions)
        # C2: kappa(MB) < threshold (condition number, v15)
        n_constr = 1 + (1 if self.use_condition_constraint else 0)
        G = np.zeros((pop_size, n_constr))

        for i in range(pop_size):
            genome_key = tuple(X[i, :].tolist())
            cached = self.evaluation_cache.get(genome_key, {})

            # C1: h(A) constraint — feasible when h(A) < 0.1
            h_A = cached.get('structure_h_A', 1.0)
            G[i, 0] = h_A - 0.1  # Feasible when <= 0

            # C2: Condition number constraint (optional)
            if self.use_condition_constraint:
                mb_indices = cached.get('markov_blanket', [])
                if len(mb_indices) > 1:
                    from jcce.validation.identifiability_diagnostics import compute_mb_condition_number
                    X_np = np.array(self.X_train)
                    kappa = compute_mb_condition_number(X_np, mb_indices)
                    G[i, 1] = kappa - self.condition_threshold
                else:
                    G[i, 1] = 0.0

        out["G"] = G

    def _decision_to_genome(self, decision_vars: np.ndarray) -> MACliteGenome:
        """Convert decision variables to MACliteGenome."""
        # Initialize genome with dummy A_topology (not used in unified approach)
        A_topology = jnp.zeros((self.n_vars, self.n_vars))

        genome = MACliteGenome(
            A_topology=A_topology,
            sl_algorithm_idx=0,  # Always GOLEM (not used)
            sl_lambda_1_idx=int(decision_vars[1]),
            sl_lambda_2_idx=int(decision_vars[2]),
            sl_lr_idx=int(decision_vars[3]),
            processor_type_idx=int(decision_vars[0]),
            mamba_d_model_idx=int(decision_vars[5]),
            mamba_d_state_idx=int(decision_vars[6]),
            mamba_d_conv_idx=int(decision_vars[7]),
            mamba_expand_idx=int(decision_vars[8]),
            transformer_d_model_idx=int(decision_vars[9]),
            transformer_n_heads_idx=int(decision_vars[10]),
            transformer_n_layers_idx=int(decision_vars[11]),
            transformer_d_ff_idx=int(decision_vars[12]),
            lstm_hidden_size_idx=0,
            lstm_n_layers_idx=0,
            lstm_bidirectional_idx=0,
            lstm_dropout_idx=0,
            gru_hidden_size_idx=0,
            gru_n_layers_idx=0,
            gru_bidirectional_idx=0,
            gru_dropout_idx=0,
            gnn_hidden_dim_idx=int(decision_vars[13]),
            gnn_n_layers_idx=int(decision_vars[14]),
            gnn_type_idx=int(decision_vars[15]),
            gnn_aggregation_idx=int(decision_vars[16]),
            elm_hidden_dim_idx=int(decision_vars[17]),
            elm_n_hidden_nodes_idx=int(decision_vars[18]),
            elm_activation_idx=int(decision_vars[19]),
            processor_lr_idx=0,
            processor_epochs_idx=int(decision_vars[4]),  # Reused for lambda_class
            processor_batch_size_idx=0,
        )

        # Add effect estimation genes dynamically (bypass frozen dataclass)
        if self.use_v7 and len(decision_vars) >= 27:
            object.__setattr__(genome, 'effect_hidden_dim_idx', int(decision_vars[20]))
            object.__setattr__(genome, 'effect_embed_dim_idx', int(decision_vars[21]))
            object.__setattr__(genome, 'lambda_effect_idx', int(decision_vars[22]))
            object.__setattr__(genome, 'effect_warmup_iter_idx', int(decision_vars[23]))
            object.__setattr__(genome, 'lambda_confound_sparse_idx', int(decision_vars[24]))
            object.__setattr__(genome, 'lambda_bow_v7_idx', int(decision_vars[25]))
            object.__setattr__(genome, 'effect_refinement_iters_idx', int(decision_vars[26]))

        return genome


def evaluate_genome_unified(
    genome: MACliteGenome,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    n_vars: int,
    true_graph: Optional[jnp.ndarray] = None,
    max_iter: int = 50,
    key: random.PRNGKey = None,
    verbose: bool = False,
    A_init: Optional[jnp.ndarray] = None,
    # Optimization parameters
    use_spectral_constraint: bool = False,
    enable_pruning: bool = False,
    use_v7: bool = False,
    Y_continuous: Optional[jnp.ndarray] = None,  # Continuous Y for effect estimation
    task: str = 'classification',
    golem_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Evaluate genome using GOLEM unified v4/v7 approach.

    Always uses golem_v4_joint (or v7 if enabled) with specified processor.
    """
    if golem_overrides is None:
        golem_overrides = {}
    if key is None:
        key = random.PRNGKey(0)

    # Get unified config (lambda_1 scales quadratically with n_vars)
    config = genome_to_unified_config(genome, use_v7=use_v7, n_vars=n_vars)

    # Import here to avoid circular dependency
    from jcce.structure_learning.jcce_learner import (
        create_processor,
        _learn_structure_legacy,
        extract_markov_blanket,
    )

    # Prepare Y
    if isinstance(Y, tuple):
        Y_continuous_sl, Y_binary_clf = Y
        Y_for_v4 = Y_binary_clf
    else:
        Y_for_v4 = Y.squeeze() if Y.ndim == 2 else Y

    # Augment X with Y for structure learning
    Y_reshaped = Y_for_v4.reshape(-1, 1).astype(jnp.float32)
    X_augmented = jnp.concatenate([X, Y_reshaped], axis=1)
    Y_idx = n_vars  # Y is last variable

    # Create processor adapter
    # Processors always take X features as input (n_vars), not X+Y
    # The adjacency matrix is (n_vars+1, n_vars+1) but processor input is just X
    n_features_for_processor = n_vars  # X features only
    key, proc_key = random.split(key)
    processor = create_processor(
        config['processor_type'],
        key=proc_key,
        n_features=n_features_for_processor,
        **config['processor_config']
    )

    # Use v7 function with effect estimation if enabled
    if use_v7:
        from jcce.structure_learning.jcce_learner import learn_structure

        key, v7_key = random.split(key)
        # Expects X only (not augmented with Y) — Y handled separately
        # Y should be (n_samples, 1) for v7
        Y_for_v7 = Y_for_v4.reshape(-1, 1) if Y_for_v4.ndim == 1 else Y_for_v4
        A_est_augmented, processor_trained, processor_params, v7_metrics = \
            learn_structure(
                data=X,  # Pass X only, not X_augmented
                Y=Y_for_v7,
                Y_idx=Y_idx,  # Still n_vars (index of Y in full variable space)
                processor=processor,
                key=v7_key,
                processor_type=config['processor_type'],
                lambda_1=config['lambda_1'],
                lambda_2_init=config['lambda_2'],
                lambda_class=config['lambda_class'],
                lr=config['lr'],
                max_iter=max_iter,
                patience=25,
                verbose=verbose,
                A_init=A_init,
                # Effect parameters
                effect_hidden_dim=config.get('effect_hidden_dim', 64),
                effect_embed_dim=config.get('effect_embed_dim', 16),
                lambda_effect=golem_overrides.get('lambda_effect', config.get('lambda_effect', 10.0)),
                effect_warmup_iter=config.get('effect_warmup_iter', 20),
                lambda_confound_sparse=config.get('lambda_confound_sparse', 0.05),
                lambda_bow=golem_overrides.get('lambda_bow', config.get('lambda_bow_v7', 0.3)),
                Y_continuous=Y_continuous,
                effect_refinement_iters=config.get('effect_refinement_iters', 50),
                n_latent_confounders=config.get('n_latent_confounders', 5),
                task=task,
                # Ablation overrides
                use_adaptive_curriculum=golem_overrides.get('use_adaptive_curriculum', True),
                curriculum_phase_splits=golem_overrides.get('curriculum_phase_splits', (0.4, 0.8)),
                lambda_ident=golem_overrides.get('lambda_ident', 0.01),
                use_amortized_effects=golem_overrides.get('use_amortized_effects', True),
                use_dragonnet=golem_overrides.get('use_dragonnet', True),
                use_structural_dml=golem_overrides.get('use_structural_dml', False),
                use_pcgrad=golem_overrides.get('use_pcgrad', False),
                enforce_outcome_sink=golem_overrides.get('enforce_outcome_sink', True),
                freeze_A=golem_overrides.get('freeze_A', False),
                proc_params_init=None,  # NSGA2 doesn't use SPR transfer; consistent with Optuna interface
            )
        v4_metrics = v7_metrics  # Use same variable name for downstream code
    else:
        # Run v4 unified optimization (legacy path)
        key, v4_key = random.split(key)
        A_est_augmented, processor_trained, processor_params, v4_metrics = \
            _learn_structure_legacy(
                data=X_augmented,
                Y=Y_for_v4,
                Y_idx=Y_idx,
                processor=processor,
                key=v4_key,
                processor_type=config['processor_type'],
                lambda_1=config['lambda_1'],
                lambda_2_init=config['lambda_2'],
                lambda_class=config['lambda_class'],
                lr=config['lr'],
                max_iter=max_iter,
                patience=15,
                verbose=verbose,
                A_init=A_init,
                use_spectral_constraint=use_spectral_constraint,
                enable_pruning=enable_pruning,
                task=task,
                lambda_bow=golem_overrides.get('lambda_bow', config.get('lambda_bow', 0.1)),
            )

    # Extract Markov Blanket - use the one from v7/v4 metrics which includes confound neighbors
    # Bug fix: Previously used extract_markov_blanket() which excluded confound_neighbors,
    # causing F1 to be computed on a different (smaller) MB than what's printed/intended
    mb_from_metrics = v4_metrics.get('markov_blanket', [])
    if mb_from_metrics:
        # Use the MB from the training function (includes parents, children, spouses, confound_neighbors)
        mb_indices = jnp.array([idx for idx in mb_from_metrics if idx != Y_idx])
    else:
        # Fallback to extract_markov_blanket if not available (shouldn't happen with v7)
        mb_indices_aug = extract_markov_blanket(A_est_augmented, Y_idx, threshold=1e-6)
        mb_indices = jnp.array([idx for idx in mb_indices_aug if idx != Y_idx])

    # Compute sparsity using continuous edge weights (not discrete MB count).
    # This matches Optuna's metric and prevents gaming where the optimizer
    # pushes all edge weights just below the MB threshold to achieve sparsity=1.0.
    n_features = X.shape[1]  # Actual number of features in X (not including Y)
    mb_size = len(mb_indices)
    A_to_Y = np.abs(np.array(A_est_augmented[:n_features, Y_idx]))
    edge_weight_to_Y = float(np.sum(A_to_Y))
    mb_sparsity = 1.0 - min(edge_weight_to_Y / n_features, 1.0)

    # Use metrics directly from GOLEM (training-data metrics as NSGA-II fitness proxy;
    # proper held-out evaluation happens in unified_cv_evaluation).
    classification_accuracy = float(v4_metrics.get('classification_accuracy', 0.0))
    classification_precision = float(v4_metrics.get('precision', 0.0))
    classification_recall = float(v4_metrics.get('recall', 0.0))
    classification_f1 = float(v4_metrics.get('f1_score', 0.0))
    classification_balanced_acc = float(v4_metrics.get('balanced_accuracy', 0.0))
    classification_roc_auc = float(v4_metrics.get('auc_roc', 0.0))

    result = {
        'fitness': 0.7 * classification_accuracy + 0.3 * mb_sparsity,
        'classification_accuracy': classification_accuracy,
        'classification_precision': classification_precision,
        'classification_recall': classification_recall,
        'classification_f1': classification_f1,
        'classification_balanced_accuracy': classification_balanced_acc,
        'classification_roc_auc': classification_roc_auc,
        'mb_sparsity': mb_sparsity,
        'mb_size': mb_size,
        'mb_indices': list(np.array(mb_indices)),
        'markov_blanket': list(np.array(mb_indices)),
        'structure_n_edges': v4_metrics.get('n_edges', 0),
        'structure_h_A': v4_metrics.get('final_h_A', 0.0),
        'structure_A_est': np.array(A_est_augmented),  # For warm-start caching!
        'v4_recon_loss': v4_metrics.get('final_recon_loss', 0.0),
        'v4_class_loss': v4_metrics.get('final_class_loss', 0.0),
        'v4_iterations': v4_metrics.get('iterations', max_iter),
        'v4_early_stopped': v4_metrics.get('early_stopped', False),
        # Store processor identity + GOLEM hyperparams for reproducibility
        'processor_type': config['processor_type'],
        'processor_config': config['processor_config'],
        'lambda_1': config['lambda_1'],
        'lambda_2': config['lambda_2'],
        'lambda_class': config['lambda_class'],
        'lr': config['lr'],
    }

    # Store trained processor + params for post-hoc counterfactual evaluation
    # These are runtime-only (JAX objects, not pickle-safe) — underscore prefix signals this
    result['_processor'] = processor_trained
    result['_processor_params'] = processor_params

    # Add effect estimation metrics
    if use_v7:
        result['effect_loss'] = v4_metrics.get('effect_loss', 1.0)
        result['causal_effects'] = v4_metrics.get('causal_effects', {})
        result['n_confound_edges'] = v4_metrics.get('n_confound_edges', 0)
        result['bow_loss'] = v4_metrics.get('bow_loss', 0.0)
        if 'A_confound' in v4_metrics:
            result['A_confound'] = np.array(v4_metrics['A_confound'])
        # Store weighted matrices for DAG visualization
        if 'A_weights' in v4_metrics:
            result['A_weights'] = np.array(v4_metrics['A_weights'])
        # Training-time diagnostics (gradient cosines, bow-free violations, residual normality)
        if 'gradient_diagnostics' in v4_metrics:
            result['gradient_diagnostics'] = v4_metrics['gradient_diagnostics']
        if 'bow_free_violations' in v4_metrics:
            result['bow_free_violations'] = v4_metrics['bow_free_violations']
        if 'residual_normality_pvals' in v4_metrics:
            result['residual_normality_pvals'] = v4_metrics['residual_normality_pvals']
        if 'A_confound_weights' in v4_metrics:
            result['A_confound_weights'] = np.array(v4_metrics['A_confound_weights'])
        # Store v7 effect config so CV evaluation detects v7 correctly
        # Without these keys, evaluate_pareto_solution_cv falls back to v4,
        # causing pos_embed shape mismatch for Transformer processors
        result['effect_hidden_dim'] = config.get('effect_hidden_dim', 64)
        result['lambda_effect'] = golem_overrides.get('lambda_effect', config.get('lambda_effect', 10.0))
        result['effect_embed_dim'] = config.get('effect_embed_dim', 16)
        result['effect_warmup_iter'] = config.get('effect_warmup_iter', 20)
        result['lambda_confound_sparse'] = config.get('lambda_confound_sparse', 0.05)
        result['lambda_bow'] = golem_overrides.get('lambda_bow', config.get('lambda_bow_v7', 0.3))

    # Add structure metrics if ground truth available
    if true_graph is not None:
        from jcce.training.metrics import evaluate_structure_recovery
        # Compare learned structure (X part only) with ground truth
        A_est_X = np.array(A_est_augmented[:n_vars, :n_vars])
        true_graph_np = np.array(true_graph)

        # Ensure same shape
        if A_est_X.shape == true_graph_np.shape:
            structure_metrics = evaluate_structure_recovery(A_est_X, true_graph_np, compute_sid=True)
            result['structure_edge_f1'] = structure_metrics['f1']
            result['structure_edge_precision'] = structure_metrics['precision']
            result['structure_edge_recall'] = structure_metrics['recall']
            result['structure_shd'] = structure_metrics['shd']
            result['structure_sid'] = structure_metrics.get('sid', 0)
            result['structure_sid_normalized'] = structure_metrics.get('sid_normalized', 0.0)
            result['structure_tp'] = structure_metrics['tp']
            result['structure_fp'] = structure_metrics['fp']
            result['structure_fn'] = structure_metrics['fn']

    # Add MB components if available
    if 'mb_components' in v4_metrics:
        result['mb_components'] = v4_metrics['mb_components']

    return result


# ============================================================================
# Run NSGA-II
# ============================================================================

def run_nsga2(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    n_vars: int,
    true_graph: Optional[jnp.ndarray] = None,
    true_mb: Optional[List[int]] = None,
    pop_size: int = 20,
    n_generations: int = 15,
    stage1_max_iter: int = 50,
    jax_key_seed: int = 0,
    verbose: bool = True,
    use_v7: bool = False,
    use_pc_warmstart: bool = False,
    negative_control_idx: Optional[int] = None,
    # Causal validation objectives
    use_dml_objective: bool = False,  # O5: DML effect variance
    use_actionability_objective: bool = False,  # O6: Actionability proxy
    classifier_for_actionability: Optional[Any] = None,
    # Identifiability constraint
    use_condition_constraint: bool = True,  # kappa(MB) < condition_threshold
    condition_threshold: float = 100.0,
    task: str = 'classification',  # 'classification' or 'regression'
    golem_overrides: Optional[Dict[str, Any]] = None,
    # Phase 3: Warm-start cache
    enable_warm_start: bool = False,
    warm_start_prob: float = 0.3,
) -> Dict[str, Any]:
    """
    Run NSGA-II for Unified Supervised Causal Discovery.

    2 objectives (balanced accuracy, sparsity) with h(A) as constraint.
    Supports both classification and regression tasks.

    Returns:
        result dict with:
            - pareto_front: List of (balanced_acc, sparsity, h_A, effect_loss, mb_f1) tuples
            - enhanced_solutions: List of solution dicts with full metrics
            - evaluation_cache: All evaluations (genome_key → metrics)
    """
    n_obj = 2
    obj_names = "R², sparsity" if task == 'regression' else "balanced_acc, sparsity"
    constraint_names = "h(A)<0.1"
    if use_condition_constraint:
        constraint_names += f", kappa<{condition_threshold}"

    if verbose:
        print("="*80)
        print(f"NSGA-II for Unified Supervised Causal Discovery")
        print("="*80)
        print(f"Exploring: {len(PROCESSOR_TYPES)} processors × GOLEM hyperparams")
        print(f"Processors: {', '.join(PROCESSOR_TYPES)}")
        print(f"Population: {pop_size}, Generations: {n_generations}")
        print(f"Task: {task}")
        print(f"Objectives: {n_obj} ({obj_names})")
        print(f"Constraints: {constraint_names}")
        if true_mb is not None:
            print(f"True MB: {true_mb} ({len(true_mb)} features)")
        print(f"Data: X{X.shape}, Y{Y.shape}")
        print("="*80 + "\n")

    # Create problem
    problem = UnifiedSCDProblem(
        X_train=X,
        Y_train=Y,
        n_vars=n_vars,
        true_graph=true_graph,
        true_mb=true_mb,
        stage1_max_iter=stage1_max_iter,
        jax_key_seed=jax_key_seed,
        verbose=verbose,
        use_v7=use_v7,
        use_pc_warmstart=use_pc_warmstart,
        negative_control_idx=negative_control_idx,
        use_dml_objective=use_dml_objective,
        use_actionability_objective=use_actionability_objective,
        classifier_for_actionability=classifier_for_actionability,
        use_condition_constraint=use_condition_constraint,
        condition_threshold=condition_threshold,
        task=task,
        golem_overrides=golem_overrides,
        # Phase 3: Warm-start cache
        enable_warm_start=enable_warm_start,
        warm_start_prob=warm_start_prob,
    )

    # Configure NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=None),
        mutation=PM(prob=1.0/problem.n_var, eta=20, vtype=float, repair=None),
        eliminate_duplicates=True,
    )

    # Run optimization
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_generations),
        seed=jax_key_seed,
        verbose=verbose,
        save_history=False,
    )

    # Extract Pareto front (2 objectives: balanced_acc, sparsity).
    # If pymoo returns result.F=None (all infeasible), reconstruct from cache.
    pareto_front = []
    if result.F is not None and result.X is not None:
        for i, objectives in enumerate(result.F):
            balanced_acc = -objectives[0]  # Convert back (was negated for minimization)
            spar = -objectives[1]

            # Also retrieve h(A) from cache for reporting
            genome_key = tuple(result.X[i, :].tolist())
            cached = problem.evaluation_cache.get(genome_key, {})
            h_A = cached.get('structure_h_A', 0.0)
            effect_loss = cached.get('effect_loss', 0.0)
            mb_f1 = cached.get('mb_f1', 0.0)

            # Entry: (balanced_acc, sparsity, h_A, effect_loss, mb_f1) for reporting
            entry = [balanced_acc, spar, h_A, effect_loss, mb_f1]
            pareto_front.append(tuple(entry))
    else:
        # Fallback: reconstruct from evaluation_cache
        print("\n[WARNING] pymoo result.F is None — reconstructing Pareto front from cache")
        print(f"  Cache contains {len(problem.evaluation_cache)} evaluated solutions")
        fallback_solutions = []
        for genome_key, metrics in problem.evaluation_cache.items():
            h_A = metrics.get('structure_h_A', 1.0)
            if h_A <= 0.1:  # feasible
                bacc = metrics.get('classification_balanced_accuracy',
                                   metrics.get('balanced_accuracy', 0.0))
                spar = metrics.get('mb_sparsity', 0.0)
                fallback_solutions.append((bacc, spar, h_A, metrics))
        # Non-dominated sort: remove dominated solutions (Deb et al. 2002)
        # A solution (bacc_a, spar_a) dominates (bacc_b, spar_b) if
        # bacc_a >= bacc_b AND spar_a >= spar_b with at least one strict.
        non_dominated = []
        for i, (bacc_i, spar_i, _, _) in enumerate(fallback_solutions):
            dominated = False
            for j, (bacc_j, spar_j, _, _) in enumerate(fallback_solutions):
                if i != j and bacc_j >= bacc_i and spar_j >= spar_i and (bacc_j > bacc_i or spar_j > spar_i):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(fallback_solutions[i])
        fallback_solutions = non_dominated
        fallback_solutions.sort(key=lambda x: -x[0])
        print(f"  Found {len(fallback_solutions)} non-dominated feasible solutions in cache")
        for bacc, spar, h_A, metrics in fallback_solutions:
            effect_loss = metrics.get('effect_loss', 0.0)
            mb_f1 = metrics.get('mb_f1', 0.0)
            pareto_front.append((bacc, spar, h_A, effect_loss, mb_f1))

    if verbose:
        print("\n" + "="*80)
        print(f"NSGA-II Complete! Pareto Front: {len(pareto_front)} solutions")
        print("="*80)

        # Pareto front display
        header = f"{'#':<5} {'BalAcc':<10} {'Sparsity':<10} {'h(A)':<12} {'EffLoss':<10} {'MB_F1':<8}"
        print(header)
        print("-"*60)

        for i, entry in enumerate(pareto_front):
            row = (f"{i+1:<5} {entry[0]:<10.4f} {entry[1]:<10.4f} "
                   f"{entry[2]:<12.6f} {entry[3]:<10.4f} {entry[4]:<8.3f}")
            print(row)

        print("="*80 + "\n")

    # Build enhanced solution data for Pareto front
    enhanced_solutions = []
    if result.F is not None and result.X is not None:
        for i in range(len(result.X)):
            genome_key = tuple(result.X[i, :].tolist())
            if genome_key in problem.evaluation_cache:
                metrics = problem.evaluation_cache[genome_key]
                enhanced_solutions.append({
                    'genome': result.X[i, :],
                    'objectives': result.F[i, :],
                    'metrics': metrics,
                })
            else:
                # Fallback if not in cache (shouldn't happen)
                enhanced_solutions.append({
                    'genome': result.X[i, :],
                    'objectives': result.F[i, :],
                    'metrics': None,
                })
    else:
        # Reconstruct from cache (matches fallback pareto_front above)
        for genome_key, metrics in problem.evaluation_cache.items():
            h_A = metrics.get('structure_h_A', 1.0)
            if h_A <= 0.1:
                genome = metrics.get('genome', np.array(list(genome_key)))
                bacc = metrics.get('classification_balanced_accuracy',
                                   metrics.get('balanced_accuracy', 0.0))
                spar = metrics.get('mb_sparsity', 0.0)
                enhanced_solutions.append({
                    'genome': genome,
                    'objectives': np.array([-bacc, -spar]),
                    'metrics': metrics,
                })

    # Post-hoc filtering: NSGA-II already filters via constraint, but apply
    # threshold again for extra safety.
    H_A_THRESHOLD = 0.1  # Matches the constraint threshold
    valid_solutions = []
    invalid_count = 0

    for sol in enhanced_solutions:
        metrics = sol.get('metrics', {})
        if metrics is None:
            metrics = {}
        h_A = metrics.get('structure_h_A', 1.0)
        if h_A <= H_A_THRESHOLD:
            valid_solutions.append(sol)
        else:
            invalid_count += 1

    if verbose and invalid_count > 0:
        print(f"\n[Post-filter] Removed {invalid_count}/{len(enhanced_solutions)} "
              f"solutions with h(A) > {H_A_THRESHOLD}")

    if len(valid_solutions) > 0:
        enhanced_solutions = valid_solutions

    # Filter pareto_front to match
    valid_pareto_front = []
    for i, entry in enumerate(pareto_front):
        h_A = entry[2]
        if h_A <= H_A_THRESHOLD or len(valid_solutions) == 0:
            valid_pareto_front.append(entry)
    pareto_front = valid_pareto_front if valid_pareto_front else pareto_front

    return {
        'X': result.X,
        'F': result.F,
        'pareto_front': pareto_front,
        'result': result,
        'use_v7': use_v7,
        'true_mb': true_mb,
        'enhanced_solutions': enhanced_solutions,
        'evaluation_cache': problem.evaluation_cache,
    }
