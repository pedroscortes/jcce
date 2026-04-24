"""
Optuna-based outer loop for JCCE hyperparameter optimization.

Replaces NSGA-II with TPESampler:
- Continuous log-uniform search for lambda_1, lambda_2, lr (vs discrete indices)
- Conditional search spaces (processor-specific params only suggested for selected processor)
- Manual multi-fidelity pruning (kills bad trials early based on running percentile)
  Note: Optuna's trial.report()/should_prune() are not supported for multi-objective studies,
  so we implement manual pruning via a shared accuracy tracker.

Output format matches run_nsga2() for downstream compatibility.
"""

import gc
import ctypes
import time
import logging
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import optuna
from optuna.samplers import TPESampler

from jcce.structure_learning.jcce_learner import (
    create_processor,
    learn_structure,
    extract_markov_blanket,
)
from jcce.structure_learning.warm_start_cache import ImprovedWarmStartCache
from jcce.structure_learning.adjacency_prior import AdjacencyPrior
from jcce.structure_learning.spr_transfer import (
    find_spr_candidate,
    spr_adjacency,
    spr_processor_params,
)

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Manual Multi-Fidelity Pruning
# ============================================================================

class PruningTracker:
    """
    Thread-safe tracker for manual pruning in multi-objective Optuna.

    Since trial.report()/should_prune() are not supported for multi-objective
    studies, we track intermediate balanced_accuracy values at fixed rungs
    and prune trials that fall below the 25th percentile of completed trials.
    """

    def __init__(self, n_startup_trials: int = 5, percentile: float = 25.0):
        self._lock = threading.Lock()
        self._rung_values: Dict[int, List[float]] = {}  # rung -> list of accuracies
        self._n_startup_trials = n_startup_trials
        self._percentile = percentile
        self._n_completed = 0

    def report(self, accuracy: float, rung: int) -> None:
        """Record accuracy at a rung."""
        with self._lock:
            self._rung_values.setdefault(rung, []).append(accuracy)

    def mark_completed(self) -> None:
        """Increment completed trials counter."""
        with self._lock:
            self._n_completed += 1

    def should_prune(self, accuracy: float, rung: int) -> bool:
        """Check if trial should be pruned at this rung."""
        with self._lock:
            if self._n_completed < self._n_startup_trials:
                return False
            values = self._rung_values.get(rung, [])
            if len(values) < self._n_startup_trials:
                return False
            threshold = float(np.percentile(values, self._percentile))
            return accuracy < threshold


# ============================================================================
# Search Space Constants
# ============================================================================

PROCESSOR_TYPES = ['elm', 'gnn', 'mlp', 'transformer', 'mamba']

PROCESSOR_SEARCH_SPACE = {
    'elm': {
        'hidden_dim': [64, 128],
        'n_hidden_nodes': [64, 128],
        'activation': ['relu', 'tanh'],
    },
    'gnn': {
        'hidden_dim': [64, 128],
        'n_layers': [2, 3],
        'gnn_type': ['gcn', 'gat'],
    },
    'mlp': {
        'hidden_dim': [64, 128],
        'n_layers': [2, 3],
        'activation': ['relu', 'tanh'],
    },
    'transformer': {
        'd_model': [64, 128],
        'n_heads': [2, 4],
    },
    'mamba': {
        'd_model': [64, 128],
        'd_state': [8, 16],
    },
}

# Fixed params not included in search (low sensitivity or single valid value)
FIXED_PROCESSOR_PARAMS = {
    'transformer': {'n_layers': 2, 'd_ff': 256},
    'mamba': {'d_conv': 4, 'expand': 2},
    'gnn': {'sage_aggregation': 'mean'},
}

# Effect estimation search values (categorical, matching NSGA-II ranges)
V7_EFFECT_HIDDEN_DIMS = [64, 128, 256]
V7_EFFECT_EMBED_DIMS = [16, 32, 64]
V7_LAMBDA_EFFECTS = [5.0, 10.0, 20.0, 50.0]
V7_EFFECT_WARMUP_ITERS = [10, 20, 30, 50]
V7_LAMBDA_CONFOUND_SPARSE = [0.01, 0.05, 0.1, 0.5]
V7_LAMBDA_BOW = [0.1, 0.3, 0.5, 1.0]
V7_EFFECT_REFINEMENT_ITERS = [0, 30, 50, 100]
V7_N_LATENT_CONFOUNDERS = [2, 3, 5, 8]


# ============================================================================
# Artifact Storage (module-level, LRU)
# ============================================================================

_MAX_ARTIFACT_ENTRIES = 200
_trial_artifacts: OrderedDict = OrderedDict()


def _store_artifacts(trial_number: int, artifacts: Dict[str, Any]) -> None:
    """Store large arrays for a trial, evicting oldest if over limit."""
    _trial_artifacts[trial_number] = artifacts
    _trial_artifacts.move_to_end(trial_number)
    while len(_trial_artifacts) > _MAX_ARTIFACT_ENTRIES:
        _trial_artifacts.popitem(last=False)


def get_trial_artifacts(trial_number: int) -> Optional[Dict[str, Any]]:
    """Retrieve stored artifacts for a trial."""
    return _trial_artifacts.get(trial_number)


# ============================================================================
# Search Space Suggestion
# ============================================================================

def suggest_hyperparams(trial: optuna.Trial, use_v7: bool = True, n_vars: int = 12) -> Dict[str, Any]:
    """
    Suggest hyperparameters for an Optuna trial.

    Uses conditional spaces: processor-specific params are only suggested
    when that processor is selected, reducing effective dimensionality.

    Args:
        trial: Optuna trial object
        use_v7: Whether to include v7 effect estimation params
        n_vars: Number of variables (features). Used to scale lambda_1 range
                with dimensionality — d=30 needs ~2.5x more sparsity than d=12.

    Returns:
        Config dict compatible with GOLEM training
    """
    # Processor type
    processor_type = trial.suggest_categorical('processor_type', PROCESSOR_TYPES)

    # GOLEM hyperparams (continuous log-uniform instead of discrete indices)
    # Scale lambda_1 upper bound with d²: possible edges grow as d(d-1), so the
    # sparsity penalty must scale quadratically to maintain per-edge pressure.
    # Coefficient 2.0 gives the optimizer room for strong sparsity at high d.
    # d=12 → 2.0, d=30 → 12.5, d=50 → 34.7, d=100 → 138.9
    lambda_1_upper = 2.0 * max(1.0, (n_vars / 12.0) ** 2)
    lambda_1 = trial.suggest_float('lambda_1', 0.001, lambda_1_upper, log=True)
    lambda_2 = trial.suggest_float('lambda_2', 0.001, 1.0, log=True)
    lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
    lambda_class = trial.suggest_categorical('lambda_class', [0.1, 0.5, 1.0, 2.0, 5.0])

    # Conditional processor-specific params
    processor_config = {}
    space = PROCESSOR_SEARCH_SPACE[processor_type]
    for param_name, values in space.items():
        key = f'{processor_type}_{param_name}'
        if isinstance(values[0], str):
            processor_config[param_name] = trial.suggest_categorical(key, values)
        else:
            processor_config[param_name] = trial.suggest_categorical(key, values)

    # Add fixed params
    if processor_type in FIXED_PROCESSOR_PARAMS:
        processor_config.update(FIXED_PROCESSOR_PARAMS[processor_type])

    config = {
        'processor_type': processor_type,
        'processor_config': processor_config,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2,
        'lambda_class': lambda_class,
        'lr': lr,
    }

    # Effect estimation params
    if use_v7:
        config['effect_hidden_dim'] = trial.suggest_categorical(
            'effect_hidden_dim', V7_EFFECT_HIDDEN_DIMS
        )
        config['effect_embed_dim'] = trial.suggest_categorical(
            'effect_embed_dim', V7_EFFECT_EMBED_DIMS
        )
        config['lambda_effect'] = trial.suggest_categorical(
            'lambda_effect', V7_LAMBDA_EFFECTS
        )
        config['effect_warmup_iter'] = trial.suggest_categorical(
            'effect_warmup_iter', V7_EFFECT_WARMUP_ITERS
        )
        config['lambda_confound_sparse'] = trial.suggest_categorical(
            'lambda_confound_sparse', V7_LAMBDA_CONFOUND_SPARSE
        )
        config['lambda_bow_v7'] = trial.suggest_categorical(
            'lambda_bow_v7', V7_LAMBDA_BOW
        )
        config['effect_refinement_iters'] = trial.suggest_categorical(
            'effect_refinement_iters', V7_EFFECT_REFINEMENT_ITERS
        )
        config['n_latent_confounders'] = trial.suggest_categorical(
            'n_latent_confounders', V7_N_LATENT_CONFOUNDERS
        )

    return config


# ============================================================================
# Constraints
# ============================================================================

def constraints_func(trial: optuna.trial.FrozenTrial) -> List[float]:
    """
    Constraint function for TPESampler.

    Feasible when all returned values <= 0.
    h_A must be < 0.1 for a valid DAG.
    """
    import math
    h_A = trial.user_attrs.get('h_A', 1.0)
    # NaN from diverged training → treat as infeasible (large positive value)
    if math.isnan(h_A):
        h_A = 10.0
    return [h_A - 0.1]


# ============================================================================
# Memory Cleanup
# ============================================================================

def _memory_cleanup():
    """Lightweight memory cleanup between trials."""
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# ============================================================================
# Objective Factory
# ============================================================================

def create_optuna_objective(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    n_vars: int,
    max_iter: int = 300,
    use_v7: bool = True,
    Y_continuous: Optional[jnp.ndarray] = None,
    task: str = 'classification',
    golem_overrides: Optional[Dict[str, Any]] = None,
    warm_start_cache: Optional[ImprovedWarmStartCache] = None,
    jax_key_seed: int = 0,
    verbose: bool = False,
    true_graph: Optional[jnp.ndarray] = None,
    pruning_tracker: Optional[PruningTracker] = None,
    adjacency_prior: Optional[AdjacencyPrior] = None,
    enable_spr: bool = False,
    spr_max_distance: float = 0.5,
    spr_lambda_A: float = 0.85,
    spr_lambda_theta: float = 0.7,
    suggest_fn: Optional[Callable] = None,
) -> Callable[[optuna.Trial], Tuple[float, float]]:
    """
    Factory that creates an Optuna objective function.

    The objective trains GOLEM with suggested hyperparameters and returns
    two values: (balanced_accuracy, mb_sparsity) for multi-objective optimization.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Target variable (n_samples,) or (n_samples, 1)
        n_vars: Number of variables (= X.shape[1])
        max_iter: Maximum GOLEM iterations per trial
        use_v7: Use v7 with effect estimation
        Y_continuous: Continuous Y for effect estimation
        task: 'classification' or 'regression'
        golem_overrides: Override dict passed to GOLEM
        warm_start_cache: Optional warm-start cache for A_init
        jax_key_seed: Base seed for JAX random keys
        verbose: Print per-trial details
        true_graph: Optional ground truth for structure metrics
        pruning_tracker: Optional PruningTracker for manual multi-fidelity pruning
        adjacency_prior: Optional AdjacencyPrior for global warm-starting
        enable_spr: Enable Shrink-Perturb-Repeat transfer from close trials
        spr_max_distance: Max HP distance for SPR candidate matching
        spr_lambda_A: SPR shrinkage factor for adjacency matrices
        spr_lambda_theta: SPR shrinkage factor for processor weights
        suggest_fn: Override for suggest_hyperparams (e.g. flat search space for C.9)

    Returns:
        objective(trial) -> (balanced_acc, mb_sparsity)
    """
    if golem_overrides is None:
        golem_overrides = {}

    # Pruning rungs: check at these iteration checkpoints.
    # Starts at 40% of max_iter to avoid pruning during Phase 1 (structure-only),
    # where BAcc=0.50 is expected before classification starts.
    _pruning_rungs = {int(max_iter * 0.4), int(max_iter * 0.7), max_iter}

    Y_idx = n_vars  # Y is appended as last variable

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        trial_start = time.time()

        # 1. Suggest hyperparams (pass n_vars for dimension-aware lambda_1 scaling)
        _suggest = suggest_fn or suggest_hyperparams
        config = _suggest(trial, use_v7=use_v7, n_vars=n_vars) if _suggest is suggest_hyperparams else _suggest(trial, use_v7=use_v7)

        # 2. Create processor
        key = random.PRNGKey(jax_key_seed + trial.number)
        key, proc_key = random.split(key)
        processor = create_processor(
            config['processor_type'],
            key=proc_key,
            n_features=n_vars,
            **config['processor_config'],
        )

        # 3. Warm-start: SPR > adjacency_prior > warm_start_cache > None
        A_init = None
        proc_params_init = None
        spr_hit = False

        # Priority 1: SPR transfer from close same-processor trial
        if enable_spr and len(_trial_artifacts) > 0:
            spr_trial = find_spr_candidate(
                config, _trial_artifacts, max_distance=spr_max_distance
            )
            if spr_trial is not None:
                donor = _trial_artifacts[spr_trial]
                donor_A = donor.get('A_weights', donor.get('A_est'))
                if donor_A is not None:
                    rng = np.random.default_rng(jax_key_seed + trial.number)
                    A_init = jnp.array(spr_adjacency(
                        np.array(donor_A),
                        lambda_A=spr_lambda_A,
                        noise_scale=0.1,
                        rng=rng,
                    ))
                    spr_hit = True
                    # Transfer processor params if available
                    donor_proc = donor.get('proc_params')
                    if donor_proc is not None:
                        proc_params_init = spr_processor_params(
                            donor_proc,
                            lambda_theta=spr_lambda_theta,
                            n_inputs=n_vars,
                            rng=rng,
                        )

        # Priority 2: Global adjacency prior
        if A_init is None and adjacency_prior is not None:
            if adjacency_prior.n_updates >= 5:
                rng = np.random.default_rng(jax_key_seed + trial.number + 1000)
                A_init = jnp.array(adjacency_prior.sample_A_init(rng=rng))

        # Priority 3: Legacy warm-start cache
        if A_init is None and warm_start_cache is not None:
            A_init = warm_start_cache.get(config)
            if A_init is None:
                A_init = warm_start_cache.sample()
            if A_init is not None:
                A_init = jnp.array(A_init)

        trial.set_user_attr('spr_hit', spr_hit)

        # 4. Build iteration callback for manual pruning
        # Note: trial.report()/should_prune() are NOT supported for multi-objective
        # Optuna studies. We use PruningTracker for manual percentile-based pruning.
        def iteration_callback(iter_num: int, metrics: Dict[str, float]) -> None:
            if iter_num in _pruning_rungs and pruning_tracker is not None:
                # Don't prune during Phase 1 (structure-only):
                # BAcc=0.50 is expected when classification hasn't started.
                if metrics.get('curriculum_phase', 0) < 2:
                    return
                acc = metrics['balanced_accuracy']
                pruning_tracker.report(acc, iter_num)
                if pruning_tracker.should_prune(acc, iter_num):
                    raise optuna.TrialPruned(
                        f"Pruned at iter {iter_num}: "
                        f"acc={acc:.3f}, "
                        f"h_A={metrics['h_A']:.4f}"
                    )

        # 5. Prepare Y
        Y_for_v7 = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # 6. Run GOLEM training
        try:
            key, train_key = random.split(key)

            golem_kwargs = {
                'data': X,
                'Y': Y_for_v7,
                'Y_idx': Y_idx,
                'processor': processor,
                'key': train_key,
                'processor_type': config['processor_type'],
                'lambda_1': config['lambda_1'],
                'lambda_2_init': config['lambda_2'],
                'lambda_class': config['lambda_class'],
                'lr': config['lr'],
                'max_iter': max_iter,
                'patience': 25,
                'verbose': 2 if verbose else 0,
                'A_init': A_init,
                'proc_params_init': proc_params_init,
                'task': task,
                'iteration_callback': iteration_callback,
                # PC constraint only makes sense for actual PC algorithm output.
                # Warm-start cache/SPR/adjacency prior provide "soft suggestions"
                # that should NOT be locked in with a structural penalty.
                'use_pc_constraint': False,
                # Overrides
                'use_adaptive_curriculum': golem_overrides.get('use_adaptive_curriculum', True),
                'curriculum_phase_splits': golem_overrides.get('curriculum_phase_splits', (0.4, 0.8)),
                'lambda_ident': golem_overrides.get('lambda_ident', 0.01),
                'use_amortized_effects': golem_overrides.get('use_amortized_effects', True),
                'use_dragonnet': golem_overrides.get('use_dragonnet', True),
                'use_structural_dml': golem_overrides.get('use_structural_dml', False),
                'use_pcgrad': golem_overrides.get('use_pcgrad', False),
                'enforce_outcome_sink': golem_overrides.get('enforce_outcome_sink', True),
                'freeze_A': golem_overrides.get('freeze_A', False),
            }

            if use_v7:
                golem_kwargs.update({
                    'effect_hidden_dim': config.get('effect_hidden_dim', 64),
                    'effect_embed_dim': config.get('effect_embed_dim', 16),
                    'lambda_effect': golem_overrides.get(
                        'lambda_effect', config.get('lambda_effect', 10.0)
                    ),
                    'effect_warmup_iter': config.get('effect_warmup_iter', 20),
                    'lambda_confound_sparse': config.get('lambda_confound_sparse', 0.05),
                    'lambda_bow': golem_overrides.get(
                        'lambda_bow', config.get('lambda_bow_v7', 0.3)
                    ),
                    'Y_continuous': Y_continuous,
                    'effect_refinement_iters': config.get('effect_refinement_iters', 50),
                    'n_latent_confounders': config.get('n_latent_confounders', 5),
                })

            A_est, processor_trained, processor_params, metrics = \
                learn_structure(**golem_kwargs)

        except optuna.TrialPruned:
            raise  # Re-raise pruning
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            trial.set_user_attr('error', str(e))
            trial.set_user_attr('h_A', 1.0)
            _memory_cleanup()
            return 0.0, 0.0

        # 7. Extract metrics
        h_A = float(metrics.get('final_h_A', 1.0))
        balanced_accuracy = float(metrics.get('balanced_accuracy', 0.0))
        n_edges = int(metrics.get('n_edges', 0))

        # Extract Markov blanket
        mb_from_metrics = metrics.get('markov_blanket', [])
        if mb_from_metrics:
            mb_indices = [idx for idx in mb_from_metrics if idx != Y_idx]
        else:
            mb_arr = extract_markov_blanket(A_est, Y_idx, threshold=1e-6)
            mb_indices = [int(idx) for idx in mb_arr if idx != Y_idx]

        n_features = X.shape[1]
        mb_size = len(mb_indices)

        # Continuous edge sparsity: penalizes total edge weight to Y, not
        # thresholded MB size. Prevents gaming where optimizer pushes weights
        # just below MB threshold while relying on the additive floor (0.01)
        # for classification. See Session 40 analysis.
        A_to_Y = np.abs(np.array(A_est[:n_features, Y_idx]))
        edge_weight_to_Y = float(np.sum(A_to_Y))
        mb_sparsity = 1.0 - min(edge_weight_to_Y / n_features, 1.0)

        # 8. Store scalar metrics as user_attrs
        trial.set_user_attr('h_A', h_A)
        trial.set_user_attr('balanced_accuracy', balanced_accuracy)
        trial.set_user_attr('mb_sparsity', mb_sparsity)
        trial.set_user_attr('mb_size', mb_size)
        trial.set_user_attr('mb_indices', mb_indices)
        trial.set_user_attr('n_edges', n_edges)
        trial.set_user_attr('processor_type', config['processor_type'])
        trial.set_user_attr('classification_accuracy',
                            float(metrics.get('classification_accuracy', 0.0)))
        trial.set_user_attr('classification_f1',
                            float(metrics.get('f1_score', 0.0)))
        trial.set_user_attr('classification_precision',
                            float(metrics.get('precision', 0.0)))
        trial.set_user_attr('classification_recall',
                            float(metrics.get('recall', 0.0)))
        trial.set_user_attr('classification_roc_auc',
                            float(metrics.get('auc_roc', 0.0)))
        trial.set_user_attr('iterations',
                            int(metrics.get('iterations', max_iter)))
        trial.set_user_attr('early_stopped',
                            bool(metrics.get('early_stopped', False)))
        trial.set_user_attr('recon_loss',
                            float(metrics.get('final_recon_loss', 0.0)))
        trial.set_user_attr('class_loss',
                            float(metrics.get('final_class_loss', 0.0)))
        trial.set_user_attr('trial_time', time.time() - trial_start)

        if use_v7:
            trial.set_user_attr('effect_loss',
                                float(metrics.get('effect_loss', 1.0)))
            trial.set_user_attr('bow_loss',
                                float(metrics.get('bow_loss', 0.0)))
            trial.set_user_attr('n_confound_edges',
                                int(metrics.get('n_confound_edges', 0)))

        # Gradient diagnostics and bow-free checks (Session 39)
        try:
            gd = metrics.get('gradient_diagnostics', [])
            if gd:
                # Ensure all values are pure Python types (not JAX/numpy scalars)
                gd_clean = [
                    {k: float(v) if isinstance(v, (float, int)) else int(v)
                     for k, v in entry.items()}
                    for entry in gd
                ]
                trial.set_user_attr('gradient_diagnostics', gd_clean)
            if 'bow_free_violations' in metrics:
                trial.set_user_attr('bow_free_violations', int(metrics['bow_free_violations']))
            if 'residuals_non_gaussian' in metrics:
                trial.set_user_attr('residuals_non_gaussian', bool(metrics['residuals_non_gaussian']))
            if 'residual_normality_pvals' in metrics:
                trial.set_user_attr('residual_normality_pvals',
                                    [float(p) for p in metrics['residual_normality_pvals']])
        except Exception as _diag_err:
            logger.warning(f"Trial {trial.number}: diagnostic forwarding failed: {_diag_err}")

        # 9. Store large arrays in module-level artifact store
        artifacts = {
            'A_est': np.array(A_est),
            'config': config,
            'markov_blanket': mb_indices,
        }

        # D.2: Store processor params + continuous A for SPR transfer
        artifacts['proc_params'] = [
            {k: np.array(v) if hasattr(v, 'shape') else v
             for k, v in p.items()}
            for p in processor_params
        ]
        if 'A_weights' in metrics:
            artifacts['A_weights'] = np.array(metrics['A_weights'])

        if use_v7:
            if 'A_confound' in metrics:
                artifacts['A_confound'] = np.array(metrics['A_confound'])
            if 'A_confound_weights' in metrics:
                artifacts['A_confound_weights'] = np.array(
                    metrics['A_confound_weights']
                )
            artifacts['causal_effects'] = metrics.get('causal_effects', {})
            # Low-rank confound model extras
            if 'B_confound' in metrics:
                artifacts['B_confound'] = np.array(metrics['B_confound'])
                artifacts['log_var_confound'] = metrics['log_var_confound']
                artifacts['n_latent_confounders'] = metrics['n_latent_confounders']

        # Structure recovery metrics if ground truth available
        if true_graph is not None:
            try:
                from jcce.training.metrics import evaluate_structure_recovery
                A_est_X = np.array(A_est)[:n_vars, :n_vars]
                true_graph_np = np.array(true_graph)
                if A_est_X.shape == true_graph_np.shape:
                    sr = evaluate_structure_recovery(
                        A_est_X, true_graph_np, compute_sid=True
                    )
                    for k, v in sr.items():
                        trial.set_user_attr(f'structure_{k}', float(v))
            except Exception:
                pass

        _store_artifacts(trial.number, artifacts)

        # 10. Update warm-start cache + adjacency prior
        if warm_start_cache is not None:
            fitness = 0.7 * balanced_accuracy + 0.3 * mb_sparsity
            warm_start_cache.add(
                config=config,
                A_matrix=np.array(A_est),
                fitness=fitness,
                generation=0,
                processor_type=config['processor_type'],
            )

        # D.1: Update adjacency prior with feasible solutions
        if adjacency_prior is not None and h_A < 0.1:
            fitness = 0.7 * balanced_accuracy + 0.3 * mb_sparsity
            A_continuous = np.array(metrics.get('A_weights', A_est))
            adjacency_prior.update(A_continuous, fitness)

        # 11. Mark completed for pruning tracker
        if pruning_tracker is not None:
            pruning_tracker.mark_completed()

        # 12. Memory cleanup
        _memory_cleanup()

        if verbose:
            print(
                f"  Trial {trial.number}: {config['processor_type']} "
                f"bacc={balanced_accuracy:.3f} sparsity={mb_sparsity:.3f} "
                f"h_A={h_A:.4f} edges={n_edges} "
                f"({time.time() - trial_start:.1f}s)"
            )

        return balanced_accuracy, mb_sparsity

    return objective


# ============================================================================
# Pareto Extraction
# ============================================================================

def extract_pareto_solutions(
    study: optuna.Study,
    use_v7: bool = True,
    h_A_threshold: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Tuple]]:
    """
    Extract Pareto-optimal solutions from an Optuna study.

    Filters infeasible solutions (h_A >= threshold) and converts to
    the same dict format as NSGA-II enhanced_solutions.

    Args:
        study: Completed Optuna study
        use_v7: Whether v7 effect params are present
        h_A_threshold: Maximum h_A for feasibility

    Returns:
        (enhanced_solutions, pareto_front) matching NSGA-II format
    """
    enhanced_solutions = []
    pareto_front = []

    for trial in study.best_trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        h_A = trial.user_attrs.get('h_A', 1.0)
        if h_A >= h_A_threshold:
            continue

        balanced_accuracy = trial.values[0]
        mb_sparsity = trial.values[1]

        # Retrieve artifacts
        artifacts = get_trial_artifacts(trial.number)
        if artifacts is None:
            continue

        config = artifacts['config']
        mb_indices = artifacts['markov_blanket']

        # Build metrics dict matching evaluate_genome_unified() output
        metrics = {
            'fitness': 0.7 * balanced_accuracy + 0.3 * mb_sparsity,
            'classification_accuracy': trial.user_attrs.get(
                'classification_accuracy', balanced_accuracy
            ),
            'classification_precision': trial.user_attrs.get(
                'classification_precision', 0.0
            ),
            'classification_recall': trial.user_attrs.get(
                'classification_recall', 0.0
            ),
            'classification_f1': trial.user_attrs.get(
                'classification_f1', 0.0
            ),
            'classification_balanced_accuracy': balanced_accuracy,
            'classification_roc_auc': trial.user_attrs.get(
                'classification_roc_auc', 0.0
            ),
            'mb_sparsity': mb_sparsity,
            'mb_size': len(mb_indices),
            'mb_indices': mb_indices,
            'markov_blanket': mb_indices,
            'structure_n_edges': trial.user_attrs.get('n_edges', 0),
            'structure_h_A': h_A,
            'structure_A_est': artifacts['A_est'],
            'v4_recon_loss': trial.user_attrs.get('recon_loss', 0.0),
            'v4_class_loss': trial.user_attrs.get('class_loss', 0.0),
            'v4_iterations': trial.user_attrs.get('iterations', 0),
            'v4_early_stopped': trial.user_attrs.get('early_stopped', False),
            'processor_type': config['processor_type'],
            'processor_config': config['processor_config'],
            'lambda_1': config['lambda_1'],
            'lambda_2': config['lambda_2'],
            'lambda_class': config['lambda_class'],
            'lr': config['lr'],
        }

        if use_v7:
            metrics['effect_loss'] = trial.user_attrs.get('effect_loss', 1.0)
            metrics['causal_effects'] = artifacts.get('causal_effects', {})
            metrics['n_confound_edges'] = trial.user_attrs.get(
                'n_confound_edges', 0
            )
            metrics['bow_loss'] = trial.user_attrs.get('bow_loss', 0.0)
            if 'A_confound' in artifacts:
                metrics['A_confound'] = artifacts['A_confound']
            if 'A_weights' in artifacts:
                metrics['A_weights'] = artifacts['A_weights']
            if 'A_confound_weights' in artifacts:
                metrics['A_confound_weights'] = artifacts['A_confound_weights']
            metrics['effect_hidden_dim'] = config.get('effect_hidden_dim', 64)
            metrics['lambda_effect'] = config.get('lambda_effect', 10.0)
            metrics['effect_embed_dim'] = config.get('effect_embed_dim', 16)
            metrics['effect_warmup_iter'] = config.get('effect_warmup_iter', 20)
            metrics['lambda_confound_sparse'] = config.get(
                'lambda_confound_sparse', 0.05
            )
            metrics['lambda_bow'] = config.get('lambda_bow_v7', 0.3)

        # Processor params (needed for CF evaluation)
        if 'proc_params' in artifacts:
            metrics['_processor_params'] = artifacts['proc_params']

        # Gradient diagnostics and bow-free checks
        for diag_key in ['gradient_diagnostics', 'bow_free_violations',
                         'residuals_non_gaussian', 'residual_normality_pvals']:
            if diag_key in trial.user_attrs:
                metrics[diag_key] = trial.user_attrs[diag_key]

        # Structure recovery metrics
        for sr_key in ['structure_edge_f1', 'structure_edge_precision',
                       'structure_edge_recall', 'structure_shd',
                       'structure_sid', 'structure_sid_normalized',
                       'structure_tp', 'structure_fp', 'structure_fn']:
            if sr_key in trial.user_attrs:
                metrics[sr_key] = trial.user_attrs[sr_key]

        enhanced_solutions.append({
            'genome': np.array(list(trial.params.values())),
            'objectives': np.array([balanced_accuracy, mb_sparsity]),
            'metrics': metrics,
        })

        effect_loss = trial.user_attrs.get('effect_loss', 0.0)
        pareto_front.append((
            balanced_accuracy,
            mb_sparsity,
            h_A,
            effect_loss,
            0.0,  # mb_f1 placeholder (requires true_mb)
        ))

    # If Pareto front is too small (< 5 solutions), supplement with feasible
    # non-dominated-adjacent solutions from ALL completed trials. This addresses
    # the issue where 2-objective Pareto gives very few solutions while NSGA-II
    # returns ~pop_size diverse solutions.
    if len(enhanced_solutions) < 5:
        _existing_trial_nums = {t.number for t in study.best_trials
                                if t.state == optuna.trial.TrialState.COMPLETE}
        _candidates = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            if trial.number in _existing_trial_nums:
                continue
            h_A = trial.user_attrs.get('h_A', 1.0)
            if h_A >= h_A_threshold:
                continue
            artifacts = get_trial_artifacts(trial.number)
            if artifacts is None:
                continue
            bacc = trial.values[0]
            sp = trial.values[1]
            # TOPSIS-like score: weighted combination
            _score = 0.7 * bacc + 0.3 * sp
            _candidates.append((trial, artifacts, _score))

        # Sort by score descending, take enough to reach 10 solutions
        _candidates.sort(key=lambda x: x[2], reverse=True)
        _n_needed = min(10 - len(enhanced_solutions), len(_candidates))
        for trial, artifacts, _score in _candidates[:_n_needed]:
            config = artifacts['config']
            mb_indices = artifacts['markov_blanket']
            balanced_accuracy = trial.values[0]
            mb_sparsity = trial.values[1]
            h_A = trial.user_attrs.get('h_A', 1.0)

            metrics = {
                'fitness': _score,
                'classification_accuracy': trial.user_attrs.get('classification_accuracy', balanced_accuracy),
                'classification_precision': trial.user_attrs.get('classification_precision', 0.0),
                'classification_recall': trial.user_attrs.get('classification_recall', 0.0),
                'classification_f1': trial.user_attrs.get('classification_f1', 0.0),
                'classification_balanced_accuracy': balanced_accuracy,
                'classification_roc_auc': trial.user_attrs.get('classification_roc_auc', 0.0),
                'mb_sparsity': mb_sparsity,
                'mb_size': len(mb_indices),
                'mb_indices': mb_indices,
                'markov_blanket': mb_indices,
                'structure_n_edges': trial.user_attrs.get('n_edges', 0),
                'structure_h_A': h_A,
                'structure_A_est': artifacts['A_est'],
                'v4_recon_loss': trial.user_attrs.get('recon_loss', 0.0),
                'v4_class_loss': trial.user_attrs.get('class_loss', 0.0),
                'v4_iterations': trial.user_attrs.get('iterations', 0),
                'v4_early_stopped': trial.user_attrs.get('early_stopped', False),
                'processor_type': config['processor_type'],
                'processor_config': config['processor_config'],
                'lambda_1': config['lambda_1'],
                'lambda_2': config['lambda_2'],
                'lambda_class': config['lambda_class'],
                'lr': config['lr'],
            }
            if use_v7:
                metrics['effect_loss'] = trial.user_attrs.get('effect_loss', 1.0)
                metrics['causal_effects'] = artifacts.get('causal_effects', {})
                metrics['n_confound_edges'] = trial.user_attrs.get('n_confound_edges', 0)
                metrics['bow_loss'] = trial.user_attrs.get('bow_loss', 0.0)
                if 'A_confound' in artifacts:
                    metrics['A_confound'] = artifacts['A_confound']
                if 'A_weights' in artifacts:
                    metrics['A_weights'] = artifacts['A_weights']
                if 'A_confound_weights' in artifacts:
                    metrics['A_confound_weights'] = artifacts['A_confound_weights']
                if 'proc_params' in artifacts:
                    metrics['_processor_params'] = artifacts['proc_params']

            enhanced_solutions.append({
                'genome': np.array(list(trial.params.values())),
                'objectives': np.array([balanced_accuracy, mb_sparsity]),
                'metrics': metrics,
            })
            effect_loss = trial.user_attrs.get('effect_loss', 0.0)
            pareto_front.append((balanced_accuracy, mb_sparsity, h_A, effect_loss, 0.0))

    return enhanced_solutions, pareto_front


# ============================================================================
# D.3: Seed Enqueuing
# ============================================================================

def _enqueue_best_as_seeds(study: optuna.Study, n_seeds: int = 3) -> None:
    """
    Enqueue top-N completed feasible trials as seed configs for new optimization.

    Useful when resuming a study with SPR: re-explores the best known
    configs so SPR can transfer from them early.

    Args:
        study: Optuna study with existing trials
        n_seeds: Number of top trials to enqueue
    """
    feasible = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get('h_A', 1.0) < 0.1
        and t.values is not None
    ]
    if not feasible:
        return

    # Sort by balanced_accuracy (first objective, descending)
    feasible.sort(key=lambda t: t.values[0], reverse=True)

    for trial in feasible[:n_seeds]:
        study.enqueue_trial(trial.params)


# ============================================================================
# Main Entry Point
# ============================================================================

def run_optuna_search(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    n_vars: int,
    n_trials: int = 100,
    max_iter: int = 300,
    use_v7: bool = True,
    Y_continuous: Optional[jnp.ndarray] = None,
    task: str = 'classification',
    golem_overrides: Optional[Dict[str, Any]] = None,
    jax_key_seed: int = 0,
    verbose: bool = True,
    true_graph: Optional[jnp.ndarray] = None,
    true_mb: Optional[List[int]] = None,
    enable_warm_start: bool = False,
    enable_adjacency_prior: bool = False,
    enable_spr: bool = False,
    spr_max_distance: float = 0.5,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Optuna multi-objective hyperparameter search.

    Drop-in replacement for run_nsga2().

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Target variable (n_samples,) or (n_samples, 1)
        n_vars: Number of variables (= X.shape[1])
        n_trials: Number of Optuna trials to run
        max_iter: Max GOLEM iterations per trial
        use_v7: Use v7 with effect estimation
        Y_continuous: Continuous Y for effect estimation
        task: 'classification' or 'regression'
        golem_overrides: Override dict for GOLEM
        jax_key_seed: Base seed for reproducibility
        verbose: Print progress
        true_graph: Optional ground truth graph
        true_mb: Optional ground truth Markov blanket
        enable_warm_start: Use warm-start cache
        enable_adjacency_prior: Use EMA adjacency prior for warm-starting
        enable_spr: Enable Shrink-Perturb-Repeat transfer
        spr_max_distance: Max HP distance for SPR candidate matching
        storage: Optuna storage URL (e.g. 'sqlite:///study.db')
        study_name: Name for the Optuna study

    Returns:
        Dict matching run_nsga2() format:
            pareto_front, enhanced_solutions, evaluation_cache,
            use_v7, true_mb, study, n_trials_completed, n_trials_pruned
    """
    if verbose:
        print(f"Optuna search: {n_trials} trials, max_iter={max_iter}, "
              f"use_v7={use_v7}, task={task}")
        print(f"Data: X={X.shape}, n_vars={n_vars}")
        print(f"Processors: {', '.join(PROCESSOR_TYPES)}")

    # Warm-start cache
    warm_start_cache = ImprovedWarmStartCache(max_size=50) if enable_warm_start else None

    # D.1: Adjacency prior
    adjacency_prior = None
    if enable_adjacency_prior:
        adjacency_prior = AdjacencyPrior(n_total=n_vars + 1)

    # Create sampler with constraints
    sampler = TPESampler(
        multivariate=True,
        group=True,
        seed=jax_key_seed,
        n_startup_trials=min(10, n_trials),
        constraints_func=constraints_func,
        constant_liar=True,
    )

    # Create study (no built-in pruner — multi-objective doesn't support trial.report)
    study = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=sampler,
        storage=storage,
        study_name=study_name or f'jcce_optuna_{jax_key_seed}',
        load_if_exists=True,
    )

    # D.3: Seed from historical trials if resuming a study with SPR
    if enable_spr and len(study.trials) > 0:
        _enqueue_best_as_seeds(study, n_seeds=3)

    # Manual pruning tracker (replaces Hyperband for multi-objective).
    # n_startup_trials=10 lets diverse processors complete before pruning activates.
    # percentile=15 avoids over-pruning during curriculum learning where early
    # BAcc is uninformative.
    pruning_tracker = PruningTracker(
        n_startup_trials=min(10, n_trials),
        percentile=15.0,
    )

    # Create objective
    objective = create_optuna_objective(
        X=X,
        Y=Y,
        n_vars=n_vars,
        max_iter=max_iter,
        use_v7=use_v7,
        Y_continuous=Y_continuous,
        task=task,
        golem_overrides=golem_overrides,
        warm_start_cache=warm_start_cache,
        jax_key_seed=jax_key_seed,
        verbose=verbose,
        true_graph=true_graph,
        pruning_tracker=pruning_tracker,
        adjacency_prior=adjacency_prior,
        enable_spr=enable_spr,
        spr_max_distance=spr_max_distance,
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Count trial states
    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.FAIL])

    if verbose:
        print(f"\nOptuna search complete:")
        print(f"  Completed: {n_completed}, Pruned: {n_pruned}, Failed: {n_failed}")

    # Extract Pareto solutions
    enhanced_solutions, pareto_front = extract_pareto_solutions(
        study, use_v7=use_v7
    )

    if verbose:
        print(f"  Pareto solutions: {len(enhanced_solutions)} (feasible, h_A < 0.1)")
        for i, sol in enumerate(enhanced_solutions):
            m = sol['metrics']
            print(f"    [{i}] {m['processor_type']}: "
                  f"bacc={m['classification_balanced_accuracy']:.3f} "
                  f"sparsity={m['mb_sparsity']:.3f} "
                  f"h_A={m['structure_h_A']:.4f}")

    # Build evaluation cache (trial_number -> metrics)
    evaluation_cache = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            cache_key = tuple(sorted(trial.params.items()))
            artifacts = get_trial_artifacts(trial.number)
            if artifacts is not None:
                evaluation_cache[cache_key] = {
                    **trial.user_attrs,
                    'trial_number': trial.number,
                }

    return {
        'pareto_front': pareto_front,
        'enhanced_solutions': enhanced_solutions,
        'evaluation_cache': evaluation_cache,
        'use_v7': use_v7,
        'true_mb': true_mb,
        'study': study,
        'n_trials_completed': n_completed,
        'n_trials_pruned': n_pruned,
        'n_trials_failed': n_failed,
    }
