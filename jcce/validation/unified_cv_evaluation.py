"""
Unified K-fold cross-validation using fθ.

Retrains GOLEM + processor on each fold and evaluates fθ on held-out
data, instead of fitting a separate LogisticRegression for evaluation.

Key design decisions:
- Warm-starting from the full-data solution reduces fold training from
  ~150 to ~50 iterations.
- fθ is the same model that discovers causal structure (unified evaluation).
- Structure stability (MB Jaccard, edge frequency) measures DAG robustness
  across folds.

Reference:
- Chernozhukov et al. (2018), "Double/debiased machine learning."
"""

import gc
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from sklearn.model_selection import StratifiedKFold

from jcce.utils.metrics import EDGE_THRESHOLD_DEFAULT


@dataclass
class UnifiedCVResult:
    """Result of K-fold cross-validation using fθ."""

    # Per-fold classification results
    accuracy_per_fold: List[float]
    precision_per_fold: List[float]
    recall_per_fold: List[float]
    f1_per_fold: List[float]
    balanced_acc_per_fold: List[float]
    roc_auc_per_fold: List[float]

    # Aggregated (for tables)
    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    balanced_acc_mean: float
    balanced_acc_std: float
    roc_auc_mean: float
    roc_auc_std: float

    # Structure stability across folds
    mb_per_fold: List[List[int]]
    mb_jaccard_mean: float
    edge_frequency: Dict[Tuple[int, int], float]

    # Model identity (which processor generated these results)
    processor_type: Optional[str] = None
    processor_config: Optional[Dict] = None
    golem_config: Optional[Dict] = None

    # Structure metrics (LUCAS only — when ground truth is available)
    shd_per_fold: Optional[List[int]] = None
    mb_f1_per_fold: Optional[List[float]] = None
    edge_f1_per_fold: Optional[List[float]] = None
    edge_precision_per_fold: Optional[List[float]] = None
    edge_recall_per_fold: Optional[List[float]] = None

    task: str = "classification"

    # Timing
    fold_times: Optional[List[float]] = None
    total_time: Optional[float] = None

    def to_dict(self) -> Dict:
        result = {
            "accuracy_per_fold": self.accuracy_per_fold,
            "precision_per_fold": self.precision_per_fold,
            "recall_per_fold": self.recall_per_fold,
            "f1_per_fold": self.f1_per_fold,
            "balanced_acc_per_fold": self.balanced_acc_per_fold,
            "roc_auc_per_fold": self.roc_auc_per_fold,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "precision_mean": self.precision_mean,
            "precision_std": self.precision_std,
            "recall_mean": self.recall_mean,
            "recall_std": self.recall_std,
            "f1_mean": self.f1_mean,
            "f1_std": self.f1_std,
            "balanced_acc_mean": self.balanced_acc_mean,
            "balanced_acc_std": self.balanced_acc_std,
            "roc_auc_mean": self.roc_auc_mean,
            "roc_auc_std": self.roc_auc_std,
            "mb_per_fold": self.mb_per_fold,
            "mb_jaccard_mean": self.mb_jaccard_mean,
            "edge_frequency": {f"{k[0]}->{k[1]}": v for k, v in self.edge_frequency.items()},
            "n_folds": len(self.accuracy_per_fold),
            "task": self.task,
            "processor_type": self.processor_type,
            "processor_config": self.processor_config,
            "golem_config": self.golem_config,
        }
        if self.shd_per_fold is not None:
            result["shd_per_fold"] = self.shd_per_fold
        if self.mb_f1_per_fold is not None:
            result["mb_f1_per_fold"] = self.mb_f1_per_fold
        if self.edge_f1_per_fold is not None:
            result["edge_f1_per_fold"] = self.edge_f1_per_fold
            result["edge_f1_mean"] = float(np.mean(self.edge_f1_per_fold))
        if self.edge_precision_per_fold is not None:
            result["edge_precision_per_fold"] = self.edge_precision_per_fold
            result["edge_precision_mean"] = float(np.mean(self.edge_precision_per_fold))
        if self.edge_recall_per_fold is not None:
            result["edge_recall_per_fold"] = self.edge_recall_per_fold
            result["edge_recall_mean"] = float(np.mean(self.edge_recall_per_fold))
        if self.fold_times is not None:
            result["fold_times"] = self.fold_times
        if self.total_time is not None:
            result["total_time"] = self.total_time
        return result

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"UNIFIED CV EVALUATION (fθ) — {self.task}",
            "=" * 60,
            f"Processor: {self.processor_type or 'unknown'}",
            f"Folds: {len(self.accuracy_per_fold)}",
            "",
        ]
        if self.task == "regression":
            lines.extend(
                [
                    "Regression (held-out):",
                    f"  R²:           {self.accuracy_mean:.4f} ± {self.accuracy_std:.4f}",
                    f"  RMSE:         {self.f1_mean:.4f} ± {self.f1_std:.4f}",
                    f"  MAE:          {self.roc_auc_mean:.4f} ± {self.roc_auc_std:.4f}",
                ]
            )
        else:
            lines.extend(
                [
                    "Classification (held-out):",
                    f"  Accuracy:     {self.accuracy_mean:.4f} ± {self.accuracy_std:.4f}",
                    f"  F1:           {self.f1_mean:.4f} ± {self.f1_std:.4f}",
                    f"  Balanced Acc: {self.balanced_acc_mean:.4f} ± {self.balanced_acc_std:.4f}",
                    f"  ROC AUC:      {self.roc_auc_mean:.4f} ± {self.roc_auc_std:.4f}",
                ]
            )
        lines.extend(
            [
                "",
                "Structure stability:",
                f"  MB Jaccard:   {self.mb_jaccard_mean:.4f}",
                f"  Unique MBs:   {len(set(tuple(sorted(mb)) for mb in self.mb_per_fold))}/"
                f"{len(self.mb_per_fold)} folds",
            ]
        )
        if self.shd_per_fold is not None:
            shd_mean = np.mean(self.shd_per_fold)
            shd_std = np.std(self.shd_per_fold)
            lines.append(f"  SHD:          {shd_mean:.1f} ± {shd_std:.1f}")
        if self.mb_f1_per_fold is not None:
            mbf1_mean = np.mean(self.mb_f1_per_fold)
            mbf1_std = np.std(self.mb_f1_per_fold)
            lines.append(f"  MB F1:        {mbf1_mean:.4f} ± {mbf1_std:.4f}")
        if self.edge_f1_per_fold is not None:
            ef1_mean = np.mean(self.edge_f1_per_fold)
            ef1_std = np.std(self.edge_f1_per_fold)
            ep_mean = np.mean(self.edge_precision_per_fold) if self.edge_precision_per_fold else 0.0
            er_mean = np.mean(self.edge_recall_per_fold) if self.edge_recall_per_fold else 0.0
            lines.append(
                f"  Edge F1:      {ef1_mean:.4f} ± {ef1_std:.4f} (P={ep_mean:.4f}, R={er_mean:.4f})"
            )
        if self.total_time is not None:
            lines.append(f"\nTotal time: {self.total_time:.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


def _compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 1.0
    return len(set_a & set_b) / len(union)


def _compute_mb_jaccard_mean(mb_per_fold: List[List[int]]) -> float:
    """Compute average pairwise Jaccard similarity of MBs across folds."""
    n_folds = len(mb_per_fold)
    if n_folds < 2:
        return 1.0

    jaccard_scores = []
    for i, j in combinations(range(n_folds), 2):
        set_i = set(mb_per_fold[i])
        set_j = set(mb_per_fold[j])
        jaccard_scores.append(_compute_jaccard(set_i, set_j))

    return float(np.mean(jaccard_scores))


def _compute_edge_frequency(
    A_per_fold: List[np.ndarray],
    threshold: float = EDGE_THRESHOLD_DEFAULT,
) -> Dict[Tuple[int, int], float]:
    """Compute edge frequency across folds (fraction of folds an edge appears in)."""
    n_folds = len(A_per_fold)
    if n_folds == 0:
        return {}

    edge_counts: Dict[Tuple[int, int], int] = {}
    for A in A_per_fold:
        n = A.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and abs(A[i, j]) > threshold:
                    edge = (i, j)
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

    return {edge: count / n_folds for edge, count in edge_counts.items()}


def _compute_shd(
    A_est: np.ndarray, A_true: np.ndarray, threshold: float = EDGE_THRESHOLD_DEFAULT
) -> int:
    """Compute Structural Hamming Distance between estimated and true DAGs."""
    B_est = (np.abs(A_est) > threshold).astype(int)
    B_true = (np.abs(A_true) > threshold).astype(int)
    return int(np.sum(B_est != B_true))


def _compute_mb_f1(predicted_mb: List[int], true_mb: List[int]) -> float:
    """Compute F1 score for Markov Blanket recovery."""
    pred_set = set(predicted_mb)
    true_set = set(true_mb)

    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _evaluate_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    hyperparams: Dict,
    processor_type: str,
    A_init: Optional[np.ndarray],
    n_features: int,
    Y_idx: int,
    fold_key: jax.Array,
    proc_key: jax.Array,
    n_folds: int,
    golem_max_iter: int,
    cold_start: bool,
    task: str,
    true_dag: Optional[np.ndarray],
    true_mb: Optional[List[int]],
    verbose: bool,
    device: Optional[Any] = None,
    freeze_structure: bool = False,
    proc_params_init: Any = None,
    A_weights_continuous: Optional[np.ndarray] = None,
    A_confound_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single CV fold. Extracted for parallel dispatch.

    Returns a dict with fold results (metrics, MB, A_est, timing).
    Thread-safe when called with explicit device pinning.

    If freeze_structure=True, A is frozen during training (only processor
    weights are updated). This isolates predictive power evaluation from
    structural instability across folds.

    A_weights_continuous: Original continuous A_direct weights from full
        training. Used for prediction weighting to match training-time
        feature importance (learn_structure uses continuous |A[:,j]| with
        0.01 floor, not binary+L1-norm).
    A_confound_weights: Original continuous confound matrix for Y weighting.
    """
    from jcce.structure_learning.jcce_learner import (
        create_processor,
        extract_markov_blanket,
        learn_structure,
    )

    ctx = jax.default_device(device) if device is not None else _nullcontext()
    with ctx:
        fold_start = time.time()

        if verbose:
            print(
                f"\n  Fold {fold_idx + 1}/{n_folds}: train={len(train_idx)}, test={len(test_idx)}"
            )

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        X_train_jax = jnp.array(X_train)
        Y_train_jax = jnp.array(Y_train).reshape(-1, 1).astype(jnp.float32)
        X_test_jax = jnp.array(X_test)

        processor_config = hyperparams.get("processor_config", {})
        processor = create_processor(
            processor_type,
            key=proc_key,
            n_features=n_features,
            **processor_config,
        )

        if freeze_structure:
            # Fixed-structure CV: A is frozen, only retrain processor
            A_warm = jnp.array(A_init) if A_init is not None else None
            fold_max_iter = golem_max_iter
        elif cold_start:
            A_warm = None
            fold_max_iter = max(golem_max_iter, 150)
        else:
            A_warm = jnp.array(A_init) if A_init is not None else None
            fold_max_iter = golem_max_iter

        result = {
            "fold_idx": fold_idx,
            "accuracy": 0.0,
            "f1": 0.0,
            "balanced_acc": 0.5 if task == "classification" else 0.0,
            "roc_auc": 0.5 if task == "classification" else 0.0,
            "mb_indices": [],
            "A_est": None,
            "shd": None,
            "mb_f1": None,
            "fold_time": 0.0,
            "success": False,
        }

        try:
            A_est, processor_trained, proc_params, fold_metrics = learn_structure(
                data=X_train_jax,
                Y=Y_train_jax,
                Y_idx=Y_idx,
                processor=processor,
                key=fold_key,
                processor_type=processor_type,
                lambda_1=hyperparams.get("lambda_1", 0.02),
                lambda_2_init=hyperparams.get("lambda_2", 0.01),
                lambda_class=hyperparams.get("lambda_class", 1.0),
                lr=hyperparams.get("lr", 0.001),
                max_iter=fold_max_iter,
                patience=15,
                verbose=0,
                A_init=A_warm,
                effect_hidden_dim=hyperparams.get("effect_hidden_dim", 64),
                effect_embed_dim=hyperparams.get("effect_embed_dim", 16),
                lambda_effect=hyperparams.get("lambda_effect", 1.0),
                effect_warmup_iter=hyperparams.get("effect_warmup_iter", 10),
                lambda_confound_sparse=hyperparams.get("lambda_confound_sparse", 0.001),
                lambda_bow=hyperparams.get("lambda_bow", 0.01),
                n_latent_confounders=hyperparams.get("n_latent_confounders", 5),
                task=task,
                use_bf16=hyperparams.get("use_bf16", False),
                freeze_A=freeze_structure,
                proc_params_init=proc_params_init,
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            warnings.warn(f"Fold {fold_idx + 1} GOLEM training failed: {e}\n{tb}")
            result["fold_time"] = time.time() - fold_start
            return result

        # Extract MB
        mb_from_metrics = fold_metrics.get("markov_blanket", [])
        if mb_from_metrics:
            mb_indices = [idx for idx in mb_from_metrics if idx != Y_idx]
        else:
            mb_indices_arr = extract_markov_blanket(A_est, Y_idx, threshold=1e-6)
            mb_indices = [int(idx) for idx in mb_indices_arr if idx != Y_idx]

        result["mb_indices"] = mb_indices
        result["A_est"] = np.array(A_est)

        # Evaluate on test data
        # Use continuous A_weights (not binary A_est) to match training-time
        # weighting in learn_structure's loss_fn (line 1212-1228).
        # Training uses |A_direct[:, j]| with a 0.01 floor, NO L1-norm.
        # With freeze_structure=True, fold A_final = A_init = binary, so
        # we must use the original continuous weights passed from the caller.
        if A_weights_continuous is not None:
            A_for_pred = jnp.array(A_weights_continuous)
        else:
            # Fallback: try fold metrics, then binary A_est
            A_continuous = fold_metrics.get("A_weights")
            if A_continuous is not None:
                A_for_pred = jnp.array(A_continuous)
            else:
                A_for_pred = A_est

        weights_Y = jnp.abs(A_for_pred[:, Y_idx])

        # Add confound edge contribution (matches post-train eval, line 5420-5423)
        A_conf = None
        if A_confound_weights is not None:
            A_conf = jnp.array(A_confound_weights)
        else:
            A_conf_from_fold = fold_metrics.get("A_confound_weights")
            if A_conf_from_fold is not None:
                A_conf = jnp.array(A_conf_from_fold)
        if A_conf is not None:
            weights_Y = weights_Y + 0.5 * jnp.abs(A_conf[:, Y_idx])

        weights_Y = weights_Y.at[Y_idx].set(0.0)
        # 0.01 floor — matches training (line 1228), NOT L1-normalize
        weights_X = jnp.maximum(weights_Y[:n_features], 0.01)
        X_test_weighted = X_test_jax * weights_X[jnp.newaxis, :]

        try:
            proc_params_Y = proc_params[Y_idx]

            # GNN needs adjacency matrix (matches learn_structure line 5430)
            if processor_trained.__class__.__name__ == "GNNAdapter":
                A_feat = A_for_pred[:n_features, :n_features]
                A_norm = A_feat / (jnp.sum(jnp.abs(A_feat), axis=0, keepdims=True) + 1e-8)
                Y_pred_logits = processor_trained.forward(X_test_weighted, proc_params_Y, A=A_norm)
            else:
                Y_pred_logits = processor_trained.forward(X_test_weighted, proc_params_Y)

            if task == "regression":
                Y_pred_np = np.array(Y_pred_logits.flatten())
            else:
                Y_pred_prob = jax.nn.sigmoid(Y_pred_logits).flatten()
                Y_pred_binary = (Y_pred_prob > 0.5).astype(jnp.float32)
                Y_pred_binary_np = np.array(Y_pred_binary)
                Y_pred_prob_np = np.array(Y_pred_prob)

        except Exception as e:
            warnings.warn(f"Fold {fold_idx + 1} prediction failed: {e}")
            result["fold_time"] = time.time() - fold_start
            return result

        # Compute metrics
        Y_test_np = np.array(Y_test).flatten()

        if task == "regression":
            ss_res = float(np.sum((Y_test_np - Y_pred_np) ** 2))
            ss_tot = float(np.sum((Y_test_np - np.mean(Y_test_np)) ** 2))
            r2_k = 1.0 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0.0
            rmse_k = float(np.sqrt(np.mean((Y_test_np - Y_pred_np) ** 2)))
            mae_k = float(np.mean(np.abs(Y_test_np - Y_pred_np)))
            result["accuracy"] = r2_k
            result["f1"] = rmse_k
            result["balanced_acc"] = r2_k
            result["roc_auc"] = mae_k
        else:
            acc_k = float(np.mean(Y_pred_binary_np == Y_test_np))
            result["accuracy"] = acc_k

            tp = np.sum((Y_pred_binary_np == 1) & (Y_test_np == 1))
            tn = np.sum((Y_pred_binary_np == 0) & (Y_test_np == 0))
            fp = np.sum((Y_pred_binary_np == 1) & (Y_test_np == 0))
            fn = np.sum((Y_pred_binary_np == 0) & (Y_test_np == 1))

            precision_k = tp / (tp + fp + 1e-10)
            recall_k = tp / (tp + fn + 1e-10)
            f1_k = 2 * precision_k * recall_k / (precision_k + recall_k + 1e-10)
            specificity_k = tn / (tn + fp + 1e-10)
            balanced_acc_k = (recall_k + specificity_k) / 2
            result["precision"] = float(precision_k)
            result["recall"] = float(recall_k)
            result["f1"] = float(f1_k)
            result["balanced_acc"] = float(balanced_acc_k)

            try:
                from sklearn.metrics import roc_auc_score

                if len(np.unique(Y_test_np)) > 1:
                    roc_auc_k = roc_auc_score(Y_test_np, Y_pred_prob_np)
                else:
                    roc_auc_k = 0.5
            except Exception:
                roc_auc_k = 0.5
            result["roc_auc"] = float(roc_auc_k)

        # Structure metrics
        if true_dag is not None:
            # A_est from learn_structure is augmented (n_features+1 x n_features+1);
            # true_dag is feature-only (n_features x n_features). Slice to match.
            A_est_np = np.array(A_est)
            if A_est_np.shape[0] > true_dag.shape[0]:
                A_est_np = A_est_np[: true_dag.shape[0], : true_dag.shape[1]]
            result["shd"] = _compute_shd(A_est_np, true_dag)
            from jcce.training.metrics import graph_precision_recall_f1

            prf = graph_precision_recall_f1(A_est_np, true_dag)
            result["edge_f1"] = prf["f1"]
            result["edge_precision"] = prf["precision"]
            result["edge_recall"] = prf["recall"]
        if true_mb is not None:
            result["mb_f1"] = _compute_mb_f1(mb_indices, true_mb)

        result["fold_time"] = time.time() - fold_start
        result["success"] = True

        if verbose:
            if task == "regression":
                print(
                    f"    R²={result['accuracy']:.4f}, RMSE={result['f1']:.4f}, "
                    f"MAE={result['roc_auc']:.4f}, MB={mb_indices}, "
                    f"time={result['fold_time']:.1f}s"
                )
            else:
                print(
                    f"    Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, "
                    f"BAcc={result['balanced_acc']:.4f}, AUC={result['roc_auc']:.4f}, "
                    f"MB={mb_indices}, time={result['fold_time']:.1f}s"
                )

    # Cleanup after fold
    jax.clear_caches()
    gc.collect()

    return result


class _nullcontext:
    """Minimal no-op context manager (Python 3.7+ compatible)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def evaluate_pareto_solution_cv(
    X: np.ndarray,
    Y: np.ndarray,
    hyperparams: Dict,
    processor_type: str,
    A_init: np.ndarray,
    processor_params_init: Any = None,
    n_folds: int = 5,
    Y_idx: int = -1,
    feature_names: Optional[List[str]] = None,
    true_mb: Optional[List[int]] = None,
    true_dag: Optional[np.ndarray] = None,
    golem_max_iter: int = 50,
    cold_start: bool = False,
    task: str = "classification",
    verbose: bool = False,
    parallel_folds: bool = False,
    freeze_structure: bool = False,
    A_weights_continuous: Optional[np.ndarray] = None,
    A_confound_weights: Optional[np.ndarray] = None,
) -> UnifiedCVResult:
    """
    Evaluate a Pareto solution using K-fold CV with the unified fθ model.

    For each fold:
    1. Retrain GOLEM (warm-start or cold-start) on fold's training data
    2. Extract Markov Blanket from fold's learned adjacency matrix
    3. Evaluate fθ on held-out test data
    4. Compute classification/regression metrics on held-out predictions

    Args:
        X: Full dataset features (n_samples, n_features)
        Y: Full dataset labels/values (n_samples,)
        hyperparams: From NSGA-II Pareto solution. Keys:
            - lambda_1, lambda_2, lambda_class, lr (required)
            - processor_config (optional, architecture-specific)
            - effect_hidden_dim, effect_embed_dim, lambda_effect, etc. (optional)
        processor_type: 'mlp', 'transformer', 'mamba', 'gnn', or 'elm'
        A_init: Full-data adjacency matrix for warm-starting (n_vars+1, n_vars+1)
        processor_params_init: Full-data processor params for warm-starting (optional)
        n_folds: Number of CV folds (K=5 default, K=3 for small datasets)
        Y_idx: Target variable index in augmented space (-1 = last column = n_features)
        feature_names: Optional feature names for reporting
        true_mb: Ground-truth MB indices for MB F1 (LUCAS only)
        true_dag: Ground-truth DAG for SHD computation (LUCAS only)
        golem_max_iter: Max GOLEM iterations per fold (50 for warm-start, 150 for cold-start)
        cold_start: If True, retrain from random init per fold (unbiased but slower).
                    If False, warm-start from full-data A (faster but mildly biased).
        verbose: Print progress
        parallel_folds: If True and >=2 GPUs detected, distribute folds across
                       GPUs via ThreadPoolExecutor. Falls back to sequential on CPU
                       or single GPU.
        freeze_structure: If True, freeze A during per-fold training (only retrain
                         processor fθ). Isolates predictive evaluation from structural
                         instability. Recommended for publishable results.

    Returns:
        UnifiedCVResult with per-fold and aggregated metrics
    """

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y).flatten()
    n_samples, n_features = X.shape

    # Resolve Y_idx
    if Y_idx == -1:
        Y_idx = n_features  # Y is appended as last column in augmented space

    # A_init should be (n_features+1 × n_features+1) — augmented space including Y.
    # learn_structure() works in augmented space internally.

    # StratifiedKFold for classification, regular KFold for regression
    if task == "regression":
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Per-fold storage
    accuracy_per_fold = []
    precision_per_fold = []
    recall_per_fold = []
    f1_per_fold = []
    balanced_acc_per_fold = []
    roc_auc_per_fold = []
    mb_per_fold = []
    A_per_fold = []
    shd_per_fold = [] if true_dag is not None else None
    mb_f1_per_fold = [] if true_mb is not None else None
    edge_f1_per_fold = [] if true_dag is not None else None
    edge_precision_per_fold = [] if true_dag is not None else None
    edge_recall_per_fold = [] if true_dag is not None else None
    fold_times = []

    total_start = time.time()
    key = random.PRNGKey(42)

    # Pre-split all PRNG keys before dispatching (JAX PRNGs not thread-safe to split concurrently)
    split_args = (X, Y) if task == "classification" else (X,)
    fold_splits = list(kfold.split(*split_args))
    fold_keys = []
    proc_keys = []
    for _ in range(n_folds):
        key, proc_key = random.split(key)
        key, fold_key = random.split(key)
        proc_keys.append(proc_key)
        fold_keys.append(fold_key)

    # Determine parallel execution mode
    use_parallel = False
    gpu_devices = []
    if parallel_folds:
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        use_parallel = len(gpu_devices) >= 2

    if use_parallel and verbose:
        print(f"  Parallel CV: {n_folds} folds across {len(gpu_devices)} GPUs")
    elif parallel_folds and verbose:
        print("  Parallel CV requested but <2 GPUs detected, falling back to sequential")

    # Common kwargs for _evaluate_single_fold
    common_kwargs = dict(
        X=X,
        Y=Y,
        hyperparams=hyperparams,
        processor_type=processor_type,
        A_init=A_init,
        n_features=n_features,
        Y_idx=Y_idx,
        n_folds=n_folds,
        golem_max_iter=golem_max_iter,
        cold_start=cold_start,
        task=task,
        true_dag=true_dag,
        true_mb=true_mb,
        verbose=verbose,
        freeze_structure=freeze_structure,
        proc_params_init=processor_params_init,
        A_weights_continuous=A_weights_continuous,
        A_confound_weights=A_confound_weights,
    )

    fold_results = [None] * n_folds

    if use_parallel:
        # Parallel execution: distribute folds across GPUs via ThreadPoolExecutor
        max_workers = min(len(gpu_devices), n_folds)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                device = gpu_devices[fold_idx % len(gpu_devices)]
                future = executor.submit(
                    _evaluate_single_fold,
                    fold_idx=fold_idx,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    fold_key=fold_keys[fold_idx],
                    proc_key=proc_keys[fold_idx],
                    device=device,
                    **common_kwargs,
                )
                futures[future] = fold_idx

            for future in as_completed(futures):
                fidx = futures[future]
                try:
                    fold_results[fidx] = future.result()
                except Exception as e:
                    warnings.warn(f"Fold {fidx + 1} parallel execution failed: {e}")
                    fold_results[fidx] = {
                        "fold_idx": fidx,
                        "accuracy": 0.0,
                        "f1": 0.0,
                        "balanced_acc": 0.5 if task == "classification" else 0.0,
                        "roc_auc": 0.5 if task == "classification" else 0.0,
                        "mb_indices": [],
                        "A_est": None,
                        "shd": None,
                        "mb_f1": None,
                        "fold_time": 0.0,
                        "success": False,
                    }
    else:
        # Sequential execution (default)
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            fold_results[fold_idx] = _evaluate_single_fold(
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                fold_key=fold_keys[fold_idx],
                proc_key=proc_keys[fold_idx],
                device=None,
                **common_kwargs,
            )

    # Collect results from fold dicts into per-fold lists
    for fr in fold_results:
        accuracy_per_fold.append(fr["accuracy"])
        precision_per_fold.append(fr.get("precision", 0.0))
        recall_per_fold.append(fr.get("recall", 0.0))
        f1_per_fold.append(fr["f1"])
        balanced_acc_per_fold.append(fr["balanced_acc"])
        roc_auc_per_fold.append(fr["roc_auc"])
        mb_per_fold.append(fr["mb_indices"])
        if fr["A_est"] is not None:
            A_per_fold.append(fr["A_est"])
        fold_times.append(fr["fold_time"])

        if shd_per_fold is not None and fr["shd"] is not None:
            shd_per_fold.append(fr["shd"])
        if mb_f1_per_fold is not None and fr["mb_f1"] is not None:
            mb_f1_per_fold.append(fr["mb_f1"])
        if edge_f1_per_fold is not None and fr.get("edge_f1") is not None:
            edge_f1_per_fold.append(fr["edge_f1"])
        if edge_precision_per_fold is not None and fr.get("edge_precision") is not None:
            edge_precision_per_fold.append(fr["edge_precision"])
        if edge_recall_per_fold is not None and fr.get("edge_recall") is not None:
            edge_recall_per_fold.append(fr["edge_recall"])

    total_time = time.time() - total_start

    # Compute aggregated metrics
    accuracy_mean = float(np.mean(accuracy_per_fold)) if accuracy_per_fold else 0.0
    accuracy_std = float(np.std(accuracy_per_fold)) if accuracy_per_fold else 0.0
    precision_mean = float(np.mean(precision_per_fold)) if precision_per_fold else 0.0
    precision_std = float(np.std(precision_per_fold)) if precision_per_fold else 0.0
    recall_mean = float(np.mean(recall_per_fold)) if recall_per_fold else 0.0
    recall_std = float(np.std(recall_per_fold)) if recall_per_fold else 0.0
    f1_mean = float(np.mean(f1_per_fold)) if f1_per_fold else 0.0
    f1_std = float(np.std(f1_per_fold)) if f1_per_fold else 0.0
    balanced_acc_mean = float(np.mean(balanced_acc_per_fold)) if balanced_acc_per_fold else 0.0
    balanced_acc_std = float(np.std(balanced_acc_per_fold)) if balanced_acc_per_fold else 0.0
    roc_auc_mean = float(np.mean(roc_auc_per_fold)) if roc_auc_per_fold else 0.0
    roc_auc_std = float(np.std(roc_auc_per_fold)) if roc_auc_per_fold else 0.0

    # Compute structure stability
    mb_jaccard_mean = _compute_mb_jaccard_mean(mb_per_fold)
    edge_frequency = _compute_edge_frequency(A_per_fold) if A_per_fold else {}

    # Build GOLEM config for traceability
    golem_config = {
        "lambda_1": hyperparams.get("lambda_1"),
        "lambda_2": hyperparams.get("lambda_2"),
        "lambda_class": hyperparams.get("lambda_class"),
        "lr": hyperparams.get("lr"),
    }

    result = UnifiedCVResult(
        accuracy_per_fold=accuracy_per_fold,
        precision_per_fold=precision_per_fold,
        recall_per_fold=recall_per_fold,
        f1_per_fold=f1_per_fold,
        balanced_acc_per_fold=balanced_acc_per_fold,
        roc_auc_per_fold=roc_auc_per_fold,
        accuracy_mean=accuracy_mean,
        accuracy_std=accuracy_std,
        precision_mean=precision_mean,
        precision_std=precision_std,
        recall_mean=recall_mean,
        recall_std=recall_std,
        f1_mean=f1_mean,
        f1_std=f1_std,
        balanced_acc_mean=balanced_acc_mean,
        balanced_acc_std=balanced_acc_std,
        roc_auc_mean=roc_auc_mean,
        roc_auc_std=roc_auc_std,
        mb_per_fold=mb_per_fold,
        mb_jaccard_mean=mb_jaccard_mean,
        edge_frequency=edge_frequency,
        processor_type=processor_type,
        processor_config=hyperparams.get("processor_config"),
        golem_config=golem_config,
        task=task,
        shd_per_fold=shd_per_fold,
        mb_f1_per_fold=mb_f1_per_fold,
        edge_f1_per_fold=edge_f1_per_fold if edge_f1_per_fold else None,
        edge_precision_per_fold=edge_precision_per_fold if edge_precision_per_fold else None,
        edge_recall_per_fold=edge_recall_per_fold if edge_recall_per_fold else None,
        fold_times=fold_times,
        total_time=total_time,
    )

    if verbose:
        print(f"\n{result.summary()}")

    return result


def evaluate_fixed_structure_cv(
    X: np.ndarray,
    Y: np.ndarray,
    hyperparams: Dict,
    processor_type: str,
    A_frozen: np.ndarray,
    processor_params_init: Any = None,
    n_folds: int = 5,
    Y_idx: int = -1,
    true_mb: Optional[List[int]] = None,
    true_dag: Optional[np.ndarray] = None,
    golem_max_iter: int = 50,
    task: str = "classification",
    verbose: bool = False,
    parallel_folds: bool = False,
) -> UnifiedCVResult:
    """
    Fixed-structure K-fold CV: freeze A, retrain only the processor per fold.

    This is the recommended CV approach. It isolates
    predictive utility from structural instability by keeping the adjacency
    matrix fixed across all folds. Only the processor (fθ) is retrained on
    each fold's training data, warm-started from full-data trained params.

    Equivalent to evaluate_pareto_solution_cv(..., freeze_structure=True).

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Target variable (n_samples,)
        hyperparams: Hyperparameter dict from Pareto solution
        processor_type: Processor architecture name
        A_frozen: Adjacency matrix from full-data training (frozen per fold)
        processor_params_init: Trained processor params for warm-starting folds
        n_folds: Number of CV folds
        Y_idx: Target variable index (-1 = last)
        true_mb: Ground truth Markov blanket (optional)
        true_dag: Ground truth DAG (optional)
        golem_max_iter: Iterations per fold (processor training only)
        task: 'classification' or 'regression'
        verbose: Print progress
        parallel_folds: Distribute folds across GPUs

    Returns:
        UnifiedCVResult with per-fold and aggregated metrics
    """
    return evaluate_pareto_solution_cv(
        X=X,
        Y=Y,
        hyperparams=hyperparams,
        processor_type=processor_type,
        A_init=A_frozen,
        processor_params_init=processor_params_init,
        n_folds=n_folds,
        Y_idx=Y_idx,
        true_mb=true_mb,
        true_dag=true_dag,
        golem_max_iter=golem_max_iter,
        cold_start=False,  # Must warm-start from A_frozen
        task=task,
        verbose=verbose,
        parallel_folds=parallel_folds,
        freeze_structure=True,
    )
