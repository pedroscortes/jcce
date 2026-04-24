"""
Standalone classifier using JCCE's processor architectures.

Wraps the same JAX processors (MLP, Transformer, Mamba, ELM, GNN)
used in JCCE's structure learning loop, but trains them as simple
supervised classifiers. This ensures a fair comparison: same
architecture, different feature selection method.

Usage:
    from jcce.baselines.classifier import train_and_evaluate_processor

    results = train_and_evaluate_processor(
        X, Y, processor_type='mlp', n_splits=5, seed=42
    )
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import Dict, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, accuracy_score,
)

from jcce.structure_learning.jcce_learner import create_processor


def _is_jax_array(x):
    """Check if x is a JAX/numpy array (trainable parameter)."""
    return isinstance(x, (jnp.ndarray, np.ndarray)) and hasattr(x, 'shape') and x.shape != ()


def _split_params(params):
    """Split params into trainable arrays and static metadata."""
    trainable = {}
    static = {}
    for k, v in params.items():
        if isinstance(v, (jnp.ndarray,)) and hasattr(v, 'shape'):
            trainable[k] = v
        elif isinstance(v, np.ndarray):
            trainable[k] = jnp.array(v)
        else:
            static[k] = v
    return trainable, static


def _merge_params(trainable, static):
    """Merge trainable and static back into full params dict."""
    params = {}
    params.update(trainable)
    params.update(static)
    return params


def _train_classifier(
    processor,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    key: random.PRNGKey,
    lr: float = 1e-3,
    n_epochs: int = 200,
    batch_size: int = 256,
    patience: int = 20,
) -> Dict:
    """Train a processor as a binary classifier via BCE loss."""
    n_train, n_features = X_train.shape
    X_jax = jnp.array(np.asarray(X_train))
    Y_jax = jnp.array(np.asarray(Y_train).ravel(), dtype=jnp.float32)

    # Initialize params and split into trainable vs static
    key, init_key = random.split(key)
    full_params = processor.init_params(n_features)
    trainable, static = _split_params(full_params)

    # Detect which kwargs the processor's forward() accepts
    import inspect
    _fwd_sig = inspect.signature(processor.forward)
    _fwd_accepts_training = 'training' in _fwd_sig.parameters

    # BCE loss — only takes trainable params as first arg (for grad)
    def loss_fn(trainable_params, X_batch, Y_batch):
        params = _merge_params(trainable_params, static)
        if _fwd_accepts_training:
            logits = processor.forward(X_batch, params, training=False, skip_centering=True)
        else:
            logits = processor.forward(X_batch, params, skip_centering=True)
        logits = jnp.clip(logits, -6, 6)
        bce = -jnp.mean(
            Y_batch * jax.nn.log_sigmoid(logits) +
            (1 - Y_batch) * jax.nn.log_sigmoid(-logits)
        )
        # L2 reg on weight matrices
        l2 = sum(jnp.sum(p ** 2) for p in jax.tree.leaves(trainable_params)
                 if hasattr(p, 'ndim') and p.ndim >= 2)
        return bce + 1e-4 * l2

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Adam state — only for trainable params
    m = jax.tree.map(jnp.zeros_like, trainable)
    v = jax.tree.map(jnp.zeros_like, trainable)

    best_loss = float('inf')
    best_trainable = trainable
    patience_counter = 0

    for epoch in range(n_epochs):
        key, batch_key = random.split(key)
        if n_train > batch_size:
            idx = random.permutation(batch_key, n_train)[:batch_size]
            X_b, Y_b = X_jax[idx], Y_jax[idx]
        else:
            X_b, Y_b = X_jax, Y_jax

        loss_val, grads = grad_fn(trainable, X_b, Y_b)

        # Adam update
        t = epoch + 1
        m = jax.tree.map(lambda mi, g: 0.9 * mi + 0.1 * g, m, grads)
        v = jax.tree.map(lambda vi, g: 0.999 * vi + 0.001 * g ** 2, v, grads)
        m_hat = jax.tree.map(lambda mi: mi / (1 - 0.9 ** t), m)
        v_hat = jax.tree.map(lambda vi: vi / (1 - 0.999 ** t), v)
        trainable = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + 1e-8),
            trainable, m_hat, v_hat,
        )

        lv = float(loss_val)
        if lv < best_loss:
            best_loss = lv
            best_trainable = jax.tree.map(lambda x: x.copy(), trainable)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return _merge_params(best_trainable, static)


def _predict(processor, params, X: np.ndarray) -> tuple:
    """Get predictions and probabilities from trained processor."""
    import inspect
    X_jax = jnp.array(np.asarray(X))
    if 'training' in inspect.signature(processor.forward).parameters:
        logits = processor.forward(X_jax, params, training=False, skip_centering=True)
    else:
        logits = processor.forward(X_jax, params, skip_centering=True)
    probs = jax.nn.sigmoid(logits)
    preds = (probs > 0.5).astype(jnp.int32)
    return np.array(preds), np.array(probs)


def train_and_evaluate_processor(
    X: np.ndarray,
    Y: np.ndarray,
    processor_type: str = 'mlp',
    n_splits: int = 5,
    seed: int = 42,
    lr: float = 1e-3,
    n_epochs: int = 200,
    processor_kwargs: Optional[Dict] = None,
) -> Dict:
    """Train and evaluate a JCCE processor as a standalone classifier.

    Uses stratified k-fold CV with standardization per fold.

    Args:
        X: (n, d) features
        Y: (n,) binary labels
        processor_type: 'mlp', 'transformer', 'mamba', 'elm', 'gnn'
        n_splits: Number of CV folds
        seed: Random seed
        lr: Learning rate
        n_epochs: Max training epochs
        processor_kwargs: Additional processor config (hidden_dim, etc.)

    Returns:
        Dict with mean/std of each metric across folds, plus per_fold details
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    key = random.PRNGKey(seed)
    kwargs = processor_kwargs or {}

    per_fold = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Standardize per fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        Y_val_int = Y_val.astype(int)

        # Create processor
        key, proc_key, train_key = random.split(key, 3)
        processor = create_processor(processor_type, proc_key, **kwargs)

        # Train
        params = _train_classifier(
            processor, X_train, Y_train, train_key,
            lr=lr, n_epochs=n_epochs,
        )

        # Predict
        y_pred, y_prob = _predict(processor, params, X_val)

        # Metrics
        fold_metrics = {
            'fold': fold_i,
            'accuracy': float(accuracy_score(Y_val_int, y_pred)),
            'balanced_acc': float(balanced_accuracy_score(Y_val_int, y_pred)),
            'f1': float(f1_score(Y_val_int, y_pred, average='macro', zero_division=0)),
            'precision': float(precision_score(Y_val_int, y_pred, average='macro', zero_division=0)),
            'recall': float(recall_score(Y_val_int, y_pred, average='macro', zero_division=0)),
        }
        try:
            fold_metrics['roc_auc'] = float(roc_auc_score(Y_val_int, y_prob))
        except ValueError:
            fold_metrics['roc_auc'] = float('nan')

        per_fold.append(fold_metrics)

    # Aggregate
    metric_keys = ['accuracy', 'balanced_acc', 'f1', 'precision', 'recall', 'roc_auc']
    summary = {'processor_type': processor_type}
    for mk in metric_keys:
        vals = [f[mk] for f in per_fold if not np.isnan(f[mk])]
        if vals:
            summary[f'{mk}_mean'] = float(np.mean(vals))
            summary[f'{mk}_std'] = float(np.std(vals))
        else:
            summary[f'{mk}_mean'] = float('nan')
            summary[f'{mk}_std'] = float('nan')
    summary['per_fold'] = per_fold

    return summary
