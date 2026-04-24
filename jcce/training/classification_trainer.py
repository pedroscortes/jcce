"""
JAX-based Classification Training for C-Lite.

Replaces sklearn's cross_val_score with JAX training loop.
Supports all processor types from classification_processors.py.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax import random
from sklearn.model_selection import StratifiedKFold

from jcce.models.classification_processors import create_classifier

# ============================================================================
# Training State
# ============================================================================


class ClassifierTrainState(train_state.TrainState):
    """Training state for classifier."""

    pass


# ============================================================================
# Loss Functions
# ============================================================================


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """
    Cross-entropy loss for classification.

    Args:
        logits: (batch_size, num_classes) predicted logits
        labels: (batch_size,) true class labels (integers)
        num_classes: Number of classes

    Returns:
        loss: Scalar loss value
    """
    # One-hot encode labels
    labels_onehot = jax.nn.one_hot(labels, num_classes)

    # Cross-entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.mean(jnp.sum(labels_onehot * log_probs, axis=-1))

    return loss


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        logits: (batch_size, num_classes) predicted logits
        labels: (batch_size,) true labels

    Returns:
        accuracy: Scalar accuracy in [0, 1]
    """
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return float(accuracy)


# ============================================================================
# Training Functions
# ============================================================================


def train_step(
    state: ClassifierTrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    num_classes: int,
    rng: random.PRNGKey,
) -> Tuple[ClassifierTrainState, Dict[str, float], random.PRNGKey]:
    """
    Single training step.

    Args:
        state: Training state
        batch: (features, labels) tuple
        num_classes: Number of classes (NOT jitted - kept as static)
        rng: Random key for dropout

    Returns:
        new_state: Updated training state
        metrics: Dictionary with loss and accuracy
        new_rng: Updated random key
    """
    features, labels = batch
    rng, dropout_rng = random.split(rng)

    def loss_fn(params):
        # Pass dropout RNG to model
        logits = state.apply_fn(
            {"params": params}, features, training=True, rngs={"dropout": dropout_rng}
        )
        loss = cross_entropy_loss(logits, labels, num_classes)
        return loss, logits

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Update parameters
    new_state = state.apply_gradients(grads=grads)

    # Compute metrics
    accuracy = compute_accuracy(logits, labels)

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
    }

    return new_state, metrics, rng


def eval_step(
    state: ClassifierTrainState, batch: Tuple[jnp.ndarray, jnp.ndarray], num_classes: int
) -> Dict[str, float]:
    """
    Single evaluation step.

    Args:
        state: Training state
        batch: (features, labels) tuple
        num_classes: Number of classes (NOT jitted - kept as static)

    Returns:
        metrics: Dictionary with loss and accuracy
    """
    features, labels = batch

    # No dropout in eval mode, so no RNG needed
    logits = state.apply_fn({"params": state.params}, features, training=False)
    loss = cross_entropy_loss(logits, labels, num_classes)
    accuracy = compute_accuracy(logits, labels)

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
    }

    return metrics


# ============================================================================
# Training Loop
# ============================================================================


def train_classifier(
    model: Any,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_val: Optional[jnp.ndarray] = None,
    Y_val: Optional[jnp.ndarray] = None,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    batch_size: int = 32,
    num_classes: Optional[int] = None,
    key: Optional[random.PRNGKey] = None,
    verbose: bool = False,
) -> Tuple[ClassifierTrainState, Dict[str, Any]]:
    """
    Train a classifier model.

    Args:
        model: Flax module (from classification_processors.py)
        X_train: (n_train, n_features) training features
        Y_train: (n_train,) training labels
        X_val: (n_val, n_features) validation features (optional)
        Y_val: (n_val,) validation labels (optional)
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_classes: Number of classes (auto-detected if None)
        key: JAX random key
        verbose: Print progress

    Returns:
        state: Final training state
        history: Training history dictionary
    """
    if key is None:
        key = random.PRNGKey(0)

    # Auto-detect number of classes
    if num_classes is None:
        num_classes = int(jnp.max(Y_train) + 1)

    # Initialize model
    key, init_key = random.split(key)
    dummy_input = jnp.ones((1, X_train.shape[1]))
    variables = model.init(init_key, dummy_input, training=False)

    # Create optimizer and training state
    tx = optax.adam(learning_rate)
    state = ClassifierTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Number of batches
    n_train = X_train.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    # Training loop
    for epoch in range(num_epochs):
        # Shuffle training data
        key, shuffle_key = random.split(key)
        perm = random.permutation(shuffle_key, n_train)
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]

        # Train on batches
        train_metrics = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)

            X_batch = X_train_shuffled[start_idx:end_idx]
            Y_batch = Y_train_shuffled[start_idx:end_idx]

            # Pass RNG to train_step and get updated RNG back
            key, step_key = random.split(key)
            state, metrics, _ = train_step(state, (X_batch, Y_batch), num_classes, step_key)
            train_metrics.append(metrics)

        # Average training metrics
        epoch_train_loss = float(np.mean([m["loss"] for m in train_metrics]))
        epoch_train_acc = float(np.mean([m["accuracy"] for m in train_metrics]))

        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_acc)

        # Validation
        if X_val is not None and Y_val is not None:
            val_metrics = eval_step(state, (X_val, Y_val), num_classes)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={epoch_train_acc:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )
        else:
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={epoch_train_acc:.4f}"
                )

    return state, history


# ============================================================================
# Cross-Validation
# ============================================================================


def cross_validate_classifier(
    processor_type: str,
    processor_config: Dict[str, Any],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    cv_folds: int = 5,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    batch_size: int = 32,
    key: Optional[random.PRNGKey] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Cross-validate a classifier using K-fold CV.

    Args:
        processor_type: Type of processor ('mlp', 'mamba', 'transformer', etc.)
        processor_config: Configuration dict for processor
        X: (n_samples, n_features) features
        Y: (n_samples,) labels
        cv_folds: Number of CV folds
        learning_rate: Learning rate
        num_epochs: Training epochs per fold
        batch_size: Batch size
        key: JAX random key
        verbose: Print progress

    Returns:
        results: Dictionary with CV scores and metrics
    """
    if key is None:
        key = random.PRNGKey(0)

    # Convert to numpy for sklearn KFold
    X_np = np.array(X)
    Y_np = np.array(Y)

    # Detect number of classes
    num_classes = int(np.max(Y_np) + 1)
    n_features = X.shape[1]

    # Use stratified K-fold for classification
    if num_classes == 2:
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        # For multiclass, also use stratified
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Cross-validation
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_np, Y_np)):
        if verbose:
            print(f"\n  CV Fold {fold_idx + 1}/{cv_folds}")

        # Split data
        X_train_fold = jnp.array(X_np[train_idx])
        Y_train_fold = jnp.array(Y_np[train_idx])
        X_val_fold = jnp.array(X_np[val_idx])
        Y_val_fold = jnp.array(Y_np[val_idx])

        # Create model
        model = create_classifier(processor_type, processor_config, n_features, num_classes)

        # Train
        key, train_key = random.split(key)
        state, history = train_classifier(
            model,
            X_train_fold,
            Y_train_fold,
            X_val_fold,
            Y_val_fold,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_classes=num_classes,
            key=train_key,
            verbose=False,  # Suppress per-epoch printing in CV
        )

        # Evaluate on validation fold
        val_metrics = eval_step(state, (X_val_fold, Y_val_fold), num_classes)
        fold_scores.append(val_metrics["accuracy"])

        if verbose:
            print(f"    Fold {fold_idx + 1} accuracy: {val_metrics['accuracy']:.4f}")

    # Aggregate results
    mean_accuracy = float(np.mean(fold_scores))
    std_accuracy = float(np.std(fold_scores))

    if verbose:
        print(f"\n  CV Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

    results = {
        "cv_scores": fold_scores,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "processor_type": processor_type,
    }

    return results


# ============================================================================
# Nested Cross-Validation (for unbiased hyperparameter tuning)
# ============================================================================


def nested_cross_validate_classifier(
    processor_type: str,
    processor_config: Dict[str, Any],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    outer_cv_folds: int = 5,
    inner_cv_folds: int = 3,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    batch_size: int = 32,
    key: Optional[random.PRNGKey] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Nested cross-validation for unbiased performance estimation.

    Outer CV: Evaluation (unbiased estimate)
    Inner CV: Hyperparameter tuning (per outer fold)

    This prevents overfitting to the validation folds during hyperparameter selection.

    Args:
        processor_type: Type of processor ('mlp', 'mamba', 'transformer', etc.)
        processor_config: Configuration dict for processor
        X: (n_samples, n_features) features
        Y: (n_samples,) labels
        outer_cv_folds: Number of outer CV folds (evaluation)
        inner_cv_folds: Number of inner CV folds (tuning)
        learning_rate: Learning rate
        num_epochs: Training epochs per fold
        batch_size: Batch size
        key: JAX random key
        verbose: Print progress

    Returns:
        results: Dictionary with nested CV scores and metrics
    """
    if key is None:
        key = random.PRNGKey(0)

    # Convert to numpy for sklearn KFold
    X_np = np.array(X)
    Y_np = np.array(Y)

    # Detect number of classes
    num_classes = int(np.max(Y_np) + 1)
    n_features = X.shape[1]

    # Outer CV: Stratified K-fold for evaluation
    outer_kfold = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)

    outer_scores = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"NESTED CV: {outer_cv_folds} outer folds × {inner_cv_folds} inner folds")
        print(f"{'=' * 60}")

    # Outer loop: Evaluation
    for outer_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_kfold.split(X_np, Y_np)):
        if verbose:
            print(f"\n[Outer Fold {outer_idx + 1}/{outer_cv_folds}]")

        # Split into outer train/test
        X_train_outer = X_np[train_outer_idx]
        Y_train_outer = Y_np[train_outer_idx]
        X_test_outer = X_np[test_outer_idx]
        Y_test_outer = Y_np[test_outer_idx]

        # Inner CV: Hyperparameter tuning on outer training set
        # For now, we just use the given config (no actual tuning)
        # This could be extended to try different learning rates, epochs, etc.
        if verbose:
            print(f"  Running inner CV ({inner_cv_folds} folds) for hyperparameter validation...")

        key, inner_key = random.split(key)
        inner_results = cross_validate_classifier(
            processor_type=processor_type,
            processor_config=processor_config,
            X=jnp.array(X_train_outer),
            Y=jnp.array(Y_train_outer),
            cv_folds=inner_cv_folds,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            key=inner_key,
            verbose=False,
        )

        if verbose:
            print(f"  Inner CV mean accuracy: {inner_results['mean_accuracy']:.4f}")

        # Train final model on entire outer training set
        if verbose:
            print("  Training final model on outer training set...")

        X_train_outer_jax = jnp.array(X_train_outer)
        Y_train_outer_jax = jnp.array(Y_train_outer)

        model = create_classifier(processor_type, processor_config, n_features, num_classes)

        key, train_key = random.split(key)
        state, history = train_classifier(
            model,
            X_train_outer_jax,
            Y_train_outer_jax,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_classes=num_classes,
            key=train_key,
            verbose=False,
        )

        # Evaluate on outer test set (unbiased estimate)
        X_test_outer_jax = jnp.array(X_test_outer)
        Y_test_outer_jax = jnp.array(Y_test_outer)

        test_metrics = eval_step(state, (X_test_outer_jax, Y_test_outer_jax), num_classes)
        outer_scores.append(test_metrics["accuracy"])

        if verbose:
            print(f"  Outer test accuracy: {test_metrics['accuracy']:.4f}")

    # Aggregate outer scores
    mean_accuracy = float(np.mean(outer_scores))
    std_accuracy = float(np.std(outer_scores))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"NESTED CV RESULT: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"{'=' * 60}")

    results = {
        "outer_scores": outer_scores,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "processor_type": processor_type,
        "nested_cv": True,
        "outer_folds": outer_cv_folds,
        "inner_folds": inner_cv_folds,
    }

    return results


# ============================================================================
# Simple Evaluate Function
# ============================================================================


def evaluate_classifier(
    state: ClassifierTrainState,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """
    Evaluate classifier on test data.

    Args:
        state: Trained classifier state
        X: (n_samples, n_features) features
        Y: (n_samples,) labels
        num_classes: Number of classes

    Returns:
        metrics: Dictionary with loss and accuracy
    """
    metrics = eval_step(state, (X, Y), num_classes)
    return metrics
