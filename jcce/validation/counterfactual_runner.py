"""
Shared counterfactual evaluation utilities.

Extracted from scripts/run_v16_ablation.py so both NSGA-II and Optuna
pipelines can run CF evaluation without code duplication.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np


def convert_params_to_numpy(processor_params: list) -> list:
    """Recursively convert JAX arrays to numpy in processor params, preserving structure."""

    def convert_item(x):
        if hasattr(x, "device"):  # JAX DeviceArray
            return np.array(x)
        elif isinstance(x, dict):
            return {k: convert_item(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [convert_item(v) for v in x]
        else:
            return x  # ints, strings, bools, PyTreeDef, etc.

    return [convert_item(p) for p in processor_params]


def convert_params_to_jax(processor_params: list) -> list:
    """Recursively convert numpy arrays to JAX in processor params, preserving structure."""
    import jax.numpy as jnp

    def convert_item(x):
        if isinstance(x, np.ndarray) and x.dtype.kind in ("f", "i", "u", "c"):
            return jnp.array(x)
        elif isinstance(x, dict):
            return {k: convert_item(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [convert_item(v) for v in x]
        else:
            return x  # ints, strings, bools, PyTreeDef, etc.

    return [convert_item(p) for p in processor_params]


def reconstruct_processor(
    processor_type: str, processor_config: dict, processor_params_numpy: list
):
    """Reconstruct a processor + JAX params from saved type/config/numpy-params."""
    from jax import random

    from jcce.structure_learning.processor_adapters import (
        ELMAdapter,
        GNNAdapter,
        MambaAdapter,
        MLPAdapter,
        TransformerAdapter,
    )

    adapters = {
        "mlp": MLPAdapter,
        "transformer": TransformerAdapter,
        "mamba": MambaAdapter,
        "elm": ELMAdapter,
        "gnn": GNNAdapter,
    }
    adapter_cls = adapters.get(processor_type)
    if adapter_cls is None:
        raise ValueError(f"Unknown processor type: {processor_type}")

    # Fix legacy checkpoints that used 'aggregation' instead of 'sage_aggregation'
    if processor_type == "gnn" and "aggregation" in processor_config:
        processor_config = dict(processor_config)
        processor_config["sage_aggregation"] = processor_config.pop("aggregation")

    key = random.PRNGKey(0)
    processor = adapter_cls(key=key, **processor_config)
    params_jax = convert_params_to_jax(processor_params_numpy)
    return processor, params_jax


def save_cf_checkpoint(
    checkpoint_path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    A_est: np.ndarray,
    A_weights: np.ndarray,
    A_confound_weights: np.ndarray,
    processor_type: str,
    processor_config: dict,
    processor_params: list,
    ds_config: dict,
):
    """Save everything needed to re-run CF without re-running NSGA-II."""
    params_numpy = convert_params_to_numpy(processor_params)

    checkpoint = {
        "X": X,
        "Y": Y,
        "A_est": A_est,
        "A_weights": A_weights,
        "A_confound_weights": A_confound_weights,
        "processor_type": processor_type,
        "processor_config": processor_config,
        "processor_params": params_numpy,
        "ds_config": ds_config,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_cf_checkpoint(checkpoint_path: Path):
    """Load CF checkpoint and reconstruct the processor."""
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    processor, params_jax = reconstruct_processor(
        ckpt["processor_type"],
        ckpt["processor_config"],
        ckpt["processor_params"],
    )

    return (
        ckpt["X"],
        ckpt["Y"],
        ckpt["A_est"],
        ckpt["A_weights"],
        ckpt["A_confound_weights"],
        processor,
        params_jax,
        ckpt["ds_config"],
    )


def run_counterfactual_evaluation(
    X: np.ndarray,
    Y: np.ndarray,
    A_est: np.ndarray,
    processor_trained,
    processor_params: list,
    ds_config: dict,
    A_weights: np.ndarray = None,
    A_confound: np.ndarray = None,
    n_instances: int = 30,
    cf_pop_size: int = 50,
    cf_n_gen: int = 50,
    verbose: bool = True,
) -> Optional[dict]:
    """
    Post-hoc counterfactual evaluation using the actual trained JCCE model.

    Uses the trained unified processor (MLP/Transformer/Mamba/GNN) from the
    best solution to generate counterfactual explanations. This ensures
    CFs explain the actual model's decisions, not a proxy.

    Pipeline:
    1. Build predict_proba from trained processor + A weights (matching training inference)
    2. Extract parents of Y from learned adjacency matrix
    3. Run NSGA-II counterfactual search with SCM propagation
    4. Compute plausibility (kNN) and causal validity metrics

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Target vector (n_samples,)
        A_est: Learned adjacency matrix (n_vars+1, n_vars+1) — augmented with Y (binary)
        processor_trained: Trained processor adapter (MLPAdapter, TransformerAdapter, etc.)
        processor_params: List of param dicts (one per variable in augmented space)
        ds_config: Dataset config dict
        A_weights: Continuous-valued adjacency matrix from training (A_final). If None, falls back to A_est.
        A_confound: Confound (bi-directed) adjacency matrix. Added with 0.5 factor to match training.
        n_instances: Number of instances to generate counterfactuals for
        cf_pop_size: NSGA-II population size for CF search
        cf_n_gen: NSGA-II generations for CF search
        verbose: Print progress

    Returns:
        Dict with CF evaluation metrics, or None if no parents found
    """
    import jax.numpy as jnp

    from jcce.counterfactual import (
        CounterfactualSearcher,
        compute_plausibility_score,
    )

    n_features = X.shape[1]
    Y_idx = A_est.shape[0] - 1  # Y is last column in augmented matrix

    # Parents of Y in the augmented matrix
    parent_weights = np.abs(A_est[:n_features, Y_idx])
    parent_mask = parent_weights > 0.01
    mb_indices = list(np.where(parent_mask)[0])

    if len(mb_indices) == 0:
        if verbose:
            print("  CF: No parents of Y found, skipping counterfactual evaluation")
        return None

    if verbose:
        feature_names = ds_config.get("feature_names")
        parent_names = [feature_names[i] if feature_names else f"X{i}" for i in mb_indices]
        print(f"\n  CF: Parents of Y: {parent_names} ({len(mb_indices)} features)")

    # Build predict_proba using the actual trained JCCE model
    # Must match the training inference path in jcce_learner.py:
    #   weights = |A_final[:, Y_idx]| + 0.5 * |A_confound[:, Y_idx]|
    # Using A_weights (continuous) instead of A_est (binary) gives correct feature scaling.
    A_for_weights = jnp.array(A_weights) if A_weights is not None else jnp.array(A_est)
    weights_Y = jnp.abs(A_for_weights[:, Y_idx])
    if A_confound is not None:
        A_conf_jax = jnp.array(A_confound)
        weights_Y = weights_Y + 0.5 * jnp.abs(A_conf_jax[:, Y_idx])
    weights_Y = weights_Y.at[Y_idx].set(0.0)
    weights_Y = weights_Y / (jnp.sum(weights_Y) + 1e-8)  # L1-normalize
    weights_X = weights_Y[:n_features]
    proc_params_Y = processor_params[Y_idx]
    A_est_jax = jnp.array(A_est)  # Still needed for SCM

    def _raw_logits(X_input):
        """Get raw logits from the processor (before any centering fix)."""
        X_jax = jnp.array(X_input)
        if X_jax.ndim == 1:
            X_jax = X_jax[jnp.newaxis, :]
        X_weighted = X_jax * weights_X[jnp.newaxis, :]

        return processor_trained.forward(X_weighted, proc_params_Y)

    # Compute training-set logit statistics to anchor predictions.
    # Some processors (e.g., ELM) apply batch-dependent mean centering internally,
    # which makes single-instance predictions always 0.5. We fix this by computing
    # the training mean logit once and using it as a fixed reference for all future
    # predictions, ensuring consistency with training-time behavior.
    train_logits = np.array(_raw_logits(X).flatten())
    _logit_mean = float(np.mean(train_logits))
    _logit_std = float(np.std(train_logits))

    # Check if the processor uses batch-dependent centering (logits cluster near 0)
    _needs_logit_recalibration = _logit_std < 0.1
    if _needs_logit_recalibration and verbose:
        print(
            f"  CF: Detected low logit variance (std={_logit_std:.4f}), "
            f"applying fixed-mean recalibration"
        )

    def predict_proba(X_input):
        """Predict class probabilities using the trained JCCE processor."""
        logits = _raw_logits(X_input)
        logits_np = np.array(logits.flatten())

        if _needs_logit_recalibration:
            # Undo per-batch centering and apply fixed training-set centering.
            # The processor internally does: output = output - mean(output)
            # We reverse this: add back batch mean, subtract training mean.
            batch_mean = np.mean(logits_np)
            logits_np = logits_np - batch_mean + _logit_mean
            # Scale up to get meaningful confidence (preserve sign/ordering)
            if _logit_std > 1e-6:
                logits_np = logits_np / _logit_std

        probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
        # Return (n_samples, 2) array for binary classification
        return np.column_stack([1 - probs, probs])

    # Verify model predictions match training labels
    Y_pred_prob = predict_proba(X)[:, 1]
    Y_pred = (Y_pred_prob > 0.5).astype(float)
    model_acc = float(np.mean(Y_pred == Y))

    if verbose:
        print(f"  CF: Trained model accuracy on structure data: {model_acc:.3f}")

        # Diagnostic: prediction confidence distribution
        confidences = np.maximum(Y_pred_prob, 1 - Y_pred_prob)
        print(
            f"  CF: Prediction confidence: mean={np.mean(confidences):.3f}, "
            f"min={np.min(confidences):.3f}, "
            f"near-boundary(<0.6)={np.sum(confidences < 0.6)}/{len(confidences)}"
        )

        # Diagnostic: batch vs single prediction consistency check
        diag_sample = min(5, len(X))
        batch_p = predict_proba(X[:diag_sample])
        single_p = np.array([predict_proba(X[i])[0] for i in range(diag_sample)])
        batch_match = np.allclose(batch_p, single_p, atol=0.01)
        print(
            f"  CF: Batch vs single predict match: {batch_match} "
            f"(batch[0]=[{batch_p[0, 0]:.4f},{batch_p[0, 1]:.4f}], "
            f"single[0]=[{single_p[0, 0]:.4f},{single_p[0, 1]:.4f}])"
        )

    # Build feature bounds from training data
    feature_bounds = np.column_stack([X.min(axis=0), X.max(axis=0)])
    # Slightly expand bounds to allow exploration
    ranges = feature_bounds[:, 1] - feature_bounds[:, 0]
    feature_bounds[:, 0] -= 0.05 * ranges
    feature_bounds[:, 1] += 0.05 * ranges

    # Immutable features from dataset config
    immutable = ds_config.get("immutable_features", [])

    # Categorical/binary features: use config if available, otherwise auto-detect
    categorical = ds_config.get("categorical_features", None)
    if categorical is None:
        # Auto-detect: features with <= 5 unique values are treated as categorical
        categorical = []
        for j in range(n_features):
            n_unique = len(np.unique(X[:, j]))
            if n_unique <= 5:
                categorical.append(j)
        if verbose and categorical:
            cat_names = [
                ds_config.get("feature_names", [f"X{i}" for i in range(n_features)])[j]
                for j in categorical
            ]
            print(f"  CF: Auto-detected {len(categorical)} categorical features: {cat_names}")

    if verbose and categorical:
        print(
            f"  CF: {len(categorical)} categorical/binary features "
            f"(will be rounded to integers in CF search)"
        )

    # The CF searcher needs the full augmented DAG including Y
    dag_augmented = np.abs(np.array(A_est[: n_features + 1, : n_features + 1]))

    # Check if DAG is acyclic — SCM propagation is unreliable with cycles
    from jcce.counterfactual.causal_constraints import topological_sort

    dag_features = dag_augmented[:n_features, :n_features]
    try:
        topological_sort(dag_features, threshold=0.1)
        dag_is_acyclic = True
    except ValueError:
        dag_is_acyclic = False

    use_scm = dag_is_acyclic  # Only use SCM when DAG is truly acyclic
    if verbose and not dag_is_acyclic:
        print(
            "  CF: Feature DAG has cycles — disabling SCM propagation (using direct perturbation)"
        )

    # Create searcher with the actual trained model
    searcher = CounterfactualSearcher(
        classifier=predict_proba,
        dag=dag_augmented,
        target_idx=Y_idx,
        feature_bounds=feature_bounds,
        immutable_features=immutable,
        categorical_features=categorical,
        use_scm_propagation=use_scm,
        scm_model_type="ridge",
    )

    # Fit SCM for Pearl's 3-step propagation (only if DAG is acyclic)
    if use_scm:
        searcher.fit_scm(X, verbose=False)

    # Select instances to explain:
    # Prefer confident predictions, but fall back to correctly-classified instances.
    # Some processors (e.g., ELM with output centering) produce accurate but
    # low-confidence predictions (logits near 0), so a strict confidence threshold
    # would exclude all instances.
    confidences_all = np.maximum(Y_pred_prob, 1 - Y_pred_prob)
    correctly_classified = Y_pred == Y

    # Try confidence thresholds in decreasing order
    for confidence_threshold in [0.55, 0.51, 0.50]:
        confident_mask = confidences_all > confidence_threshold
        positive_idx = np.where((Y == 1) & confident_mask)[0]
        negative_idx = np.where((Y == 0) & confident_mask)[0]
        if len(positive_idx) + len(negative_idx) >= n_instances:
            break

    # Final fallback: use correctly-classified instances regardless of confidence
    if len(positive_idx) + len(negative_idx) < n_instances:
        confidence_threshold = None
        positive_idx = np.where((Y == 1) & correctly_classified)[0]
        negative_idx = np.where((Y == 0) & correctly_classified)[0]

    if verbose:
        n_confident = int(np.sum(confidences_all > 0.55))
        n_correct = int(np.sum(correctly_classified))
        if confidence_threshold is not None:
            print(
                f"  CF: {n_confident}/{len(Y)} instances with confidence > 0.55, "
                f"using threshold={confidence_threshold:.2f}"
            )
        else:
            print(
                f"  CF: {n_confident}/{len(Y)} confident, using {n_correct}/{len(Y)} "
                f"correctly-classified instances instead"
            )

    # Mix: some positives (flip to 0) and some negatives (flip to 1)
    n_pos = min(n_instances // 2, len(positive_idx))
    n_neg = min(n_instances - n_pos, len(negative_idx))

    rng = np.random.RandomState(42)
    selected_pos = (
        rng.choice(positive_idx, size=n_pos, replace=False)
        if n_pos > 0
        else np.array([], dtype=int)
    )
    selected_neg = (
        rng.choice(negative_idx, size=n_neg, replace=False)
        if n_neg > 0
        else np.array([], dtype=int)
    )
    selected_idx = np.concatenate([selected_pos, selected_neg]).astype(int)

    if verbose:
        print(
            f"  CF: Searching counterfactuals for {len(selected_idx)} instances "
            f"({n_pos} pos→neg, {n_neg} neg→pos)..."
        )

    # Run CF search per instance
    all_results = []
    n_valid = 0
    sparsities = []
    distances = []
    causal_validities = []
    plausibility_ratios = []
    n_plausible = 0

    for i, idx in enumerate(selected_idx):
        x_instance = X[idx]

        result = searcher.search(
            x_original=x_instance,
            pop_size=cf_pop_size,
            n_gen=cf_n_gen,
            seed=42 + i,
            verbose=False,
        )

        if result.best_sparse is not None:
            n_valid += 1
            best = result.best_sparse
            sparsities.append(best.objectives["sparsity"])
            distances.append(best.objectives["distance"])
            causal_validities.append(best.objectives["causal_invalidity"])

            # Plausibility check
            _, plaus_details = compute_plausibility_score(
                best.x_counterfactual,
                X,
                k=5,
            )
            plausibility_ratios.append(plaus_details["plausibility_ratio"])
            if plaus_details["is_plausible"]:
                n_plausible += 1

            all_results.append(
                {
                    "instance_idx": int(idx),
                    "original_class": int(Y[idx]),
                    "target_class": result.target_prediction,
                    "n_candidates": len(result.candidates),
                    "n_valid": result.n_valid,
                    "best_sparsity": best.objectives["sparsity"],
                    "best_distance": best.objectives["distance"],
                    "best_causal_invalidity": best.objectives["causal_invalidity"],
                    "changed_features": best.changed_features,
                    "plausibility_ratio": plaus_details["plausibility_ratio"],
                    "is_plausible": plaus_details["is_plausible"],
                }
            )

        if verbose and (i + 1) % 10 == 0:
            print(
                f"    CF progress: {i + 1}/{len(selected_idx)} instances ({n_valid} valid so far)"
            )

    # Aggregate results
    validity_rate = n_valid / max(len(selected_idx), 1)
    avg_sparsity = float(np.mean(sparsities)) if sparsities else 0.0
    avg_distance = float(np.mean(distances)) if distances else 0.0
    avg_causal_validity = float(np.mean(causal_validities)) if causal_validities else 0.0
    avg_plausibility = float(np.mean(plausibility_ratios)) if plausibility_ratios else 0.0

    if verbose:
        print("\n  CF RESULTS:")
        print(f"    Validity:     {n_valid}/{len(selected_idx)} ({validity_rate:.1%})")
        print(f"    Avg sparsity: {avg_sparsity:.1f} features changed")
        print(f"    Avg distance: {avg_distance:.3f}")
        print(f"    Avg causal validity: {avg_causal_validity:.3f} (0=perfect)")
        print(f"    Plausible:    {n_plausible}/{n_valid} (ratio={avg_plausibility:.2f})")

        # Show example counterfactual
        if all_results:
            ex = all_results[0]
            feature_names = ds_config.get("feature_names")
            changed_names = [
                feature_names[f] if feature_names else f"X{f}" for f in ex["changed_features"]
            ]
            print(
                f"    Example CF (instance {ex['instance_idx']}, "
                f"class {ex['original_class']}→{ex['target_class']}): "
                f"changed {changed_names}"
            )

    return {
        "n_instances": len(selected_idx),
        "n_valid": n_valid,
        "validity_rate": validity_rate,
        "avg_sparsity": avg_sparsity,
        "avg_distance": avg_distance,
        "avg_causal_validity": avg_causal_validity,
        "avg_plausibility_ratio": avg_plausibility,
        "n_plausible": n_plausible,
        "per_instance_results": all_results,
        "mb_indices": mb_indices,
        "model_accuracy": model_acc,
    }
