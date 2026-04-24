"""
Causal Constraints for Counterfactual Generation

Provides utilities for:
- DAG queries (ancestors, descendants, parents, children)
- Causal validity scoring for counterfactuals
- Actionability proxy computation (O6 objective for NSGA-II)

The key insight is that valid counterfactual explanations should only
modify CAUSES of the target variable, not EFFECTS or unrelated features.
"""

import numpy as np
from typing import Set, List, Dict, Optional, Tuple, Callable
from collections import deque
from sklearn.neighbors import NearestNeighbors


def get_parents(dag: np.ndarray, node: int, threshold: float = 0.1) -> Set[int]:
    """
    Get direct parents of a node in the DAG.

    Args:
        dag: Adjacency matrix where dag[i,j] > 0 means i -> j
        node: Node index to query
        threshold: Minimum edge weight to consider

    Returns:
        Set of parent node indices
    """
    # Parents are nodes i where dag[i, node] > threshold
    parents = set(np.where(dag[:, node] > threshold)[0])
    parents.discard(node)  # Remove self-loops
    return parents


def get_children(dag: np.ndarray, node: int, threshold: float = 0.1) -> Set[int]:
    """
    Get direct children of a node in the DAG.

    Args:
        dag: Adjacency matrix where dag[i,j] > 0 means i -> j
        node: Node index to query
        threshold: Minimum edge weight to consider

    Returns:
        Set of child node indices
    """
    # Children are nodes j where dag[node, j] > threshold
    children = set(np.where(dag[node, :] > threshold)[0])
    children.discard(node)  # Remove self-loops
    return children


def get_ancestors(dag: np.ndarray, node: int, threshold: float = 0.1) -> Set[int]:
    """
    Get all ancestors of a node (transitive closure of parents).

    An ancestor is any node from which there is a directed path to the target node.
    These are the features that CAUSE the target - valid intervention points.

    Args:
        dag: Adjacency matrix
        node: Target node index
        threshold: Minimum edge weight

    Returns:
        Set of ancestor node indices (excludes the node itself)
    """
    ancestors = set()
    queue = deque(get_parents(dag, node, threshold))

    while queue:
        current = queue.popleft()
        if current not in ancestors and current != node:
            ancestors.add(current)
            # Add parents of current to queue
            for parent in get_parents(dag, current, threshold):
                if parent not in ancestors:
                    queue.append(parent)

    return ancestors


def get_descendants(dag: np.ndarray, node: int, threshold: float = 0.1) -> Set[int]:
    """
    Get all descendants of a node (transitive closure of children).

    A descendant is any node reachable via directed path from the source node.
    These are EFFECTS of the target - should NOT be modified in counterfactuals.

    Args:
        dag: Adjacency matrix
        node: Source node index
        threshold: Minimum edge weight

    Returns:
        Set of descendant node indices (excludes the node itself)
    """
    descendants = set()
    queue = deque(get_children(dag, node, threshold))

    while queue:
        current = queue.popleft()
        if current not in descendants and current != node:
            descendants.add(current)
            # Add children of current to queue
            for child in get_children(dag, current, threshold):
                if child not in descendants:
                    queue.append(child)

    return descendants


def get_markov_blanket_from_dag(
    dag: np.ndarray,
    node: int,
    threshold: float = 0.1
) -> Set[int]:
    """
    Compute Markov Blanket from DAG structure.

    MB(X) = Parents(X) ∪ Children(X) ∪ Parents(Children(X))

    Args:
        dag: Adjacency matrix
        node: Target node index
        threshold: Minimum edge weight

    Returns:
        Set of Markov Blanket node indices
    """
    parents = get_parents(dag, node, threshold)
    children = get_children(dag, node, threshold)

    # Spouses = parents of children (excluding self)
    spouses = set()
    for child in children:
        child_parents = get_parents(dag, child, threshold)
        spouses.update(child_parents)
    spouses.discard(node)

    mb = parents | children | spouses
    return mb


def topological_sort(dag: np.ndarray, threshold: float = 0.1) -> List[int]:
    """
    Compute topological ordering of DAG nodes.

    Args:
        dag: Adjacency matrix
        threshold: Minimum edge weight

    Returns:
        List of node indices in topological order

    Raises:
        ValueError: If DAG contains cycles
    """
    n = dag.shape[0]
    in_degree = np.sum(dag > threshold, axis=0)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for child in get_children(dag, node, threshold):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != n:
        raise ValueError("DAG contains cycles - topological sort failed")

    return order


def compute_causal_validity_score(
    x_original: np.ndarray,
    x_counterfactual: np.ndarray,
    dag: np.ndarray,
    target_idx: int,
    immutable_features: Optional[Set[int]] = None,
    threshold: float = 0.1,
    change_threshold: float = 0.01,
) -> Tuple[float, Dict]:
    """
    Compute causal validity score for a counterfactual.

    A valid counterfactual should only modify:
    - Ancestors of the target (causes)
    - NOT descendants (effects)
    - NOT immutable features

    Args:
        x_original: Original instance
        x_counterfactual: Counterfactual instance
        dag: Adjacency matrix (includes target as last column typically)
        target_idx: Index of target variable in DAG
        immutable_features: Set of feature indices that cannot change
        threshold: DAG edge threshold
        change_threshold: Minimum change to consider a feature "modified"

    Returns:
        Tuple of (validity_score, details_dict)
        - validity_score: 0 = perfectly valid, higher = more violations
        - details_dict: Breakdown of penalties
    """
    immutable_features = immutable_features or set()

    # Get causal roles
    ancestors = get_ancestors(dag, target_idx, threshold)
    descendants = get_descendants(dag, target_idx, threshold)

    # Identify changed features
    n_features = len(x_original)
    changes = np.abs(x_counterfactual - x_original)

    # Normalize by feature range if possible
    feature_ranges = np.maximum(np.abs(x_original), 1.0)
    relative_changes = changes / feature_ranges

    total_penalty = 0.0
    details = {
        'n_immutable_violations': 0,
        'n_descendant_violations': 0,
        'n_non_causal_violations': 0,
        'n_valid_changes': 0,
        'changed_features': [],
        'violation_features': [],
    }

    for i in range(n_features):
        if i == target_idx:
            continue  # Skip target variable

        if relative_changes[i] > change_threshold:
            details['changed_features'].append(i)

            if i in immutable_features:
                # Heavy penalty for immutable
                total_penalty += 100.0
                details['n_immutable_violations'] += 1
                details['violation_features'].append((i, 'immutable'))

            elif i in descendants:
                # Medium penalty for descendants (effects)
                total_penalty += 10.0
                details['n_descendant_violations'] += 1
                details['violation_features'].append((i, 'descendant'))

            elif i not in ancestors:
                # Light penalty for non-causal features
                total_penalty += 2.0
                details['n_non_causal_violations'] += 1
                details['violation_features'].append((i, 'non_causal'))

            else:
                # Valid change (ancestor)
                details['n_valid_changes'] += 1

    details['total_changes'] = len(details['changed_features'])
    details['validity_ratio'] = (
        details['n_valid_changes'] / max(details['total_changes'], 1)
    )

    return total_penalty, details


def compute_actionability_proxy(
    dag: np.ndarray,
    markov_blanket: List[int],
    target_idx: int,
    X: Optional[np.ndarray] = None,
    classifier: Optional[Callable] = None,
    n_samples: int = 50,
    threshold: float = 0.1,
) -> Tuple[float, Dict]:
    """
    Compute actionability proxy for NSGA-II objective O6.

    This is a cheap approximation of counterfactual feasibility that can
    be computed during the main optimization loop.

    Components:
    1. Actionability ratio: What fraction of MB features are valid intervention targets?
    2. (Optional) Avg decision margin: How far are samples from decision boundary?
    3. (Optional) Feature elasticity: How sensitive is Y to MB feature changes?

    Lower proxy score = better actionability = easier to generate counterfactuals

    Args:
        dag: Adjacency matrix
        markov_blanket: List of MB feature indices
        target_idx: Target variable index in DAG
        X: Optional data matrix for computing margins/elasticity
        classifier: Optional classifier for computing margins
        n_samples: Number of samples for margin/elasticity estimation
        threshold: DAG edge threshold

    Returns:
        Tuple of (proxy_score, details_dict)
    """
    if not markov_blanket:
        return float('inf'), {'error': 'Empty Markov Blanket'}

    # 1. Actionability ratio
    ancestors = get_ancestors(dag, target_idx, threshold)
    actionable_mb = set(markov_blanket) & ancestors
    descendants = get_descendants(dag, target_idx, threshold)
    non_actionable_mb = set(markov_blanket) & descendants

    actionability_ratio = len(actionable_mb) / len(markov_blanket)

    details = {
        'actionability_ratio': actionability_ratio,
        'n_actionable_features': len(actionable_mb),
        'n_non_actionable_features': len(non_actionable_mb),
        'actionable_features': list(actionable_mb),
        'non_actionable_features': list(non_actionable_mb),
    }

    # Base proxy score (inverted actionability)
    proxy_score = 1.0 - actionability_ratio

    # 2. Optional: Average decision margin
    if X is not None and classifier is not None:
        try:
            sample_idx = np.random.choice(
                len(X), size=min(n_samples, len(X)), replace=False
            )
            X_sample = X[sample_idx]

            if hasattr(classifier, 'decision_function'):
                margins = np.abs(classifier.decision_function(X_sample))
            elif hasattr(classifier, 'predict_proba'):
                proba = classifier.predict_proba(X_sample)
                if proba.shape[1] == 2:
                    margins = np.abs(proba[:, 1] - 0.5) * 2
                else:
                    margins = np.max(proba, axis=1) - np.partition(proba, -2, axis=1)[:, -2]
            else:
                margins = None

            if margins is not None:
                avg_margin = np.mean(margins)
                details['avg_decision_margin'] = float(avg_margin)
                # Higher margin = harder to flip = worse actionability
                proxy_score += 0.3 * avg_margin

        except Exception as e:
            details['margin_error'] = str(e)

    # 3. Optional: Feature elasticity (how much Y changes with MB features)
    if X is not None and classifier is not None and actionable_mb:
        try:
            sample_idx = np.random.choice(
                len(X), size=min(n_samples, len(X)), replace=False
            )
            X_sample = X[sample_idx].copy()

            elasticities = []
            for feat_idx in actionable_mb:
                delta = 0.1 * np.std(X[:, feat_idx])
                if delta < 1e-6:
                    continue

                X_perturbed = X_sample.copy()
                X_perturbed[:, feat_idx] += delta

                if hasattr(classifier, 'predict_proba'):
                    y_original = classifier.predict_proba(X_sample)[:, -1]
                    y_perturbed = classifier.predict_proba(X_perturbed)[:, -1]
                    elasticity = np.mean(np.abs(y_perturbed - y_original)) / delta
                    elasticities.append(elasticity)

            if elasticities:
                avg_elasticity = np.mean(elasticities)
                details['avg_elasticity'] = float(avg_elasticity)
                # Higher elasticity = easier to change Y = better actionability
                proxy_score -= 0.1 * min(avg_elasticity, 1.0)

        except Exception as e:
            details['elasticity_error'] = str(e)

    details['proxy_score'] = float(proxy_score)
    return float(proxy_score), details


def identify_intervention_targets(
    dag: np.ndarray,
    target_idx: int,
    feature_names: Optional[List[str]] = None,
    immutable_indices: Optional[Set[int]] = None,
    threshold: float = 0.1,
) -> Dict:
    """
    Identify valid intervention targets from the DAG.

    Useful for understanding what features can be modified in counterfactuals.

    Args:
        dag: Adjacency matrix
        target_idx: Target variable index
        feature_names: Optional list of feature names
        immutable_indices: Features that cannot be changed
        threshold: DAG edge threshold

    Returns:
        Dictionary with intervention analysis
    """
    immutable_indices = immutable_indices or set()
    n_features = dag.shape[0]

    ancestors = get_ancestors(dag, target_idx, threshold)
    descendants = get_descendants(dag, target_idx, threshold)
    parents = get_parents(dag, target_idx, threshold)

    # Categorize all features
    categories = {
        'direct_causes': [],      # Parents of Y - strongest intervention targets
        'indirect_causes': [],    # Other ancestors - also valid
        'effects': [],            # Descendants - should NOT modify
        'non_causal': [],         # No path to/from Y
        'immutable': [],          # Marked as immutable
    }

    for i in range(n_features):
        if i == target_idx:
            continue

        name = feature_names[i] if feature_names else f'X{i}'

        if i in immutable_indices:
            categories['immutable'].append((i, name))
        elif i in parents:
            categories['direct_causes'].append((i, name))
        elif i in ancestors:
            categories['indirect_causes'].append((i, name))
        elif i in descendants:
            categories['effects'].append((i, name))
        else:
            categories['non_causal'].append((i, name))

    return {
        'categories': categories,
        'n_actionable': len(categories['direct_causes']) + len(categories['indirect_causes']),
        'n_non_actionable': len(categories['effects']) + len(categories['immutable']),
        'recommendation': (
            f"Prioritize interventions on {len(categories['direct_causes'])} direct causes. "
            f"Avoid modifying {len(categories['effects'])} effect variables."
        )
    }


def compute_plausibility_score(
    x_counterfactual: np.ndarray,
    X_train: np.ndarray,
    k: int = 5,
) -> Tuple[float, Dict]:
    """
    Compute plausibility of a counterfactual via k-nearest neighbor distance.

    A plausible counterfactual should lie close to the training data manifold.
    High distance indicates an unrealistic/off-manifold counterfactual.

    Args:
        x_counterfactual: Counterfactual instance (1D array)
        X_train: Training data (n_samples, n_features)
        k: Number of nearest neighbors

    Returns:
        Tuple of (avg_knn_distance, details_dict)
    """
    x_cf = np.atleast_2d(x_counterfactual)
    k = min(k, X_train.shape[0])

    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_train)
    distances, indices = nn.kneighbors(x_cf)

    avg_dist = float(np.mean(distances[0]))
    max_dist = float(np.max(distances[0]))
    min_dist = float(np.min(distances[0]))

    # Compare to typical inter-sample distances (baseline)
    sample_idx = np.random.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    baseline_distances, _ = nn.kneighbors(X_train[sample_idx])
    baseline_avg = float(np.mean(baseline_distances[:, -1]))  # k-th neighbor distance

    # Ratio > 1 means CF is farther from data than typical samples
    plausibility_ratio = avg_dist / max(baseline_avg, 1e-10)

    return avg_dist, {
        'avg_knn_distance': avg_dist,
        'min_knn_distance': min_dist,
        'max_knn_distance': max_dist,
        'baseline_avg_knn_distance': baseline_avg,
        'plausibility_ratio': plausibility_ratio,
        'is_plausible': plausibility_ratio < 2.0,  # Within 2x typical distance
        'k': k,
    }
