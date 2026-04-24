"""
Classical Markov Blanket estimation baselines.

Implements data-driven MB discovery algorithms that use conditional independence
tests to identify MB(target) directly from data (no DAG required).

Algorithms:
  - IAMB (Incremental Association Markov Blanket) — Tsamardinos et al. (2003)
  - Fast-IAMB — Yaramakala & Margaritis (2005)
  - HITON-MB — Aliferis et al. (2003)
  - Inter-IAMB — variant with interleaved forward/backward

CI Testing:
  - Partial correlation with Fisher's z-test (Gaussian assumption)
  - Permutation-based CI test (nonparametric fallback)

Data Generation:
  - Linear-Gaussian SEM from known DAG
  - Nonlinear SEM (quadratic + tanh) from known DAG

Dependencies: numpy, scipy (NO JAX, NO external causal discovery libraries)

References:
  - Tsamardinos et al. (2003). Algorithms for Large Scale Markov Blanket Discovery. FLAIRS.
  - Yaramakala & Margaritis (2005). Speculative Markov Blanket Discovery. AAAI.
  - Aliferis et al. (2003). HITON: A Novel Markov Blanket Algorithm. AMIA.
  - Aliferis et al. (2010). Local Causal and Markov Blanket Induction. JMLR 11.
"""

import numpy as np
from scipy import linalg, stats

# =============================================================================
# Conditional Independence Testing
# =============================================================================


def partial_correlation(X: np.ndarray, i: int, j: int, cond_set: list[int] | None = None) -> float:
    """
    Compute partial correlation between X[:,i] and X[:,j] given X[:,cond_set].

    Uses the recursive formula via the precision matrix of the relevant submatrix.

    Args:
        X: (n, d) data matrix.
        i, j: variable indices.
        cond_set: conditioning set indices. None or [] means marginal correlation.

    Returns:
        Partial correlation coefficient in [-1, 1].
    """
    if cond_set is None or len(cond_set) == 0:
        r = np.corrcoef(X[:, i], X[:, j])[0, 1]
        return r if np.isfinite(r) else 0.0

    # Submatrix of [i, j, cond_set]
    idx = [i, j] + list(cond_set)
    sub_X = X[:, idx]
    cov = np.cov(sub_X, rowvar=False)
    cov += np.eye(len(idx)) * 1e-8  # regularize

    if not np.all(np.isfinite(cov)):
        return 0.0

    try:
        prec = linalg.inv(cov)
    except linalg.LinAlgError:
        return 0.0

    if not np.all(np.isfinite(prec)):
        return 0.0

    # Partial corr of first two variables = -P[0,1] / sqrt(P[0,0]*P[1,1])
    denom = np.sqrt(abs(prec[0, 0] * prec[1, 1]))
    if denom < 1e-10:
        return 0.0

    pcorr = -prec[0, 1] / denom
    return float(np.clip(pcorr, -1.0, 1.0))


def fisher_z_test(
    X: np.ndarray, i: int, j: int, cond_set: list[int] | None = None, alpha: float = 0.05
) -> tuple[bool, float]:
    """
    Fisher's z-test for conditional independence.

    Tests H0: X_i ⊥ X_j | X_cond_set using partial correlation + Fisher's z.

    Args:
        X: (n, d) data matrix.
        i, j: variable indices.
        cond_set: conditioning set.
        alpha: significance level.

    Returns:
        (independent, p_value): True if CI holds at level alpha.
    """
    n = X.shape[0]
    k = len(cond_set) if cond_set else 0

    r = partial_correlation(X, i, j, cond_set)

    # Fisher's z-transformation
    r_clipped = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    se = 1.0 / np.sqrt(max(n - k - 3, 1))

    z_stat = abs(z) / se
    p_value = 2 * (1 - stats.norm.cdf(z_stat))

    return p_value > alpha, float(p_value)


def association_score(X: np.ndarray, i: int, j: int, cond_set: list[int] | None = None) -> float:
    """
    Association score: |partial_correlation| — used for IAMB ranking.
    Higher = stronger association = more likely MB member.
    """
    return abs(partial_correlation(X, i, j, cond_set))


# =============================================================================
# IAMB — Incremental Association Markov Blanket
# =============================================================================


def iamb(
    X: np.ndarray,
    target: int,
    alpha: float = 0.05,
    max_mb_size: int | None = None,
) -> set[int]:
    """
    IAMB algorithm for Markov blanket discovery.

    Phase 1 (Forward — Growing): Greedily add the variable with the
    highest association score with target given current MB, until no
    variable passes the CI test.

    Phase 2 (Backward — Shrinking): Remove any variable from MB that
    becomes conditionally independent of target given the rest of MB.

    Args:
        X: (n, d) data matrix.
        target: target variable index.
        alpha: significance level for CI tests.
        max_mb_size: maximum MB size (early stopping).

    Returns:
        Set of variable indices in estimated MB(target).
    """
    n, d = X.shape
    if max_mb_size is None:
        max_mb_size = d - 1

    mb = set()
    candidates = set(range(d)) - {target}

    # Phase 1: Forward (Growing)
    changed = True
    while changed and len(mb) < max_mb_size:
        changed = False
        best_score = 0.0
        best_var = None

        for x in candidates - mb:
            score = association_score(X, target, x, list(mb))
            if score > best_score:
                best_score = score
                best_var = x

        if best_var is not None:
            # Test if best_var is dependent on target given MB
            independent, p_val = fisher_z_test(X, target, best_var, list(mb), alpha)
            if not independent:
                mb.add(best_var)
                changed = True

    # Phase 2: Backward (Shrinking)
    for x in list(mb):
        cond = list(mb - {x})
        independent, p_val = fisher_z_test(X, target, x, cond, alpha)
        if independent:
            mb.discard(x)

    return mb


# =============================================================================
# Fast-IAMB — Speculative Markov Blanket Discovery
# =============================================================================


def fast_iamb(
    X: np.ndarray,
    target: int,
    alpha: float = 0.05,
    max_mb_size: int | None = None,
) -> set[int]:
    """
    Fast-IAMB: Speculative variant of IAMB.

    Difference from IAMB: in the forward phase, adds ALL variables that
    pass the CI test in a single pass (speculative addition), then does
    backward shrinking. Faster but may include more false positives
    initially.

    Args:
        X: (n, d) data matrix.
        target: target variable index.
        alpha: significance level.
        max_mb_size: maximum MB size.

    Returns:
        Set of variable indices in estimated MB(target).
    """
    n, d = X.shape
    if max_mb_size is None:
        max_mb_size = d - 1

    mb = set()
    candidates = set(range(d)) - {target}

    # Phase 1: Forward (Speculative Growing)
    changed = True
    while changed and len(mb) < max_mb_size:
        changed = False
        # Score all remaining candidates
        scored = []
        for x in candidates - mb:
            score = association_score(X, target, x, list(mb))
            scored.append((score, x))

        scored.sort(reverse=True)

        # Add all that pass CI test (speculative)
        for score, x in scored:
            if len(mb) >= max_mb_size:
                break
            independent, p_val = fisher_z_test(X, target, x, list(mb), alpha)
            if not independent:
                mb.add(x)
                changed = True
            else:
                # Once we hit an independent variable in sorted order, stop
                break

    # Phase 2: Backward (Shrinking)
    for x in list(mb):
        cond = list(mb - {x})
        independent, p_val = fisher_z_test(X, target, x, cond, alpha)
        if independent:
            mb.discard(x)

    return mb


# =============================================================================
# HITON-MB — HITON Markov Blanket
# =============================================================================


def hiton_pc(
    X: np.ndarray,
    target: int,
    alpha: float = 0.05,
    max_k: int = 3,
) -> set[int]:
    """
    HITON-PC: Find the Parents-and-Children set of target.

    Uses progressive conditioning: test X_i ⊥ T | S for increasing
    subsets S of the current candidate set.

    Args:
        X: (n, d) data matrix.
        target: target variable index.
        alpha: significance level.
        max_k: maximum conditioning set size.

    Returns:
        Set of variable indices in PC(target).
    """
    n, d = X.shape
    candidates = set(range(d)) - {target}

    # Sort by marginal association (descending) for efficiency
    scored = [(association_score(X, target, x), x) for x in candidates]
    scored.sort(reverse=True)

    pc_set = set()

    # Forward: add candidates that are dependent on target
    for _, x in scored:
        independent, _ = fisher_z_test(X, target, x, [], alpha)
        if not independent:
            pc_set.add(x)

    # Progressive backward elimination with increasing conditioning set size
    for k in range(1, max_k + 1):
        for x in list(pc_set):
            others = list(pc_set - {x})
            if len(others) < k:
                continue

            # Test with subsets of size k
            removed = False
            for subset in _combinations(others, k):
                independent, _ = fisher_z_test(X, target, x, list(subset), alpha)
                if independent:
                    pc_set.discard(x)
                    removed = True
                    break
            if removed:
                continue

    return pc_set


def hiton_mb(
    X: np.ndarray,
    target: int,
    alpha: float = 0.05,
    max_k: int = 3,
) -> set[int]:
    """
    HITON-MB: Full Markov blanket via HITON-PC + spouse detection.

    Step 1: Find PC(target) using HITON-PC.
    Step 2: For each child in PC(target), find PC(child).
    Step 3: Spouses = nodes in PC(child) that are dependent on target
            given PC(target).

    Args:
        X: (n, d) data matrix.
        target: target variable index.
        alpha: significance level.
        max_k: maximum conditioning set size for HITON-PC.

    Returns:
        Set of variable indices in MB(target).
    """
    pc_target = hiton_pc(X, target, alpha, max_k)
    mb = set(pc_target)

    # For each potential child, find spouses
    for child in list(pc_target):
        pc_child = hiton_pc(X, child, alpha, max_k)
        # Potential spouses: in PC(child) but not in PC(target) and not target
        potential_spouses = pc_child - pc_target - {target}

        for spouse in potential_spouses:
            # Spouse if dependent on target given PC(target)
            independent, _ = fisher_z_test(X, target, spouse, list(pc_target), alpha)
            if not independent:
                mb.add(spouse)

    return mb


# =============================================================================
# Inter-IAMB — Interleaved IAMB
# =============================================================================


def inter_iamb(
    X: np.ndarray,
    target: int,
    alpha: float = 0.05,
    max_mb_size: int | None = None,
    max_iter: int = 10,
) -> set[int]:
    """
    Inter-IAMB: Interleaved forward/backward IAMB.

    Alternates between adding the best candidate and removing
    the weakest member. More conservative than IAMB — less prone
    to false positives in finite samples.

    Args:
        X: (n, d) data matrix.
        target: target variable index.
        alpha: significance level.
        max_mb_size: maximum MB size.
        max_iter: maximum iterations.

    Returns:
        Set of variable indices in estimated MB(target).
    """
    n, d = X.shape
    if max_mb_size is None:
        max_mb_size = d - 1

    mb = set()
    candidates = set(range(d)) - {target}

    for iteration in range(max_iter):
        # Forward: add best candidate
        best_score = 0.0
        best_var = None
        for x in candidates - mb:
            score = association_score(X, target, x, list(mb))
            if score > best_score:
                best_score = score
                best_var = x

        if best_var is not None and len(mb) < max_mb_size:
            independent, _ = fisher_z_test(X, target, best_var, list(mb), alpha)
            if not independent:
                mb.add(best_var)
            else:
                break  # No more candidates pass
        else:
            break

        # Backward: remove weakest member (interleaved)
        if len(mb) > 1:
            weakest_score = float("inf")
            weakest_var = None
            for x in mb:
                cond = list(mb - {x})
                score = association_score(X, target, x, cond)
                if score < weakest_score:
                    weakest_score = score
                    weakest_var = x

            if weakest_var is not None:
                cond = list(mb - {weakest_var})
                independent, _ = fisher_z_test(X, target, weakest_var, cond, alpha)
                if independent:
                    mb.discard(weakest_var)

    # Final backward pass
    for x in list(mb):
        cond = list(mb - {x})
        independent, _ = fisher_z_test(X, target, x, cond, alpha)
        if independent:
            mb.discard(x)

    return mb


# =============================================================================
# Data Generation from Known DAGs
# =============================================================================


def generate_linear_sem_data(
    A: np.ndarray,
    n_samples: int = 1000,
    noise_std: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate data from a linear-Gaussian SEM.

    X_j = sum_i A[i,j] * X_i + eps_j,  eps_j ~ N(0, noise_std^2)

    Args:
        A: (d, d) DAG adjacency matrix. A[i,j] > 0 means i → j.
        n_samples: number of data points.
        noise_std: noise standard deviation.
        seed: random seed.

    Returns:
        X: (n_samples, d) data matrix.
    """
    d = A.shape[0]
    rng = np.random.default_rng(seed)
    order = _topological_sort(A)

    X = np.zeros((n_samples, d))
    for sample_idx in range(n_samples):
        noise = rng.normal(0, noise_std, d)
        x = np.zeros(d)
        for node in order:
            x[node] = A[:, node] @ x + noise[node]
        X[sample_idx] = x

    return X


def generate_nonlinear_sem_data(
    A: np.ndarray,
    n_samples: int = 1000,
    noise_std: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate data from a nonlinear SEM (quadratic + tanh).

    X_j = tanh(sum_i A[i,j] * X_i) + 0.5 * (sum_i A[i,j] * X_i)^2 + eps_j

    Args:
        A: (d, d) DAG adjacency matrix.
        n_samples: number of data points.
        noise_std: noise standard deviation.
        seed: random seed.

    Returns:
        X: (n_samples, d) data matrix.
    """
    d = A.shape[0]
    rng = np.random.default_rng(seed)
    order = _topological_sort(A)

    X = np.zeros((n_samples, d))
    for sample_idx in range(n_samples):
        noise = rng.normal(0, noise_std, d)
        x = np.zeros(d)
        for node in order:
            parent_sum = A[:, node] @ x
            # Clip parent_sum to prevent overflow in quadratic term
            parent_sum = np.clip(parent_sum, -5.0, 5.0)
            x[node] = np.tanh(parent_sum) + 0.3 * parent_sum**2 + noise[node]
        X[sample_idx] = x

    return X


def true_markov_blanket(A: np.ndarray, target: int, threshold: float = 0.1) -> set[int]:
    """
    Extract ground-truth Markov blanket from a DAG adjacency matrix.

    MB(target) = parents ∪ children ∪ spouses(other parents of children).
    """
    d = A.shape[0]
    A_bin = (np.abs(A) > threshold).astype(int)

    parents = set(np.where(A_bin[:, target] > 0)[0])
    children = set(np.where(A_bin[target, :] > 0)[0])

    spouses = set()
    for child in children:
        child_parents = set(np.where(A_bin[:, child] > 0)[0])
        spouses |= child_parents

    return (parents | children | spouses) - {target}


def mb_metrics(predicted: set, true_mb: set) -> dict:
    """Compute precision, recall, F1 for MB prediction."""
    if len(predicted) == 0 and len(true_mb) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "predicted_size": 0, "true_size": 0}

    tp = len(predicted & true_mb)
    fp = len(predicted - true_mb)
    fn = len(true_mb - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_size": len(predicted),
        "true_size": len(true_mb),
    }


# =============================================================================
# Convenience: run all MB algorithms
# =============================================================================


def run_all_mb_algorithms(
    X: np.ndarray,
    target: int,
    alpha: float = 0.05,
) -> dict[str, set[int]]:
    """
    Run all classical MB algorithms on a dataset.

    Returns dict mapping algorithm name → predicted MB set.
    """
    return {
        "IAMB": iamb(X, target, alpha),
        "Fast-IAMB": fast_iamb(X, target, alpha),
        "Inter-IAMB": inter_iamb(X, target, alpha),
        "HITON-MB": hiton_mb(X, target, alpha),
    }


# =============================================================================
# Internal helpers
# =============================================================================


def _topological_sort(A: np.ndarray, threshold: float = 0.1) -> list[int]:
    """Kahn's algorithm for topological sorting."""
    d = A.shape[0]
    A_bin = (np.abs(A) > threshold).astype(int)
    in_degree = A_bin.sum(axis=0).copy()
    queue = [i for i in range(d) if in_degree[i] == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in range(d):
            if A_bin[node, child] > 0:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    # Handle any remaining nodes (cycles — shouldn't happen in a DAG)
    remaining = set(range(d)) - set(order)
    order.extend(sorted(remaining))

    return order


def _combinations(items: list, k: int):
    """Generate all k-sized combinations from items."""
    if k == 0:
        yield ()
        return
    if k > len(items):
        return
    for i, item in enumerate(items):
        for rest in _combinations(items[i + 1 :], k - 1):
            yield (item,) + rest
