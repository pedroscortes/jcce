"""
Statistical significance testing for experimental results.

This module provides statistical tests to determine if performance differences
between variants are statistically significant.
"""

import itertools
from typing import Dict, Tuple

import numpy as np
from scipy import stats


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Paired t-test for comparing two models.

    Tests the null hypothesis that two related samples have identical average values.

    Args:
        scores_a: Scores for model A (n_trials,)
        scores_b: Scores for model B (n_trials,)
        alpha: Significance level (default: 0.05)

    Returns:
        result: Dictionary with test statistics
    """
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    # Effect size (Cohen's d for paired samples)
    diff = scores_a - scores_b
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    result = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "cohens_d": float(cohens_d),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "winner": "A" if np.mean(scores_a) > np.mean(scores_b) else "B",
    }

    return result


def wilcoxon_signed_rank_test(
    scores_a: np.ndarray, scores_b: np.ndarray, alpha: float = 0.05
) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Tests whether two related samples come from the same distribution.
    More robust to outliers and non-normal distributions.

    Args:
        scores_a: Scores for model A (n_trials,)
        scores_b: Scores for model B (n_trials,)
        alpha: Significance level

    Returns:
        result: Dictionary with test statistics
    """
    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)

    result = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_diff": float(np.mean(scores_a - scores_b)),
        "median_diff": float(np.median(scores_a - scores_b)),
        "winner": "A" if np.median(scores_a) > np.median(scores_b) else "B",
    }

    return result


def bootstrap_confidence_interval(
    scores: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Args:
        scores: Sample scores (n_samples,)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)

    Returns:
        mean: Sample mean
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
    """
    n = len(scores)
    bootstrap_means = []

    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(np.mean(scores)), float(ci_lower), float(ci_upper)


def anova_test(scores_dict: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict:
    """
    One-way ANOVA test for comparing multiple models.

    Tests the null hypothesis that all groups have the same population mean.

    Args:
        scores_dict: Dictionary mapping variant names to scores
        alpha: Significance level

    Returns:
        result: Dictionary with test statistics
    """
    # Extract scores
    variant_names = list(scores_dict.keys())
    scores_list = [scores_dict[name] for name in variant_names]

    # Perform one-way ANOVA
    statistic, p_value = stats.f_oneway(*scores_list)

    result = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "variant_means": {name: float(np.mean(scores_dict[name])) for name in variant_names},
    }

    return result


def post_hoc_pairwise_tests(
    scores_dict: Dict[str, np.ndarray], alpha: float = 0.05, correction: str = "bonferroni"
) -> Dict:
    """
    Post-hoc pairwise comparisons with multiple testing correction.

    Performs pairwise t-tests between all variants with correction for
    multiple comparisons.

    Args:
        scores_dict: Dictionary mapping variant names to scores
        alpha: Significance level
        correction: Multiple testing correction ('bonferroni', 'holm', 'none')

    Returns:
        results: Dictionary with pairwise comparison results
    """
    variant_names = list(scores_dict.keys())
    n_comparisons = len(variant_names) * (len(variant_names) - 1) // 2

    # Adjusted alpha for multiple comparisons
    if correction == "bonferroni":
        alpha_adjusted = alpha / n_comparisons
    elif correction == "holm":
        # Will be applied after sorting p-values
        alpha_adjusted = alpha
    else:
        alpha_adjusted = alpha

    # Perform all pairwise tests
    pairwise_results = {}
    p_values = []
    comparisons = []

    for variant_a, variant_b in itertools.combinations(variant_names, 2):
        scores_a = scores_dict[variant_a]
        scores_b = scores_dict[variant_b]

        # Paired t-test
        test_result = paired_t_test(scores_a, scores_b, alpha=alpha_adjusted)

        comparison_name = f"{variant_a} vs {variant_b}"
        pairwise_results[comparison_name] = test_result

        p_values.append(test_result["p_value"])
        comparisons.append(comparison_name)

    # Apply Holm-Bonferroni correction if requested
    if correction == "holm":
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        for rank, idx in enumerate(sorted_indices):
            alpha_holm = alpha / (n_comparisons - rank)
            comparison_name = comparisons[idx]
            pairwise_results[comparison_name]["significant"] = p_values[idx] < alpha_holm
            pairwise_results[comparison_name]["alpha_adjusted"] = alpha_holm

    results = {
        "pairwise_comparisons": pairwise_results,
        "n_comparisons": n_comparisons,
        "correction": correction,
        "alpha": alpha,
    }

    return results


def friedman_test(scores_dict: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict:
    """
    Friedman test (non-parametric alternative to repeated measures ANOVA).

    Tests whether k related samples have different distributions.

    Args:
        scores_dict: Dictionary mapping variant names to scores
                    All variants must have same number of samples
        alpha: Significance level

    Returns:
        result: Dictionary with test statistics
    """
    # Extract scores
    variant_names = list(scores_dict.keys())
    scores_list = [scores_dict[name] for name in variant_names]

    # Stack scores into matrix (n_samples x n_variants)
    scores_matrix = np.column_stack(scores_list)

    # Perform Friedman test
    statistic, p_value = stats.friedmanchisquare(*scores_list)

    # Compute mean ranks
    ranks = stats.rankdata(scores_matrix, axis=1)
    mean_ranks = {name: float(np.mean(ranks[:, i])) for i, name in enumerate(variant_names)}

    result = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "mean_ranks": mean_ranks,
        "best_variant": min(mean_ranks, key=mean_ranks.get) if p_value < alpha else None,
    }

    return result


def compare_variants_statistical(
    results: Dict, metric_name: str = "loss", alpha: float = 0.05, n_bootstrap: int = 1000
) -> Dict:
    """
    Comprehensive statistical comparison of all variants.

    Performs multiple statistical tests to compare variants on a given metric.

    Args:
        results: Dictionary with results for each variant
        metric_name: Metric to compare ('loss', 'mcc', 'mig', etc.)
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples for confidence intervals

    Returns:
        statistical_results: Dictionary with all statistical test results
    """
    print(f"\n{'=' * 60}")
    print(f"STATISTICAL COMPARISON: {metric_name.upper()}")
    print(f"{'=' * 60}")

    variant_names = list(results.keys())

    # Extract metric values (for now, use final test metrics)
    # In practice, you'd want multiple runs to get distributions
    scores_dict = {}
    for variant in variant_names:
        if metric_name in ["loss", "recon_loss", "kl_div"]:
            # These are from final_test_metrics
            score = results[variant]["final_test_metrics"][metric_name]
            # Create pseudo-distribution (in practice, run multiple times)
            scores_dict[variant] = np.array([score] * 10)  # Placeholder
        else:
            # Disentanglement metrics
            score = results[variant]["disentanglement"][metric_name]
            scores_dict[variant] = np.array([score] * 10)  # Placeholder

    print("\nNote: For proper statistical testing, run experiments multiple times")
    print("      with different random seeds to get score distributions.")
    print("\nMean scores:")
    for variant in variant_names:
        mean_score = np.mean(scores_dict[variant])
        print(f"  {variant:15s}: {mean_score:.4f}")

    # 1. ANOVA / Friedman test
    print(f"\n{'-' * 60}")
    print("1. One-Way ANOVA Test")
    print(f"{'-' * 60}")
    anova_result = anova_test(scores_dict, alpha=alpha)
    print(f"F-statistic: {anova_result['statistic']:.4f}")
    print(f"p-value: {anova_result['p_value']:.4f}")
    print(f"Significant difference: {anova_result['significant']}")

    # 2. Post-hoc pairwise comparisons
    print(f"\n{'-' * 60}")
    print("2. Post-Hoc Pairwise Comparisons (Bonferroni correction)")
    print(f"{'-' * 60}")
    pairwise_results = post_hoc_pairwise_tests(scores_dict, alpha=alpha, correction="bonferroni")

    for comparison_name, test_result in pairwise_results["pairwise_comparisons"].items():
        print(f"\n{comparison_name}:")
        print(f"  Mean difference: {test_result['mean_diff']:.4f}")
        print(f"  p-value: {test_result['p_value']:.4f}")
        print(f"  Significant: {test_result['significant']}")
        print(f"  Cohen's d: {test_result['cohens_d']:.4f}")

    # 3. Bootstrap confidence intervals
    print(f"\n{'-' * 60}")
    print("3. Bootstrap 95% Confidence Intervals")
    print(f"{'-' * 60}")
    ci_results = {}
    for variant in variant_names:
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(
            scores_dict[variant], n_bootstrap=n_bootstrap
        )
        ci_results[variant] = {"mean": mean, "ci_lower": ci_lower, "ci_upper": ci_upper}
        print(f"{variant:15s}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    statistical_results = {
        "metric": metric_name,
        "scores": {k: v.tolist() for k, v in scores_dict.items()},
        "anova": anova_result,
        "pairwise": pairwise_results,
        "confidence_intervals": ci_results,
    }

    return statistical_results


def print_significance_matrix(pairwise_results: Dict, alpha: float = 0.05):
    """
    Print a matrix showing which pairwise comparisons are significant.

    Args:
        pairwise_results: Results from post_hoc_pairwise_tests
        alpha: Significance level
    """
    # Extract variant names
    comparisons = pairwise_results["pairwise_comparisons"]
    variant_names = set()
    for comparison in comparisons.keys():
        a, b = comparison.split(" vs ")
        variant_names.add(a)
        variant_names.add(b)

    variant_names = sorted(list(variant_names))
    n_variants = len(variant_names)

    print(f"\n{'=' * 60}")
    print("SIGNIFICANCE MATRIX")
    print(f"{'=' * 60}")
    print("Legend: Y = significant difference, N = no significant difference\n")

    # Print header
    print(f"{'':15s}", end="")
    for name in variant_names:
        print(f"{name:15s}", end="")
    print()

    # Print matrix
    for i, variant_a in enumerate(variant_names):
        print(f"{variant_a:15s}", end="")
        for j, variant_b in enumerate(variant_names):
            if i == j:
                print(f"{'—':^15s}", end="")
            elif i < j:
                # Try both orderings
                comparison_name_1 = f"{variant_a} vs {variant_b}"
                comparison_name_2 = f"{variant_b} vs {variant_a}"

                if comparison_name_1 in comparisons:
                    is_sig = comparisons[comparison_name_1]["significant"]
                elif comparison_name_2 in comparisons:
                    is_sig = comparisons[comparison_name_2]["significant"]
                else:
                    is_sig = False  # Fallback

                symbol = "Y" if is_sig else "N"
                print(f"{symbol:^15s}", end="")
            else:
                print(f"{'':^15s}", end="")
        print()

    print()
