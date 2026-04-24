"""Training utilities for JCCE models."""

from jcce.training.vae_trainer import (
    TrainState,
    create_train_state,
    train_step,
    train_epoch,
    train_vae,
    evaluate_vae,
)

from jcce.training.causal_vae_trainer import (
    CausalVAETrainState,
    create_causal_vae_train_state,
    causal_vae_train_step,
    causal_vae_train_epoch,
    train_causal_vae,
    evaluate_causal_vae,
)

# New unified trainer for Milestone 7
from jcce.training.trainer import (
    compute_vae_loss,
    train_epoch as unified_train_epoch,
    evaluate as unified_evaluate,
)

from jcce.training.metrics import (
    mean_correlation_coefficient,
    mutual_information_gap,
    sap_score,
    compute_r2_score,
    evaluate_disentanglement,
    structural_hamming_distance,
    graph_precision_recall_f1,
    evaluate_structure_recovery,
)

from jcce.training.statistical_tests import (
    paired_t_test,
    wilcoxon_signed_rank_test,
    bootstrap_confidence_interval,
    anova_test,
    post_hoc_pairwise_tests,
    friedman_test,
    compare_variants_statistical,
    print_significance_matrix,
)

__all__ = [
    # Baseline VAE
    "TrainState",
    "create_train_state",
    "train_step",
    "train_epoch",
    "train_vae",
    "evaluate_vae",
    # CausalVAE
    "CausalVAETrainState",
    "create_causal_vae_train_state",
    "causal_vae_train_step",
    "causal_vae_train_epoch",
    "train_causal_vae",
    "evaluate_causal_vae",
    # Unified trainer (Milestone 7)
    "compute_vae_loss",
    "unified_train_epoch",
    "unified_evaluate",
    # Metrics
    "mean_correlation_coefficient",
    "mutual_information_gap",
    "sap_score",
    "compute_r2_score",
    "evaluate_disentanglement",
    "structural_hamming_distance",
    "graph_precision_recall_f1",
    "evaluate_structure_recovery",
    # Statistical tests
    "paired_t_test",
    "wilcoxon_signed_rank_test",
    "bootstrap_confidence_interval",
    "anova_test",
    "post_hoc_pairwise_tests",
    "friedman_test",
    "compare_variants_statistical",
    "print_significance_matrix",
]
