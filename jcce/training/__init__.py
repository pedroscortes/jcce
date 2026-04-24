"""Training utilities for JCCE models."""

from jcce.training.causal_vae_trainer import (
    CausalVAETrainState,
    causal_vae_train_epoch,
    causal_vae_train_step,
    create_causal_vae_train_state,
    evaluate_causal_vae,
    train_causal_vae,
)
from jcce.training.metrics import (
    compute_r2_score,
    evaluate_disentanglement,
    evaluate_structure_recovery,
    graph_precision_recall_f1,
    mean_correlation_coefficient,
    mutual_information_gap,
    sap_score,
    structural_hamming_distance,
)
from jcce.training.statistical_tests import (
    anova_test,
    bootstrap_confidence_interval,
    compare_variants_statistical,
    friedman_test,
    paired_t_test,
    post_hoc_pairwise_tests,
    print_significance_matrix,
    wilcoxon_signed_rank_test,
)

# New unified trainer for Milestone 7
from jcce.training.trainer import (
    compute_vae_loss,
)
from jcce.training.trainer import (
    evaluate as unified_evaluate,
)
from jcce.training.trainer import (
    train_epoch as unified_train_epoch,
)
from jcce.training.vae_trainer import (
    TrainState,
    create_train_state,
    evaluate_vae,
    train_epoch,
    train_step,
    train_vae,
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
