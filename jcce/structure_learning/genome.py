"""
Memetic Algorithm - C-Lite Version Utilities.

Simplified genome for Option C-Lite (classification-guided causal discovery):
1. Structure learning hyperparameters
2. Processor architecture (optional - can use fixed MLP for C-Lite-Simple)
3. Latent confounder hyperparameters (L genes)

NO VAE components - direct classification on Markov Blanket features.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import random

from .memetic_utils import crossover, mutate

# Import processor options from MA-Full
from .memetic_utils_full import (
    ELM_ACTIVATIONS,
    ELM_HIDDEN_DIMS,
    ELM_N_HIDDEN_NODES,
    GNN_AGGREGATIONS,
    GNN_HIDDEN_DIMS,
    GNN_N_LAYERS,
    GNN_TYPES,
    GRU_BIDIRECTIONALS,
    GRU_DROPOUTS,
    GRU_HIDDEN_SIZES,
    GRU_N_LAYERS,
    LSTM_BIDIRECTIONALS,
    LSTM_DROPOUTS,
    LSTM_HIDDEN_SIZES,
    LSTM_N_LAYERS,
    MAMBA_D_CONVS,
    MAMBA_D_MODELS,
    MAMBA_D_STATES,
    MAMBA_EXPANDS,
    PROCESSOR_BATCH_SIZES,
    PROCESSOR_EPOCHS,
    PROCESSOR_LRS,
    PROCESSOR_TYPES,
    STRUCTURE_ALGORITHMS,
    TRANSFORMER_D_FFS,
    TRANSFORMER_D_MODELS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_N_LAYERS,
)

# ============================================================================
# Latent Confounder (L) Hyperparameter Options
# ============================================================================

LATENT_RANK_K = [2, 3, 5, 8, 10]  # Number of latent factors
LAMBDA_L = [0.01, 0.02, 0.05, 0.1, 0.2]  # Nuclear norm penalty on L
LAMBDA_BOW = [0.01, 0.05, 0.1, 0.2, 0.5]  # Bow-free penalty (A ⊙ Ω)
WARM_START_L_ITERS = [10, 15, 20, 30]  # Iterations before activating L


# ============================================================================
# C-Lite Genome
# ============================================================================


@dataclass
class MACliteGenome:
    """
    Simplified genome for Option C-Lite: Classification-guided causal discovery.

    Components (~30 total):
    - 1: Graph topology (A_topology)
    - 4: Structure learning hyperparams (algorithm, lambda_1, lambda_2, lr)
    - 1: Processor type selector
    - 21: Processor-specific architectures (all 6 types)
    - 3: Processor training hyperparameters

    NO VAE components - this is the key simplification!
    """

    # ========== Component 1: Graph Structure ==========
    A_topology: jnp.ndarray  # Binary adjacency matrix (n_vars, n_vars)

    # ========== Components 2-5: Structure Learning Hyperparameters ==========
    sl_algorithm_idx: int  # Index into STRUCTURE_ALGORITHMS (0=NOTEARS, 1=GOLEM, 2=DAGMA)
    sl_lambda_1_idx: int  # Index into lambda_1 range [0.01, 0.5]
    sl_lambda_2_idx: int  # Index into lambda_2 range [0.0001, 0.01]
    sl_lr_idx: int  # Index into learning rate range [0.001, 0.01]

    # ========== Component 6: Processor Type Selector ==========
    processor_type_idx: int  # Index into PROCESSOR_TYPES

    # ========== Components 7-27: Processor-Specific Architectures ==========
    # Mamba (4 components)
    mamba_d_model_idx: int
    mamba_d_state_idx: int
    mamba_d_conv_idx: int
    mamba_expand_idx: int

    # Transformer (4 components)
    transformer_d_model_idx: int
    transformer_n_heads_idx: int
    transformer_n_layers_idx: int
    transformer_d_ff_idx: int

    # LSTM (4 components)
    lstm_hidden_size_idx: int
    lstm_n_layers_idx: int
    lstm_bidirectional_idx: int
    lstm_dropout_idx: int

    # GRU (4 components)
    gru_hidden_size_idx: int
    gru_n_layers_idx: int
    gru_bidirectional_idx: int
    gru_dropout_idx: int

    # GNN (4 components)
    gnn_hidden_dim_idx: int
    gnn_n_layers_idx: int
    gnn_type_idx: int
    gnn_aggregation_idx: int

    # ELM (3 components)
    elm_hidden_dim_idx: int
    elm_n_hidden_nodes_idx: int
    elm_activation_idx: int

    # ========== Components 28-30: Processor Training Hyperparameters ==========
    processor_lr_idx: int
    processor_epochs_idx: int
    processor_batch_size_idx: int

    # ========== Components 31-34: Latent Confounder (L) Hyperparameters ==========
    latent_rank_k_idx: int = 2  # Index into LATENT_RANK_K (default: k=5)
    lambda_L_idx: int = 2  # Index into LAMBDA_L (default: 0.05)
    lambda_bow_idx: int = 2  # Index into LAMBDA_BOW (default: 0.1)
    warm_start_L_iters_idx: int = 1  # Index into WARM_START_L_ITERS (default: 15)

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for logging."""
        return {
            "topology_edges": int(jnp.sum(self.A_topology)),
            "sl_algorithm": STRUCTURE_ALGORITHMS[int(self.sl_algorithm_idx)],
            "sl_lambda_1_idx": int(self.sl_lambda_1_idx),
            "sl_lambda_2_idx": int(self.sl_lambda_2_idx),
            "sl_lr_idx": int(self.sl_lr_idx),
            "processor_type": PROCESSOR_TYPES[int(self.processor_type_idx)],
            # Latent confounder genes
            "latent_rank_k": LATENT_RANK_K[int(self.latent_rank_k_idx)],
            "lambda_L": LAMBDA_L[int(self.lambda_L_idx)],
            "lambda_bow": LAMBDA_BOW[int(self.lambda_bow_idx)],
            "warm_start_L_iters": WARM_START_L_ITERS[int(self.warm_start_L_iters_idx)],
        }


# ============================================================================
# Genome Initialization
# ============================================================================


def initialize_genome_clite(
    n_vars: int,
    edge_prob: float = 0.3,
    key: random.PRNGKey = None,
    # Processor type override (for C-Lite-Simple with fixed MLP)
    fixed_processor_type: Optional[str] = None,
) -> MACliteGenome:
    """
    Initialize a random C-Lite genome.

    Args:
        n_vars: Number of variables in causal graph
        edge_prob: Probability of adding each edge to DAG
        key: Random key
        fixed_processor_type: If provided, fix processor type (e.g., 'mlp' for C-Lite-Simple)

    Returns:
        Initialized MACliteGenome
    """
    if key is None:
        key = random.PRNGKey(0)

    # Generate random DAG topology
    from .memetic_utils import random_dag

    key, subkey = random.split(key)
    A_topology = random_dag(n_vars, edge_prob=edge_prob, key=subkey)

    # Structure learning hyperparameters
    key, subkey = random.split(key)
    sl_algorithm_idx = int(random.randint(subkey, (), 0, len(STRUCTURE_ALGORITHMS)))

    key, subkey = random.split(key)
    sl_lambda_1_idx = int(random.randint(subkey, (), 0, 10))

    key, subkey = random.split(key)
    sl_lambda_2_idx = int(random.randint(subkey, (), 0, 10))

    key, subkey = random.split(key)
    sl_lr_idx = int(random.randint(subkey, (), 0, 10))

    # Processor type selector
    key, subkey = random.split(key)
    if fixed_processor_type is not None and fixed_processor_type in PROCESSOR_TYPES:
        processor_type_idx = PROCESSOR_TYPES.index(fixed_processor_type)
    else:
        processor_type_idx = int(random.randint(subkey, (), 0, len(PROCESSOR_TYPES)))

    # Processor architectures (random for all types)
    key, subkey = random.split(key)
    mamba_d_model_idx = int(random.randint(subkey, (), 0, len(MAMBA_D_MODELS)))
    key, subkey = random.split(key)
    mamba_d_state_idx = int(random.randint(subkey, (), 0, len(MAMBA_D_STATES)))
    key, subkey = random.split(key)
    mamba_d_conv_idx = int(random.randint(subkey, (), 0, len(MAMBA_D_CONVS)))
    key, subkey = random.split(key)
    mamba_expand_idx = int(random.randint(subkey, (), 0, len(MAMBA_EXPANDS)))

    key, subkey = random.split(key)
    transformer_d_model_idx = int(random.randint(subkey, (), 0, len(TRANSFORMER_D_MODELS)))
    key, subkey = random.split(key)
    transformer_n_heads_idx = int(random.randint(subkey, (), 0, len(TRANSFORMER_N_HEADS)))
    key, subkey = random.split(key)
    transformer_n_layers_idx = int(random.randint(subkey, (), 0, len(TRANSFORMER_N_LAYERS)))
    key, subkey = random.split(key)
    transformer_d_ff_idx = int(random.randint(subkey, (), 0, len(TRANSFORMER_D_FFS)))

    key, subkey = random.split(key)
    lstm_hidden_size_idx = int(random.randint(subkey, (), 0, len(LSTM_HIDDEN_SIZES)))
    key, subkey = random.split(key)
    lstm_n_layers_idx = int(random.randint(subkey, (), 0, len(LSTM_N_LAYERS)))
    key, subkey = random.split(key)
    lstm_bidirectional_idx = int(random.randint(subkey, (), 0, len(LSTM_BIDIRECTIONALS)))
    key, subkey = random.split(key)
    lstm_dropout_idx = int(random.randint(subkey, (), 0, len(LSTM_DROPOUTS)))

    key, subkey = random.split(key)
    gru_hidden_size_idx = int(random.randint(subkey, (), 0, len(GRU_HIDDEN_SIZES)))
    key, subkey = random.split(key)
    gru_n_layers_idx = int(random.randint(subkey, (), 0, len(GRU_N_LAYERS)))
    key, subkey = random.split(key)
    gru_bidirectional_idx = int(random.randint(subkey, (), 0, len(GRU_BIDIRECTIONALS)))
    key, subkey = random.split(key)
    gru_dropout_idx = int(random.randint(subkey, (), 0, len(GRU_DROPOUTS)))

    key, subkey = random.split(key)
    gnn_hidden_dim_idx = int(random.randint(subkey, (), 0, len(GNN_HIDDEN_DIMS)))
    key, subkey = random.split(key)
    gnn_n_layers_idx = int(random.randint(subkey, (), 0, len(GNN_N_LAYERS)))
    key, subkey = random.split(key)
    gnn_type_idx = int(random.randint(subkey, (), 0, len(GNN_TYPES)))
    key, subkey = random.split(key)
    gnn_aggregation_idx = int(random.randint(subkey, (), 0, len(GNN_AGGREGATIONS)))

    key, subkey = random.split(key)
    elm_hidden_dim_idx = int(random.randint(subkey, (), 0, len(ELM_HIDDEN_DIMS)))
    key, subkey = random.split(key)
    elm_n_hidden_nodes_idx = int(random.randint(subkey, (), 0, len(ELM_N_HIDDEN_NODES)))
    key, subkey = random.split(key)
    elm_activation_idx = int(random.randint(subkey, (), 0, len(ELM_ACTIVATIONS)))

    # Processor training hyperparameters
    key, subkey = random.split(key)
    processor_lr_idx = int(random.randint(subkey, (), 0, len(PROCESSOR_LRS)))
    key, subkey = random.split(key)
    processor_epochs_idx = int(random.randint(subkey, (), 0, len(PROCESSOR_EPOCHS)))
    key, subkey = random.split(key)
    processor_batch_size_idx = int(random.randint(subkey, (), 0, len(PROCESSOR_BATCH_SIZES)))

    # Latent confounder hyperparameters
    key, subkey = random.split(key)
    latent_rank_k_idx = int(random.randint(subkey, (), 0, len(LATENT_RANK_K)))
    key, subkey = random.split(key)
    lambda_L_idx = int(random.randint(subkey, (), 0, len(LAMBDA_L)))
    key, subkey = random.split(key)
    lambda_bow_idx = int(random.randint(subkey, (), 0, len(LAMBDA_BOW)))
    key, subkey = random.split(key)
    warm_start_L_iters_idx = int(random.randint(subkey, (), 0, len(WARM_START_L_ITERS)))

    return MACliteGenome(
        A_topology=A_topology,
        sl_algorithm_idx=sl_algorithm_idx,
        sl_lambda_1_idx=sl_lambda_1_idx,
        sl_lambda_2_idx=sl_lambda_2_idx,
        sl_lr_idx=sl_lr_idx,
        processor_type_idx=processor_type_idx,
        mamba_d_model_idx=mamba_d_model_idx,
        mamba_d_state_idx=mamba_d_state_idx,
        mamba_d_conv_idx=mamba_d_conv_idx,
        mamba_expand_idx=mamba_expand_idx,
        transformer_d_model_idx=transformer_d_model_idx,
        transformer_n_heads_idx=transformer_n_heads_idx,
        transformer_n_layers_idx=transformer_n_layers_idx,
        transformer_d_ff_idx=transformer_d_ff_idx,
        lstm_hidden_size_idx=lstm_hidden_size_idx,
        lstm_n_layers_idx=lstm_n_layers_idx,
        lstm_bidirectional_idx=lstm_bidirectional_idx,
        lstm_dropout_idx=lstm_dropout_idx,
        gru_hidden_size_idx=gru_hidden_size_idx,
        gru_n_layers_idx=gru_n_layers_idx,
        gru_bidirectional_idx=gru_bidirectional_idx,
        gru_dropout_idx=gru_dropout_idx,
        gnn_hidden_dim_idx=gnn_hidden_dim_idx,
        gnn_n_layers_idx=gnn_n_layers_idx,
        gnn_type_idx=gnn_type_idx,
        gnn_aggregation_idx=gnn_aggregation_idx,
        elm_hidden_dim_idx=elm_hidden_dim_idx,
        elm_n_hidden_nodes_idx=elm_n_hidden_nodes_idx,
        elm_activation_idx=elm_activation_idx,
        processor_lr_idx=processor_lr_idx,
        processor_epochs_idx=processor_epochs_idx,
        processor_batch_size_idx=processor_batch_size_idx,
        # Latent confounder genes
        latent_rank_k_idx=latent_rank_k_idx,
        lambda_L_idx=lambda_L_idx,
        lambda_bow_idx=lambda_bow_idx,
        warm_start_L_iters_idx=warm_start_L_iters_idx,
    )


# ============================================================================
# Genome Conversion Functions
# ============================================================================


def genome_to_sl_config_clite(genome: MACliteGenome) -> Dict[str, float]:
    """Convert genome to structure learning configuration."""
    # Lambda 1: [0.1, 2.0] with 10 discrete levels - INCREASED for stronger sparsity
    lambda_1_range = jnp.array([0.1, 2.0])  # Was [0.01, 0.5]
    lambda_1 = lambda_1_range[0] + (genome.sl_lambda_1_idx / 9) * (
        lambda_1_range[1] - lambda_1_range[0]
    )

    # Lambda 2: [0.001, 0.1] with log spacing - INCREASED for stronger acyclicity
    lambda_2_values = jnp.logspace(-3, -1, 10)  # Was logspace(-4, -2, 10)
    lambda_2 = lambda_2_values[int(genome.sl_lambda_2_idx)]

    # Learning rate: [0.001, 0.01] with 10 discrete levels
    lr_range = jnp.array([0.001, 0.01])
    lr = lr_range[0] + (genome.sl_lr_idx / 9) * (lr_range[1] - lr_range[0])

    return {
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "learning_rate": float(lr),
    }


def genome_to_sl_algorithm_clite(genome: MACliteGenome) -> str:
    """Convert genome to structure learning algorithm name."""
    return STRUCTURE_ALGORITHMS[int(genome.sl_algorithm_idx)]


def genome_to_processor_config_clite(genome: MACliteGenome) -> Dict[str, Any]:
    """
    Convert genome to processor configuration dictionary.

    Same as MA-Full processor config extraction.
    """
    processor_type = PROCESSOR_TYPES[int(genome.processor_type_idx)]

    config = {
        "processor_type": processor_type,
        "learning_rate": PROCESSOR_LRS[int(genome.processor_lr_idx)],
        "epochs": PROCESSOR_EPOCHS[int(genome.processor_epochs_idx)],
        "batch_size": PROCESSOR_BATCH_SIZES[int(genome.processor_batch_size_idx)],
    }

    if processor_type == "mamba":
        config.update(
            {
                "d_model": MAMBA_D_MODELS[int(genome.mamba_d_model_idx)],
                "d_state": MAMBA_D_STATES[int(genome.mamba_d_state_idx)],
                "d_conv": MAMBA_D_CONVS[int(genome.mamba_d_conv_idx)],
                "expand": MAMBA_EXPANDS[int(genome.mamba_expand_idx)],
            }
        )
    elif processor_type == "transformer":
        config.update(
            {
                "d_model": TRANSFORMER_D_MODELS[int(genome.transformer_d_model_idx)],
                "n_heads": TRANSFORMER_N_HEADS[int(genome.transformer_n_heads_idx)],
                "n_layers": TRANSFORMER_N_LAYERS[int(genome.transformer_n_layers_idx)],
                "d_ff": TRANSFORMER_D_FFS[int(genome.transformer_d_ff_idx)],
            }
        )
    elif processor_type == "lstm":
        config.update(
            {
                "hidden_size": LSTM_HIDDEN_SIZES[int(genome.lstm_hidden_size_idx)],
                "n_layers": LSTM_N_LAYERS[int(genome.lstm_n_layers_idx)],
                "bidirectional": LSTM_BIDIRECTIONALS[int(genome.lstm_bidirectional_idx)],
                "dropout": LSTM_DROPOUTS[int(genome.lstm_dropout_idx)],
            }
        )
    elif processor_type == "gru":
        config.update(
            {
                "hidden_size": GRU_HIDDEN_SIZES[int(genome.gru_hidden_size_idx)],
                "n_layers": GRU_N_LAYERS[int(genome.gru_n_layers_idx)],
                "bidirectional": GRU_BIDIRECTIONALS[int(genome.gru_bidirectional_idx)],
                "dropout": GRU_DROPOUTS[int(genome.gru_dropout_idx)],
            }
        )
    elif processor_type == "gnn":
        config.update(
            {
                "hidden_dim": GNN_HIDDEN_DIMS[int(genome.gnn_hidden_dim_idx)],
                "n_layers": GNN_N_LAYERS[int(genome.gnn_n_layers_idx)],
                "gnn_type": GNN_TYPES[int(genome.gnn_type_idx)],
                "aggregation": GNN_AGGREGATIONS[int(genome.gnn_aggregation_idx)],
            }
        )
    elif processor_type == "elm":
        config.update(
            {
                "hidden_dim": ELM_HIDDEN_DIMS[int(genome.elm_hidden_dim_idx)],
                "n_hidden_nodes": ELM_N_HIDDEN_NODES[int(genome.elm_n_hidden_nodes_idx)],
                "activation": ELM_ACTIVATIONS[int(genome.elm_activation_idx)],
            }
        )

    return config


def genome_to_latent_config_clite(genome: MACliteGenome) -> Dict[str, Any]:
    """
    Convert genome to latent confounder (L) configuration.

    Extracts L hyperparameters from genome for use in GOLEM.

    Returns:
        Dict with:
            - latent_rank_k: int (number of latent factors)
            - lambda_L: float (nuclear norm penalty)
            - lambda_bow: float (bow-free penalty)
            - warm_start_L_iters: int (warm-start iterations)
    """
    return {
        "latent_rank_k": LATENT_RANK_K[int(genome.latent_rank_k_idx)],
        "lambda_L": LAMBDA_L[int(genome.lambda_L_idx)],
        "lambda_bow": LAMBDA_BOW[int(genome.lambda_bow_idx)],
        "warm_start_L_iters": WARM_START_L_ITERS[int(genome.warm_start_L_iters_idx)],
    }


# ============================================================================
# Genetic Operators (Crossover & Mutation)
# ============================================================================


def crossover_clite(
    parent1: MACliteGenome, parent2: MACliteGenome, key: random.PRNGKey
) -> Tuple[MACliteGenome, MACliteGenome]:
    """
    Crossover two C-Lite genomes to create two offspring.

    Strategy: Uniform crossover for all components.
    """
    key, subkey1, subkey2 = random.split(key, 3)

    # Crossover graph topologies
    A1_new, A2_new = crossover(parent1.A_topology, parent2.A_topology, subkey1, method="uniform")

    # Crossover all index-based components (uniform probability)
    n_indices = (
        35  # Total indices (4 SL + 1 proc_type + 23 proc_params + 3 training params + 4 L genes)
    )
    key, subkey = random.split(key)
    mask = random.bernoulli(subkey, p=0.5, shape=(n_indices,))

    # Extract all indices from both parents
    p1_indices = [
        parent1.sl_algorithm_idx,
        parent1.sl_lambda_1_idx,
        parent1.sl_lambda_2_idx,
        parent1.sl_lr_idx,
        parent1.processor_type_idx,
        parent1.mamba_d_model_idx,
        parent1.mamba_d_state_idx,
        parent1.mamba_d_conv_idx,
        parent1.mamba_expand_idx,
        parent1.transformer_d_model_idx,
        parent1.transformer_n_heads_idx,
        parent1.transformer_n_layers_idx,
        parent1.transformer_d_ff_idx,
        parent1.lstm_hidden_size_idx,
        parent1.lstm_n_layers_idx,
        parent1.lstm_bidirectional_idx,
        parent1.lstm_dropout_idx,
        parent1.gru_hidden_size_idx,
        parent1.gru_n_layers_idx,
        parent1.gru_bidirectional_idx,
        parent1.gru_dropout_idx,
        parent1.gnn_hidden_dim_idx,
        parent1.gnn_n_layers_idx,
        parent1.gnn_type_idx,
        parent1.gnn_aggregation_idx,
        parent1.elm_hidden_dim_idx,
        parent1.elm_n_hidden_nodes_idx,
        parent1.elm_activation_idx,
        parent1.processor_lr_idx,
        parent1.processor_epochs_idx,
        parent1.processor_batch_size_idx,
        # Latent confounder genes
        parent1.latent_rank_k_idx,
        parent1.lambda_L_idx,
        parent1.lambda_bow_idx,
        parent1.warm_start_L_iters_idx,
    ]
    p2_indices = [
        parent2.sl_algorithm_idx,
        parent2.sl_lambda_1_idx,
        parent2.sl_lambda_2_idx,
        parent2.sl_lr_idx,
        parent2.processor_type_idx,
        parent2.mamba_d_model_idx,
        parent2.mamba_d_state_idx,
        parent2.mamba_d_conv_idx,
        parent2.mamba_expand_idx,
        parent2.transformer_d_model_idx,
        parent2.transformer_n_heads_idx,
        parent2.transformer_n_layers_idx,
        parent2.transformer_d_ff_idx,
        parent2.lstm_hidden_size_idx,
        parent2.lstm_n_layers_idx,
        parent2.lstm_bidirectional_idx,
        parent2.lstm_dropout_idx,
        parent2.gru_hidden_size_idx,
        parent2.gru_n_layers_idx,
        parent2.gru_bidirectional_idx,
        parent2.gru_dropout_idx,
        parent2.gnn_hidden_dim_idx,
        parent2.gnn_n_layers_idx,
        parent2.gnn_type_idx,
        parent2.gnn_aggregation_idx,
        parent2.elm_hidden_dim_idx,
        parent2.elm_n_hidden_nodes_idx,
        parent2.elm_activation_idx,
        parent2.processor_lr_idx,
        parent2.processor_epochs_idx,
        parent2.processor_batch_size_idx,
        # Latent confounder genes
        parent2.latent_rank_k_idx,
        parent2.lambda_L_idx,
        parent2.lambda_bow_idx,
        parent2.warm_start_L_iters_idx,
    ]

    # Apply crossover mask
    c1_indices = [p1_indices[i] if mask[i] else p2_indices[i] for i in range(n_indices)]
    c2_indices = [p2_indices[i] if mask[i] else p1_indices[i] for i in range(n_indices)]

    # Create child genomes
    child1 = MACliteGenome(
        A_topology=A1_new,
        sl_algorithm_idx=c1_indices[0],
        sl_lambda_1_idx=c1_indices[1],
        sl_lambda_2_idx=c1_indices[2],
        sl_lr_idx=c1_indices[3],
        processor_type_idx=c1_indices[4],
        mamba_d_model_idx=c1_indices[5],
        mamba_d_state_idx=c1_indices[6],
        mamba_d_conv_idx=c1_indices[7],
        mamba_expand_idx=c1_indices[8],
        transformer_d_model_idx=c1_indices[9],
        transformer_n_heads_idx=c1_indices[10],
        transformer_n_layers_idx=c1_indices[11],
        transformer_d_ff_idx=c1_indices[12],
        lstm_hidden_size_idx=c1_indices[13],
        lstm_n_layers_idx=c1_indices[14],
        lstm_bidirectional_idx=c1_indices[15],
        lstm_dropout_idx=c1_indices[16],
        gru_hidden_size_idx=c1_indices[17],
        gru_n_layers_idx=c1_indices[18],
        gru_bidirectional_idx=c1_indices[19],
        gru_dropout_idx=c1_indices[20],
        gnn_hidden_dim_idx=c1_indices[21],
        gnn_n_layers_idx=c1_indices[22],
        gnn_type_idx=c1_indices[23],
        gnn_aggregation_idx=c1_indices[24],
        elm_hidden_dim_idx=c1_indices[25],
        elm_n_hidden_nodes_idx=c1_indices[26],
        elm_activation_idx=c1_indices[27],
        processor_lr_idx=c1_indices[28],
        processor_epochs_idx=c1_indices[29],
        processor_batch_size_idx=c1_indices[30],
        # Latent confounder genes
        latent_rank_k_idx=c1_indices[31],
        lambda_L_idx=c1_indices[32],
        lambda_bow_idx=c1_indices[33],
        warm_start_L_iters_idx=c1_indices[34],
    )

    child2 = MACliteGenome(
        A_topology=A2_new,
        sl_algorithm_idx=c2_indices[0],
        sl_lambda_1_idx=c2_indices[1],
        sl_lambda_2_idx=c2_indices[2],
        sl_lr_idx=c2_indices[3],
        processor_type_idx=c2_indices[4],
        mamba_d_model_idx=c2_indices[5],
        mamba_d_state_idx=c2_indices[6],
        mamba_d_conv_idx=c2_indices[7],
        mamba_expand_idx=c2_indices[8],
        transformer_d_model_idx=c2_indices[9],
        transformer_n_heads_idx=c2_indices[10],
        transformer_n_layers_idx=c2_indices[11],
        transformer_d_ff_idx=c2_indices[12],
        lstm_hidden_size_idx=c2_indices[13],
        lstm_n_layers_idx=c2_indices[14],
        lstm_bidirectional_idx=c2_indices[15],
        lstm_dropout_idx=c2_indices[16],
        gru_hidden_size_idx=c2_indices[17],
        gru_n_layers_idx=c2_indices[18],
        gru_bidirectional_idx=c2_indices[19],
        gru_dropout_idx=c2_indices[20],
        gnn_hidden_dim_idx=c2_indices[21],
        gnn_n_layers_idx=c2_indices[22],
        gnn_type_idx=c2_indices[23],
        gnn_aggregation_idx=c2_indices[24],
        elm_hidden_dim_idx=c2_indices[25],
        elm_n_hidden_nodes_idx=c2_indices[26],
        elm_activation_idx=c2_indices[27],
        processor_lr_idx=c2_indices[28],
        processor_epochs_idx=c2_indices[29],
        processor_batch_size_idx=c2_indices[30],
        # Latent confounder genes
        latent_rank_k_idx=c2_indices[31],
        lambda_L_idx=c2_indices[32],
        lambda_bow_idx=c2_indices[33],
        warm_start_L_iters_idx=c2_indices[34],
    )

    return child1, child2


def mutate_clite(
    genome: MACliteGenome, key: random.PRNGKey, mutation_rate: float = 0.15
) -> MACliteGenome:
    """
    Mutate a C-Lite genome.

    Strategy: Mutate each component with probability mutation_rate.
    """
    key, subkey = random.split(key)

    # Mutate graph topology
    A_mutated = mutate(genome.A_topology, subkey, mutation_rate=mutation_rate, mutation_type="flip")

    # Mutate each index-based component
    def mutate_index(idx: int, max_val: int, key: random.PRNGKey) -> int:
        """Mutate a single index with probability mutation_rate."""
        key, subkey1, subkey2 = random.split(key, 3)
        should_mutate = random.bernoulli(subkey1, p=mutation_rate)
        new_idx = random.randint(subkey2, (), 0, max_val)
        return int(jnp.where(should_mutate, new_idx, idx))

    # Mutate all indices
    key, subkey = random.split(key)
    sl_algorithm_idx = mutate_index(genome.sl_algorithm_idx, len(STRUCTURE_ALGORITHMS), subkey)
    key, subkey = random.split(key)
    sl_lambda_1_idx = mutate_index(genome.sl_lambda_1_idx, 10, subkey)
    key, subkey = random.split(key)
    sl_lambda_2_idx = mutate_index(genome.sl_lambda_2_idx, 10, subkey)
    key, subkey = random.split(key)
    sl_lr_idx = mutate_index(genome.sl_lr_idx, 10, subkey)

    key, subkey = random.split(key)
    processor_type_idx = mutate_index(genome.processor_type_idx, len(PROCESSOR_TYPES), subkey)

    # Mutate processor architectures
    key, subkey = random.split(key)
    mamba_d_model_idx = mutate_index(genome.mamba_d_model_idx, len(MAMBA_D_MODELS), subkey)
    key, subkey = random.split(key)
    mamba_d_state_idx = mutate_index(genome.mamba_d_state_idx, len(MAMBA_D_STATES), subkey)
    key, subkey = random.split(key)
    mamba_d_conv_idx = mutate_index(genome.mamba_d_conv_idx, len(MAMBA_D_CONVS), subkey)
    key, subkey = random.split(key)
    mamba_expand_idx = mutate_index(genome.mamba_expand_idx, len(MAMBA_EXPANDS), subkey)

    key, subkey = random.split(key)
    transformer_d_model_idx = mutate_index(
        genome.transformer_d_model_idx, len(TRANSFORMER_D_MODELS), subkey
    )
    key, subkey = random.split(key)
    transformer_n_heads_idx = mutate_index(
        genome.transformer_n_heads_idx, len(TRANSFORMER_N_HEADS), subkey
    )
    key, subkey = random.split(key)
    transformer_n_layers_idx = mutate_index(
        genome.transformer_n_layers_idx, len(TRANSFORMER_N_LAYERS), subkey
    )
    key, subkey = random.split(key)
    transformer_d_ff_idx = mutate_index(genome.transformer_d_ff_idx, len(TRANSFORMER_D_FFS), subkey)

    key, subkey = random.split(key)
    lstm_hidden_size_idx = mutate_index(genome.lstm_hidden_size_idx, len(LSTM_HIDDEN_SIZES), subkey)
    key, subkey = random.split(key)
    lstm_n_layers_idx = mutate_index(genome.lstm_n_layers_idx, len(LSTM_N_LAYERS), subkey)
    key, subkey = random.split(key)
    lstm_bidirectional_idx = mutate_index(
        genome.lstm_bidirectional_idx, len(LSTM_BIDIRECTIONALS), subkey
    )
    key, subkey = random.split(key)
    lstm_dropout_idx = mutate_index(genome.lstm_dropout_idx, len(LSTM_DROPOUTS), subkey)

    key, subkey = random.split(key)
    gru_hidden_size_idx = mutate_index(genome.gru_hidden_size_idx, len(GRU_HIDDEN_SIZES), subkey)
    key, subkey = random.split(key)
    gru_n_layers_idx = mutate_index(genome.gru_n_layers_idx, len(GRU_N_LAYERS), subkey)
    key, subkey = random.split(key)
    gru_bidirectional_idx = mutate_index(
        genome.gru_bidirectional_idx, len(GRU_BIDIRECTIONALS), subkey
    )
    key, subkey = random.split(key)
    gru_dropout_idx = mutate_index(genome.gru_dropout_idx, len(GRU_DROPOUTS), subkey)

    key, subkey = random.split(key)
    gnn_hidden_dim_idx = mutate_index(genome.gnn_hidden_dim_idx, len(GNN_HIDDEN_DIMS), subkey)
    key, subkey = random.split(key)
    gnn_n_layers_idx = mutate_index(genome.gnn_n_layers_idx, len(GNN_N_LAYERS), subkey)
    key, subkey = random.split(key)
    gnn_type_idx = mutate_index(genome.gnn_type_idx, len(GNN_TYPES), subkey)
    key, subkey = random.split(key)
    gnn_aggregation_idx = mutate_index(genome.gnn_aggregation_idx, len(GNN_AGGREGATIONS), subkey)

    key, subkey = random.split(key)
    elm_hidden_dim_idx = mutate_index(genome.elm_hidden_dim_idx, len(ELM_HIDDEN_DIMS), subkey)
    key, subkey = random.split(key)
    elm_n_hidden_nodes_idx = mutate_index(
        genome.elm_n_hidden_nodes_idx, len(ELM_N_HIDDEN_NODES), subkey
    )
    key, subkey = random.split(key)
    elm_activation_idx = mutate_index(genome.elm_activation_idx, len(ELM_ACTIVATIONS), subkey)

    key, subkey = random.split(key)
    processor_lr_idx = mutate_index(genome.processor_lr_idx, len(PROCESSOR_LRS), subkey)
    key, subkey = random.split(key)
    processor_epochs_idx = mutate_index(genome.processor_epochs_idx, len(PROCESSOR_EPOCHS), subkey)
    key, subkey = random.split(key)
    processor_batch_size_idx = mutate_index(
        genome.processor_batch_size_idx, len(PROCESSOR_BATCH_SIZES), subkey
    )

    # Mutate latent confounder genes
    key, subkey = random.split(key)
    latent_rank_k_idx = mutate_index(genome.latent_rank_k_idx, len(LATENT_RANK_K), subkey)
    key, subkey = random.split(key)
    lambda_L_idx = mutate_index(genome.lambda_L_idx, len(LAMBDA_L), subkey)
    key, subkey = random.split(key)
    lambda_bow_idx = mutate_index(genome.lambda_bow_idx, len(LAMBDA_BOW), subkey)
    key, subkey = random.split(key)
    warm_start_L_iters_idx = mutate_index(
        genome.warm_start_L_iters_idx, len(WARM_START_L_ITERS), subkey
    )

    return MACliteGenome(
        A_topology=A_mutated,
        sl_algorithm_idx=sl_algorithm_idx,
        sl_lambda_1_idx=sl_lambda_1_idx,
        sl_lambda_2_idx=sl_lambda_2_idx,
        sl_lr_idx=sl_lr_idx,
        processor_type_idx=processor_type_idx,
        mamba_d_model_idx=mamba_d_model_idx,
        mamba_d_state_idx=mamba_d_state_idx,
        mamba_d_conv_idx=mamba_d_conv_idx,
        mamba_expand_idx=mamba_expand_idx,
        transformer_d_model_idx=transformer_d_model_idx,
        transformer_n_heads_idx=transformer_n_heads_idx,
        transformer_n_layers_idx=transformer_n_layers_idx,
        transformer_d_ff_idx=transformer_d_ff_idx,
        lstm_hidden_size_idx=lstm_hidden_size_idx,
        lstm_n_layers_idx=lstm_n_layers_idx,
        lstm_bidirectional_idx=lstm_bidirectional_idx,
        lstm_dropout_idx=lstm_dropout_idx,
        gru_hidden_size_idx=gru_hidden_size_idx,
        gru_n_layers_idx=gru_n_layers_idx,
        gru_bidirectional_idx=gru_bidirectional_idx,
        gru_dropout_idx=gru_dropout_idx,
        gnn_hidden_dim_idx=gnn_hidden_dim_idx,
        gnn_n_layers_idx=gnn_n_layers_idx,
        gnn_type_idx=gnn_type_idx,
        gnn_aggregation_idx=gnn_aggregation_idx,
        elm_hidden_dim_idx=elm_hidden_dim_idx,
        elm_n_hidden_nodes_idx=elm_n_hidden_nodes_idx,
        elm_activation_idx=elm_activation_idx,
        processor_lr_idx=processor_lr_idx,
        processor_epochs_idx=processor_epochs_idx,
        processor_batch_size_idx=processor_batch_size_idx,
        # Latent confounder genes
        latent_rank_k_idx=latent_rank_k_idx,
        lambda_L_idx=lambda_L_idx,
        lambda_bow_idx=lambda_bow_idx,
        warm_start_L_iters_idx=warm_start_L_iters_idx,
    )
