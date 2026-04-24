"""
Reduced Hyperparameter Space for Memory-Constrained NSGA-II.

Reduces processor hyperparameter combinations from ~624 to ~32 variants
by focusing on the most important hyperparameters for each processor type.

Total variants: 4 (Mamba) + 4 (Transformer) + 8 (GNN) + 8 (MLP) + 8 (ELM) = 32
"""

# Processor types (same as before)
PROCESSOR_TYPES = ['elm', 'gnn', 'mamba', 'mlp', 'transformer']

# GOLEM hyperparameters (same as before - these are cheap to vary)
GOLEM_LAMBDA_1 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]  # Sparsity
GOLEM_LAMBDA_2 = [0.001, 0.01, 0.1, 1.0]  # DAG constraint init
GOLEM_LAMBDA_CLASS = [0.1, 0.5, 1.0, 2.0, 5.0]  # Classification weight
GOLEM_LRS = [0.0001, 0.0003, 0.001, 0.003, 0.01]

# ============================================================================
# REDUCED Processor-Specific Hyperparameters
# ============================================================================

# Mamba: 2×2 = 4 variants (was 144)
# Focus on d_model (most important) and d_state
MAMBA_D_MODELS = [64, 128]          # Reduced from [64, 128, 256, 512]
MAMBA_D_STATES = [8, 16]            # Reduced from [8, 16, 32, 64]
MAMBA_D_CONVS = [4]                 # Fixed (less critical)
MAMBA_EXPANDS = [2]                 # Fixed (less critical)

# Transformer: 2×2 = 4 variants (was 320)
# Focus on d_model and n_heads
TRANSFORMER_D_MODELS = [64, 128]    # Reduced from [64, 128, 256, 512]
TRANSFORMER_N_HEADS = [2, 4]        # Reduced from [2, 4, 8, 16]
TRANSFORMER_N_LAYERS = [2]          # Fixed to 2 (good balance)
TRANSFORMER_D_FFS = [256]           # Fixed, proportional to d_model

# GNN: 2×2×2 = 8 variants (was ~64)
GNN_HIDDEN_DIMS = [64, 128]         # Reduced from [32, 64, 128, 256]
GNN_N_LAYERS = [2, 3]               # Reduced from [1, 2, 3, 4]
GNN_TYPES = ['gcn', 'gat']          # Reduced from ['gcn', 'gat', 'sage']
GNN_AGGREGATIONS = ['mean']         # Fixed (SAGE-specific, less critical)

# MLP: 2×2×2 = 8 variants (was 48)
MLP_HIDDEN_DIMS = [64, 128]         # Reduced from [32, 64, 128, 256]
MLP_N_LAYERS = [2, 3]               # Reduced from [1, 2, 3, 4]
MLP_ACTIVATIONS = ['relu', 'tanh']  # Reduced from ['relu', 'tanh', 'gelu']

# ELM: 2×2×2 = 8 variants (was ~48)
ELM_HIDDEN_DIMS = [64, 128]         # Reduced from [32, 64, 128, 256]
ELM_N_HIDDEN_NODES = [64, 128]      # Reduced from [32, 64, 128, 256]
ELM_ACTIVATIONS = ['relu', 'tanh']  # Reduced from ['relu', 'tanh', 'sigmoid']
