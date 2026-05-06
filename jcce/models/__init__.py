"""Per-variable processor architectures and structural-equation building blocks.

The processors (`mlp`, `gnn`, `mamba`, `topo_mamba`, `elm`,
`dag_attention_transformer`, `processor_wrappers`) parameterise
`X_j = f_j(X_Pa(j)) + noise` and are interchangeable in the joint-loss
training loop. Auxiliary modules (`encoder`, `decoder`, `causal_layer`,
`causal_vae`, `epsilon_projection`) provide the structural-equation
machinery used internally by the processors and by the DAG-Attention
Transformer variant.
"""
