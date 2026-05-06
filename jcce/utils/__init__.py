"""General-purpose utilities for graph operations, GPU configuration, and metrics.

`graph_utils` provides adjacency-matrix operations and graph statistics;
`gpu_config` handles JAX device selection and parallel evaluation; `metrics`
implements structural-comparison primitives (SHD, SID, edge F1) shared by
the evaluation submodule.
"""
