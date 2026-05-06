"""Backward-compatibility shim: ``jcce.models.causal_mamba`` is the former
name of :mod:`jcce.models.topo_mamba`, renamed to avoid collision with
unrelated 2025 papers (Zhan & Cheng, arXiv:2510.17318; Bae & Cha,
arXiv:2511.16191). New code should import from :mod:`jcce.models.topo_mamba`.

This module re-exports the public API under the old names so existing
callers continue to work without modification.
"""

from jcce.models.topo_mamba import (  # noqa: F401
    CausalMambaProcessor,
    TopoMambaProcessor,
    ancestral_depth_scores,
    sinkhorn_topological_sort,
    topological_sort_from_adjacency,
)
