"""Backward-compatibility shim: ``jcce.models.causal_mamba`` was renamed to
``jcce.models.topo_mamba`` on 2026-04-28 due to naming collisions with two
unrelated 2025 papers (Zhan & Cheng, arXiv:2510.17318; Bae & Cha,
arXiv:2511.16191). New code should import from ``jcce.models.topo_mamba``.

This module re-exports the public API under the old names so existing
scripts, paper-1 reproductions, and the parallel TopoMamba worktree continue
to work without modification.
"""

from jcce.models.topo_mamba import (  # noqa: F401
    CausalMambaProcessor,
    TopoMambaProcessor,
    ancestral_depth_scores,
    sinkhorn_topological_sort,
    topological_sort_from_adjacency,
)
