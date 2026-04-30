"""JCCE counterfactual module.

Currently exports:
- Causal-graph helpers (parents/children/ancestors/descendants/topological sort,
  scoring functions for actionability and plausibility) from
  ``causal_constraints``.
- ``AAPCounterfactual`` plus the post-hoc heads ``AAPCounterfactualOLS`` and
  ``AAPCounterfactualLogistic`` (the architectural-fix mini-Sprint deliverable).
- The diagnostic probe API in ``probe`` (``make_f_Y``, ``autodiff_sensitivity``,
  ``grid_sweep``) — the reusable functions used to surface JCCE's f_Y collapse.

Earlier drafts referenced ``counterfactual_search`` and ``scm_propagation``
modules that never landed in this branch; those imports are removed so the
package is importable without the importlib bypass used by the AAP scripts.
"""

from .aap import AAPCounterfactual
from .aap_ols import AAPCounterfactualLogistic, AAPCounterfactualOLS
from .attention_probe import (
    compare_attention_to_A,
    extract_dagattn_causal_graph,
    extract_topomamba_hidden_attention,
)
from .causal_constraints import (
    compute_actionability_proxy,
    compute_causal_validity_score,
    compute_plausibility_score,
    get_ancestors,
    get_children,
    get_descendants,
    get_markov_blanket_from_dag,
    get_parents,
    identify_intervention_targets,
    topological_sort,
)
from .probe import autodiff_sensitivity, grid_sweep, make_f_Y
from .sparsity import edge_set_agreement, ste_hard_parents

__all__ = [
    "AAPCounterfactual",
    "AAPCounterfactualLogistic",
    "AAPCounterfactualOLS",
    "autodiff_sensitivity",
    "compare_attention_to_A",
    "compute_actionability_proxy",
    "compute_causal_validity_score",
    "compute_plausibility_score",
    "edge_set_agreement",
    "extract_dagattn_causal_graph",
    "extract_topomamba_hidden_attention",
    "get_ancestors",
    "get_children",
    "get_descendants",
    "get_markov_blanket_from_dag",
    "get_parents",
    "grid_sweep",
    "identify_intervention_targets",
    "make_f_Y",
    "ste_hard_parents",
    "topological_sort",
]
