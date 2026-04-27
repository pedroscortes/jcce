"""JCCE counterfactual module.

Currently exports:
- Causal-graph helpers (parents/children/ancestors/descendants/topological sort,
  scoring functions for actionability and plausibility) from
  ``causal_constraints``.
- ``AAPCounterfactual``: Pearl Abduction-Action-Prediction over a learned
  JCCE SCM (adjacency A + per-variable processors).

Earlier drafts referenced ``counterfactual_search`` and ``scm_propagation``
modules that never landed in this branch; those imports are removed so the
package is importable without the importlib bypass used by the AAP scripts.
"""

from .aap import AAPCounterfactual
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
from .sparsity import edge_set_agreement, ste_hard_parents

__all__ = [
    "AAPCounterfactual",
    "compute_actionability_proxy",
    "compute_causal_validity_score",
    "compute_plausibility_score",
    "edge_set_agreement",
    "get_ancestors",
    "get_children",
    "get_descendants",
    "get_markov_blanket_from_dag",
    "get_parents",
    "identify_intervention_targets",
    "ste_hard_parents",
    "topological_sort",
]
