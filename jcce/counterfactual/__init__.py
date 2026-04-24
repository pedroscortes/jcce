"""
JCCE Counterfactual Module

Provides causally-constrained counterfactual explanations:
- Causal validity checking (only modify ancestors of Y)
- SCM-based propagation (Pearl's 3-step process)
- Multi-objective counterfactual search via NSGA-II
- Ensemble counterfactuals across Pareto-optimal DAGs
"""

from .causal_constraints import (
    get_ancestors,
    get_descendants,
    get_parents,
    get_children,
    compute_causal_validity_score,
    compute_actionability_proxy,
    compute_plausibility_score,
    identify_intervention_targets,
)

from .scm_propagation import (
    StructuralEquationModel,
    CounterfactualResult,
    fit_scm_from_golem_solution,
)

from .counterfactual_search import (
    CounterfactualSearcher,
    CounterfactualSearchResult,
    CounterfactualCandidate,
    CausalCounterfactualProblem,
    generate_counterfactual_explanation,
)

__all__ = [
    # Causal constraints
    'get_ancestors',
    'get_descendants',
    'get_parents',
    'get_children',
    'compute_causal_validity_score',
    'compute_actionability_proxy',
    'compute_plausibility_score',
    'identify_intervention_targets',
    # SCM propagation
    'StructuralEquationModel',
    'CounterfactualResult',
    'fit_scm_from_golem_solution',
    # Counterfactual search
    'CounterfactualSearcher',
    'CounterfactualSearchResult',
    'CounterfactualCandidate',
    'CausalCounterfactualProblem',
    'generate_counterfactual_explanation',
]
