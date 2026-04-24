"""
JCCE Validation Module

Provides causal-aware validation strategies:
- DML cross-fitting for robust effect estimation
- Refutation tests (placebo, random cause, subset, dummy outcome)
- Bootstrap stability analysis
- Identifiability diagnostics (BIC, condition number, SVD spectrum)
- Unified K-fold CV evaluation using fθ (v15)
"""

from .dml_crossfitting import DMLCrossFitter, DMLResult
from .refutation_suite import RefutationSuite, RefutationResult
from .unified_cv_evaluation import UnifiedCVResult, evaluate_pareto_solution_cv
from .identifiability_diagnostics import (
    IdentifiabilityReport,
    IdentifiabilitySummary,
    compute_mb_condition_number,
    compute_svd_diagnostics,
    compute_simple_bic,
    compute_neural_bic,
    compute_bic_for_dag,
    compute_full_identifiability_report,
    d_optimality_penalty,
    hutchinson_edf,
)

__all__ = [
    'DMLCrossFitter',
    'DMLResult',
    'RefutationSuite',
    'RefutationResult',
    'IdentifiabilityReport',
    'IdentifiabilitySummary',
    'compute_mb_condition_number',
    'compute_svd_diagnostics',
    'compute_simple_bic',
    'compute_neural_bic',
    'compute_bic_for_dag',
    'compute_full_identifiability_report',
    'd_optimality_penalty',
    'hutchinson_edf',
    'UnifiedCVResult',
    'evaluate_pareto_solution_cv',
]
