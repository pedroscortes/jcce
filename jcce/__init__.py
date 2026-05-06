"""JCCE — Joint Causal Structure Learning, Classification, and Effect Estimation.

A JAX/Flax library that learns a DAG, fits per-variable structural equations,
and estimates treatment effects in a single end-to-end joint-loss training run.

The headline entry points are:

    >>> from jcce import learn_structure, create_processor
    >>> import jax.random as random
    >>> processor = create_processor("linear_head", random.PRNGKey(0))
    >>> A_est, processor, params, metrics = learn_structure(
    ...     data=X, Y=Y, Y_idx=d, processor=processor, key=random.PRNGKey(0),
    ...     processor_type="linear_head",
    ... )

Lower-level functionality lives in submodules:

    jcce.structure_learning   DAG learning core (jcce_learner, optuna_search)
    jcce.models               Processor architectures (Linear, MLP, Transformer, Mamba, GNN, ELM)
    jcce.training             Training loops and curriculum
    jcce.optimization         Multi-objective outer loop (Optuna TPE, NSGA-II)
    jcce.validation           DML, sensitivity, bootstrap, LOVO, refutations
    jcce.counterfactual       Counterfactual generation on the learned SCM
    jcce.evaluation           Metrics (SHD, F1, BAcc, ATE bias, PEHE)
    jcce.data                 Dataset loaders and synthetic SCM generators
    jcce.gbs                  Dequantized Gaussian Boson Sampling kernels
    jcce.analysis             Post-hoc analysis tooling
    jcce.visualization        DAG plotting, Pareto front plots
    jcce.utils                Utilities (graph metrics, GPU configuration)
"""

from jcce.structure_learning.jcce_learner import create_processor, learn_structure

__version__ = "0.1.0"

__all__ = [
    "create_processor",
    "learn_structure",
    "__version__",
]
