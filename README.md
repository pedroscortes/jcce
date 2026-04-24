# jcce

Joint Causal Structure Learning, Classification, and Effect Estimation via Multi-Objective Optimization.

Part of the PhD thesis *Information Geometry of Causal Inference* (see `../../plan/master_plan.md`). This is the Phase 1 empirical anchor (under review at *Expert Systems with Applications*, 2026; earlier circulated as GOLEM-GA / JACE).

## What this is

A single differentiable training loop that simultaneously solves three coupled problems on observational tabular data:

1. **Structure learning** — a weighted adjacency matrix `A` under a DAGMA-style acyclicity constraint.
2. **Per-variable structural equations** — neural "processors" learning `X_j = f_j(X_Pa_j) + noise` across five interchangeable architectures (MLP, Transformer, GNN, Mamba, ELM).
3. **Effect estimation** — CATE / ATE via structural DML with AIPW doubly-robust estimation and influence-function standard errors.

On top of the core loop:

- Multi-objective outer loop (Optuna TPE / NSGA-II) producing Pareto fronts over (accuracy, sparsity).
- 3-phase curriculum (structure → classification → effects) with PCGrad gradient surgery.
- 70/30 sample splitting + 7-step post-hoc validation chain (fixed-structure CV, DML, Cinelli sensitivity, bootstrap stability, LOVO, refutations).
- Outcome-sink constraint and bow-free low-rank latent confounder modelling.

## Package layout

```
jcce/
├── analysis/             # post-hoc analysis tooling
├── baselines/            # baseline methods (SDCD, DAGMA, NOTEARS, DiffAN, classical MB)
├── counterfactual/       # counterfactual generation (AAP on learned SCM)
├── data/                 # dataset loaders and generators
├── evaluation/           # metrics (SHD, F1, ATE bias, etc.)
├── gbs/                  # dequantized Gaussian Boson Sampling kernels (Pareto comparison)
├── models/               # processor architectures (MLP/Transformer/GNN/Mamba/ELM)
├── optimization/         # multi-objective outer loop (NSGA-II, Optuna TPE)
├── structure_learning/   # DAG learning core (jcce_learner, optuna_search, experiment_runner)
├── training/             # training loops and curriculum
├── utils/                # utilities
├── validation/           # DML, Cinelli sensitivity, bootstrap, LOVO, refutations
└── visualization/        # DAG plotting, Pareto front plots
```

## Status

Phase 1 paper under review at *Expert Systems with Applications*, 2026. Three follow-on extensions in progress:

- **DAG-Attention Transformer** — structure-aware Transformer processor (bidirectional attention / DAG consistency loss).
- **Topological Mamba** — Sinkhorn soft topological sort + DAG-gated state transitions.
- **Dequantized GBS Pareto kernel** — polynomial-time quantum-inspired DAG comparison.

See `next_steps/` (not tracked in git) for planning documents.

## Installation

Requires Python 3.11+ and `uv`.

```bash
uv venv --python 3.11
uv sync
```

For GPU (JAX + CUDA 12):

```bash
uv sync --extra dev
```

## Policies

This repo follows [`../../POLICIES.md`](../../POLICIES.md) for code style, commit messages, and testing.

## Relation to other repos

- `../geodag/` — natural-gradient causal discovery with Fisher geometry. Phase 3 Track A; extends the training loop here with Fisher-natural gradient + Fisher-PCGrad + Fisher-information Pareto-front distance.
- `../geokoop/` — Fisher-Koopman manifold for dynamical causal inference. Phase 3 Track B; lifts the static programme here to time-series.

## References

- Bello, Aragam, Ravikumar (2022). DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization. NeurIPS.
- Chernozhukov et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*.
- Geffner et al. (2022). Deep End-to-end Causal Inference. *TMLR*.
- Liu, Bellamy, Beam (2024). DAG-Aware Transformer for Causal Effect Estimation. arXiv:2410.10044.

See `bib/ref.bib` for the full reference set.

## License

MIT.
