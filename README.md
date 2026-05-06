# jcce

**Joint Causal Structure Learning, Classification, and Effect Estimation** — a JAX/Flax library that learns a DAG, fits per-variable structural equations, and estimates treatment effects in a single end-to-end joint-loss training run.

## What this is

A single differentiable training loop that simultaneously solves three coupled problems on observational tabular data:

1. **Structure learning** — a weighted adjacency matrix `A` under a DAGMA-style acyclicity constraint.
2. **Per-variable structural equations** — neural "processors" learning `X_j = f_j(X_Pa(j)) + noise` across interchangeable architectures (LinearHead, MLP, Transformer, GNN, Mamba, ELM, plus the more recent DAG-Attention and Topological Mamba variants).
3. **Effect estimation** — CATE / ATE via structural DML with AIPW doubly-robust estimation and influence-function standard errors.

## Installation

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.11
uv sync
```

For GPU (JAX + CUDA 12):

```bash
uv sync --extra dev
```

## Repository structure

```
jcce/                       # Core library (the implementation)
├── analysis/               # Post-hoc analysis tooling
├── baselines/              # Baseline methods (SDCD, DAGMA, NOTEARS, DiffAN, classical MB)
├── counterfactual/         # Counterfactual generation (AAP on the learned SCM)
├── data/                   # Dataset loaders and synthetic generators
├── evaluation/             # Metrics (SHD, F1, BAcc, ATE bias, etc.)
├── gbs/                    # Dequantized Gaussian Boson Sampling kernels (Pareto comparison)
├── models/                 # Processor architectures (Linear/MLP/Transformer/GNN/Mamba/ELM)
├── optimization/           # Multi-objective outer loop (Optuna TPE, NSGA-II)
├── structure_learning/     # DAG learning core (jcce_learner, optuna_search, runner)
├── training/               # Training loops and curriculum
├── utils/                  # Utilities
├── validation/             # DML, Cinelli sensitivity, bootstrap, LOVO, refutations
└── visualization/          # DAG plotting, Pareto front plots

tests/                      # Library unit tests
bib/                        # Bibliography (BibTeX)
```

Reference benchmark datasets are not bundled in the repo. The loaders
in `jcce.data.benchmark_loader` point at canonical sources (UCI ML
Repository, bnlearn, the GANITE Twins distribution, etc.) and emit
download instructions when called on a missing file. Place downloaded
data under `data/` at the repo root; the directory is gitignored.

## References

- Bello, K., Aragam, B., Ravikumar, P. (2022). *DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization.* NeurIPS.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal, 21(1), C1–C68.
- Geffner, T., Antoran, J., Foster, A., Gong, W., Ma, C., Kiciman, E., Sharma, A., Lamb, A., Kukla, M., Pawlowski, N., Allamanis, M., Zhang, C. (2024). *Deep End-to-End Causal Inference.* Transactions on Machine Learning Research.
- Kyono, T., Zhang, Y., van der Schaar, M. (2020). *CASTLE: Regularization via Auxiliary Causal Graph Discovery.* Advances in Neural Information Processing Systems.
- Liu, Y., Bellamy, D., Beam, A. (2024). *DAG-Aware Transformer for Causal Effect Estimation.* arXiv:2410.10044.
- Pezeshki, M., Kaba, S.-O., Bengio, Y., Courville, A., Precup, D., Lajoie, G. (2021). *Gradient Starvation: A Learning Proclivity in Neural Networks.* NeurIPS.

Full reference set in `bib/ref.bib`.

## License

MIT.
