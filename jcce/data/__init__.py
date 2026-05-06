"""Dataset loaders and synthetic SCM generators.

Reference real-world benchmarks live in `benchmark_loader` (Heart Disease,
LUCAS, Sachs, Diabetes, Asia, Breast Cancer, Alarm, Child, Insurance,
Neuropathic Pain) and `twins_loader` (Twins CATE benchmark). Synthetic
generators (`scm`, `synthetic_dataset`, `synthetic_nonlinear`,
`dag_generator`) produce linear and nonlinear ER-DAG SCMs at configurable
dimensionality, edge density, and noise scale.
"""
