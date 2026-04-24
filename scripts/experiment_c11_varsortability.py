#!/usr/bin/env python
"""
Experiment C.11: Varsortability Check (Reisach et al., NeurIPS 2021).

Computes varsortability across noise levels and graph types. Shows that
standardized data removes the variance-ordering shortcut that GOLEM may
exploit.

Usage:
    uv run python scripts/experiment_c11_varsortability.py --n-vars 5 --n-samples 200
    uv run python scripts/experiment_c11_varsortability.py \
        --n-vars 10 --n-samples 1000 --graph-types erdos_renyi chain scale_free
"""

import argparse
import json
import sys
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jcce.data.dag_generator import DAGConfig, generate_dag, count_edges
from jcce.data.scm import SCMConfig, LinearSCM
from jcce.utils.metrics import compute_varsortability


def main():
    parser = argparse.ArgumentParser(description='C.11: Varsortability Check')
    parser.add_argument('--n-vars', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--graph-types', nargs='+',
                        default=['erdos_renyi', 'chain', 'scale_free'])
    parser.add_argument('--noise-scales', nargs='+', type=float,
                        default=[0.1, 0.3, 0.5, 1.0, 2.0])
    parser.add_argument('--expected-degree', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print(f"VARSORTABILITY CHECK (C.11)")
    print(f"=" * 70)

    results = []

    for gt_idx, graph_type in enumerate(args.graph_types):
        # Generate ONE DAG per graph_type (shared across noise levels)
        dag_config = DAGConfig(
            num_nodes=args.n_vars,
            graph_type=graph_type,
            expected_degree=args.expected_degree,
            seed=args.seed + gt_idx * 100,
        )
        A = generate_dag(dag_config)
        A_np = np.array(A)
        n_edges = int(np.sum(np.abs(A_np) > 1e-6))

        print(f"\nGraph: {graph_type.upper()}, d={args.n_vars}, {n_edges} edges | SCM: linear Gaussian")

        for noise_scale in args.noise_scales:
            scm_config = SCMConfig(noise_scale=noise_scale)
            scm = LinearSCM(A, scm_config)
            key = random.PRNGKey(args.seed + gt_idx * 1000 + int(noise_scale * 100))
            X = scm.sample(args.n_samples, key)
            X_np = np.array(X)

            # Raw varsortability
            v_raw = compute_varsortability(X_np, A_np)

            # Standardized control
            X_std = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-10)
            v_std = compute_varsortability(X_std, A_np)

            results.append({
                'graph_type': graph_type,
                'noise_scale': noise_scale,
                'n_edges': n_edges,
                'varsort_raw': v_raw,
                'varsort_std': v_std,
            })

    # Print table
    print(f"\n{'=' * 70}")
    print(f"{'Graph Type':<16} {'Noise':>6} {'Varsort(raw)':>13} {'Varsort(std)':>13}")
    print(f"{'-' * 16} {'-' * 6} {'-' * 13} {'-' * 13}")

    for r in results:
        print(f"{r['graph_type']:<16} {r['noise_scale']:>6.1f} "
              f"{r['varsort_raw']:>13.3f} {r['varsort_std']:>13.3f}")

    # Warnings and recommendations
    print(f"\n{'=' * 70}")
    high_varsort = [r for r in results if r['varsort_raw'] > 0.9]
    if high_varsort:
        conditions = set((r['graph_type'], r['noise_scale']) for r in high_varsort)
        print(f"WARNING: {len(high_varsort)} condition(s) have varsortability > 0.9:")
        for gt, ns in sorted(conditions):
            print(f"  {gt}, noise_scale={ns}")
        print(f"\nGOLEM may exploit variance ordering in these settings.")
        print(f"Recommendation: Use noise_scale >= 0.5 or report both raw + standardized results.")
    else:
        print(f"OK: No conditions have varsortability > 0.9.")

    # Save results
    if args.output:
        output = {
            'args': vars(args),
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
