#!/usr/bin/env python
"""
Post-hoc diagnostic extraction for JCCE results.

Extracts gradient cosines from logs, computes bow-free violations and
residual normality from saved pkl data. Works on completed runs where
these diagnostics were logged but not persisted to pkl files.

Usage:
    # Analyze all completed datasets
    uv run python scripts/posthoc_diagnostics.py

    # Single dataset
    uv run python scripts/posthoc_diagnostics.py --dataset lucas

    # Custom paths
    uv run python scripts/posthoc_diagnostics.py \
        --results-dir results/optuna_v16 \
        --log-dir results/full_run_full_20260316_001427
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# 1. Gradient Cosine Extraction (from log files)
# ============================================================================

def parse_gradient_logs(log_path: Path) -> dict:
    """Parse [Gradient] lines from an Optuna log file.

    Returns:
        dict with 'entries' (list of dicts) and 'summary' (aggregate stats)
    """
    pattern = re.compile(
        r'\[Gradient\] cos\(recon,class\)=([-\d.]+)'
        r'(?:, cos\(r,e\)=([-\d.]+))?'
        r'(?:, cos\(c,e\)=([-\d.]+))?'
    )

    entries = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                entries.append({
                    'cos_rc': float(m.group(1)),
                    'cos_re': float(m.group(2)) if m.group(2) else 0.0,
                    'cos_ce': float(m.group(3)) if m.group(3) else 0.0,
                })

    if not entries:
        return {'entries': [], 'summary': None}

    cos_rc = np.array([e['cos_rc'] for e in entries])
    cos_re = np.array([e['cos_re'] for e in entries])
    cos_ce = np.array([e['cos_ce'] for e in entries])

    def _stats(arr, name):
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'pct_negative': float(np.mean(arr < 0) * 100),
            'pct_strong_conflict': float(np.mean(arr < -0.3) * 100),
            'pct_strong_aligned': float(np.mean(arr > 0.3) * 100),
        }

    summary = {
        'n_entries': len(entries),
        'cos_recon_class': _stats(cos_rc, 'rc'),
        'cos_recon_effect': _stats(cos_re, 're'),
        'cos_class_effect': _stats(cos_ce, 'ce'),
    }

    return {'entries': entries, 'summary': summary}


# ============================================================================
# 2. Bow-Free Violations (from pkl A matrices)
# ============================================================================

def compute_bow_free_violations(sol: dict, threshold: float = 0.3) -> dict:
    """Count variable pairs with both directed AND bi-directed edges.

    Uses A_weights (continuous) and A_confound_weights from the pkl.
    """
    A_direct = sol.get('A_weights')
    if A_direct is None:
        A_direct = sol.get('A_est')
    A_confound = sol.get('A_confound_weights')
    if A_confound is None:
        A_confound = sol.get('A_confound')

    if A_direct is None or A_confound is None:
        return {'error': 'missing A matrices'}

    A_dir = np.abs(np.array(A_direct, dtype=float))
    A_conf = np.abs(np.array(A_confound, dtype=float))

    # Align shapes (A_direct may be augmented with Y)
    min_d = min(A_dir.shape[0], A_conf.shape[0])
    A_dir = A_dir[:min_d, :min_d]
    A_conf = A_conf[:min_d, :min_d]

    # Symmetrize confound (should already be symmetric)
    A_conf = (A_conf + A_conf.T) / 2

    both_mask = (A_dir > threshold) & (A_conf > threshold)
    n_violations = int(np.sum(both_mask))

    # Identify which pairs
    violation_pairs = []
    if n_violations > 0:
        rows, cols = np.where(both_mask)
        for r, c in zip(rows, cols):
            if r < c:  # avoid double-counting
                violation_pairs.append({
                    'i': int(r), 'j': int(c),
                    'directed_weight': float(A_dir[r, c]),
                    'confound_weight': float(A_conf[r, c]),
                })

    return {
        'n_violations': n_violations,
        'n_unique_pairs': len(violation_pairs),
        'pairs': violation_pairs,
        'n_directed_edges': int(np.sum(A_dir > threshold)),
        'n_confound_edges': int(np.sum(A_conf > threshold) // 2),  # symmetric
    }


# ============================================================================
# 3. Residual Normality (from pkl A + data)
# ============================================================================

def compute_residual_normality(sol: dict, X: np.ndarray,
                                n_test: int = 5, n_sample: int = 200) -> dict:
    """Test residual normality via Shapiro-Wilk.

    Uses a linear reconstruction model (A-weighted sum of parents) as a
    proxy for the full processor forward pass. This is conservative —
    the actual nonlinear processor would produce even more non-Gaussian
    residuals.
    """
    try:
        from scipy.stats import shapiro
    except ImportError:
        return {'error': 'scipy not available'}

    A_weights = sol.get('A_weights')
    if A_weights is None:
        A_weights = sol.get('A_est')
    if A_weights is None:
        return {'error': 'no A_weights in solution'}

    A = np.abs(np.array(A_weights, dtype=float))
    n_vars = X.shape[1]

    # Use feature-only submatrix
    A_feat = A[:n_vars, :n_vars]

    pvals = []
    var_results = []
    for j in range(min(n_test, n_vars)):
        weights = A_feat[:, j].copy()
        weights[j] = 0  # no self-loop

        if np.sum(weights) < 0.01:
            var_results.append({'var': j, 'status': 'no_parents', 'pval': None})
            continue

        # Linear reconstruction: weighted sum of parent values
        X_j_pred = X @ weights
        residuals = X[:, j] - X_j_pred

        n_unique = len(np.unique(residuals[:n_sample]))
        if n_unique <= 3:
            var_results.append({'var': j, 'status': 'low_variance', 'pval': None})
            continue

        _, p = shapiro(residuals[:n_sample])
        pvals.append(float(p))
        var_results.append({'var': j, 'status': 'tested', 'pval': float(p)})

    if not pvals:
        return {'error': 'no testable variables', 'details': var_results}

    return {
        'n_tested': len(pvals),
        'pvals': pvals,
        'min_pval': min(pvals),
        'max_pval': max(pvals),
        'all_non_gaussian': all(p < 0.05 for p in pvals),
        'fraction_non_gaussian': sum(1 for p in pvals if p < 0.05) / len(pvals),
        'details': var_results,
    }


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_dataset(dataset_name: str, pkl_path: Path,
                    log_path: Path = None, X: np.ndarray = None) -> dict:
    """Full diagnostic analysis for one dataset."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()}")
    print(f"{'='*60}")

    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    sols = result.get('pareto_solutions', [])
    n_trials = result.get('n_trials_completed', '?')
    print(f"  Trials: {n_trials}, Pareto: {len(sols)}")

    dataset_results = {
        'dataset': dataset_name,
        'n_trials': n_trials,
        'n_pareto': len(sols),
    }

    # --- 1. Gradient cosines ---
    if log_path and log_path.exists():
        gd = parse_gradient_logs(log_path)
        dataset_results['gradient'] = gd['summary']

        if gd['summary']:
            s = gd['summary']
            print(f"\n  Gradient Cosines ({s['n_entries']} entries from log):")
            for name, key in [('recon vs class', 'cos_recon_class'),
                              ('recon vs effect', 'cos_recon_effect'),
                              ('class vs effect', 'cos_class_effect')]:
                st = s[key]
                print(f"    {name:20s}: mean={st['mean']:+.4f} std={st['std']:.4f} "
                      f"[{st['min']:+.3f}, {st['max']:+.3f}] "
                      f"neg={st['pct_negative']:.0f}% strong_conflict={st['pct_strong_conflict']:.0f}%")

            # Interpretation
            rc_conflict = s['cos_recon_class']['pct_strong_conflict']
            ce_conflict = s['cos_class_effect']['pct_strong_conflict']
            if rc_conflict < 5 and ce_conflict < 5:
                print("    => No significant gradient conflicts (curriculum sufficient)")
            elif rc_conflict < 15 and ce_conflict < 15:
                print("    => Mild gradient tension (curriculum handles it)")
            else:
                print("    => SIGNIFICANT gradient conflicts detected — consider PCGrad/CAGrad")
        else:
            print(f"\n  Gradient Cosines: no entries in log")
    else:
        print(f"\n  Gradient Cosines: no log file")

    # --- 2. Bow-free violations ---
    print(f"\n  Bow-Free Violations (threshold=0.3):")
    bow_results = []
    for i, sol in enumerate(sols[:5]):
        proc = sol.get('processor_type', '?')
        bacc = sol.get('balanced_accuracy') or sol.get('classification_balanced_accuracy', 0)
        bfv = compute_bow_free_violations(sol)
        bow_results.append(bfv)

        if 'error' in bfv:
            print(f"    Sol {i} ({proc:12s} BAcc={bacc:.3f}): {bfv['error']}")
        else:
            status = 'CLEAN' if bfv['n_unique_pairs'] == 0 else f"{bfv['n_unique_pairs']} PAIRS"
            print(f"    Sol {i} ({proc:12s} BAcc={bacc:.3f}): {status} "
                  f"(directed={bfv['n_directed_edges']}, confound={bfv['n_confound_edges']})")
            if bfv['pairs']:
                for p in bfv['pairs'][:3]:
                    print(f"      pair ({p['i']},{p['j']}): "
                          f"dir={p['directed_weight']:.3f} conf={p['confound_weight']:.3f}")
    dataset_results['bow_free'] = bow_results

    # --- 3. Residual normality ---
    if X is not None:
        print(f"\n  Residual Normality (Shapiro-Wilk, linear proxy):")
        norm_results = []
        for i, sol in enumerate(sols[:3]):
            proc = sol.get('processor_type', '?')
            rn = compute_residual_normality(sol, X)
            norm_results.append(rn)

            if 'error' in rn:
                print(f"    Sol {i} ({proc:12s}): {rn['error']}")
            else:
                ng_str = 'NON-GAUSSIAN' if rn['all_non_gaussian'] else 'MIXED'
                print(f"    Sol {i} ({proc:12s}): {ng_str} "
                      f"({rn['n_tested']} tested, "
                      f"{rn['fraction_non_gaussian']*100:.0f}% reject normality, "
                      f"pvals=[{rn['min_pval']:.2e}, {rn['max_pval']:.2e}])")

        # Interpretation for thesis
        if norm_results and not any('error' in r for r in norm_results):
            all_ng = all(r.get('all_non_gaussian', False) for r in norm_results)
            if all_ng:
                print("    => All solutions: non-Gaussian residuals")
                print("       Cite Wang & Drton 2023 for bow-free ADMG identifiability")
            else:
                print("    => Mixed: some solutions have Gaussian residuals")
                print("       Bow-free identifiability claim requires caution")
        dataset_results['residual_normality'] = norm_results
    else:
        print(f"\n  Residual Normality: skipped (dataset not loadable)")

    return dataset_results


def main():
    parser = argparse.ArgumentParser(description='Post-hoc JCCE diagnostics')
    parser.add_argument('--results-dir', type=str, default='results/optuna_v16')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Log directory (auto-detected if not set)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset (default: all available)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--source', type=str, default='optuna',
                        choices=['optuna', 'nsga2', 'both'],
                        help='Which results to analyze (default: optuna)')
    args = parser.parse_args()

    sources = ['optuna', 'nsga2'] if args.source == 'both' else [args.source]

    # Auto-detect log dir
    log_dir = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        candidates = sorted(Path('results').glob('full_run_full_*'))
        if candidates:
            log_dir = candidates[-1]
            print(f"Auto-detected log dir: {log_dir}")

    all_results = {}

    for source in sources:
        if source == 'optuna':
            results_dir = Path(args.results_dir)
            pkl_pattern = 'optuna_seed42.pkl'
            log_prefix = 'optuna_'
        else:
            results_dir = Path('results/ablation_v16')
            pkl_pattern = 'B6.pkl'
            log_prefix = 'nsga2_'

        if not results_dir.exists():
            print(f"\n[SKIP] {source}: {results_dir} not found")
            continue

        print(f"\n{'#'*60}")
        print(f"  SOURCE: {source.upper()}")
        print(f"  Dir: {results_dir}")
        print(f"{'#'*60}")

        # Find datasets
        if args.dataset:
            datasets = [args.dataset]
        else:
            datasets = sorted(d.name for d in results_dir.iterdir()
                              if d.is_dir() and (d / pkl_pattern).exists())

        print(f"Datasets to analyze: {datasets}")

        for ds in datasets:
            pkl_path = results_dir / ds / pkl_pattern
            log_path = log_dir / f'{log_prefix}{ds}.log' if log_dir else None

            # Load data
            X = None
            try:
                from jcce.data.benchmark_loader import load_dataset
                X, _, _ = load_dataset(ds)
            except Exception as e:
                print(f"\n  Warning: could not load {ds} data: {e}")

            result_key = f'{source}_{ds}' if args.source == 'both' else ds
            all_results[result_key] = analyze_dataset(ds, pkl_path, log_path, X)

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':18s} {'Grad Entries':>12s} {'cos(r,c)':>10s} {'cos(c,e)':>10s} "
          f"{'BowFree':>8s} {'NonGauss':>10s}")
    print(f"{'-'*18} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for ds, r in all_results.items():
        g = r.get('gradient', {})
        n_grad = g.get('n_entries', 0) if g else 0
        cos_rc = f"{g['cos_recon_class']['mean']:+.3f}" if g else '-'
        cos_ce = f"{g['cos_class_effect']['mean']:+.3f}" if g else '-'

        bfv_list = r.get('bow_free', [])
        bfv = '-'
        if bfv_list and 'n_unique_pairs' in bfv_list[0]:
            bfv = str(bfv_list[0]['n_unique_pairs'])

        rn_list = r.get('residual_normality', [])
        ng = '-'
        if rn_list and 'all_non_gaussian' in rn_list[0]:
            ng = 'YES' if rn_list[0]['all_non_gaussian'] else 'no'

        print(f"{ds:18s} {n_grad:>12d} {cos_rc:>10s} {cos_ce:>10s} {bfv:>8s} {ng:>10s}")

    # Save
    if args.save:
        # Convert numpy types for JSON
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.save, 'w') as f:
            json.dump(all_results, f, indent=2, default=_convert)
        print(f"\nResults saved to {args.save}")

    print("\nDone.")


if __name__ == '__main__':
    main()
