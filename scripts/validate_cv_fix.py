#!/usr/bin/env python
"""
Quick validation script to verify CV works on GPU with ground-truth datasets.

Tests the exact code path that was failing: evaluate_pareto_solution_cv()
with a real A_init (augmented, 12x12 for LUCAS) and true_dag (11x11).

Run on server AFTER deploying fixes:
    uv run python scripts/validate_cv_fix.py

Expected: All tests PASS with CV BAcc != 0.500
If any test shows 0.500±0.000 or "FAILED", the fix didn't work.
"""

import numpy as np
import time
import sys


def test_cv_with_dataset(dataset_name, n_folds=2, max_iter=5):
    """Test CV on a single dataset. Returns True if CV works."""
    from jcce.data.benchmark_loader import load_dataset
    from jcce.validation.unified_cv_evaluation import evaluate_pareto_solution_cv

    X, Y, config = load_dataset(dataset_name)
    n_vars = X.shape[1]
    n_total = n_vars + 1
    has_gt = config.get('true_dag') is not None

    # Simulate a Pareto solution's A_init (augmented space)
    rng = np.random.RandomState(42)
    A_init = rng.randn(n_total, n_total).astype(np.float32) * 0.1

    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name} (d={n_vars}, n={X.shape[0]})")
    print(f"  A_init shape: {A_init.shape} (augmented)")
    print(f"  true_dag: {'shape=' + str(config['true_dag'].shape) if has_gt else 'None'}")
    print(f"  This {'HAS' if has_gt else 'does NOT have'} ground truth")

    t0 = time.time()
    try:
        result = evaluate_pareto_solution_cv(
            X=X, Y=Y,
            hyperparams={
                'lambda_1': 0.02, 'lambda_2': 0.01,
                'lambda_class': 1.0, 'lr': 0.001,
                'processor_config': {
                    'hidden_dim': 64, 'n_hidden_nodes': 128,
                    'activation': 'relu'
                },
            },
            processor_type='elm',
            A_init=A_init,
            n_folds=n_folds,
            true_dag=config.get('true_dag'),
            true_mb=config.get('true_mb'),
            golem_max_iter=max_iter,
            freeze_structure=True,
            task='classification',
            verbose=True,
        )
        elapsed = time.time() - t0
        bacc = result.balanced_acc_mean

        if abs(bacc - 0.500) < 0.001 and result.balanced_acc_std < 0.001:
            print(f"\n  FAILED: CV BAcc={bacc:.3f}±{result.balanced_acc_std:.3f} "
                  f"(random baseline — CV silently failed)")
            return False
        else:
            print(f"\n  PASSED: CV BAcc={bacc:.3f}±{result.balanced_acc_std:.3f} ({elapsed:.1f}s)")
            return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED with exception ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Verify all datasets load without errors."""
    from jcce.data.benchmark_loader import load_dataset, list_datasets

    print("Testing dataset loading...")
    all_ok = True
    for name in list_datasets():
        try:
            X, Y, config = load_dataset(name)
            dag = config.get('true_dag')
            dag_info = f"dag={dag.shape}, {int(dag.sum())}e" if dag is not None else "no GT"
            print(f"  {name:20s} X={str(X.shape):12s} Y_bal={Y.mean():.2f} {dag_info}")
        except Exception as e:
            print(f"  {name:20s} LOAD FAILED: {e}")
            all_ok = False
    return all_ok


def main():
    print("JCCE CV Validation Script")
    print("=" * 60)

    # Step 1: All datasets load
    if not test_dataset_loading():
        print("\nABORT: Dataset loading failed")
        sys.exit(1)

    # Step 2: CV on a GT dataset (the one that was failing)
    lucas_ok = test_cv_with_dataset('lucas', n_folds=2, max_iter=5)

    # Step 3: CV on a non-GT dataset (should always work since true_dag=None skips comparison)
    heart_ok = test_cv_with_dataset('heart_disease', n_folds=2, max_iter=5)

    # Step 4: CV on a new GT dataset
    child_ok = test_cv_with_dataset('child', n_folds=2, max_iter=5)

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    results = {
        'Dataset loading': True,
        'LUCAS CV (GT, d=11)': lucas_ok,
        'Heart Disease CV (no GT, d=13)': heart_ok,
        'CHILD CV (GT, d=19)': child_ok,
    }
    all_pass = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test:40s} {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED. Safe to run full experiment suite.")
    else:
        print("\nSome tests FAILED. DO NOT start full run until fixed.")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
