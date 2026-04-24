#!/usr/bin/env python3
"""Plot Pareto fronts for all datasets (Optuna vs NSGA-II) — ESWA paper figure."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OPTUNA_DIR = Path('/home/user/phd/repos/jcce/official_results/optuna_v16')
NSGA2_DIR = Path('/home/user/phd/repos/jcce/results/results_server/ablation_v16')

DATASETS_ORDERED = [
    ('asia', 'Asia', 7),
    ('diabetes', 'Diabetes', 8),
    ('sachs', 'Sachs', 10),
    ('lucas', 'LUCAS', 11),
    ('heart_disease', 'Heart Disease', 13),
    ('child', 'CHILD', 19),
    ('insurance', 'Insurance', 26),
    ('neuropathic_pain', 'Neuropathic Pain', 28),
    ('breast_cancer', 'Breast Cancer', 30),
    ('alarm', 'ALARM', 36),
]


def extract_pareto_points(sols, metric='balanced_accuracy'):
    """Return arrays of (mb_size, accuracy, cv_accuracy) from Pareto solutions."""
    mb_sizes, train_accs, cv_accs = [], [], []
    for sol in sols:
        mb = sol.get('mb_indices', [])
        mb_size = len(mb) if mb else sol.get('markov_blanket_size', 0)
        acc = sol.get(metric, sol.get('accuracy', 0))
        cv = sol.get('cv', {})
        cv_acc = cv.get('balanced_acc_mean', None)
        mb_sizes.append(mb_size)
        train_accs.append(acc)
        cv_accs.append(cv_acc if cv_acc is not None else 0)
    return np.array(mb_sizes), np.array(train_accs), np.array(cv_accs)


def plot_dataset(ax, ds_key, label, d):
    """Plot one dataset's Pareto fronts."""
    # Optuna
    opt_path = OPTUNA_DIR / ds_key / 'optuna_seed42.pkl'
    nsga_path = NSGA2_DIR / ds_key / 'B6.pkl'

    opt_exists = opt_path.exists()
    nsga_exists = nsga_path.exists()

    opt_mb, opt_train, opt_cv = (np.array([]),) * 3
    if opt_exists:
        with open(opt_path, 'rb') as f:
            od = pickle.load(f)
        opt_mb, opt_train, opt_cv = extract_pareto_points(od.get('pareto_solutions', []))

    nsga_mb, nsga_train, nsga_cv = (np.array([]),) * 3
    if nsga_exists:
        with open(nsga_path, 'rb') as f:
            nd = pickle.load(f)
        nsga_mb, nsga_train, nsga_cv = extract_pareto_points(nd.get('pareto_solutions', []))

    # Plot — CV BAcc as primary, training as secondary
    # Optuna: filled circles (train) + diamonds (CV)
    if len(opt_mb) > 0:
        ax.scatter(opt_mb, opt_train, c='#1f77b4', marker='o', s=35, alpha=0.5,
                   edgecolor='none', label='Optuna (train)', zorder=2)
        valid = opt_cv > 0
        if valid.any():
            ax.scatter(opt_mb[valid], opt_cv[valid], c='#1f77b4', marker='D', s=45,
                       edgecolor='black', linewidth=0.5, label='Optuna (CV)', zorder=4)

    # NSGA-II: gray crosses (train) + gray squares (CV)
    if len(nsga_mb) > 0:
        ax.scatter(nsga_mb, nsga_train, c='gray', marker='x', s=30, alpha=0.5,
                   label='NSGA-II (train)', zorder=1)
        valid = nsga_cv > 0
        if valid.any():
            ax.scatter(nsga_mb[valid], nsga_cv[valid], c='gray', marker='s', s=35,
                       edgecolor='black', linewidth=0.5, alpha=0.8,
                       label='NSGA-II (CV)', zorder=3)

    ax.set_title(f'{label} ($d={d}$)', fontsize=10)
    ax.set_xlabel('|MB|', fontsize=9)
    ax.set_ylabel('BAcc', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=8)

    # Annotate best CV
    if len(opt_mb) > 0 and (opt_cv > 0).any():
        valid = opt_cv > 0
        idx = np.argmax(opt_cv[valid])
        best_mb = opt_mb[valid][idx]
        best_cv = opt_cv[valid][idx]
        ax.annotate(f'{best_cv:.2f}', xy=(best_mb, best_cv),
                    xytext=(3, 3), textcoords='offset points', fontsize=7,
                    color='#1f77b4')


def main():
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, (ds_key, label, d) in enumerate(DATASETS_ORDERED):
        plot_dataset(axes[i], ds_key, label, d)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    plt.suptitle('Pareto fronts: classification accuracy vs.\\ Markov blanket size (BAcc axis)',
                 fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_png = Path('/home/user/phd/repos/jcce/docs/article/figuras/pareto_fronts_all.png')
    out_pdf = Path('/home/user/phd/repos/jcce/docs/article/figuras/pareto_fronts_all.pdf')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
