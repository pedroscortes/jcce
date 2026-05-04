"""Figure 6 (v2) — DAG comparison heatmap on the Tier-21 redesigned synthetic.

Same visual structure as v1 but uses the Tier-21 generator
(`generate_clean_synthetic` with deterministic Y from full weighted-parent
contribution and edge weights in [0.5, 2.0]) instead of the deprecated Tier-9
stochastic-binarisation generator.

Output: docs/article/figures/fig6_dag_comparison.{pdf,png}  (overwrites v1)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from sklearn.model_selection import train_test_split

from jcce.structure_learning.jcce_learner import create_processor, learn_structure

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory"))
from test_synthetic_jpc_v2 import generate_clean_synthetic

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"

D = 10
N_SAMPLES = 1500  # smaller for CPU
SEED = 0
MAX_ITER = 100
PROCESSORS = ["dag_transformer", "linear_head", "mlp_head"]
PROC_SHORT = {"dag_transformer": "DAG-Transformer", "linear_head": "LinearHead",
               "mlp_head": "MLPHead"}


def make_processor(processor_type, seed):
    if processor_type == "dag_transformer":
        return create_processor(processor_type, random.PRNGKey(seed * 100),
                                  d_model=16, n_heads=2, n_layers=1, temperature=5.0)
    if processor_type == "linear_head":
        return create_processor(processor_type, random.PRNGKey(seed * 100))
    if processor_type == "mlp_head":
        return create_processor(processor_type, random.PRNGKey(seed * 100), hidden_dim=16)


def train_and_get_A(X_tr, Y_tr, T_idx, processor_type, seed, warm_iters=0,
                    lambda_class=1.0):
    n_features = X_tr.shape[1]
    Y_idx = n_features
    proc = make_processor(processor_type, seed)
    A_est, processor, params, metrics = learn_structure(
        data=jnp.array(X_tr, dtype=jnp.float32),
        Y=jnp.array(Y_tr.reshape(-1, 1), dtype=jnp.float32),
        Y_idx=Y_idx, processor=proc, key=random.PRNGKey(seed),
        processor_type=processor_type,
        max_iter=MAX_ITER, lr=0.003, lambda_1=0.05, lambda_class=lambda_class,
        task="classification", verbose=0,
        use_amortized_effects=False, use_confound_matrix=False,
        T_idx_for_loss=T_idx,
        warm_start_fY_iters=warm_iters,
    )
    return np.abs(np.array(metrics["A_weights"]))


def main():
    print(f"Generating Tier-21 ER-SCM data: d={D}, n={N_SAMPLES}, seed={SEED}")
    X, Y, T_idx, A_Y_true_vec, true_w = generate_clean_synthetic(
        d=D, n=N_SAMPLES, seed=SEED, scm_type="linear", noise_scale=0.5,
    )
    n_features = X.shape[1]

    print("Training 6 models on CPU (~5 min)...")
    learned_A = {}
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=SEED)
    for proc in PROCESSORS:
        for fix in ["no_fix", "warm_start"]:
            warm = 50 if fix == "warm_start" else 0
            t0 = time.time()
            A = train_and_get_A(X_tr, Y_tr, T_idx, proc, SEED, warm_iters=warm)
            elapsed = time.time() - t0
            print(f"  {proc:>17s} + {fix:>10s}: |A|.mean={A.mean():.4f}, "
                  f"|A[:,Y]|.mean={A[:n_features, n_features].mean():.4f}, {elapsed:.0f}s")
            learned_A[(proc, fix)] = A

    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.10]})
    all_vals = np.concatenate([A.flatten() for A in learned_A.values()])
    vmax = max(0.05, float(np.percentile(all_vals, 95)))
    Y_col_idx = n_features

    for row, fix in enumerate(["no_fix", "warm_start"]):
        for col, proc in enumerate(PROCESSORS):
            ax = axes[row, col]
            A = learned_A[(proc, fix)]
            im = ax.imshow(A[:Y_col_idx + 1, :Y_col_idx + 1], cmap="viridis",
                            vmin=0, vmax=vmax, aspect="equal")
            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(False)
            ax.set_xticks(range(Y_col_idx + 1))
            ax.set_yticks(range(Y_col_idx + 1))
            xticklabels = [f"x{i}" for i in range(Y_col_idx)] + ["Y"]
            ax.set_xticklabels(xticklabels, fontsize=7)
            ax.set_yticklabels(xticklabels, fontsize=7)
            ax.add_patch(plt.Rectangle((Y_col_idx - 0.5, -0.5), 1, Y_col_idx + 1,
                                          fill=False, edgecolor="red", linewidth=1.5))
            fix_label = "no fix" if fix == "no_fix" else "warm-start"
            ax.set_title(f"{PROC_SHORT[proc]} ({fix_label})", fontsize=9)
            if col == 0:
                ax.set_ylabel("Source variable", fontsize=8)
            if row == 1:
                ax.set_xlabel("Target variable", fontsize=8)

    for col, proc in enumerate(PROCESSORS):
        ax = axes[2, col]
        x = np.arange(len(A_Y_true_vec))
        ax.bar(x, A_Y_true_vec, color="#444", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"x{i}" for i in range(len(A_Y_true_vec))], fontsize=7)
        # Explicit y-label on every column (not just first) so panel reads cleanly.
        ax.set_ylabel("|A_true[:, Y]|", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"Tier-21 ground-truth Y-parents (seed={SEED})", fontsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    cbar_ax = axes[0, 3]
    cbar_ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[0:2, 3].tolist(), fraction=1.0, pad=0.02)
    cbar.set_label("|A| (learned weight)", fontsize=9)
    axes[1, 3].axis("off")
    axes[2, 3].axis("off")

    fig.suptitle(
        f"Figure 6. JCCE-learned $|A|$ matrices (top: no-fix baseline, middle: warm-start) vs "
        f"Tier-21 ground-truth Y-parents (bottom).\n"
        f"Tier-21 redesigned synthetic ER-SCM at $d{{=}}{D}$ ($n{{=}}{N_SAMPLES}$, deterministic "
        f"$Y = \\mathbb{{1}}[X \\cdot A_{{\\text{{true}}}}[:, Y] > \\text{{median}}]$), seed {SEED}, "
        f"single training run per cell. Red box highlights the Y column.\n"
        f"All three processors recover Y-column structure on Tier-21's signal-rich data — the "
        f"DAG-Tr-only pattern from the deprecated Tier-9 generator does not reproduce here, "
        f"consistent with §6.III's Tier-21 finding that post-hoc recovers $\\geq 92\\%$ of "
        f"ceiling across all three processors.",
        y=1.00, fontsize=10,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig6_dag_comparison.pdf"
    out_png = OUT_DIR / "fig6_dag_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
