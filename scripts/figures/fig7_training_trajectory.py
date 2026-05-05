"""Figure 7 (Hero candidate) — Training trajectory: gradient starvation in action.

Visualizes the §6.III reconstruction-dominance hypothesis directly:
- ||∇recon|| dominates ||∇bce|| by 10-50x throughout training under default
  lambda_class=1.0 — gradient starvation in the sense of Pezeshki et al. 2021.
- Increasing lambda_class to 10.0 multiplies the BCE contribution to the total
  gradient but, post-Adam-normalization, has limited effect on the parameter
  trajectory. The dominance is structural, not just a hyperparameter knob.

Three panels:
- (a) Loss components over training (recon, BCE, total) — log-y axis.
- (b) Raw gradient norms ||∇recon|| vs ||∇bce|| — log-y axis. The starvation gap.
- (c) Weighted contributions ||∇recon|| vs ||λ_class · ∇bce|| — overlays both
      lambda_class values to show the contribution gap that the lambda_class
      knob is supposed to close.

Data source: results/server/q14_trajectory_*.json (from test_trajectory_logging.py).

Output: docs/article/figures/fig7_training_trajectory.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "results" / "server"
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trajectory(dataset, processor, lambda_class):
    path = LOG_DIR / f"q14_trajectory_{dataset}_{processor}_lc{lambda_class}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # Pick representative dataset + processor (override with CLI args later if needed)
    dataset = "heart_disease"
    processor = "linear_head"
    lcs = [1.0, 10.0]

    data = {lc: load_trajectory(dataset, processor, lc) for lc in lcs}
    if not all(data[lc] for lc in lcs):
        missing = [lc for lc in lcs if not data[lc]]
        print(f"Missing trajectory data for lambda_class={missing}")
        print(f"Run: bash scripts/theory/launch_tier14_trajectory.sh (or the underlying script)")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.0))

    colors = {1.0: "#1f77b4", 10.0: "#d62728"}
    labels = {1.0: r"$\lambda_{\mathrm{class}}{=}1$ (default)", 10.0: r"$\lambda_{\mathrm{class}}{=}10$"}

    # Panel (a): Loss components, only for default lambda_class
    traj1 = data[1.0]["trajectory"]
    iters1 = [e["iter"] for e in traj1]
    ax1.plot(iters1, [e["recon_loss"] for e in traj1], label="reconstruction", color="#1f77b4", linewidth=2)
    ax1.plot(iters1, [e["bce_loss"] for e in traj1], label="BCE classification", color="#ff7f0e", linewidth=2)
    ax1.plot(iters1, [e["sparsity_loss"] for e in traj1], label="sparsity", color="#2ca02c", linewidth=1.5, linestyle=":")
    ax1.plot(iters1, [e["h_A_sq"] for e in traj1], label="DAG penalty $h(A)^2$", color="#9467bd", linewidth=1.5, linestyle=":")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training iteration", fontsize=10)
    ax1.set_ylabel("Loss component (log scale)", fontsize=10)
    ax1.set_title(rf"(a) Loss components, $\lambda_{{\mathrm{{class}}}}{{=}}1$", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, frameon=False)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Raw gradient norms — show the starvation gap directly
    for lc in lcs:
        traj = data[lc]["trajectory"]
        iters = [e["iter"] for e in traj]
        ax2.plot(iters, [e["grad_recon_norm"] for e in traj],
                  label=f"||∇recon||  ({labels[lc]})", color=colors[lc], linewidth=2, linestyle="-")
        ax2.plot(iters, [e["grad_bce_norm"] for e in traj],
                  label=f"||∇BCE||  ({labels[lc]})", color=colors[lc], linewidth=2, linestyle="--")
    ax2.set_yscale("log")
    ax2.set_xlabel("Training iteration", fontsize=10)
    ax2.set_ylabel("Raw gradient norm (log scale)", fontsize=10)
    ax2.set_title("(b) Reconstruction gradient dominates BCE gradient", fontsize=10)
    ax2.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, frameon=False)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Weighted contributions ||∇recon|| vs ||lambda * ∇BCE||
    for lc in lcs:
        traj = data[lc]["trajectory"]
        iters = [e["iter"] for e in traj]
        recon_norms = np.array([e["grad_recon_norm"] for e in traj])
        bce_norms = np.array([e["grad_bce_norm"] for e in traj])
        weighted_bce = lc * bce_norms
        ax3.plot(iters, recon_norms, label=f"||∇recon||  ({labels[lc]})", color=colors[lc], linewidth=2, linestyle="-")
        ax3.plot(iters, weighted_bce, label=rf"||$\lambda$·∇BCE||  ({labels[lc]})", color=colors[lc], linewidth=2, linestyle="--")
    ax3.set_yscale("log")
    ax3.set_xlabel("Training iteration", fontsize=10)
    ax3.set_ylabel("Effective gradient contribution (log scale)", fontsize=10)
    ax3.set_title(r"(c) Effective gradient: $\lambda_{\mathrm{class}}$ closes some of the gap", fontsize=10)
    ax3.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, frameon=False)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Figure 7. Gradient starvation during JCCE joint training on {dataset} ({processor}, seed=0). "
        f"\nReconstruction gradient dominates BCE gradient by 1-2 orders of magnitude throughout; "
        rf"$\lambda_{{\mathrm{{class}}}}{{=}}10$ partially closes the contribution gap (panel c) but does "
        f"not eliminate it.",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig7_training_trajectory.pdf"
    out_png = OUT_DIR / "fig7_training_trajectory.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # Console summary
    print(f"\nGradient norm summary at last iteration:")
    for lc in lcs:
        last = data[lc]["trajectory"][-1]
        print(f"  lambda_class={lc}: ||∇recon||={last['grad_recon_norm']:.5f}  "
              f"||∇BCE||={last['grad_bce_norm']:.5f}  "
              f"||lambda·∇BCE||={lc*last['grad_bce_norm']:.5f}  "
              f"ratio (lambda·BCE / recon)={lc*last['grad_bce_norm']/last['grad_recon_norm']:.4f}")


if __name__ == "__main__":
    main()
