"""Figure 10 — Calibration of joint-trained logits vs post-hoc head.

Shows the predicted-probability distribution for a representative collapsed cell
(Asia / LinearHead) under JCCE no-fix vs the post-hoc fix. JCCE's logits cluster
around the prior (visible as a near-degenerate spike); post-hoc spreads them out
into a discriminative distribution. Pairs with §6.III calibrated-threshold note
and §7.5 finding 6.

Data: results/server/q17_extended_*.npz (raw test logits/probabilities per seed).

Output: docs/article/figures/fig10_calibration.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
NPZ_DIR = REPO_ROOT / "results" / "server"
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Three representative cells: collapsed (Asia/Linear), bimodal (Heart/MLP),
# and a no-collapse reference (LUCAS/Linear).
CELLS = [
    ("asia", "linear_head", "Asia / Linear (sign-flipped JPC)"),
    ("heart_disease", "mlp_head", "Heart / MLP (bimodal collapse)"),
    ("lucas", "linear_head", "LUCAS / Linear (escape)"),
]
SEED = 0


def load_cell(dataset: str, processor: str, seed: int) -> dict:
    path = NPZ_DIR / f"q17_extended_{dataset}_{processor}_seed{seed}.npz"
    return dict(np.load(path, allow_pickle=True))


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    handles = None
    for col, (ds, proc, label) in enumerate(CELLS):
        d = load_cell(ds, proc, SEED)
        y = d["Y_te"]
        p_jcce = d["Y_prob_jcce"]
        p_phoc = d["Y_prob_post_hoc"]

        # Top: predicted-probability histogram, JCCE vs post-hoc, split by true class.
        ax = axes[0, col]
        bins = np.linspace(0, 1, 26)
        h0 = ax.hist(
            p_jcce[y == 0], bins=bins, alpha=0.5, color="#4477AA",
            label="true y = 0", density=True,
        )
        h1 = ax.hist(
            p_jcce[y == 1], bins=bins, alpha=0.5, color="#BB5566",
            label="true y = 1", density=True,
        )
        ax.set_xlabel("Predicted P(y=1) — JCCE no-fix", fontsize=9)
        if col == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if handles is None:
            handles = [h0[2][0], h1[2][0]]

        # Bottom: same histogram for post-hoc fix
        ax = axes[1, col]
        ax.hist(
            p_phoc[y == 0], bins=bins, alpha=0.5, color="#4477AA",
            density=True,
        )
        ax.hist(
            p_phoc[y == 1], bins=bins, alpha=0.5, color="#BB5566",
            density=True,
        )
        ax.set_xlabel("Predicted P(y=1) — post-hoc fix", fontsize=9)
        if col == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single shared legend at top of figure
    fig.legend(handles=handles, labels=["true y = 0", "true y = 1"],
                 loc="upper center", bbox_to_anchor=(0.5, 1.02),
                 ncol=2, fontsize=10, frameon=False)

    fig.suptitle(
        "Figure 10. Predicted-probability distributions on the test split, JCCE no-fix (top) vs "
        "post-hoc fix (bottom).\n"
        "Asia/Linear: JCCE collapses both classes onto P≈0.4 (sign-flipped), post-hoc separates them. "
        "Heart/MLP: JCCE shows a single bimodal mode mixing classes; post-hoc separates by class. "
        "LUCAS/Linear: JCCE already separates (no JPC), post-hoc preserves separation.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig10_calibration.pdf"
    out_png = OUT_DIR / "fig10_calibration.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
