"""Figure 11 — Three-framework Sachs trajectory (the headline cross-framework visual).

Bar chart showing JCCE no-fix, JCCE+post-hoc, CASTLE, DECI on Sachs. The headline
result of the paper: architectural sophistication monotonically reduces JPC severity,
with DECI escaping JPC on the cell that defeats both JCCE (universal collapse) and
CASTLE (60% collapse rate). B2 (sklearn baseline) shown as dashed reference.

Source: numbers already in §6.III tables — no log parsing needed.

Output: docs/article/figures/fig11_three_framework_sachs.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"


def main():
    methods = [
        "JCCE\nno fix",
        "JCCE\n+ post-hoc",
        "CASTLE",
        "DECI",
    ]
    bacc = [0.500, 0.767, 0.627, 0.785]
    bacc_std = [0.000, 0.021, 0.061, 0.030]
    collapse_rate = [1.00, 0.00, 0.60, 0.00]  # fraction of seeds collapsed
    colors = ["#bf3636", "#3d7a36", "#d09642", "#1f6fb3"]

    B2_ref = 0.873  # sklearn MLP baseline on Sachs
    prior = 0.500

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(methods))
    width = 0.55

    bars = ax.bar(x, bacc, width, color=colors, alpha=0.85,
                    edgecolor="black", linewidth=0.5,
                    yerr=bacc_std, capsize=5)

    # Hatched overlay for collapse rate (visual encoding of 'how broken')
    for i, (bar, cr) in enumerate(zip(bars, collapse_rate)):
        if cr > 0:
            # Hatch the bar based on collapse fraction (visual-only cue;
            # collapse rate is reported in the LaTeX caption).
            hatch_density = "////" if cr >= 0.5 else "//"
            ax.bar(bar.get_x(), bar.get_height(), bar.get_width(),
                    color="none", edgecolor="black", linewidth=0,
                    hatch=hatch_density, alpha=0.6, align="edge")

    # Reference lines (legend will be placed below the axes — no overlap)
    ax.axhline(B2_ref, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"B2 sklearn ceiling ({B2_ref})")
    ax.axhline(prior, color="gray", linestyle=":", linewidth=0.8, alpha=0.7,
                label=f"class prior ({prior})")

    # Value labels under bars
    for xi, val in zip(x, bacc):
        ax.text(xi, 0.06, f"{val:.3f}",
                 ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Test BAcc on Sachs (5 seeds)", fontsize=11)
    ax.set_ylim(0, 1.0)
    # Reference-line legend below the axes — no overlap with bars/labels
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
               fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # In-figure title and arrow annotation removed — captions go in the LaTeX caption.
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig11_three_framework_sachs.pdf"
    out_png = OUT_DIR / "fig11_three_framework_sachs.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
