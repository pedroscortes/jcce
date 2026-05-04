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
            # Hatch the bar based on collapse fraction
            hatch_density = "////" if cr >= 0.5 else "//"
            label_y = bar.get_height() + 0.018
            ax.bar(bar.get_x(), bar.get_height(), bar.get_width(),
                    color="none", edgecolor="black", linewidth=0,
                    hatch=hatch_density, alpha=0.6, align="edge")
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                     f"{int(cr * 100)}% seeds\ncollapse",
                     ha="center", fontsize=8, color="#990000", style="italic")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                     "0% collapse",
                     ha="center", fontsize=8, color="#1a4d2e", style="italic")

    # Reference lines
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
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Trajectory arrow
    ax.annotate(
        "",
        xy=(3.0, 0.93), xytext=(0.0, 0.93),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.5),
    )
    ax.text(1.5, 0.96, "architectural sophistication →",
             ha="center", fontsize=10, style="italic", color="#444")

    fig.suptitle(
        "Figure 11. Three-framework JPC trajectory on Sachs ($n{=}7466$, $d{=}10$).\n"
        "JCCE no-fix collapses 7/7 datasets including Sachs (BAcc $0.50$); JCCE+post-hoc fix "
        "lifts to $0.77$. CASTLE \\cite{kyono2020} ($d{+}1$ subnet decoupling) escapes "
        "JPC on 6 of 7 datasets but collapses on Sachs ($60\\%$ of seeds, BAcc $0.63$). "
        "DECI \\cite{geffner2024} (variational graph + spline noise) escapes on all 7, "
        "reaching BAcc $0.79$ on the same cell.\n"
        "Pairwise Fisher exact tests on 35-cell-per-framework totals with Holm-Bonferroni at "
        "$\\alpha{=}0.05$: JCCE vs DECI $p{=}1.8 \\times 10^{-20}$ (reject $H_0$), "
        "JCCE vs CASTLE $p{=}1.5 \\times 10^{-16}$ (reject), CASTLE vs DECI $p{=}0.24$ (fail to "
        "reject — direction-consistent but underpowered at $n{=}5$ seeds).",
        y=1.05, fontsize=9,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig11_three_framework_sachs.pdf"
    out_png = OUT_DIR / "fig11_three_framework_sachs.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
