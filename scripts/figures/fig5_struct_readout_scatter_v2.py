"""Figure 5 (v2) — Structure-vs-readout decoupling on the Tier-21 redesigned synthetic.

Reads `results/server/fig5_tier21_data.json` (produced by
`regenerate_fig5_data_tier21.py`) which has per-cell (test_bacc, struct_corr)
on linear ER-SCM d=20, 5 seeds × 3 processors × 3 fixes = 45 cells.

Updated qualitative story vs Tier-11 v1:
  - On Tier-21, post-hoc fix recovers BAcc to 0.92+ across all 3 processors,
    not just DAG-Tr — because the redesigned generator gives ALL processors
    a recoverable |A|. The decoupling pattern (high BAcc despite low
    struct_corr) is more pronounced on warm-start cells.

Output: docs/article/figures/fig5_struct_readout_scatter.{pdf,png}  (overwrites v1)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "results" / "server" / "fig5_tier21_data.json"
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"


def main():
    rows = json.loads(DATA_PATH.read_text())
    print(f"Loaded {len(rows)} cells from {DATA_PATH}")

    PROCS = ["linear_head", "mlp_head", "dag_transformer"]
    PROC_SHORT = {"linear_head": "Linear", "mlp_head": "MLP",
                   "dag_transformer": "DAG-Tr"}
    FIXES = ["no_fix", "post_hoc", "warm_start"]
    FIX_SHORT = {"no_fix": "no-fix", "post_hoc": "post-hoc",
                  "warm_start": "warm-start"}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    proc_colors = {"linear_head": "#ff7f0e", "mlp_head": "#2ca02c",
                    "dag_transformer": "#1f77b4"}
    fix_markers = {"no_fix": "o", "post_hoc": "s", "warm_start": "^"}

    for proc in PROCS:
        for fix in FIXES:
            sub = [r for r in rows if r["processor"] == proc and r["fix"] == fix
                    and not (r["struct_corr"] is None or np.isnan(r["struct_corr"]))]
            if not sub:
                continue
            xs = [r["struct_corr"] for r in sub]
            ys = [r["test_bacc"] for r in sub]
            ax.scatter(
                xs, ys, c=proc_colors[proc], marker=fix_markers[fix],
                s=90, alpha=0.75, edgecolors="black", linewidths=0.5,
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)

    ax.set_xlabel(
        r"struct_corr: Pearson $\rho$(learned $|A[:, Y]|$, "
        r"ground truth $|A_{\mathrm{true}}[:, Y]|$)",
        fontsize=10,
    )
    ax.set_ylabel("Per-seed test BAcc", fontsize=10)
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(0.30, 1.00)

    # Compact two-axis legend: color = processor, marker = fix.
    from matplotlib.lines import Line2D
    proc_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=proc_colors[p],
                markeredgecolor="black", markersize=9, label=PROC_SHORT[p])
        for p in PROCS
    ]
    fix_handles = [
        Line2D([0], [0], marker=fix_markers[f], color="w", markerfacecolor="#888888",
                markeredgecolor="black", markersize=9, label=FIX_SHORT[f])
        for f in FIXES
    ]
    leg1 = ax.legend(handles=proc_handles, title="Processor (color)",
                      fontsize=8, title_fontsize=8.5, loc="upper center",
                      bbox_to_anchor=(0.30, -0.13), ncol=3, frameon=False,
                      handletextpad=0.3, columnspacing=1.0)
    ax.add_artist(leg1)
    ax.legend(handles=fix_handles, title="Fix (marker)",
              fontsize=8, title_fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.78, -0.13), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.0)

    # Effective-region shading in neutral gray (green clashed with mlp_head color)
    ax.axvspan(0.0, 1.0, alpha=0.08, color="#888888")
    ax.text(0.5, 1.01, "post-hoc effective region (struct_corr > 0)",
             fontsize=8, ha="center", transform=ax.get_xaxis_transform(),
             color="#444444", style="italic", alpha=0.9)

    fig.suptitle(
        "Figure 5. Structure-vs-readout decoupling on the Tier-21 redesigned synthetic\n"
        "(linear ER-SCM, $d{=}20$, $n{=}2000$, 5 seeds × 3 processors × 3 fixes).\n"
        "Post-hoc (squares) lifts BAcc to $\\geq 0.92$ on cells with struct_corr $> 0$ "
        "across ALL three processors — the structure-readout decoupling that licenses "
        "post-hoc as the universal fix.",
        y=1.04, fontsize=10,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig5_struct_readout_scatter.pdf"
    out_png = OUT_DIR / "fig5_struct_readout_scatter.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # Per-cell summary
    print("\nPer (processor, fix) summary:")
    for proc in PROCS:
        for fix in FIXES:
            sub = [r for r in rows if r["processor"] == proc and r["fix"] == fix
                    and not (r["struct_corr"] is None or np.isnan(r["struct_corr"]))]
            if not sub:
                continue
            sc_mean = float(np.mean([r["struct_corr"] for r in sub]))
            ba_mean = float(np.mean([r["test_bacc"] for r in sub]))
            print(f"  {PROC_SHORT[proc]:>6s} + {FIX_SHORT[fix]:>10s}: n={len(sub)}, "
                  f"struct_corr={sc_mean:+.3f}, test_bacc={ba_mean:.3f}")


if __name__ == "__main__":
    main()
