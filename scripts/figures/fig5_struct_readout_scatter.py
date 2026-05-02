"""Figure 5 — Structure-vs-readout decoupling (per-seed scatter on synthetic SCM).

Visualizes the §7.5 finding 4 + finding 5 directly:
- x-axis: per-seed `struct_corr` (Pearson rho between learned |A[:,Y]| and ground truth)
- y-axis: per-seed test BAcc
- Color: processor (DAG-Tr, Linear, MLP)
- Marker: fix variant (no-fix, post-hoc, warm-start, KL-bottleneck)

Tells the story: post-hoc fix lifts BAcc when struct_corr > 0 (bypasses readout,
uses structurally-informative |A| weights); warm-start/KL lift BAcc independently
of struct_corr (sidestep f_Y bimodality without requiring |A| structure).

Data source: Tier-11 q11_scm_fixes_linear_d20.log per-seed table.

Output: docs/article/figures/fig5_struct_readout_scatter.{pdf,png}
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "results" / "server"
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"


def parse_tier11(path: Path):
    """Returns list of dicts with keys: processor, fix, seed, test_bacc, struct_corr."""
    text = path.read_text()
    out = []
    current_proc = None
    current_fix = None
    in_table = False
    for line in text.splitlines():
        m_proc = re.match(r"\s*Processor:\s+(\w+)", line)
        if m_proc:
            current_proc = m_proc.group(1)
            current_fix = None
            in_table = False
            continue
        m_fix = re.match(r"\s*Fix:\s+(\w+)", line)
        if m_fix:
            current_fix = m_fix.group(1)
            in_table = False
            continue
        if "test_bacc" in line and "struct_corr" in line:
            in_table = True
            continue
        if not in_table or current_proc is None or current_fix is None:
            continue
        toks = line.strip().split()
        if len(toks) != 7:
            continue
        try:
            seed = int(toks[0])
            test_bacc = float(toks[1])
            # toks[2] = abs_dT, toks[3] = A[T,Y], toks[4] = struct_corr, toks[5] = collapsed, toks[6] = sec
            struct_corr = float(toks[4]) if toks[4] != "nan" else float("nan")
        except (ValueError, IndexError):
            continue
        out.append({
            "processor": current_proc,
            "fix": current_fix,
            "seed": seed,
            "test_bacc": test_bacc,
            "struct_corr": struct_corr,
        })
    return out


def main():
    rows = parse_tier11(LOG_DIR / "q11_scm_fixes_linear_d20.log")
    print(f"Parsed {len(rows)} rows from Tier-11 log")

    PROCS = ["dag_transformer", "linear_head", "mlp_head"]
    PROC_SHORT = {"dag_transformer": "DAG-Tr", "linear_head": "Linear", "mlp_head": "MLP"}
    FIXES = ["no_fix", "post_hoc", "warm_start", "kl_bottleneck"]
    FIX_SHORT = {"no_fix": "no-fix", "post_hoc": "post-hoc",
                  "warm_start": "warm-start", "kl_bottleneck": "KL"}

    # Plot — single panel scatter
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    proc_colors = {"dag_transformer": "#1f77b4", "linear_head": "#ff7f0e", "mlp_head": "#2ca02c"}
    fix_markers = {"no_fix": "o", "post_hoc": "s", "warm_start": "^", "kl_bottleneck": "D"}

    for proc in PROCS:
        for fix in FIXES:
            sub = [r for r in rows if r["processor"] == proc and r["fix"] == fix
                    and not np.isnan(r["struct_corr"])]
            if not sub:
                continue
            xs = [r["struct_corr"] for r in sub]
            ys = [r["test_bacc"] for r in sub]
            ax.scatter(xs, ys, c=proc_colors[proc], marker=fix_markers[fix],
                        s=80, alpha=0.7, edgecolors="black", linewidths=0.5,
                        label=f"{PROC_SHORT[proc]}, {FIX_SHORT[fix]}")

    # Reference lines
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.text(0.95, 0.51, "random", fontsize=8, color="gray", ha="right")

    ax.set_xlabel(r"struct_corr: Pearson $\rho$(learned $|A[:, Y]|$, ground truth $|A_{\mathrm{true}}[:, Y]|$)",
                   fontsize=10)
    ax.set_ylabel("Per-seed test BAcc", fontsize=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.30, 0.85)
    ax.legend(fontsize=7, ncol=4, loc="upper left", frameon=True, handletextpad=0.3,
               columnspacing=0.6)

    # Annotation: "post-hoc lift requires struct_corr > 0" region
    ax.axvspan(0.0, 1.0, alpha=0.06, color="green")
    ax.text(0.5, 0.83, "post-hoc effective region", fontsize=9, ha="center",
             color="green", style="italic", alpha=0.85)

    fig.suptitle(
        "Figure 5. Structure-vs-readout decoupling (Tier-11 synthetic ER-SCM, $d{=}20$, 5 seeds × 3 procs × 4 fixes).\n"
        "Post-hoc (squares) lifts BAcc only on DAG-Tr (blue), where struct_corr > 0; warm-start/KL "
        "(triangles, diamonds) lift BAcc on Linear/MLP regardless of struct_corr.",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig5_struct_readout_scatter.pdf"
    out_png = OUT_DIR / "fig5_struct_readout_scatter.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # Summary
    print("\nPer (processor, fix) summary:")
    for proc in PROCS:
        for fix in FIXES:
            sub = [r for r in rows if r["processor"] == proc and r["fix"] == fix
                    and not np.isnan(r["struct_corr"])]
            if not sub:
                continue
            sc_mean = float(np.mean([r["struct_corr"] for r in sub]))
            ba_mean = float(np.mean([r["test_bacc"] for r in sub]))
            print(f"  {PROC_SHORT[proc]:>6s} + {FIX_SHORT[fix]:>10s}: "
                  f"n={len(sub)}, struct_corr={sc_mean:+.3f}, test_bacc={ba_mean:.3f}")


if __name__ == "__main__":
    main()
