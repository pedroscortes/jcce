"""Figure 1 — Joint-loss posterior collapse (JPC) rate panel: real vs synthetic.

Visualizes the universality finding: collapse rate (fraction of seeds with
|df_Y/dT| < 1e-3) on (i) real benchmarks at the no-fix baseline configuration
vs (ii) synthetic ER-SCM data across configurations. Tells the "mechanism-driven,
not data-driven" story in one frame.

Data sources:
- Real: Tier-7 variance-reg log at lam=0 cells (q31_varreg_70_30_split[01].log)
        — per-seed |df_Y/dT| at lam_var=0 (no fix) for 6 datasets × 3 procs
- Synthetic: Tier-9 SCM logs (scm_linear_default.log + scm_linear_lownoise.log
        + scm_nonlinear.log + scm_linear_n10k.log) — explicit "collapsed" flag
        per (config, d, processor, seed)

Output: docs/article/figures/fig1_jpc_collapse_rate.{pdf,png}
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "results" / "server"
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_REAL = ["heart_disease", "lucas", "sachs", "diabetes", "asia", "breast_cancer"]
DATASETS_REAL_SHORT = ["Heart", "LUCAS", "Sachs", "Diabetes", "Asia", "BC"]
PROCESSORS = ["dag_transformer", "linear_head", "mlp_head"]
PROC_SHORT = {"dag_transformer": "DAG-Tr", "linear_head": "Linear", "mlp_head": "MLP"}

COLLAPSE_THRESHOLD = 1e-3


def parse_varreg_per_seed(path: Path):
    """Parse per-seed |df_Y/dT| at lam=0 from a varreg log file.

    Returns dict[(dataset, processor)] -> list of abs_dT values across seeds.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    # Match "Dataset: <name>" then within each dataset block, "Processor: <name>"
    # then per-seed rows: "  lam  seed   train   test    y_var   |df_Y/dT|   |A[T,Y]|   sec"
    out = {}
    current_dataset = None
    current_proc = None
    for line in text.splitlines():
        m_ds = re.match(r"\s*Dataset:\s+(\S+)", line)
        if m_ds:
            current_dataset = m_ds.group(1).strip().rstrip(",")
            current_proc = None
            continue
        m_proc = re.match(r"\s*Processor:\s+(\w+)", line)
        if m_proc:
            current_proc = m_proc.group(1)
            continue
        if current_dataset is None or current_proc is None:
            continue
        # Per-seed row: floats separated by whitespace; want lam=0.00 specifically
        toks = line.strip().split()
        if len(toks) < 7:
            continue
        try:
            lam = float(toks[0])
            seed = int(toks[1])
            # train, test, y_var, abs_dT, signed_dT (or A_T_Y), sec
            # Format varies; |df_Y/dT| is column 5 (index 5, 0-indexed)
            abs_dT = float(toks[5])
        except (ValueError, IndexError):
            continue
        if abs(lam) > 1e-9:
            continue  # only the no-fix baseline (lam=0)
        key = (current_dataset, current_proc)
        out.setdefault(key, []).append(abs_dT)
    return out


def parse_scm_per_seed(path: Path):
    """Parse per-seed collapsed flag from a Tier-9 SCM log.

    Returns dict[(d, processor)] -> list of collapsed booleans.
    Header rows are skipped; per-seed rows look like:
        seed  test_bacc  abs_dT  A[T,Y]  struct_corr  collapsed  sec
    """
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    current_d = None
    current_proc = None
    for line in text.splitlines():
        m_d = re.match(r"\s*d\s*=\s*(\d+),", line)
        if m_d:
            current_d = int(m_d.group(1))
            current_proc = None
            continue
        m_proc = re.match(r"\s*Processor:\s+(\w+)", line)
        if m_proc:
            current_proc = m_proc.group(1)
            continue
        if current_d is None or current_proc is None:
            continue
        toks = line.strip().split()
        # per-seed row has 7 cols; collapsed is the 6th (index 5)
        if len(toks) != 7:
            continue
        try:
            seed = int(toks[0])
            collapsed_str = toks[5]
        except (ValueError, IndexError):
            continue
        if collapsed_str not in ("True", "False"):
            continue
        key = (current_d, current_proc)
        out.setdefault(key, []).append(collapsed_str == "True")
    return out


def collapse_rate(abs_dT_list):
    """Fraction of values below COLLAPSE_THRESHOLD."""
    if not abs_dT_list:
        return float("nan")
    return float(np.mean([float(v) < COLLAPSE_THRESHOLD for v in abs_dT_list]))


def main():
    # --- Real benchmarks: from Tier-7 varreg logs ---
    real_data = {}  # (dataset, proc) -> collapse_rate
    for split_idx in (0, 1):
        path = LOG_DIR / f"q31_varreg_70_30_split{split_idx}.log"
        per_seed = parse_varreg_per_seed(path)
        for (ds, proc), values in per_seed.items():
            real_data[(ds, proc)] = collapse_rate(values)

    # --- Synthetic: from Tier-9 SCM logs ---
    scm_files = {
        "default (n=2000, noise=0.5)": "scm_linear_default.log",
        "low-noise (noise=0.1)":         "scm_linear_lownoise.log",
        "nonlinear MLP-SCM":             "scm_nonlinear.log",
        "data-rich (n=10000)":           "scm_linear_n10k.log",
    }
    synthetic_data = {}  # (config, d, proc) -> collapse_rate
    for cfg_label, fname in scm_files.items():
        per_seed = parse_scm_per_seed(LOG_DIR / fname)
        for (d, proc), flags in per_seed.items():
            # flags are already booleans; rate = fraction of True
            synthetic_data[(cfg_label, d, proc)] = float(np.mean(flags)) if flags else float("nan")

    # --- Plot ---
    fig, (ax_real, ax_syn) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.0, 1.4]}
    )
    proc_colors = {
        "dag_transformer": "#1f77b4",
        "linear_head":     "#ff7f0e",
        "mlp_head":        "#2ca02c",
    }

    # Real panel — grouped bars per dataset
    bar_w = 0.25
    x_real = np.arange(len(DATASETS_REAL))
    for i, proc in enumerate(PROCESSORS):
        rates = [real_data.get((ds, proc), float("nan")) for ds in DATASETS_REAL]
        offset = (i - 1) * bar_w
        bars = ax_real.bar(x_real + offset, rates, bar_w,
                            label=PROC_SHORT[proc], color=proc_colors[proc])
        for x, r in zip(x_real + offset, rates):
            if not np.isnan(r):
                ax_real.text(x, r + 0.02, f"{r:.0%}", ha="center", fontsize=7)
    ax_real.set_xticks(x_real)
    ax_real.set_xticklabels(DATASETS_REAL_SHORT, rotation=20, ha="right", fontsize=9)
    ax_real.set_ylabel("JPC collapse rate", fontsize=10)
    ax_real.set_ylim(0, 1.10)
    ax_real.set_title("(a) Real benchmarks", fontsize=10)
    ax_real.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax_real.legend(fontsize=8, loc="upper right", frameon=True)

    # Synthetic panel — grouped by config × processor
    cfg_labels = list(scm_files.keys())
    n_cfg = len(cfg_labels)
    x_syn = np.arange(n_cfg)
    for i, proc in enumerate(PROCESSORS):
        rates = []
        for cfg in cfg_labels:
            # If config has multiple d values, take mean rate
            cfg_rates = [v for (c, d, p), v in synthetic_data.items()
                         if c == cfg and p == proc]
            rates.append(float(np.mean(cfg_rates)) if cfg_rates else float("nan"))
        offset = (i - 1) * bar_w
        ax_syn.bar(x_syn + offset, rates, bar_w,
                    label=PROC_SHORT[proc], color=proc_colors[proc])
        for x, r in zip(x_syn + offset, rates):
            if not np.isnan(r):
                ax_syn.text(x, r + 0.02, f"{r:.0%}", ha="center", fontsize=7)
    ax_syn.set_xticks(x_syn)
    ax_syn.set_xticklabels(cfg_labels, rotation=15, ha="right", fontsize=8)
    ax_syn.set_ylim(0, 1.10)
    ax_syn.set_title("(b) Synthetic ER-SCM (4 configurations, mean across d)", fontsize=10)
    ax_syn.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)

    fig.suptitle(
        "Figure 1. JPC collapse rate: real benchmarks (40–60%) vs synthetic ER-SCM (100%).",
        y=1.02, fontsize=11,
    )
    fig.text(0.5, -0.05,
             "Collapse defined as $|\\partial f_Y/\\partial T| < 10^{-3}$ on the trained model. "
             "Real benchmark rates from Tier-7 variance-reg sweep at $\\lambda$=0 (no-fix baseline). "
             "Synthetic from Tier-9 extended ER-SCM sweep.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig1_jpc_collapse_rate.pdf"
    out_png = OUT_DIR / "fig1_jpc_collapse_rate.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

    # Console summary
    print("\nReal benchmarks — JPC collapse rate at no-fix baseline:")
    for ds in DATASETS_REAL:
        row = "  " + ds.ljust(18)
        for proc in PROCESSORS:
            r = real_data.get((ds, proc), float("nan"))
            row += f"  {PROC_SHORT[proc]}: {r*100:5.1f}%" if not np.isnan(r) else f"  {PROC_SHORT[proc]}: ----- "
        print(row)
    print("\nSynthetic — JPC collapse rate per config × processor:")
    for cfg in cfg_labels:
        for proc in PROCESSORS:
            cfg_rates = [v for (c, d, p), v in synthetic_data.items()
                         if c == cfg and p == proc]
            if cfg_rates:
                print(f"  {cfg:>30s}  {PROC_SHORT[proc]:>6s}: "
                      f"mean={np.mean(cfg_rates)*100:5.1f}% "
                      f"(d-cells={len(cfg_rates)})")


if __name__ == "__main__":
    main()
