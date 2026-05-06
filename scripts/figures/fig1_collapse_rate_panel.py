"""Figure 1 — JPC collapse rate panel: real benchmarks (per-processor) +
cross-framework triangulation (JCCE / CASTLE / DECI).

Two panels:
- Left: collapse rate on 6 real benchmarks × 3 JCCE processors at fixed-default
  hyperparameters (|df_Y/dT| < 1e-3 threshold).
- Right: cross-framework collapse-rate summary on the 7-dataset cross-framework
  suite (Heart, LUCAS, Sachs, Alarm, Child, Insurance, Neuropathic Pain) —
  JCCE, CASTLE [Kyono et al. 2020], DECI [Geffner et al. 2024]. Cross-framework
  collapse rates are the headline §6.III finding.

Data sources:
- Real: variance-reg log at lam=0 cells (q31_varreg_70_30_split[01].log)
        — per-seed |df_Y/dT| at lam_var=0 (no fix) for 6 datasets × 3 procs.
- Cross-framework: hardcoded from §6.III tables (CASTLE and DECI sweeps).

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
    """Parse per-seed collapsed flag from a synthetic SCM log.

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
    # --- Real benchmarks: from variance-reg logs ---
    real_data = {}  # (dataset, proc) -> collapse_rate
    for split_idx in (0, 1):
        path = LOG_DIR / f"q31_varreg_70_30_split{split_idx}.log"
        per_seed = parse_varreg_per_seed(path)
        for (ds, proc), values in per_seed.items():
            real_data[(ds, proc)] = collapse_rate(values)

    # --- Cross-framework collapse rates (from §6.III tables) ---
    # JCCE: 7/7 datasets collapse on all seeds = 100%
    # CASTLE: 6/7 escape; only Sachs collapses at 60% seed-level → mean 8.6%
    # DECI: 0/7 collapse on all 35 cells = 0%
    crossfw_datasets = ["Heart", "LUCAS", "Sachs", "Alarm", "Child", "Insurance", "NeurPain"]
    crossfw = {
        "JCCE":   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "CASTLE": [0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.00],
        "DECI":   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    }

    # --- Plot ---
    fig, (ax_real, ax_xfw) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.0, 1.2]}
    )
    proc_colors = {
        "dag_transformer": "#1f77b4",
        "linear_head":     "#ff7f0e",
        "mlp_head":        "#2ca02c",
    }
    fw_colors = {"JCCE": "#bf3636", "CASTLE": "#d09642", "DECI": "#1f6fb3"}

    # Real panel — grouped bars per dataset
    bar_w = 0.25
    x_real = np.arange(len(DATASETS_REAL))
    for i, proc in enumerate(PROCESSORS):
        rates = [real_data.get((ds, proc), float("nan")) for ds in DATASETS_REAL]
        offset = (i - 1) * bar_w
        ax_real.bar(x_real + offset, rates, bar_w,
                     label=PROC_SHORT[proc], color=proc_colors[proc], alpha=0.9)
    ax_real.set_xticks(x_real)
    ax_real.set_xticklabels(DATASETS_REAL_SHORT, rotation=15, ha="right", fontsize=9)
    ax_real.set_ylabel("JPC collapse rate", fontsize=10)
    ax_real.set_ylim(0, 1.18)
    ax_real.set_title("(a) JCCE — 6 real benchmarks × 3 processors", fontsize=10)
    ax_real.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax_real.spines["top"].set_visible(False)
    ax_real.spines["right"].set_visible(False)
    # Processor legend on left axes only — outside top
    ax_real.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                    ncol=3, frameon=False)

    # Cross-framework panel — grouped bars per dataset × framework
    x_xfw = np.arange(len(crossfw_datasets))
    for i, fw in enumerate(["JCCE", "CASTLE", "DECI"]):
        offset = (i - 1) * bar_w
        ax_xfw.bar(x_xfw + offset, crossfw[fw], bar_w,
                    label=fw, color=fw_colors[fw], alpha=0.9)
    ax_xfw.set_xticks(x_xfw)
    ax_xfw.set_xticklabels(crossfw_datasets, rotation=15, ha="right", fontsize=9)
    ax_xfw.set_ylim(0, 1.18)
    ax_xfw.set_title("(b) Cross-framework — JCCE → CASTLE → DECI on 7 datasets", fontsize=10)
    ax_xfw.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax_xfw.spines["top"].set_visible(False)
    ax_xfw.spines["right"].set_visible(False)
    ax_xfw.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                    ncol=3, frameon=False)

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
    print("\nCross-framework collapse rates (right panel):")
    for fw in ["JCCE", "CASTLE", "DECI"]:
        rates = crossfw[fw]
        n_collapsed = sum(1 for r in rates if r > 0)
        mean_rate = float(np.mean(rates))
        print(f"  {fw:>8s}: {n_collapsed}/{len(rates)} datasets affected, "
              f"mean rate {mean_rate*100:5.1f}%")


if __name__ == "__main__":
    main()
