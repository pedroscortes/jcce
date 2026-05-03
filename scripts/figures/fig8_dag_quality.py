"""Figure 8 — DAG quality (SHD, F1) on Bayesian-network benchmarks.

Visualises that JCCE's structural learning is largely independent of the
predictive collapse: across Asia, LUCAS, Sachs (the three BN benchmarks with
ground-truth DAGs), SHD and F1 are stable across processors even when their
joint-trained f_Y collapses. Pairs with the §7.5 finding-4 mechanistic claim
that struct_corr is preserved while f_Y collapses (structure-readout decoupling).

Data source: results/server/q17_extended_metrics.log (TIER-17 SUMMARY block).

Output: docs/article/figures/fig8_dag_quality.{pdf,png}
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG = REPO_ROOT / "results" / "server" / "q17_extended_metrics.log"
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BN_DATASETS = ["asia", "lucas", "sachs"]
DS_LABEL = {"asia": "Asia (d=8)", "lucas": "LUCAS (d=11)", "sachs": "Sachs (d=10)"}
PROCESSORS = ["linear_head", "mlp_head", "dag_transformer"]
PROC_LABEL = {"linear_head": "Linear", "mlp_head": "MLP", "dag_transformer": "DAG-Tr"}
PROC_COLOR = {"linear_head": "#4477AA", "mlp_head": "#DDAA33", "dag_transformer": "#BB5566"}


def parse_summary(path: Path) -> dict:
    """Return dict[(dataset, processor)] -> dict with shd, shd_std, f1, f1_std, bacc_jcce, bacc_phoc."""
    text = path.read_text()
    m = re.search(r"TIER-17 SUMMARY.*?Reading guide", text, re.DOTALL)
    if not m:
        raise RuntimeError("could not locate TIER-17 SUMMARY block")
    out = {}
    for line in m.group(0).splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        ds = parts[0]
        proc = parts[1]
        if ds not in BN_DATASETS or proc not in PROCESSORS:
            continue

        def parse_pm(token: str):
            try:
                a, b = token.split("+/-")
                return float(a), float(b)
            except (ValueError, AttributeError):
                return float("nan"), float("nan")

        bacc_jcce = parse_pm(parts[2])
        bacc_phoc = parse_pm(parts[3])
        # parts[4] = AUC_jcce; parts[5] = SHD; parts[6] = F1
        shd = parse_pm(parts[5]) if "+/-" in parts[5] else (float("nan"), float("nan"))
        f1 = parse_pm(parts[6]) if "+/-" in parts[6] else (float("nan"), float("nan"))
        out[(ds, proc)] = {
            "bacc_jcce": bacc_jcce,
            "bacc_phoc": bacc_phoc,
            "shd": shd,
            "f1": f1,
        }
    return out


def main() -> None:
    rows = parse_summary(LOG)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: SHD bars
    x = np.arange(len(BN_DATASETS))
    w = 0.26
    for i, proc in enumerate(PROCESSORS):
        vals = [rows[(ds, proc)]["shd"][0] for ds in BN_DATASETS]
        errs = [rows[(ds, proc)]["shd"][1] for ds in BN_DATASETS]
        ax1.bar(
            x + (i - 1) * w, vals, w, yerr=errs, capsize=3,
            color=PROC_COLOR[proc], label=PROC_LABEL[proc], alpha=0.85,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([DS_LABEL[d] for d in BN_DATASETS])
    ax1.set_ylabel("SHD vs ground-truth DAG (lower is better)", fontsize=10)
    ax1.set_title("(a) Structural Hamming Distance", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Panel B: F1 bars
    for i, proc in enumerate(PROCESSORS):
        vals = [rows[(ds, proc)]["f1"][0] for ds in BN_DATASETS]
        errs = [rows[(ds, proc)]["f1"][1] for ds in BN_DATASETS]
        ax2.bar(
            x + (i - 1) * w, vals, w, yerr=errs, capsize=3,
            color=PROC_COLOR[proc], label=PROC_LABEL[proc], alpha=0.85,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([DS_LABEL[d] for d in BN_DATASETS])
    ax2.set_ylabel("F1 vs ground-truth DAG (higher is better)", fontsize=10)
    ax2.set_title("(b) Edge-recovery F1", fontsize=11)
    ax2.set_ylim(0, max(0.3, ax2.get_ylim()[1]))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="y", alpha=0.3, linestyle=":")

    fig.suptitle(
        "Figure 8. JCCE structural learning vs ground-truth DAG on Bayesian-network benchmarks "
        "(Tier-17, 5 seeds).\n"
        "All processors recover similar-quality DAGs (SHD constant within ±2; F1 mostly 0–0.12), "
        "even when their joint-trained $f_Y$ collapses (Linear and DAG-Tr on Sachs, all on Asia).\n"
        "Pairs with §7.5 finding 4: structure-readout decoupling.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig8_dag_quality.pdf"
    out_png = OUT_DIR / "fig8_dag_quality.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
