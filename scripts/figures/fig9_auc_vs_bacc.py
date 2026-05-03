"""Figure 9 — AUC vs BAcc divergence: per-cell sign-flipped JPC visualisation.

Implements §7.5 finding 6: cells with both AUC < 0.5 and BAcc < 0.5 are
sign-flipped JPC (Type IIIb); cells with AUC ≈ 0.5 and BAcc ≈ prior are
constant-output JPC (Type IIIa). Post-hoc fix moves cells diagonally up-right.

Data: per-seed table from results/server/q17_extended_metrics.log.

Output: docs/article/figures/fig9_auc_vs_bacc.{pdf,png}
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG = REPO_ROOT / "results" / "server" / "q17_extended_metrics.log"
OUT_DIR = REPO_ROOT / "docs" / "article" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["heart_disease", "lucas", "sachs", "diabetes", "asia", "breast_cancer"]
PROCESSORS = ["linear_head", "mlp_head", "dag_transformer"]
PROC_LABEL = {"linear_head": "Linear", "mlp_head": "MLP", "dag_transformer": "DAG-Tr"}
PROC_MARKER = {"linear_head": "o", "mlp_head": "s", "dag_transformer": "^"}


def parse_per_seed(path: Path) -> list[dict]:
    """Return list of per-seed dicts with bacc_jcce, auc_jcce, bacc_phoc, auc_phoc, etc."""
    text = path.read_text()
    rows = []
    blocks = re.split(r"={5,}\n  (\w+) / (\w+)", text)
    # blocks: [pre, dataset0, proc0, body0, dataset1, proc1, body1, ...]
    for i in range(1, len(blocks) - 2, 3):
        ds = blocks[i]
        proc = blocks[i + 1]
        body = blocks[i + 2]
        if ds not in DATASETS or proc not in PROCESSORS:
            continue
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                seed = int(parts[0])
                bacc_jcce = float(parts[1])
                bacc_phoc = float(parts[2])
                auc_jcce = float(parts[3])
                auc_phoc = float(parts[4])
            except (ValueError, IndexError):
                continue
            rows.append({
                "dataset": ds, "processor": proc, "seed": seed,
                "bacc_jcce": bacc_jcce, "bacc_phoc": bacc_phoc,
                "auc_jcce": auc_jcce, "auc_phoc": auc_phoc,
            })
    return rows


def main() -> None:
    rows = parse_per_seed(LOG)
    assert rows, "no per-seed rows parsed"

    bacc_j = np.array([r["bacc_jcce"] for r in rows])
    auc_j = np.array([r["auc_jcce"] for r in rows])
    bacc_p = np.array([r["bacc_phoc"] for r in rows])
    auc_p = np.array([r["auc_phoc"] for r in rows])
    procs = [r["processor"] for r in rows]

    # Tally regimes
    iiib_mask = (auc_j < 0.5) & (bacc_j < 0.5)
    iiia_mask = (~iiib_mask) & (np.abs(bacc_j - 0.5) < 0.05) & (auc_j < 0.6) & (auc_j >= 0.5)
    escape_mask = ~(iiib_mask | iiia_mask)
    n_iiib = int(iiib_mask.sum())
    n_iiia = int(iiia_mask.sum())
    n_esc = int(escape_mask.sum())
    n = len(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)

    # Panel A: JCCE no-fix
    # Shaded regimes
    for ax in (ax1, ax2):
        ax.add_patch(Rectangle((0, 0), 0.5, 0.5, color="#BB5566", alpha=0.10, zorder=0))
        ax.add_patch(Rectangle((0.45, 0.5), 0.10, 0.10, color="#DDAA33", alpha=0.10, zorder=0))
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.7)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    for proc in PROCESSORS:
        idx = [i for i, p in enumerate(procs) if p == proc]
        ax1.scatter(
            bacc_j[idx], auc_j[idx],
            marker=PROC_MARKER[proc], label=PROC_LABEL[proc],
            s=40, alpha=0.65, edgecolor="black", linewidth=0.5,
        )
        ax2.scatter(
            bacc_p[idx], auc_p[idx],
            marker=PROC_MARKER[proc], label=PROC_LABEL[proc],
            s=40, alpha=0.65, edgecolor="black", linewidth=0.5,
        )

    ax1.set_xlabel("Test BAcc", fontsize=10)
    ax1.set_ylabel("Test AUC", fontsize=10)
    ax1.set_title(
        f"(a) JCCE no-fix\nIIIb (AUC<0.5 ∧ BAcc<0.5): {n_iiib}/{n} cells; "
        f"IIIa (BAcc≈prior): {n_iiia}; escape: {n_esc}",
        fontsize=10,
    )
    ax2.set_xlabel("Test BAcc (post-hoc fix)", fontsize=10)
    ax2.set_ylabel("Test AUC (post-hoc fix)", fontsize=10)
    n_iiib_p = int(((auc_p < 0.5) & (bacc_p < 0.5)).sum())
    ax2.set_title(
        f"(b) Post-hoc fix applied\nIIIb cells remaining: {n_iiib_p}/{n}",
        fontsize=10,
    )

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.05, 0.45, "IIIb\n(sign-flipped)", color="#BB5566", fontsize=8, alpha=0.8)
        ax.text(0.55, 0.05, "(a-only)", color="gray", fontsize=7, alpha=0.5)

    fig.suptitle(
        "Figure 9. AUC vs BAcc per cell (6 datasets × 3 processors × 5 seeds = 90 cells; Tier-17). "
        "Cells in the lower-left red band have both metrics below 0.5 — sign-flipped JPC (Type IIIb). "
        "Post-hoc fix moves cells diagonally toward the upper-right.",
        fontsize=10, y=1.00,
    )
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig9_auc_vs_bacc.pdf"
    out_png = OUT_DIR / "fig9_auc_vs_bacc.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Type IIIb (JCCE): {n_iiib}/{n}; Type IIIa: {n_iiia}; escape: {n_esc}")
    print(f"Type IIIb (post-hoc): {n_iiib_p}/{n}")


if __name__ == "__main__":
    main()
