#!/usr/bin/env bash
# Polish server batch — A + B + C in sequence.
#
# A) LLC computation on LinearHead/Sachs
#    Strongest reviewer-bulletproofing: upgrades §6.III.E from rank-only
#    to LLC measurement (Lau et al. 2025).
#
# B) DECI ablation on Sachs
#    Identifies WHICH DECI component prevents JPC. Sharpens the §6.III
#    cross-framework claim from "DECI escapes" to "DECI's <X> is the
#    active ingredient."
#
# C) Bias-init ablation expansion
#    Brings the bias-init ablation panel from 3 datasets to 6.
#
# Usage on the server (after `git pull`):
#   cd /data/jcce
#   git pull
#   bash scripts/launch_phase3_polish_server.sh 2>&1 | \
#     tee results/server/phase3_polish_server.log
#
# Total expected wall-clock on 2× RTX 4090: ~12-16 hours
# (A: ~30min, B: ~6-8h on CPU since causica is locked to CPU per the
# DECI JPC probe's working pattern, C: ~1-2h)

set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
REPO_ROOT="$(pwd)"
RESULTS_DIR="${REPO_ROOT}/results/server"
mkdir -p "${RESULTS_DIR}"

echo "==============================================================="
echo "Polish — A + B + C"
echo "Started: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "==============================================================="

run_step() {
  local label="$1"; shift
  local logfile="$1"; shift
  echo
  echo ">>> [$label] $(date -Iseconds)"
  echo ">>> Log: $logfile"
  if "$@" > "$logfile" 2>&1; then
    echo ">>> [$label] OK ($(date -Iseconds))"
  else
    local rc=$?
    echo "!!! [$label] FAILED (exit $rc) — see $logfile"
    return $rc
  fi
}

# ----- (A) LLC on LinearHead/Sachs -----
echo
echo "----- (A) LLC computation -----"
source .venv/bin/activate
run_step "LLC smoke" "${RESULTS_DIR}/q35_llc_smoke.log" \
  python scripts/theory/test_llc_linearhead.py --smoke
run_step "LLC full" "${RESULTS_DIR}/q35_llc_full.log" \
  python scripts/theory/test_llc_linearhead.py \
    --n-seeds 5 --num-chains 4 --num-draws 200 --num-burnin 100 \
    --out "${RESULTS_DIR}/q35_llc_linearhead_sachs.json"
deactivate

# ----- (C) Bias-init expansion (run before B since it's faster) -----
echo
echo "----- (C) Bias-init ablation expansion -----"
source .venv/bin/activate
run_step "Bias-init expansion" \
  "${RESULTS_DIR}/q33b_bias_init_extra.log" \
  python scripts/theory/test_bias_init_ablation.py \
    --datasets diabetes,asia,breast_cancer \
    --n-seeds 5 --max-iter 150 \
    --out "${RESULTS_DIR}/q33b_bias_init_ablation_extra.json"
deactivate

# ----- (B) DECI ablation -----
echo
echo "----- (B) DECI variational/spline ablation -----"
source .deci_venv/bin/activate
# Discovery already informed the kwarg choices on the first server run
# (logged at results/server/q36_deci_discover.log). Re-run for archive.
run_step "DECI API discovery" \
  "${RESULTS_DIR}/q36_deci_discover.log" \
  python scripts/theory/test_deci_ablation.py --discover || \
  echo ">>> [B] discovery non-fatal warnings — continuing."
# Smoke: 'full' + 'no_spline' (both implemented).
run_step "DECI ablation smoke" \
  "${RESULTS_DIR}/q36_deci_smoke.log" \
  python scripts/theory/test_deci_ablation.py \
    --datasets sachs --ablations full,no_spline \
    --n-seeds 1 --max-epochs 50 \
    --out "${RESULTS_DIR}/q36_deci_ablation_sachs_smoke.json" || \
  echo ">>> [B] smoke had errors — see log."
# Full ablation: full + no_spline + no_variational (linear_sem is NotImpl
# and skipped at script level).
run_step "DECI ablation full" \
  "${RESULTS_DIR}/q36_deci_full.log" \
  python scripts/theory/test_deci_ablation.py \
    --datasets sachs --ablations full,no_spline,no_variational \
    --n-seeds 5 --max-epochs 500 \
    --out "${RESULTS_DIR}/q36_deci_ablation_sachs.json" || \
  echo ">>> [B] full pass had errors — see log."
deactivate

echo
echo "==============================================================="
echo "Polish — DONE"
echo "Finished: $(date -Iseconds)"
echo "==============================================================="
echo
echo "Outputs:"
ls -la "${RESULTS_DIR}"/q33b_*.json \
       "${RESULTS_DIR}"/q35_*.json \
       "${RESULTS_DIR}"/q36_*.json 2>/dev/null
