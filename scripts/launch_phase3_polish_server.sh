#!/usr/bin/env bash
# Phase 2 polish server batch — A + B + C in sequence.
#
# A) LLC computation on LinearHead/Sachs (Tier-35)
#    Strongest reviewer-bulletproofing: upgrades §6.III.E from rank-only
#    to LLC measurement (Lau et al. 2025).
#
# B) DECI ablation on Sachs (Tier-36)
#    Identifies WHICH DECI component prevents JPC. Sharpens the §6.III
#    cross-framework claim from "DECI escapes" to "DECI's <X> is the
#    active ingredient."
#
# C) Bias-init ablation expansion (Tier-33b)
#    Brings Tier-33's panel from 3 datasets to 6.
#
# Usage on the server (after `git pull`):
#   cd /data/jcce
#   git pull
#   bash scripts/launch_phase3_polish_server.sh 2>&1 | \
#     tee results/server/phase3_polish_server.log
#
# Total expected wall-clock on 2× RTX 4090: ~12-16 hours
# (A: ~30min, B: ~6-8h on CPU since causica is locked to CPU per
# Tier-19's working pattern, C: ~1-2h)

set -euo pipefail
cd "$(dirname "$0")/.."  # repo root
REPO_ROOT="$(pwd)"
RESULTS_DIR="${REPO_ROOT}/results/server"
mkdir -p "${RESULTS_DIR}"

echo "==============================================================="
echo "Phase 2 polish — A + B + C"
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
run_step "LLC smoke (Tier-35)" "${RESULTS_DIR}/q35_llc_smoke.log" \
  python scripts/theory/test_llc_linearhead.py --smoke
run_step "LLC full (Tier-35)" "${RESULTS_DIR}/q35_llc_full.log" \
  python scripts/theory/test_llc_linearhead.py \
    --n-seeds 5 --num-chains 4 --num-draws 200 --num-burnin 100 \
    --out "${RESULTS_DIR}/q35_llc_linearhead_sachs.json"
deactivate

# ----- (C) Bias-init expansion (run before B since it's faster) -----
echo
echo "----- (C) Bias-init ablation expansion -----"
source .venv/bin/activate
run_step "Bias-init expansion (Tier-33b)" \
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
# First: discovery (so the log records what kwargs are available).
run_step "DECI API discovery (Tier-36)" \
  "${RESULTS_DIR}/q36_deci_discover.log" \
  python scripts/theory/test_deci_ablation.py --discover
# Smoke (only 'full' + 'no_variational' — verifies the wiring before the
# long run. NoImpl on the latter is expected on first server pass; the
# log captures it so Pedro can fill the kwargs.)
run_step "DECI ablation smoke (Tier-36)" \
  "${RESULTS_DIR}/q36_deci_smoke.log" \
  python scripts/theory/test_deci_ablation.py --smoke || \
  echo ">>> Smoke had NotImplementedErrors as expected on first pass; check the log."
# Full ablation: only run if the script's TODOs are filled in. The
# `set -e` at the top would otherwise abort on NotImplementedError;
# we run conditionally and capture the failure as data.
run_step "DECI ablation full (Tier-36)" \
  "${RESULTS_DIR}/q36_deci_full.log" \
  python scripts/theory/test_deci_ablation.py \
    --datasets sachs --ablations full,no_variational,no_spline,linear_sem \
    --n-seeds 5 --max-epochs 500 \
    --out "${RESULTS_DIR}/q36_deci_ablation_sachs.json" || \
  echo ">>> [B] DECI ablation full pass had NotImplementedErrors; fill TODOs in test_deci_ablation.py and re-run."
deactivate

echo
echo "==============================================================="
echo "Phase 2 polish — DONE"
echo "Finished: $(date -Iseconds)"
echo "==============================================================="
echo
echo "Outputs:"
ls -la "${RESULTS_DIR}"/q33b_*.json \
       "${RESULTS_DIR}"/q35_*.json \
       "${RESULTS_DIR}"/q36_*.json 2>/dev/null
