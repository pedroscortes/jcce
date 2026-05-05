#!/usr/bin/env bash
# Fisher-rank bootstrap CIs server launcher.
#
# Runs Tier-34 across the 3 processors sequentially. Addresses the v3
# panel's "Fisher rank lacks bootstrap CIs and seed variance" concern
# (raised by Roles 1, 3, 4, 5, 7 across multiple LLMs).
#
# Each step:
#   - Trains JCCE 5 seeds × 2 configs (default + Sachs Optuna best) ×
#     2 datasets (Sachs, Tier-21 d=20)
#   - Computes empirical Fisher Jacobian (200 held-in samples)
#   - Bootstrap-resamples 1000× → 95% rank CI per seed
#
# Usage on server (after git pull):
#   cd /data/jcce
#   git pull
#   source .venv/bin/activate
#   bash scripts/launch_fisher_bootstrap_server.sh 2>&1 | \
#     tee results/server/fisher_bootstrap_server.log
#
# Total expected wall-clock on 2× RTX 4090: ~6–10 hours
# (DAG-Tr's 9808-param Jacobian is the bottleneck).

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

REPO_ROOT="$(pwd)"
RESULTS_DIR="${REPO_ROOT}/results/server"
mkdir -p "${RESULTS_DIR}"

echo "==============================================================="
echo "Fisher-rank bootstrap CIs (Tier-34) — v3 panel response"
echo "Started: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "==============================================================="

run_step() {
  local label="$1"; shift
  local logfile="$1"; shift
  echo
  echo ">>> [$label] $(date -Iseconds)"
  echo ">>> Command: $*"
  echo ">>> Log: $logfile"
  if "$@" > "$logfile" 2>&1; then
    echo ">>> [$label] OK ($(date -Iseconds))"
  else
    echo "!!! [$label] FAILED (exit $?) — see $logfile"
    return 1
  fi
}

# Step 1 — LinearHead (small Fisher matrix, fastest)
run_step "Fisher boot LinearHead" "${RESULTS_DIR}/q34a_fisher_rank_bootstrap_linear.log" \
  python scripts/theory/test_fisher_rank_bootstrap.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type linear_head \
    --n-bootstrap 1000 \
    --out "${RESULTS_DIR}/q34a_fisher_rank_bootstrap_linear.json"

# Step 2 — MLPHead (medium; 193-337 params; the headline rank-doubling cell)
run_step "Fisher boot MLPHead" "${RESULTS_DIR}/q34b_fisher_rank_bootstrap_mlp.log" \
  python scripts/theory/test_fisher_rank_bootstrap.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type mlp_head \
    --n-bootstrap 1000 \
    --out "${RESULTS_DIR}/q34b_fisher_rank_bootstrap_mlp.json"

# Step 3 — DAG-Transformer (largest; 9808-9952 params; slowest)
run_step "Fisher boot DAG-Tr" "${RESULTS_DIR}/q34c_fisher_rank_bootstrap_dagtr.log" \
  python scripts/theory/test_fisher_rank_bootstrap.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type dag_transformer \
    --n-bootstrap 1000 \
    --out "${RESULTS_DIR}/q34c_fisher_rank_bootstrap_dagtr.json"

echo
echo "==============================================================="
echo "Fisher-rank bootstrap CIs — DONE"
echo "Finished: $(date -Iseconds)"
echo "==============================================================="
echo
echo "Outputs ready to copy back:"
ls -la "${RESULTS_DIR}"/q34a_*.json "${RESULTS_DIR}"/q34b_*.json "${RESULTS_DIR}"/q34c_*.json 2>/dev/null
