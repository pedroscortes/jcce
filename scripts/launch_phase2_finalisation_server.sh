#!/usr/bin/env bash
# Phase 2 finalisation server launcher.
#
# Runs the GPU-friendly experiments from the Phase 2 finalisation campaign
# in sequence:
#   - Run 11 cross-processor extension (LinearHead already done locally;
#     here we run MLPHead and DAG-Transformer — needs GPU for DAG-Tr's
#     attention)
#   - B1 NOTEARS-MLP comparison (8 datasets, 5 seeds; minimal JAX impl)
#   - B3 Fisher rank preview at JPC-collapsed vs Optuna-best configs
#
# CPU-runnable experiments (already done on local machine, no need to
# re-run on server):
#   - Run 11 LinearHead diagnostic (q27_actual_learner_diagnostic.json)
#   - A2 Optuna reconciliation (q28_optuna_jpc_reconciliation.json)
#   - A4 Twins PEHE bootstrap CI (analysis on existing logs)
#   - C2 lambda_class sweep (q29_lambda_class_sweep.json)
#   - C3 GES + LiNGAM classical baselines (q30_classical_baselines.json,
#     CPU-only by design — running locally as of commit a6e5311+)
#
# Excluded from this launcher (need separate setup):
#   - B2 BayesDAG: needs microsoft/causica install + paradigm-mismatch
#     coupling discussion (deferred pending advisor decision)
#   - C1 DECI ablation: needs DECI venv + code modification (deferred)
#
# Usage on server:
#   cd /data/jcce  # or wherever the repo lives
#   git pull
#   source .venv/bin/activate     # JAX/Flax venv
#   bash scripts/launch_phase2_finalisation_server.sh 2>&1 | \
#     tee results/server/phase2_finalisation_server.log
#
# Then: scp results/server/q27b_*.json q27c_*.json q31_*.json q32_*.json
#       back to the laptop, and `git pull` of any committed scripts.
#
# Total expected wall-clock: ~3–6 hours on 2× RTX 4090.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

REPO_ROOT="$(pwd)"
RESULTS_DIR="${REPO_ROOT}/results/server"
mkdir -p "${RESULTS_DIR}"

echo "==============================================================="
echo "Phase 2 finalisation server campaign"
echo "Started: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "==============================================================="

# Short helper
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

# ============================================================================
# Step 1 — Run 11 cross-processor extension: MLPHead
# ============================================================================
run_step "Run-11 MLP" "${RESULTS_DIR}/q27b_actual_learner_diagnostic_mlp.log" \
  python scripts/theory/test_actual_learner_gradient_diagnostic.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type mlp_head \
    --out "${RESULTS_DIR}/q27b_actual_learner_diagnostic_mlp.json"

# ============================================================================
# Step 2 — Run 11 cross-processor extension: DAG-Transformer
# ============================================================================
run_step "Run-11 DAG-Tr" "${RESULTS_DIR}/q27c_actual_learner_diagnostic_dagtr.log" \
  python scripts/theory/test_actual_learner_gradient_diagnostic.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type dag_transformer \
    --out "${RESULTS_DIR}/q27c_actual_learner_diagnostic_dagtr.json"

# ============================================================================
# Step 3 — B1 NOTEARS-MLP comparison (7 real-data benchmarks)
# ============================================================================
run_step "B1 NOTEARS-MLP" "${RESULTS_DIR}/q31_notears_mlp_comparison.log" \
  python scripts/theory/test_notears_mlp_comparison.py \
    --datasets sachs,heart_disease,lucas,asia,alarm,child,insurance,neuropathic_pain \
    --n-seeds 5 \
    --out "${RESULTS_DIR}/q31_notears_mlp_comparison.json"

# ============================================================================
# Step 4 — B3 Fisher rank preview (Sachs + Tier-21, default vs Optuna-best)
#         Three processors: linear_head (Sanity check; low-dim; rank-deficit
#         signal expected to be small) → MLP (medium-dim; rank-deficit story
#         can show clearly) → DAG-Transformer (high-dim; canonical processor).
# ============================================================================
run_step "B3 Fisher rank Linear" "${RESULTS_DIR}/q32a_fisher_rank_linear.log" \
  python scripts/theory/test_fisher_rank_preview.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type linear_head \
    --out "${RESULTS_DIR}/q32a_fisher_rank_linear.json"

run_step "B3 Fisher rank MLP" "${RESULTS_DIR}/q32b_fisher_rank_mlp.log" \
  python scripts/theory/test_fisher_rank_preview.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type mlp_head \
    --out "${RESULTS_DIR}/q32b_fisher_rank_mlp.json"

run_step "B3 Fisher rank DAG-Tr" "${RESULTS_DIR}/q32c_fisher_rank_dagtr.log" \
  python scripts/theory/test_fisher_rank_preview.py \
    --datasets sachs,tier21_d20 \
    --n-seeds 5 \
    --max-iter 150 \
    --processor-type dag_transformer \
    --out "${RESULTS_DIR}/q32c_fisher_rank_dagtr.json"

echo
echo "==============================================================="
echo "Phase 2 finalisation server campaign — DONE"
echo "Finished: $(date -Iseconds)"
echo "==============================================================="
echo
echo "Outputs ready to copy back:"
ls -la "${RESULTS_DIR}"/q27b_*.json "${RESULTS_DIR}"/q27c_*.json \
       "${RESULTS_DIR}"/q31_*.json "${RESULTS_DIR}"/q32a_*.json \
       "${RESULTS_DIR}"/q32b_*.json "${RESULTS_DIR}"/q32c_*.json 2>/dev/null

echo
echo "Suggested rsync (run on laptop):"
echo "  rsync -av --include='q27b_*' --include='q27c_*' --include='q31_*' \\"
echo "    --include='q32*_*' --exclude='*' \\"
echo "    server:${REPO_ROOT}/results/server/ ./results/server/"
