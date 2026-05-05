#!/usr/bin/env bash
# Tier-37 — Fill the 19 missing cells in Figure 3 (fix-compatibility grid).
#
# Figure 3 currently has 71 of 90 cells filled (79%). Missing:
#   - post-hoc:   3 cells (alarm, child, neuropathic_pain × dag_transformer)
#   - warm-start: 8 cells (4 new datasets × {dag_transformer, mlp_head})
#   - var-reg:    8 cells (4 new datasets × {dag_transformer, mlp_head})
#
# This launcher runs the three sweeps that produce parser-compatible logs:
#   - test_post_hoc_fix_70_30_rerun.py → q37_posthoc_4new_dagtr.log
#   - test_warm_start_sweep.py         → q37_warmstart_4new.log
#   - test_y_variance_sweep.py         → q37_varreg_4new.log
#
# Sized to run on a SINGLE GPU (default: GPU 1 — leaves GPU 0 free for the
# concurrent Phase 2 polish launcher's LLC step). Override with:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/launch_tier37_fig3_fill.sh
#
# Total expected wall-clock: ~3-5 hours on a single RTX 4090.
#
# Usage on server:
#   cd /data/jcce
#   git pull
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/launch_tier37_fig3_fill.sh \
#     > results/server/tier37_fig3_fill_launcher.log 2>&1 &
#   tail -f results/server/tier37_fig3_fill_launcher.log
#
# After completion: re-run Fig 3 to confirm 90/90 cells filled.

set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

REPO_ROOT="$(pwd)"
RESULTS_DIR="${REPO_ROOT}/results/server"
mkdir -p "${RESULTS_DIR}"

NEW_DATASETS="alarm,child,neuropathic_pain,insurance"
NEW_PROCS_FOR_NONLINEAR="dag_transformer,mlp_head"   # warm-start, var-reg fills
POSTHOC_DATASETS_DAGTR_ONLY="alarm,child,neuropathic_pain"  # the only 3 post-hoc gaps
XLA_OPTS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"

echo "==============================================================="
echo "Tier-37 — Fig 3 fill (19 missing cells)"
echo "Started: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
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
    echo "!!! [$label] FAILED (exit $?) — see $logfile"
    return 1
  fi
}

# Step 1 — Post-hoc fix on the 3 missing dag_transformer cells.
# Cheap (~10 min): 3 datasets × 1 proc × 5 seeds × 150 iter.
run_step "Post-hoc 4new (dag_tr only)" \
  "${RESULTS_DIR}/q37_posthoc_4new_dagtr.log" \
  bash -c "XLA_FLAGS='${XLA_OPTS}' uv run python scripts/theory/test_post_hoc_fix_70_30_rerun.py \
    --datasets ${POSTHOC_DATASETS_DAGTR_ONLY} \
    --processors dag_transformer \
    --n-seeds 5 --max-iter 150"

# Step 2 — Warm-start sweep on 4 new × {dag_tr, mlp}.
# Heaviest single step: 4 datasets × 2 procs × 5 warm × 5 seeds = 200 runs.
run_step "Warm-start 4new (dag_tr + mlp)" \
  "${RESULTS_DIR}/q37_warmstart_4new.log" \
  bash -c "XLA_FLAGS='${XLA_OPTS}' uv run python scripts/theory/test_warm_start_sweep.py \
    --datasets ${NEW_DATASETS} \
    --processors ${NEW_PROCS_FOR_NONLINEAR} \
    --warm-iters-list 0,25,50,100,200 \
    --n-seeds 5 --max-iter 150"

# Step 3 — Variance-reg sweep on 4 new × {dag_tr, mlp}.
# 4 × 2 × 6 lambdas × 3 seeds = 144 runs.
run_step "Var-reg 4new (dag_tr + mlp)" \
  "${RESULTS_DIR}/q37_varreg_4new.log" \
  bash -c "XLA_FLAGS='${XLA_OPTS}' uv run python scripts/theory/test_y_variance_sweep.py \
    --datasets ${NEW_DATASETS} \
    --processors ${NEW_PROCS_FOR_NONLINEAR} \
    --lambdas 0.0,0.1,0.3,1.0,3.0,10.0 \
    --n-seeds 3 --max-iter 150"

echo
echo "==============================================================="
echo "Tier-37 — DONE"
echo "Finished: $(date -Iseconds)"
echo "==============================================================="
echo
echo "New logs:"
ls -la "${RESULTS_DIR}"/q37_*.log 2>/dev/null
echo
echo "Next: re-render Fig 3 to confirm 90/90 cells filled:"
echo "  source .venv/bin/activate"
echo "  python scripts/figures/fig3_fix_compatibility_grid.py"
echo "  # output should report: 'Cells filled: 90/90 (100%)'"
