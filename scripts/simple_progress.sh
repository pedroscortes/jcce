#!/bin/bash
# Simple one-liner progress check

RUN_DIR="${1:-results/comprehensive_study/run_20251106_152320}"
COMPLETED=$(find "$RUN_DIR" -name "vae_*.npy" 2>/dev/null | wc -l)
echo "Progress: $COMPLETED/486 ($(echo "scale=1; $COMPLETED * 100 / 486" | bc)%)"
