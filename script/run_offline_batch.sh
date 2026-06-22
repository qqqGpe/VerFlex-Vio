#!/usr/bin/env bash
# One-click batch VIO simulation + evo evaluation (no ROS).
#
# Runs all EuRoC cases through build/vio_offline, headless (Pangolin is forced
# off in the per-case temp configs), with SE(3)-aligned APE + RPE evaluation.
# Mirrors the verified full-dataset batch run.
#
# Usage:
#   ./run_offline_batch.sh                         # all cases, defaults below
#   ./run_offline_batch.sh /path/to/euroc_mav      # override dataset root
#   JOBS=4 CASES=MH_01_easy,MH_02_easy ./run_offline_batch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                  # src/VerFlex-Vio
BINARY="$PKG_DIR/build/vio_offline"
CONFIG="$PKG_DIR/config/euroc_stereo.yaml"

# Dataset root: first positional arg, else $DATASET_ROOT, else default.
DATASET_ROOT="${1:-${DATASET_ROOT:-$HOME/dataset/euroc_mav}}"

# Concurrency and case list are overridable via env; defaults match the
# reference run (all 11 EuRoC cases at once on a 28-core machine).
JOBS="${JOBS:-11}"
CASES="${CASES:-all}"

if [ ! -x "$BINARY" ]; then
    echo "Error: vio_offline not found or not executable: $BINARY" >&2
    echo "       Build it first with script/build.sh" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 1
fi
if [ ! -d "$DATASET_ROOT" ]; then
    echo "Error: dataset root not found: $DATASET_ROOT" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/run_offline_batch.py" \
    --binary "$BINARY" \
    --config "$CONFIG" \
    --dataset_root "$DATASET_ROOT" \
    --cases "$CASES" --jobs "$JOBS" --align --metric both --timeout 1200
