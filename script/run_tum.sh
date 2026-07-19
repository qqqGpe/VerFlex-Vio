#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
CONFIG="$PROJECT_DIR/config/tumvi_room_stereo.yaml"
DATASET_DIR="${1:-$HOME/dataset/tum_vi/dataset-room1_1024_16}"

if [ ! -f "$BUILD_DIR/vio_offline" ]; then
    echo "Error: vio_offline not found. Run build.sh first."
    exit 1
fi

# TUM-VI rooms ship as ~5 GB .tar files; auto-extract if the dir is missing but
# the tar exists, so room2-6 run out of the box.
if [ ! -d "$DATASET_DIR" ]; then
    TAR_FILE="$(dirname "$DATASET_DIR")/$(basename "$DATASET_DIR").tar"
    if [ -f "$TAR_FILE" ]; then
        echo "Extracting $TAR_FILE ..."
        tar -xf "$TAR_FILE" -C "$(dirname "$DATASET_DIR")"
    else
        echo "Error: dataset dir not found: $DATASET_DIR"
        echo "       and no tar at: $TAR_FILE"
        exit 1
    fi
fi

# Run from PROJECT_DIR so the relative log_path ("log_tumvi") lands predictably.
cd "$PROJECT_DIR"

# Force the X11/XWayland backend for the Pangolin viewer. On NVIDIA + GNOME-Wayland,
# Pangolin's GLFW hangs inside EGL context creation. Harmless when the viewer is
# disabled (enable_pangolin_viewer: false).
export WAYLAND_DISPLAY=nonexistent
export QT_QPA_PLATFORM=offscreen

LOG_DIR="$PROJECT_DIR/log_tumvi"
"$BUILD_DIR/vio_offline" --config "$CONFIG" --dataset "$DATASET_DIR"

# ATE evaluation hint (evo is not installed; use the self-contained Umeyama script).
EST_TRAJ="$(ls -t "$LOG_DIR"/*_tum.csv 2>/dev/null | head -1)"
echo ""
echo "Done. Estimate: $EST_TRAJ"
echo "Evaluate ATE:"
echo "  python3 $SCRIPT_DIR/evaluate/tumvi_ate_eval.py \"$EST_TRAJ\" \"$LOG_DIR/gt_tum.txt\""
