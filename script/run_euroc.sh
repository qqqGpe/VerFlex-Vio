#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
CONFIG="$PROJECT_DIR/config/euroc_stereo.yaml"
DATASET_DIR="${1:-$HOME/dataset/euroc_mav/MH_01_easy}"

if [ ! -f "$BUILD_DIR/vio_offline" ]; then
    echo "Error: vio_offline not found. Run build.sh first."
    exit 1
fi

"$BUILD_DIR/vio_offline" --config "$CONFIG" --dataset "$DATASET_DIR"
