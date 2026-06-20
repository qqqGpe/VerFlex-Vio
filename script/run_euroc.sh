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

# Force the X11/XWayland backend for the Pangolin viewer. On NVIDIA + GNOME-Wayland,
# Pangolin's GLFW hangs inside EGL context creation (NVIDIA exposes no /dev/dri render
# node that Mesa EGL can open). Routing through XWayland + the NVIDIA GLX driver works.
# This is harmless when the viewer is disabled (enable_pangolin_viewer: false).
export WAYLAND_DISPLAY=nonexistent
# No Qt GUI is used by the viewer; keep Qt's platform-plugin probe quiet (OpenCV's
# HighGUI lib pulls in Qt5, which otherwise prints harmless wayland-plugin warnings).
export QT_QPA_PLATFORM=offscreen

"$BUILD_DIR/vio_offline" --config "$CONFIG" --dataset "$DATASET_DIR"
