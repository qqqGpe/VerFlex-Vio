#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_DIR" \
    -DBUILD_WITH_ROS=OFF \
    -DBUILD_TESTS=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

make -j$(nproc)

echo "Build completed. Executable: $BUILD_DIR/vio_offline"
