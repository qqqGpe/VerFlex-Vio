<!--
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-08-31 02:06:34
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
-->
# VerFlex-Vio: Versatile and Flexible Visual-Inertial Odometry System

A high-performance Visual Inertial Odometry (VIO) system implemented in C++ with ROS integration, featuring real-time state estimation using camera and IMU data.

## 🎬 Demo

| EuRoC MH_01_easy | EuRoC V1_01_easy |
| :---: | :---: |
| ![MH_01_easy](docs/viewer_demo.gif) | ![V1_01_easy](docs/viewer_demo_v1_01.gif) |

Real-time Pangolin 3D viewer: trajectory (red), current body pose (green frustum), active clone-window keyframes (blue frustums), MSCKF feature points (blue = in window, black = marginalized), and the live feature-tracking frame (inset). Built-in chase-cam follows the camera from behind; world Z points down to match the VIO frame. Enable with `enable_pangolin_viewer: true` (build with `-DUSE_PANGOLIN=ON`).

## 🚀 Features
- **Multiple camera models**: monocular and stereo camera models
- **Dual operation modes**: online/offline processing
- **Flexible initialization**: static/dynamic visual initialization
- **Multiple estimators**: support ESKF, Sqrt-ESKF solvers
- **Multiple frontends**: traditional KLT-based tracking & NN descriptor-based tracking
- **Threading options**: multi-threaded/single-threaded execution
- **Schmidt ESKF**: First clone pose anchoring to reduce trajectory drift
- **Numerically stable**: Joseph form covariance update for improved robustness
- **Dual feature support**: MSCKF features for batch updates & SLAM features for landmark-based updates
- **Sequential ESKF**: Sequential ESKF update for SLAM features to boost processing speed

## 🛠️ Installation

### Prerequisites

- **C++17** compatible compiler (tested on Ubuntu 20.04 / GCC 9)
- **CMake 3.15+**
- **Pangolin** — required only for the 3D viewer (`-DUSE_PANGOLIN=ON`, which `build.sh` enables); install or build from source if not packaged
- **ROS Noetic** — *optional*, only for the ROS adapter (`BUILD_WITH_ROS=ON`); the standalone `vio_offline` build needs no ROS
- **CUDA 11.0+** — *optional*, only for neural (SuperPoint) feature extraction, which is currently disabled

### System Dependencies

```bash
# Install system dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libopencv-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libceres-dev \
    libsophus-dev \
    libyaml-cpp-dev \
    libfmt-dev \
    python3-pip
```

### Building from Source

```bash
# Clone the repository
git clone git@github.com:qqqGpe/VerFlex-Vio.git
cd VerFlex-Vio

# Standalone build (no ROS required) — produces build/vio_offline
./script/build.sh
```

`build.sh` configures CMake with `-DBUILD_WITH_ROS=OFF -DUSE_PANGOLIN=ON`. Notable options:

- `BUILD_WITH_ROS` (default `OFF`) — build the ROS adapter & nodes (requires catkin)
- `USE_PANGOLIN` (default `OFF`) — build the Pangolin 3D viewer (`build.sh` enables it)
- `BUILD_TESTS` (default `ON`) — build GTest unit tests

```bash
# Manual equivalent of build.sh:
mkdir build && cd build
cmake .. -DBUILD_WITH_ROS=OFF -DUSE_PANGOLIN=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Optional ROS (catkin) build — only if you need the ROS nodes:
catkin build vio && source devel/setup.bash
```

## 🚀 Quick Start

### 1. Basic Usage with Euroc Dataset

Download a EuRoC MAV sequence (e.g. `MH_01_easy`) and run the standalone offline binary on it:

```bash
# Single sequence with the Pangolin 3D viewer
# (default dataset: ~/dataset/euroc_mav/MH_01_easy)
./script/run_euroc.sh ~/dataset/euroc_mav/MH_01_easy

# Or invoke vio_offline directly:
#   build/vio_offline --config config/euroc_stereo.yaml --dataset ~/dataset/euroc_mav/MH_01_easy
# The dataset dir may also be set in config/euroc_stereo.yaml (dataset_dir).
```

For the ROS nodes, see `launch/euroc_serial_backend.launch` (requires the catkin build).

### 2. Batch simulatation with Euroc dataset
```bash
# One-click batch simulation + evo evaluation (APE & RPE, SE(3)-aligned).
# Headless, runs through build/vio_offline — no ROS required.
./script/run_offline_batch.sh

# Override dataset root (positional arg) or concurrency/cases (env vars):
#   ./script/run_offline_batch.sh /path/to/euroc_mav
#   JOBS=4 CASES=MH_01_easy,MH_02_easy ./script/run_offline_batch.sh

# Or invoke the Python script directly:
python3 script/run_offline_batch.py \
    --binary build/vio_offline \
    --config config/euroc_stereo.yaml \
    --dataset_root ~/dataset/euroc_mav --cases all --jobs 11 --align --metric both

# An example of running results for batch simulation on EuRoC dataset (APE, SE(3)-aligned)
# All 11 sequences run to completion (0 crashes).
# Histogram equalization is OFF by default. Rows marked (*) were run with stereo
# ================================================================================
# VIO SIMULATION RESULTS SUMMARY (APE, SE(3)-aligned, meters)
# ================================================================================
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | Dataset           |   RMSE |   Mean |   Median |    Std |    Min |     Max |     SSE |
# +===================+========+========+==========+========+========+=========+=========+
# | MH_01_easy        | 0.0643 | 0.0563 |   0.0453 | 0.0310 | 0.0137 |  0.1882 |   15.02 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | MH_02_easy        | 0.1113 | 0.0966 |   0.0769 | 0.0552 | 0.0327 |  0.2375 |   37.09 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | MH_03_medium      | 0.1676 | 0.1567 |   0.1625 | 0.0595 | 0.0045 |  0.3058 |   73.10 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | MH_04_difficult   | 0.5742 | 0.4761 |   0.4688 | 0.3210 | 0.0818 |  2.6276 |  574.95 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | MH_05_difficult * | 1.6721 | 1.0477 |   0.8169 | 1.3031 | 0.1841 | 21.5603 | 5681.08 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | V1_01_easy        | 0.0955 | 0.0820 |   0.0724 | 0.0489 | 0.0031 |  0.6124 |   25.80 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | V1_02_medium      | 0.2845 | 0.1507 |   0.1180 | 0.2413 | 0.0236 |  3.2869 |  125.79 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | V1_03_difficult * | 0.1471 | 0.1305 |   0.1162 | 0.0678 | 0.0134 |  0.3933 |   44.21 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | V2_01_easy        | 0.0712 | 0.0606 |   0.0557 | 0.0375 | 0.0010 |  0.5665 |   11.23 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | V2_02_medium      | 0.0937 | 0.0826 |   0.0786 | 0.0442 | 0.0178 |  0.5715 |   19.92 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
# | V2_03_difficult * | 0.6613 | 0.5625 |   0.4999 | 0.3477 | 0.0167 |  1.9256 |  800.62 |
# +-------------------+--------+--------+----------+--------+--------+---------+---------+
```

**Note**: This is a research-grade VIO system. For production use, additional testing and validation is recommended.
