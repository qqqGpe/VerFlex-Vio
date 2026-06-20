<!--
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-08-31 02:06:34
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
-->
# VerFlex-Vio: Versatile and Flexible Visual-Inertial Odometry System

A high-performance Visual Inertial Odometry (VIO) system implemented in C++ with ROS integration, featuring real-time state estimation using camera and IMU data.

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

- **ROS Noetic** (Ubuntu 20.04)
- **CUDA 11.0+** (for neural feature extraction)
- **C++17** compatible compiler
- **CMake 3.15+**

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
    python3-pip
```

### Building from Source

```bash
# Clone the repository
git clone git@github.com:qqqGpe/VerFlex-Vio.git
cd catkin_ws

# Quickly build the project with script
./src/VerFlex-Vio/script/build_vio_backend.sh

# Source the workspace
source devel/setup.bash
```

## 🚀 Quick Start

### 1. Basic Usage with Euroc Dataset

```bash
# Launch VIO with Euroc dataset with offline rosbag
roslaunch vio euroc_serial_backend.launch

# Launch VIO with Euroc dataset with rosbag playback
roslaunch vio euroc_serial_backend_subscribe.launch

# Launch VIO with Intel Realsense Camera
roslaunch vio realsense_serial_online.launch
```

### 2. Batch simulatation with Euroc dataset
```bash
# Batch simulation and evaluation
source ./devel/setup.zsh

# Run all data
python3 src/vio_backend/script/run_vio_batch.py --align --dataset_dir ~/dataset/euroc_mav --cases all

# Run specified data
python3 src/vio_backend/script/run_vio_batch.py --align --dataset_dir ~/dataset/euroc_mav --cases MH_01_easy

# An example of running results for batch simulation on EuRoC dataset (APE, SE(3)-aligned)
# Config: rotation-compensated warp KLT (frontend) + projected adaptive Mahalanobis chi-square
# gate (backend). MH_05_difficult diverges (structural chi-square limitation: R dominates S so
# the projected gate reduces to the reprojection gate and rejects informative high-parallax
# features); the other 8 sequences are sub-meter to sub-2m. V1_02_medium was 2.65 m before the
# warp fix and is now 0.285 m.
# ================================================================================
# VIO SIMULATION RESULTS SUMMARY (APE, SE(3)-aligned, meters)
# ================================================================================
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | Dataset        |      RMSE |      Mean |    Median |       Std |       Min |       Max |       SSE |
# +================+===========+===========+===========+===========+===========+===========+===========+
# | MH_01_easy     |  0.0676   |  0.0599   |  0.0550   |  0.0314   |  0.0042   |  0.2013   |    16.61  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | MH_02_easy     |  0.1036   |  0.0924   |  0.0822   |  0.0468   |  0.0135   |  0.2338   |    32.13  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | MH_03_medium   |  0.1519   |  0.1338   |  0.1157   |  0.0719   |  0.0341   |  0.3229   |    60.07  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | MH_04_difficult|  0.8022   |  0.6264   |  0.5238   |  0.5011   |  0.1185   |  4.6132   |  1120.33  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | MH_05_difficult| 147.3162  | 60.9185   | 32.5338   |134.1305   | 25.8099   |1010.7689  | 7.834e6   |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | V1_01_easy     |  0.1130   |  0.1000   |  0.0915   |  0.0527   |  0.0211   |  0.6138   |    36.16  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | V1_02_medium   |  0.2852   |  0.1501   |  0.1182   |  0.2426   |  0.0134   |  3.3141   |   126.43  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | V2_01_easy     |  0.0646   |  0.0533   |  0.0460   |  0.0365   |  0.0016   |  0.5981   |     9.26  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
# | V2_02_medium   |  0.1092   |  0.0983   |  0.0963   |  0.0475   |  0.0186   |  0.5533   |    27.06  |
# +----------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
```

**Note**: This is a research-grade VIO system. For production use, additional testing and validation is recommended.
