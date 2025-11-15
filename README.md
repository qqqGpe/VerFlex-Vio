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
basedir=#PATH_TO_WS

source ./devel/setup.zsh

python3 ${basedir}/script/run_batch_sim.py --ros_node "vio" --launch_file "euroc_serial_backend.launch"

# An example of running results for batch simulation on EuRoc Dataset
# ================================================================================
# VIO SIMULATION RESULTS SUMMARY
# ================================================================================
# +--------------+----------+----------+----------+----------+----------+----------+----------+
# | Dataset      |     RMSE |     Mean |   Median |      Std |      Min |      Max |      SSE |
# +==============+==========+==========+==========+==========+==========+==========+==========+
# | MH_01_easy   | 0.284176 | 0.256443 | 0.252405 | 0.122446 | 0.072402 | 0.724794 | 292.417  |
# +--------------+----------+----------+----------+----------+----------+----------+----------+
# | MH_02_easy   | 0.261663 | 0.22578  | 0.198678 | 0.132252 | 0.026258 | 0.753107 | 203.485  |
# +--------------+----------+----------+----------+----------+----------+----------+----------+
# | MH_03_medium | 0.18343  | 0.163888 | 0.150588 | 0.082384 | 0.01175  | 0.571727 |  88.2884 |
# +--------------+----------+----------+----------+----------+----------+----------+----------+
# | V1_01_easy   | 0.216476 | 0.18816  | 0.173854 | 0.10704  | 0.023342 | 0.41278  | 132.385  |
# +--------------+----------+----------+----------+----------+----------+----------+----------+
# | V1_02_medium | 0.222966 | 0.201963 | 0.181061 | 0.094472 | 0.032893 | 0.811831 |  82.5253 |
# +--------------+----------+----------+----------+----------+----------+----------+----------+
# | V2_01_easy   | 0.095293 | 0.085423 | 0.07872  | 0.042233 | 0.012368 | 0.234315 |  20.3046 |
# +--------------+----------+----------+----------+----------+----------+----------+----------+
# | V2_02_medium | 0.261529 | 0.245031 | 0.229532 | 0.091417 | 0.005754 | 0.542613 | 157.519  |
# +--------------+----------+----------+----------+----------+----------+----------+----------+
```

**Note**: This is a research-grade VIO system. For production use, additional testing and validation is recommended.
