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
# | MH_01_easy   | 0.121111 | 0.105462 | 0.085949 | 0.059545 | 0.015374 | 0.355678 | 53.0098  |
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

## ⚙️ Parameters

### Solver Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver_type` | int | 0 | Estimator type: 0=ESKF, 1=Sqrt-ESKF |
| `enable_schmidt_eskf` | bool | false | Enable Schmidt ESKF to anchor first clone pose and reduce drift |
| `use_fej` | bool | false | Enable First-Estimate Jacobian for consistency |

### Feature Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_slam_feature` | bool | false | Enable SLAM features for landmark-based updates |
| `max_slam_feature` | int | 25 | Maximum number of SLAM features in state |

### Schmidt ESKF

When `enable_schmidt_eskf=true`, the first clone pose in the sliding window is anchored:
- The first frame remains in the state vector and contributes to innovation covariance S
- Its Kalman gain is zeroed, so the state is never updated
- This anchors the trajectory to a reference frame, reducing drift
- Reference: [Schmidt-EKF for Visual-Inertial SLAM (arxiv:1903.08636)](https://arxiv.org/pdf/1903.08636)

Example launch configuration:
```xml
<!-- Enable Schmidt ESKF -->
<param name="solver_type" value="0" />
<param name="enable_schmidt_eskf" value="true" />

<!-- Enable SLAM features -->
<param name="use_slam_feature" value="true" />
<param name="max_slam_feature" value="50" />
```
