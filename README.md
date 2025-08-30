# VIO Backend - Visual Inertial Odometry System

A high-performance Visual Inertial Odometry (VIO) system implemented in C++ with ROS integration, featuring real-time state estimation using camera and IMU data.

## 🚀 Features

- **Real-time VIO**: High-frequency state estimation with camera and IMU fusion
- **Multiple Frontend Options**: Traditional feature-based and neural network-based feature extraction
- **Robust Backend**: MSCKF (Multi-State Constraint Kalman Filter) implementation
- **SFM Integration**: Structure from Motion for initialization and mapping
- **ROS Integration**: Full ROS/ROS2 support with standard message types
- **Multiple Dataset Support**: Euroc MAV, TUM RGB-D, and custom datasets
- **Performance Optimized**: GPU-accelerated neural feature extraction with SuperPoint
- **Comprehensive Testing**: Unit tests and performance benchmarks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [Performance](#performance)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- **ROS Noetic** (Ubuntu 20.04) or **ROS2 Foxy** (Ubuntu 20.04)
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
git clone <repository-url>
cd catkin_ws

# Build the project
catkin_make

# Source the workspace
source devel/setup.bash
```

## 🚀 Quick Start

### 1. Basic Usage with Euroc Dataset

```bash
# Launch VIO with Euroc dataset
roslaunch vio euroc_serial_backend.launch \
    dataset_path:=/path/to/euroc/dataset \
    config_file:=config/euroc_config.yaml
```

### 2. Real-time Processing with RealSense Camera

```bash
# Launch VIO with RealSense camera
roslaunch vio realsense_serial_online.launch
```

### 3. Neural Feature Extraction

```bash
# Launch with SuperPoint neural features
roslaunch vio euroc_serial_backend.launch \
    use_neural_features:=true \
    neural_config:=config/neural_matcher_config.yaml
```


## 🙏 Acknowledgments

- **MSCKF**: Multi-State Constraint Kalman Filter implementation
- **SuperPoint**: Neural feature extraction
- **LightGlue**: Neural feature matching
- **Ceres Solver**: Nonlinear optimization
- **Sophus**: Lie groups for robotics
- **Eigen**: Linear algebra library


**Note**: This is a research-grade VIO system. For production use, additional testing and validation is recommended.
