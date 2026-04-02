---
name: vio-backend
description: VerFlex-VIO - ROS Visual-Inertial Odometry system with C++17, Eigen, Ceres, Sophus. Build with catkin, test with gtest.
license: MIT
compatibility: opencode
metadata:
  audience: developers
  language: c++
  framework: ros-noetic
---

## Project Overview
- **VerFlex-VIO**: Visual-Inertial Odometry system with ROS integration
- **Language**: C++17
- **Build System**: catkin (ROS)
- **Key Dependencies**: Eigen3, Ceres Solver, Sophus, OpenCV 4, glog, gtest

## Build Commands
```bash
cd /home/gao/ws/catkin_ws

# Full build (recommended)
./src/vio_backend/script/build_vio_backend.sh

# Or using catkin directly
catkin build vio

# Clean build
rm -rf build devel && catkin build vio

# Source workspace after build
source devel/setup.bash

# Build with tests enabled
catkin build vio -DCATKIN_ENABLE_TESTING=1
```

## Test Commands
```bash
cd /home/gao/ws/catkin_ws

# Run all test executables (in devel/lib/vio/)
./devel/lib/vio/test_triangulation
./devel/lib/vio/test_givens_rotation
./devel/lib/vio/test_pnp_ransac
./devel/lib/vio/test_rotation_matrix_to_euler
./devel/lib/vio/test_sfm
./devel/lib/vio/test_bch
./devel/lib/vio/test_undistort_points
./devel/lib/vio/test_logger
./devel/lib/vio/test_slam_sequential_update

# Run single test with gtest filter
./devel/lib/vio/test_logger --gtest_filter=LoggerTest.*
```

## Code Style Guidelines

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `VioManager`, `ImuManager` |
| Functions/Methods | camelCase | `TryFrontendTrack()` |
| Member Variables | `_suffix` | `_imu_manager`, `params_` |
| Constants | kCamelCase | `kRad2Deg` |
| Enums | PascalCase | `SolverType::ESKF` |
| Namespaces | lowercase | `utils`, `utils::math` |

### Include Order
```cpp
// 1. System/external headers (alphabetical)
#include <Eigen/Core>
#include <glog/logging.h>
#include <memory>

// 2. ROS headers
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

// 3. Project headers (relative paths)
#include "vioManager.h"
#include "parameter.h"
```

### Formatting
- **clang-format** configured (`.clang-format` exists)
- Column limit: 120 characters
- Indent width: 4 spaces
- C++11 braced list style enabled
- Run `clang-format -i <file.cpp>` before committing

### Class Structure
```cpp
class ClassName
{
public:
    ClassName() = default;
    explicit ClassName(const Param& params);
    ~ClassName() {}

    void PublicMethod();

private:
    void PrivateMethod();
    Param params_;
    std::shared_ptr<State> state_;
};
```

### Error Handling (glog)
```cpp
LOG(INFO) << "Initialization successful";
LOG(WARNING) << "Not enough features: " << count;
LOG(ERROR) << "Failed to load parameters!";
LOG(FATAL) << "Unsupported configuration!";  // Exits program
```

### Eigen Types
```cpp
Eigen::Vector3d position;
Eigen::Matrix3d rotation;
Eigen::Quaterniond quat;
```

## Test Patterns (gtest)
```cpp
#include <gtest/gtest.h>

class MyTest : public ::testing::Test
{
protected:
    void SetUp() override { /* setup */ }
    void TearDown() override { /* cleanup */ }
};

TEST_F(MyTest, TestName_ExpectedBehavior)
{
    EXPECT_EQ(actual, expected);
}
```

## Run Commands
```bash
# Offline rosbag processing
roslaunch vio euroc_serial_backend.launch

# Online subscription (rosbag playback)
roslaunch vio euroc_serial_backend_subscribe.launch

# RealSense camera (online)
roslaunch vio realsense_serial_online.launch
```

## Important Notes
1. **C++17** required
2. **Thread safety**: Use `std::mutex` for shared resources
3. **ROS parameters**: Config loaded from launch files
4. **Coordinate frames**: IMU is the body frame
5. **Timestamps**: All in seconds (double)
