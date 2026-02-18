# AGENTS.md - VerFlex-VIO Codebase Guide

> This file provides essential context for AI coding agents working in this repository.

## Project Overview

**VerFlex-VIO** is a Visual-Inertial Odometry system with ROS integration. It supports monocular/stereo cameras, multiple initialization modes (static/dynamic), and multiple estimators (ESKF, Sqrt-ESKF).

## Build Commands

```bash
# Full build (from catkin workspace root)
cd /home/gao/ws/catkin_ws
catkin build vio

# Clean build
rm -rf build devel && catkin build vio

# Source the workspace after build
source devel/setup.bash
```

## Test Commands

```bash
# Run all tests
cd /home/gao/ws/catkin_ws
catkin build vio -DCATKIN_ENABLE_TESTING=1
./devel/lib/vio/test_logger        # Run specific test
./devel/lib/vio/test_triangulation # Run specific test

# Run single test with gtest filter
./devel/lib/vio/test_logger --gtest_filter=LoggerTest.GetInstance_SameFilename_ReturnsSameInstance
```

## Run Commands

```bash
# Launch with Euroc dataset (offline rosbag)
roslaunch vio euroc_serial_backend.launch

# Launch with Euroc dataset (rosbag playback)
roslaunch vio euroc_serial_backend_subscribe.launch

# Launch with Intel Realsense Camera
roslaunch vio realsense_serial_online.launch
```

## Code Style Guidelines

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `VioManager`, `ImuManager`, `VisualManager` |
| Functions/Methods | camelCase | `TryFrontendTrack()`, `SaveResultsToFile()` |
| Member Variables | underscore suffix or prefix | `params_`, `_imu_manager`, `state` |
| Constants | kCamelCase | `kRad2Deg`, `kMinVisualFeaturesForUpdate` |
| Enums | PascalCase | `SolverType::ESKF`, `InitializerType::kStatic` |
| Namespaces | lowercase | `utils`, `utils::math` |

### File Organization

```
src/
├── core/           # Main VIO components (vioManager, initializer, frontend)
├── types/          # State types (Type, Vec, Pose, Quat, ImuState)
├── utils/          # Utilities (logger, mathematical_tools, utils)
├── solver/         # ESKF and Sqrt-ESKF solvers
├── test/           # Unit tests
├── vio_serial_node.cpp      # Main node (offline rosbag)
└── vio_subscribe_node.cpp   # Main node (online subscription)
```

### Include Order

```cpp
// 1. System/external headers (alphabetical)
#include <Eigen/Core>
#include <glog/logging.h>
#include <memory>
#include <vector>

// 2. ROS headers
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

// 3. Project headers (relative paths)
#include "vioManager.h"
#include "parameter.h"
#include "logger.h"
```

### Header Guards

Use `#ifndef __NAME__` / `#define __NAME__` / `#endif` pattern:

```cpp
#ifndef __VIO_MANAGER__
#define __VIO_MANAGER__
// ... content
#endif
```

### Class Structure

```cpp
class ClassName
{
   public:
    ClassName() = default;
    explicit ClassName(const Param& params);
    ~ClassName() {}

    // Public methods
    void PublicMethod();

   private:
    // Private methods
    void PrivateMethod();

    // Member variables
    Param params_;
    std::shared_ptr<State> state;
};
```

### Smart Pointers

Prefer `std::shared_ptr` for shared ownership:

```cpp
std::shared_ptr<State> state = std::make_shared<State>(params);
std::shared_ptr<ImuManager> _imu_manager;
```

### Error Handling

Use `glog` for logging:

```cpp
#include <glog/logging.h>

LOG(INFO) << "Initialization successful";
LOG(WARNING) << "Not enough features: " << count;
LOG(ERROR) << "Failed to load parameters!";
LOG(FATAL) << "Unsupported configuration!";  // Exits program
```

### Constants

Define constants in anonymous namespaces:

```cpp
namespace
{
constexpr double kRad2Deg = 180.0 / M_PI;
constexpr uint32_t kMinVisualFeaturesForUpdate = 10;
}  // namespace
```

### Eigen Types

Use Eigen for matrix/vector operations:

```cpp
Eigen::Vector3d position;      // 3D vector
Eigen::Matrix3d rotation;      // 3x3 matrix
Eigen::Quaterniond quat;       // Quaternion
Eigen::Ref<Eigen::MatrixXd>    // Reference to matrix
```

## Test Patterns

```cpp
// Test file structure (gtest)
#include <gtest/gtest.h>

class MyTest : public ::testing::Test
{
protected:
    void SetUp() override { /* setup code */ }
    void TearDown() override { /* cleanup code */ }
};

TEST_F(MyTest, TestName_ExpectedBehavior)
{
    // Arrange
    int value = 42;

    // Act
    int result = value * 2;

    // Assert
    EXPECT_EQ(result, 84);
}
```

## Key Dependencies

- **ROS Noetic** - Robot Operating System
- **Eigen3** - Linear algebra
- **Ceres Solver** - Nonlinear optimization
- **Sophus** - Lie groups (SO3, SE3)
- **OpenCV 4** - Image processing
- **glog** - Logging
- **gtest** - Unit testing

## Common Patterns

### Singleton Pattern (for utilities)

```cpp
class Logger {
public:
    static Logger* GetInstance(const std::string& filename);
    bool SaveValues(const LogValue& value);
private:
    Logger(const std::string& filename);
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};
```

### Parameter Loading

```cpp
Param params(nh);
params.load_params();  // Loads from ROS parameter server
```

### State Management

```cpp
std::shared_ptr<State> state = std::make_shared<State>(params);
state->_imu_state->p()->vec();  // Position vector
state->_imu_state->q()->Rot();  // Rotation matrix
```

## Important Notes

1. **C++17** is required
2. **Thread safety**: Use `std::mutex` for shared resources
3. **ROS parameters**: All config loaded from launch files
4. **Coordinate frames**: IMU frame is the body frame
5. **Timestamps**: All timestamps in seconds (double)
