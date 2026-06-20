# =============================================================================
# Project Configuration
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(vio)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo) # Release Debug RelWithDebInfo
endif()

# Prefer system libraries over conda to avoid conflicts
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(PREPEND CMAKE_PREFIX_PATH "/usr/local" "/usr")

# =============================================================================
# Options
# =============================================================================

option(BUILD_WITH_ROS "Build with ROS support (requires catkin)" OFF)
option(BUILD_TESTS "Build unit tests" ON)

# Auto-detect catkin environment (only if BUILD_WITH_ROS not explicitly set)
if(NOT DEFINED BUILD_WITH_ROS_EXPLICIT)
    if(DEFINED CATKIN_DEVEL_PREFIX)
        set(BUILD_WITH_ROS ON)
        message(STATUS "Catkin environment detected, enabling ROS support")
    endif()
endif()

# =============================================================================
# Core Dependencies (always required)
# =============================================================================

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(glog REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CURL QUIET)

# =============================================================================
# Core Library (ROS-independent)
# =============================================================================

# Utils library
aux_source_directory(src/utils utils_files)
add_library(vio_utils STATIC ${utils_files})
target_include_directories(vio_utils PUBLIC
    src/utils/
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vio_utils PUBLIC
    glog::glog
    ${OpenCV_LIBRARIES}
)

# Core algorithm library
aux_source_directory(src/core vio_core_files)
add_library(vio_core STATIC ${vio_core_files})
target_include_directories(vio_core PUBLIC
    src/types/
    src/utils/
    src/core/
    src/solver/
    ${EIGEN3_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
target_link_libraries(vio_core PUBLIC
    vio_utils
    fmt::fmt
    glog::glog
    yaml-cpp::yaml-cpp
    ${OpenCV_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${CERES_LIBRARIES}
    $<$<BOOL:${CURL_FOUND}>:CURL::libcurl>
)

# =============================================================================
# Pure C++ Offline Executable
# =============================================================================

add_executable(vio_offline src/app/vio_offline_entry.cpp)
target_include_directories(vio_offline PRIVATE src/dataloader/)
target_link_libraries(vio_offline PRIVATE vio_core fmt::fmt)

# =============================================================================
# ROS Support (optional)
# =============================================================================

if(BUILD_WITH_ROS)
    find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        geometry_msgs
        image_transport
        nav_msgs
        rosbag
        roscpp
        sensor_msgs
        std_msgs
        tf2
        tf2_ros
        tf2_eigen
        tf2_geometry_msgs
        message_filters
    )

    catkin_package(
        LIBRARIES vio_core
        INCLUDE_DIRS src/core src/types src/utils
        CATKIN_DEPENDS
            cv_bridge
            geometry_msgs
            nav_msgs
            image_transport
            rosbag
            roscpp
            sensor_msgs
            std_msgs
    )

    # ROS adapter library
    add_library(vio_ros STATIC
        src/ros/ros_visualizer.cpp
    )
    target_include_directories(vio_ros PUBLIC
        src/ros/
        src/core/
        src/types/
        src/utils/
        src/solver/
        ${catkin_INCLUDE_DIRS}
    )
    target_link_libraries(vio_ros PUBLIC
        vio_core
        ${catkin_LIBRARIES}
    )

    # ROS serial node (offline rosbag)
    add_executable(vio_serial_node src/ros/ros_node_serial.cpp)
    target_include_directories(vio_serial_node PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        src/ros/
    )
    target_link_libraries(vio_serial_node PRIVATE vio_ros)

    # ROS subscribe node (online)
    add_executable(vio_subscribe_node src/ros/ros_node_subscribe.cpp)
    target_include_directories(vio_subscribe_node PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        src/ros/
    )
    target_link_libraries(vio_subscribe_node PRIVATE vio_ros)

    message(STATUS "ROS support enabled: building vio_serial_node, vio_subscribe_node")
else()
    message(STATUS "ROS support disabled: building vio_offline only")
endif()

# =============================================================================
# Tests
# =============================================================================

if(BUILD_TESTS)
    if(BUILD_WITH_ROS AND CATKIN_ENABLE_TESTING)
        add_subdirectory(src/test)
    elseif(NOT BUILD_WITH_ROS)
        enable_testing()
        find_package(GTest REQUIRED)
        add_subdirectory(src/test)
    endif()
endif()
