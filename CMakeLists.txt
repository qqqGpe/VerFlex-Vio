# =============================================================================
# Project Configuration
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(vio)

# Compiler settings
add_compile_options(-std=c++17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)  # Options: Debug, Release, RelWithDebInfo

# =============================================================================
# Package Dependencies
# =============================================================================

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

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(glog REQUIRED)

# =============================================================================
# Message Generation
# =============================================================================

add_service_files(FILES nnFeatures.srv)
generate_messages(DEPENDENCIES sensor_msgs)

# =============================================================================
# Catkin Package Configuration
# =============================================================================

catkin_package(
    LIBRARIES vio
    INCLUDE_DIRS src/core src/types src/utils
    CATKIN_DEPENDS
        cv_bridge
        geometry_msgs
        nav_msgs
        image_transport
        nav_msgs
        rosbag
        roscpp
        sensor_msgs
        std_msgs
        message_runtime
)

catkin_install_python(PROGRAMS script/nnFeatureSrv.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# =============================================================================
# Include Directories
# =============================================================================

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    src/types/
    src/utils/
    src/core/
    /usr/include/glog/
)

# =============================================================================
# Third-party Libraries
# =============================================================================

list(APPEND thirdparty_libraries
    fmt
    glog::glog
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${CERES_LIBRARIES}
)

# =============================================================================
# Core Library Target
# =============================================================================
add_subdirectory(src/utils)

aux_source_directory(src/core msckf_core_files)
add_library(msckf_core STATIC
    ${msckf_core_files}
)

target_include_directories(msckf_core PUBLIC
    src/types/
    src/utils/
    src/solver
)

target_link_libraries(msckf_core PUBLIC
    utils
    ${thirdparty_libraries}
)
# =============================================================================
# Main Executable Targets
# =============================================================================

add_executable(${PROJECT_NAME}_serial_node src/vio_serial_node.cpp)
target_link_libraries(${PROJECT_NAME}_serial_node
    msckf_core
)

add_executable(${PROJECT_NAME}_subscribe_node src/vio_subscribe_node.cpp)
target_link_libraries(${PROJECT_NAME}_subscribe_node
    msckf_core
)

# =============================================================================
# Test Targets
# =============================================================================

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS AND CATKIN_ENABLE_TESTING)
    add_subdirectory(src/test)
endif()