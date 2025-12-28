# =============================================================================
# Project Configuration
# =============================================================================

cmake_minimum_required(VERSION 3.15)
project(vio)

# Compiler settings
add_compile_options(-std=c++17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)  # Options: Debug, Release, RelWithDebInfo


# Enable AddressSanitizer for memory leak detection
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
#     add_link_options(-fsanitize=address)
# endif()

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
    ${GTEST_INCLUDE_DIRS}
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

if(CATKIN_ENABLE_TESTING)

    # Unit test for triangulation
    catkin_add_gtest(test_triangulation src/test/test_triangulation.cpp)
    if (TARGET test_triangulation)
        target_include_directories(test_triangulation PRIVATE
            src/types
            src/core
            src/utils
        )
        target_link_libraries(test_triangulation
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for Givens rotation
    catkin_add_gtest(test_givens_rotation src/test/test_givens_rotation.cpp)
    if (TARGET test_givens_rotation)
        target_include_directories(test_givens_rotation PRIVATE
            src/types
            src/core
            src/utils
        )
        target_link_libraries(test_givens_rotation
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for PnP RANSAC
    catkin_add_gtest(test_pnp_ransac src/test/test_pnp_ransac.cpp)
    if (TARGET test_pnp_ransac)
        target_include_directories(test_pnp_ransac PRIVATE
            src/types
            src/core
            src/utils
        )
        target_link_libraries(test_pnp_ransac
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for rotation matrix to euler angles
    catkin_add_gtest(test_rotation_matrix_to_euler src/test/test_rotation_matrix_to_euler.cpp)
    if (TARGET test_rotation_matrix_to_euler)
        target_include_directories(test_rotation_matrix_to_euler PRIVATE
            src/types
            src/core
            src/utils
        )
        target_link_libraries(test_rotation_matrix_to_euler
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for structure from motion
    catkin_add_gtest(test_sfm src/test/test_sfm.cpp)
    if (TARGET test_sfm)
        target_include_directories(test_sfm PRIVATE
            src/types
            src/core
            src/utils
        )
        target_link_libraries(test_sfm
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for BCH code
    catkin_add_gtest(test_bch src/test/test_bch.cpp)
    if (TARGET test_bch)
        target_include_directories(test_bch PRIVATE
            src/utils
        )
        target_link_libraries(test_bch
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for undistortPoints back-projection
    catkin_add_gtest(test_undistort_points src/test/test_undistort_points.cpp)
    if (TARGET test_undistort_points)
        target_link_libraries(test_undistort_points
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()
endif()