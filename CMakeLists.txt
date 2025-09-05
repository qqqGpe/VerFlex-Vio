cmake_minimum_required(VERSION 3.15)
project(vio)

add_compile_options(-std=c++17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_BUILD_TYPE RelWithDebInfo) # Debug Release
set(CMAKE_BUILD_TYPE Debug) # Debug Release
# set(CMAKE_BUILD_TYPE Release) # Debug Release
# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall")


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
find_package(GTest REQUIRED)

add_service_files(FILES nnFeatures.srv)
generate_messages(DEPENDENCIES sensor_msgs)

catkin_package(
 LIBRARIES vio
 INCLUDE_DIRS src/core src/types src/utils
 CATKIN_DEPENDS cv_bridge geometry_msgs nav_msgs image_transport nav_msgs rosbag roscpp sensor_msgs std_msgs message_runtime
)

catkin_install_python(PROGRAMS script/nnFeatureSrv.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

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

list(APPEND thirdparty_libraries
        fmt
        glog::glog
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${Sophus_LIBRARIES}
        ${CERES_LIBRARIES}
)

file(GLOB_RECURSE CORE_FILE_LIST "src/core/*.cpp")
message(STATUS "CORE_FILE_LIST = ${CORE_FILE_LIST}")
add_library(
        msckf_core STATIC
        ${CORE_FILE_LIST}
)

target_include_directories(msckf_core PUBLIC
src/types/
src/utils/
src/solver)
target_link_libraries(msckf_core glog::glog)

add_subdirectory(src/core)

add_executable(${PROJECT_NAME}_serial_node src/vio_serial_node.cpp)
target_link_libraries(${PROJECT_NAME}_serial_node
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(${PROJECT_NAME}_subscribe_node src/vio_subscribe_node.cpp)
target_link_libraries(${PROJECT_NAME}_subscribe_node
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_triangulation src/test/test_triangulation.cpp)
target_include_directories(test_triangulation PRIVATE
    src/types
    src/core
    src/utils
)
target_link_libraries(test_triangulation
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
                      GTest::GTest
)

add_executable(test_givens_rotation src/test/test_givens_rotation.cpp)
target_include_directories(test_givens_rotation PRIVATE
    src/types
    src/core
    src/utils
)
target_link_libraries(test_givens_rotation
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
                      GTest::GTest
)

add_executable(test_pnp_ransac src/test/test_pnp_ransac.cpp)
target_include_directories(test_pnp_ransac PRIVATE
    src/types
    src/core
    src/utils
)
target_link_libraries(test_pnp_ransac
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
                      GTest::GTest
  )

add_executable(test_rotation_matrix_to_euler src/test/test_rotation_matrix_to_euler.cpp)
target_include_directories(test_rotation_matrix_to_euler PRIVATE
    src/types
    src/core
    src/utils
)
target_link_libraries(test_rotation_matrix_to_euler
                      msckf_core
                      ${thirdparty_libraries}
                      GTest::GTest)

add_executable(test_sfm src/test/test_sfm.cpp)
target_include_directories(test_sfm PRIVATE
    src/types
    src/core
    src/utils
)
target_link_libraries(test_sfm
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
                      GTest::GTest)

add_executable(test_bch src/test/test_bch.cpp)
target_include_directories(test_bch PRIVATE
    src/types
    src/core
    src/utils
)
target_link_libraries(test_bch
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
                      GTest::GTest)