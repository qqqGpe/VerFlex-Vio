cmake_minimum_required(VERSION 3.0.2)
project(vio_backend)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug) # Debug Releasea

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  nav_msgs
  rosbag
  roscpp
  sensor_msgs
  std_msgs
  tf
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(glog 0.7.0 REQUIRED)

catkin_package(
 LIBRARIES vio_backend
 INCLUDE_DIRS src/core src/states src/types src/utils
 CATKIN_DEPENDS cv_bridge geometry_msgs image_transport nav_msgs rosbag roscpp sensor_msgs std_msgs tf
)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  src/states/
  src/types/
  src/utils/
  src/core/
)

list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        glog::glog
)

file(GLOB_RECURSE CORE_DIR_LIST "src/core/*.cpp")
message(STATUS "CORE_DIR_LIST = ${CORE_DIR_LIST}")
add_library(
        CORE_LIB STATIC
        ${CORE_DIR_LIST}
)
target_include_directories(CORE_LIB PUBLIC
src/states/
src/types/
src/utils/)
target_link_libraries(CORE_LIB glog::glog)

# add_subdirectory(src/states)
# add_subdirectory(src/types)
add_subdirectory(src/core)
# add_subdirectory(src/utils)

# aux_source_directory(src DIR_SRCS)
# message(lib_list)
# add_library(backend_lib lib_list)

add_executable(${PROJECT_NAME}_node src/vio_backend_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
                      CORE_LIB
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_triangulation src/test_triangulation.cpp)
target_link_libraries(test_triangulation
                      CORE_LIB
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_givens_rotation src/test_givens_rotation.cpp)
target_link_libraries(test_givens_rotation
                      CORE_LIB
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_feature_jacobian src/test_feature_jacobian.cpp)
target_link_libraries(test_feature_jacobian
                      CORE_LIB
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_pnp_ransac src/test_pnp_ransac.cpp)
target_link_libraries(test_pnp_ransac
                      CORE_LIB
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)