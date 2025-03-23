cmake_minimum_required(VERSION 3.5)
project(vio_backend)

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

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(glog REQUIRED)

catkin_package(
 LIBRARIES vio_backend
 INCLUDE_DIRS src/core src/states src/types src/utils
 CATKIN_DEPENDS cv_bridge geometry_msgs nav_msgs image_transport nav_msgs rosbag roscpp sensor_msgs std_msgs tf
)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  src/states/
  src/types/
  src/utils/
  src/core/
  /usr/include/glog/
)

list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${Sophus_LIBRARIES}
        fmt
        glog::glog
)

file(GLOB_RECURSE CORE_FILE_LIST "src/core/*.cpp")
message(STATUS "CORE_FILE_LIST = ${CORE_FILE_LIST}")
add_library(
        msckf_core STATIC
        ${CORE_FILE_LIST}
)
target_include_directories(msckf_core PUBLIC
src/states/
src/types/
src/utils/)
target_link_libraries(msckf_core glog::glog)

add_subdirectory(src/core)

add_executable(${PROJECT_NAME}_node src/vio_backend_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_triangulation src/test_triangulation.cpp)
target_link_libraries(test_triangulation
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_givens_rotation src/test_givens_rotation.cpp)
target_link_libraries(test_givens_rotation
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_feature_jacobian src/test_feature_jacobian.cpp)
target_link_libraries(test_feature_jacobian
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

add_executable(test_pnp_ransac src/test_pnp_ransac.cpp)
target_link_libraries(test_pnp_ransac
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries})

add_executable(test_rotation_matrix_to_euler src/test_rotation_matrix_to_euler.cpp)

add_executable(test_show_mat src/test_mat_show.cpp)
target_link_libraries(test_show_mat
                      msckf_core
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)
# target_link_libraries(test_rotation_matrix_to_euler
#                       ${thirdparty_libraries})