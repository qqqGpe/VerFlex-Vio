cmake_minimum_required(VERSION 3.0.2)
project(vio_backend)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

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
 INCLUDE_DIRS src/
 CATKIN_DEPENDS cv_bridge geometry_msgs image_transport nav_msgs rosbag roscpp sensor_msgs std_msgs tf
)

add_subdirectory(src/states)
add_subdirectory(src/types)

include_directories(
          src
          ${EIGEN3_INCLUDE_DIR}
          ${Boost_INCLUDE_DIRS}
          ${catkin_INCLUDE_DIRS}
)

list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        glog::glog
)

add_executable(${PROJECT_NAME}_node src/vio_backend_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
                      ${catkin_LIBRARIES}
                      ${thirdparty_libraries}
)

