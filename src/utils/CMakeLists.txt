aux_source_directory(./ utils_files)

add_library(utils STATIC
    ${utils_files}
)

target_include_directories(utils PUBLIC
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(utils
    glog::glog
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Eigen3_LIBRARIES}
)
