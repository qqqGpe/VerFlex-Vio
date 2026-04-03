# =============================================================================
# Test Targets
# =============================================================================

if(BUILD_WITH_ROS AND CATKIN_ENABLE_TESTING)
    # ROS/catkin test mode
    catkin_add_gtest(test_triangulation test_triangulation.cpp)
    if (TARGET test_triangulation)
        target_link_libraries(test_triangulation vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_givens_rotation test_givens_rotation.cpp)
    if (TARGET test_givens_rotation)
        target_link_libraries(test_givens_rotation vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_pnp_ransac test_pnp_ransac.cpp)
    if (TARGET test_pnp_ransac)
        target_link_libraries(test_pnp_ransac vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_rotation_matrix_to_euler test_rotation_matrix_to_euler.cpp)
    if (TARGET test_rotation_matrix_to_euler)
        target_link_libraries(test_rotation_matrix_to_euler vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_sfm test_sfm.cpp)
    if (TARGET test_sfm)
        target_link_libraries(test_sfm vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_bch test_bch.cpp)
    if (TARGET test_bch)
        target_link_libraries(test_bch vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_undistort_points test_undistort_points.cpp)
    if (TARGET test_undistort_points)
        target_link_libraries(test_undistort_points vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_logger test_logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cpp
    )
    if (TARGET test_logger)
        target_include_directories(test_logger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
        target_link_libraries(test_logger vio_core ${GTEST_LIBRARIES} pthread)
    endif()

    catkin_add_gtest(test_slam_sequential_update test_slam_sequential_update.cpp)
    if (TARGET test_slam_sequential_update)
        target_link_libraries(test_slam_sequential_update vio_core ${GTEST_LIBRARIES})
    endif()

    catkin_add_gtest(test_sqrt_eskf_slam test_sqrt_eskf_slam.cpp)
    if (TARGET test_sqrt_eskf_slam)
        target_link_libraries(test_sqrt_eskf_slam vio_core ${GTEST_LIBRARIES})
    endif()

else()
    # Pure CMake/GTest mode (no ROS)
    macro(add_vio_test test_name test_src)
        add_executable(${test_name} ${test_src} ${ARGN})
        target_link_libraries(${test_name} PRIVATE vio_core GTest::GTest GTest::Main pthread)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endmacro()

    add_vio_test(test_triangulation test_triangulation.cpp)
    add_vio_test(test_givens_rotation test_givens_rotation.cpp)
    add_vio_test(test_pnp_ransac test_pnp_ransac.cpp)
    add_vio_test(test_rotation_matrix_to_euler test_rotation_matrix_to_euler.cpp)
    add_vio_test(test_sfm test_sfm.cpp)
    add_vio_test(test_bch test_bch.cpp)
    add_vio_test(test_undistort_points test_undistort_points.cpp)
    add_vio_test(test_logger test_logger.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cpp)
    add_vio_test(test_slam_sequential_update test_slam_sequential_update.cpp)
    add_vio_test(test_sqrt_eskf_slam test_sqrt_eskf_slam.cpp)
endif()
