# =============================================================================
# Test Targets
# =============================================================================

if(CATKIN_ENABLE_TESTING)

    # Unit test for triangulation
    catkin_add_gtest(test_triangulation test_triangulation.cpp)
    if (TARGET test_triangulation)
        target_include_directories(test_triangulation PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../types
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        )
        target_link_libraries(test_triangulation
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for Givens rotation
    catkin_add_gtest(test_givens_rotation test_givens_rotation.cpp)
    if (TARGET test_givens_rotation)
        target_include_directories(test_givens_rotation PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../types
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        )
        target_link_libraries(test_givens_rotation
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for PnP RANSAC
    catkin_add_gtest(test_pnp_ransac test_pnp_ransac.cpp)
    if (TARGET test_pnp_ransac)
        target_include_directories(test_pnp_ransac PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../types
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        )
        target_link_libraries(test_pnp_ransac
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for rotation matrix to euler angles
    catkin_add_gtest(test_rotation_matrix_to_euler test_rotation_matrix_to_euler.cpp)
    if (TARGET test_rotation_matrix_to_euler)
        target_include_directories(test_rotation_matrix_to_euler PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../types
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        )
        target_link_libraries(test_rotation_matrix_to_euler
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for structure from motion
    catkin_add_gtest(test_sfm test_sfm.cpp)
    if (TARGET test_sfm)
        target_include_directories(test_sfm PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../types
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        )
        target_link_libraries(test_sfm
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for BCH code
    catkin_add_gtest(test_bch test_bch.cpp)
    if (TARGET test_bch)
        target_include_directories(test_bch PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        )
        target_link_libraries(test_bch
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for undistortPoints back-projection
    catkin_add_gtest(test_undistort_points test_undistort_points.cpp)
    if (TARGET test_undistort_points)
        target_link_libraries(test_undistort_points
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for Logger singleton (TDD RED Phase)
    catkin_add_gtest(test_logger test_logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cpp
    )
    if (TARGET test_logger)
        target_include_directories(test_logger PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        )
        target_link_libraries(test_logger
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
            pthread  # Required for std::thread
        )
    endif()

    # Unit test for sequential SLAM feature update
    catkin_add_gtest(test_slam_sequential_update test_slam_sequential_update.cpp)
    if (TARGET test_slam_sequential_update)
        target_include_directories(test_slam_sequential_update PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../types
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
            ${CMAKE_CURRENT_SOURCE_DIR}/../solver
        )
        target_link_libraries(test_slam_sequential_update
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

    # Unit test for sqrt-ESKF SLAM feature lifecycle
    catkin_add_gtest(test_sqrt_eskf_slam test_sqrt_eskf_slam.cpp)
    if (TARGET test_sqrt_eskf_slam)
        target_include_directories(test_sqrt_eskf_slam PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../types
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../utils
            ${CMAKE_CURRENT_SOURCE_DIR}/../solver
        )
        target_link_libraries(test_sqrt_eskf_slam
            msckf_core
            ${catkin_LIBRARIES}
            ${thirdparty_libraries}
            ${GTEST_LIBRARIES}
        )
    endif()

endif()
