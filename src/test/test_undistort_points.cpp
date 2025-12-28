/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-12-21 17:07:52
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
 */
#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

namespace {

cv::Mat KFrom(double fx, double fy, double cx, double cy) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    return K;
}

}  // namespace

TEST(UndistortPoints, BackProjectToNormalizedPlaneWith4DistParams) {
    // Intrinsics
    const double fx = 460.0;
    const double fy = 460.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const cv::Mat K = KFrom(fx, fy, cx, cy);

    // Distortion: k1, k2, p1, p2 (no k3)
    const double k1 = -0.12;
    const double k2 = 0.02;
    const double p1 = 0.001;
    const double p2 = -0.0005;
    const cv::Mat D = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);

    // Identity camera pose for projection
    const cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    const cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

    // A few 3D points in front of the camera
    std::vector<cv::Point3d> pts3d = {
        {0.0, 0.0, 3.0},
        {0.2, 0.1, 4.0},
        {-0.3, 0.15, 5.5},
        {0.1, -0.2, 2.5},
        {-0.15, -0.05, 6.0},
    };

    // 1) Project with distortion -> pixel points
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(pts3d, rvec, tvec, K, D, pixels);
    ASSERT_EQ(pixels.size(), pts3d.size());

    // 2) Undistort back -> normalized coordinates (x, y)
    std::vector<cv::Point2d> undist_norm;
    cv::undistortPoints(pixels, undist_norm, K, D, cv::noArray(), cv::noArray());
    ASSERT_EQ(undist_norm.size(), pts3d.size());

    // 3) Compare with ground-truth normalized coords (X/Z, Y/Z)
    const double tol = 1e-6;
    for (size_t i = 0; i < pts3d.size(); ++i) {
        const double x_gt = pts3d[i].x / pts3d[i].z;
        const double y_gt = pts3d[i].y / pts3d[i].z;
        EXPECT_NEAR(undist_norm[i].x, x_gt, tol) << "i=" << i;
        EXPECT_NEAR(undist_norm[i].y, y_gt, tol) << "i=" << i;
    }
}
