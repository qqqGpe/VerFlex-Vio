/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-11-07 01:40:59
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
 */
#include "camModel.h"

void CamModel::Init(const Parameter params)
{
    camera_num_ = params.camera_num;
    image_size_ = cv::Size(params.img_width, params.img_height);

    for (int i = 0; i < camera_num_; i++)
    {
        vRic_.push_back(params.Ric[i]);
        vTic_.push_back(params.tic[i]);
        vD_.push_back(params.distortion[i]);
        cv::Mat map1, map2;
        Eigen::Matrix3d K_undistort = params.intrinsics[i];
        rectify_map_.push_back(map1);
        rectify_map_.push_back(map2);
        vK_.push_back(K_undistort);
    }
}

// For debug
void CamModel::SetCameraIntrinsicMatrix(const std::vector<double>& intrinsic_coeff)
{
    vK_.clear();
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = intrinsic_coeff[0];
    K(1, 1) = intrinsic_coeff[1];
    K(0, 2) = intrinsic_coeff[2];
    K(1, 2) = intrinsic_coeff[3];
    vK_.push_back(K);
}

void CamModel::compute_distort_jacobian(const int& cam_id, const Eigen::Vector2d& uv_norm, Eigen::MatrixXd& H_dz_dzn)
{
    double fx = vK_[cam_id](0, 0);
    double fy = vK_[cam_id](1, 1);
    double cx = vK_[cam_id](0, 2);
    double cy = vK_[cam_id](1, 2);
    double k1 = vD_[cam_id](0);
    double k2 = vD_[cam_id](1);
    double p1 = vD_[cam_id](2);
    double p2 = vD_[cam_id](3);

    // Get our camera parameters
    Eigen::VectorXd cam_d(8);
    cam_d << fx, fy, cx, cy, k1, k2, p1, p2;

    // Calculate distorted coordinates for radial
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double r_2 = r * r;
    double r_4 = r_2 * r_2;

    // Jacobian of distorted pixel to normalized pixel
    H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
    double x = uv_norm(0);
    double y = uv_norm(1);
    double x_2 = uv_norm(0) * uv_norm(0);
    double y_2 = uv_norm(1) * uv_norm(1);
    double x_y = uv_norm(0) * uv_norm(1);
    H_dz_dzn(0, 0) = cam_d(0) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * x_2 + 4 * cam_d(5) * x_2 * r_2) + 2 * cam_d(6) * y +
                                 (2 * cam_d(7) * x + 4 * cam_d(7) * x));
    H_dz_dzn(0, 1) = cam_d(0) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
    H_dz_dzn(1, 0) = cam_d(1) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
    H_dz_dzn(1, 1) = cam_d(1) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * y_2 + 4 * cam_d(5) * y_2 * r_2) + 2 * cam_d(7) * x +
                                 (2 * cam_d(6) * y + 4 * cam_d(6) * y));
}

Eigen::Vector2d CamModel::project_distort(const uint32_t cam_id, const Eigen::Vector3d& p3d) const
{
    double fx = vK_[cam_id](0, 0);
    double fy = vK_[cam_id](1, 1);
    double cx = vK_[cam_id](0, 2);
    double cy = vK_[cam_id](1, 2);
    double k1 = vD_[cam_id](0);
    double k2 = vD_[cam_id](1);
    double p1 = vD_[cam_id](2);
    double p2 = vD_[cam_id](3);
    Eigen::VectorXd cam_d(8);
    cam_d << fx, fy, cx, cy, k1, k2, p1, p2;

    Eigen::Vector2d uv_norm;
    uv_norm(0) = p3d(0) / p3d(2);
    uv_norm(1) = p3d(1) / p3d(2);

    // Calculate distorted coordinates for radial
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double r_2 = r * r;
    double r_4 = r_2 * r_2;
    double x1 = uv_norm(0) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * uv_norm(0) * uv_norm(1) +
                cam_d(7) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
    double y1 = uv_norm(1) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
                2 * cam_d(7) * uv_norm(0) * uv_norm(1);

    // Return the distorted point
    Eigen::Vector2d uv_dist;
    uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
    uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
    return uv_dist;
}

void CamModel::back_project_undistort(CameraObs& obs) const
{
    Eigen::Vector2d uv_distort;
    for (int cam_id = 0; cam_id < camera_num_; cam_id++)
    {
        Eigen::Vector2d uv = obs.uv.at(cam_id);
        // Use OpenCV to undistort the normalized coordinates
        std::vector<cv::Point2f> src_pts(1);
        std::vector<cv::Point2f> dst_pts(1);
        src_pts[0] = cv::Point2f(uv(0), uv(1));
        cv::Mat K_cv, D_cv;
        cv::eigen2cv(vK_[cam_id], K_cv);
        cv::eigen2cv(vD_[cam_id], D_cv);
        cv::undistortPoints(src_pts, dst_pts, K_cv, D_cv);
        uv_distort(0) = dst_pts[0].x;
        uv_distort(1) = dst_pts[0].y;
        obs.uv_norm[cam_id] = uv_distort;
    }
}

void CamModel::RectifyImage(const int32_t cam_id, const cv::Mat& img_raw_ptr, cv::Mat* img_rectified_ptr)
{
    // Skip rectification if maps are not initialized
    if (rectify_map_.size() <= cam_id * 2 + 1 ||
        rectify_map_[cam_id * 2].empty() || rectify_map_[cam_id * 2 + 1].empty())
    {
        *img_rectified_ptr = img_raw_ptr.clone();
        return;
    }
    cv::remap(img_raw_ptr, *img_rectified_ptr, rectify_map_[cam_id * 2], rectify_map_[cam_id * 2 + 1], cv::INTER_LINEAR);
}
