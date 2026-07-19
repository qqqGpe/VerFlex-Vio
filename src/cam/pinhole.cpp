#include "pinhole.h"

void Pinhole::initImpl(const Parameter& params)
{
    initCommon(params);
    // Pinhole + Brown-Conrady: keep raw K/D. No rectify map is generated, so
    // distortion is handled analytically per feature in project_distort /
    // compute_distort_jacobian / back_project_undistort.
    for (int i = 0; i < camera_num_; i++)
    {
        vK_.push_back(params.intrinsics[i]);
        vD_.push_back(params.distortion[i]);
    }
}

void Pinhole::compute_distort_jacobian(const int& cam_id, const Eigen::Vector2d& uv_norm, Eigen::MatrixXd& H_dz_dzn)
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

Eigen::Vector2d Pinhole::project_distort(const uint32_t cam_id, const Eigen::Vector3d& p3d) const
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

void Pinhole::back_project_undistort(CameraObs& obs) const
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
