#include "fisheye.h"

void Fisheye::initImpl(const Parameter& params)
{
    initCommon(params);
    // Fisheye (Kannala-Brandt equidistant, same model as cv::fisheye): keep the
    // raw fisheye intrinsics and [k1,k2,k3,k4]. No pre-rectification — the KB
    // model projects directly on the raw fisheye image. (fisheye_balance unused.)
    for (int i = 0; i < camera_num_; i++)
    {
        vK_.push_back(params.intrinsics[i]);
        vD_.push_back(params.distortion[i]);
    }
}

Eigen::Vector2d Fisheye::project_distort(const uint32_t cam_id, const Eigen::Vector3d& p3d) const
{
    double fx = vK_[cam_id](0, 0);
    double fy = vK_[cam_id](1, 1);
    double cx = vK_[cam_id](0, 2);
    double cy = vK_[cam_id](1, 2);
    double k1 = vD_[cam_id](0);
    double k2 = vD_[cam_id](1);
    double k3 = vD_[cam_id](2);
    double k4 = vD_[cam_id](3);

    // Normalized coordinates of the 3D feature.
    Eigen::Vector2d uv_norm;
    uv_norm(0) = p3d(0) / p3d(2);
    uv_norm(1) = p3d(1) / p3d(2);

    // KB forward distortion (Kannala-Brandt / cv::fisheye), following OpenVINS CamEqui.
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double theta = std::atan(r);
    double theta_d = theta + k1 * std::pow(theta, 3) + k2 * std::pow(theta, 5) +
                     k3 * std::pow(theta, 7) + k4 * std::pow(theta, 9);

    // Handle when r is small (point near the camera principal point).
    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

    double x1 = uv_norm(0) * cdist;
    double y1 = uv_norm(1) * cdist;

    Eigen::Vector2d uv_dist;
    uv_dist(0) = fx * x1 + cx;
    uv_dist(1) = fy * y1 + cy;
    return uv_dist;
}

void Fisheye::back_project_undistort(CameraObs& obs) const
{
    // cv::fisheye::undistortPoints, like cv::undistortPoints, returns normalized
    // (intrinsics-removed) undistorted coordinates when R/P are omitted — matching
    // the pinhole path and the semantics of obs.uv_norm. (Same as OpenVINS
    // CamEqui::undistort_f.) Verified by the numerical self-check in the test step.
    Eigen::Vector2d uv_norm_pt;
    for (int cam_id = 0; cam_id < camera_num_; cam_id++)
    {
        Eigen::Vector2d uv = obs.uv.at(cam_id);
        std::vector<cv::Point2f> src_pts(1);
        std::vector<cv::Point2f> dst_pts(1);
        src_pts[0] = cv::Point2f(uv(0), uv(1));
        cv::Mat K_cv, D_cv;
        cv::eigen2cv(vK_[cam_id], K_cv);
        cv::eigen2cv(vD_[cam_id], D_cv);
        cv::fisheye::undistortPoints(src_pts, dst_pts, K_cv, D_cv);
        uv_norm_pt(0) = dst_pts[0].x;
        uv_norm_pt(1) = dst_pts[0].y;
        obs.uv_norm[cam_id] = uv_norm_pt;
    }
}

void Fisheye::compute_distort_jacobian(const int& cam_id, const Eigen::Vector2d& uv_norm, Eigen::MatrixXd& H_dz_dzn)
{
    // Jacobian d(distorted pixel)/d(normalized coord) for the KB model.
    // Structure follows OpenVINS CamEqui::compute_distort_jacobian (chain-rule
    // decomposition); the H_dz_dzeta (intrinsics) term is omitted since this
    // project does not calibrate intrinsics online.
    double fx = vK_[cam_id](0, 0);
    double fy = vK_[cam_id](1, 1);
    double k1 = vD_[cam_id](0);
    double k2 = vD_[cam_id](1);
    double k3 = vD_[cam_id](2);
    double k4 = vD_[cam_id](3);

    // KB distortion (shared with project_distort).
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double theta = std::atan(r);
    double theta_d = theta + k1 * std::pow(theta, 3) + k2 * std::pow(theta, 5) +
                     k3 * std::pow(theta, 7) + k4 * std::pow(theta, 9);

    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;

    // Jacobian of distorted pixel wrt distorted "normalized" pixel xy
    Eigen::Matrix2d duv_dxy = Eigen::Matrix2d::Zero();
    duv_dxy << fx, 0, 0, fy;

    // Jacobian of "normalized" pixel wrt normalized pixel (direct term)
    Eigen::Matrix2d dxy_dxyn = Eigen::Matrix2d::Zero();
    dxy_dxyn << theta_d * inv_r, 0, 0, theta_d * inv_r;

    // Jacobian of "normalized" pixel wrt r
    Eigen::Vector2d dxy_dr;
    dxy_dr << -uv_norm(0) * theta_d * inv_r * inv_r, -uv_norm(1) * theta_d * inv_r * inv_r;

    // Jacobian of r wrt normalized pixel
    Eigen::Matrix<double, 1, 2> dr_dxyn;
    dr_dxyn << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

    // Jacobian of "normalized" pixel wrt theta_d
    Eigen::Vector2d dxy_dthd;
    dxy_dthd << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

    // Jacobian of theta_d wrt theta
    double dthd_dth = 1 + 3 * k1 * std::pow(theta, 2) + 5 * k2 * std::pow(theta, 4) +
                      7 * k3 * std::pow(theta, 6) + 9 * k4 * std::pow(theta, 8);

    // Jacobian of theta wrt r
    double dth_dr = 1.0 / (r * r + 1.0);

    // Total Jacobian wrt normalized pixel coordinates
    H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
    H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);
}
