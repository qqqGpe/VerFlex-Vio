#include "camModel.h"

void CamModel::Init(const Param params)
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
        CalculateUndistortRectifyMap(params.intrinsics[i], params.distortion[i], K_undistort, map1, map2);
        rectify_map_.push_back(map1);
        rectify_map_.push_back(map2);
        vK_.push_back(K_undistort);
    }
}

void CamModel::CalculateUndistortRectifyMap(const Eigen::Matrix3d& K,
                                               const Eigen::VectorXd& D,
                                               Eigen::Matrix3d& K_undistort,
                                               cv::Mat& map1,
                                               cv::Mat& map2)
{
    cv::Mat K_cv, D_cv;
    cv::eigen2cv(K, K_cv);
    cv::eigen2cv(D, D_cv);

    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat new_K = cv::getOptimalNewCameraMatrix(K_cv, D_cv, image_size_, 0, image_size_, 0);
    cv::initUndistortRectifyMap(K_cv, D_cv, R, new_K, image_size_, CV_32FC1, map1, map2);
    cv::cv2eigen(new_K, K_undistort);
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

Eigen::Vector2d CamModel::project(const uint32_t cam_id, const Eigen::Vector3d& p3d_norm) const
{
    Eigen::Vector2d uv;
    uv(0) = vK_[cam_id](0, 0) * (p3d_norm(0) / p3d_norm(2)) + vK_[cam_id](0, 2);
    uv(1) = vK_[cam_id](1, 1) * (p3d_norm(1) / p3d_norm(2)) + vK_[cam_id](1, 2);
    return uv;
}

void CamModel::back_project(CameraObs& obs) const
{
    Eigen::Vector2d uv_norm;
    for (int cam_id = 0; cam_id < camera_num_; cam_id++)
    {
        Eigen::Vector2d uv = obs.uv.at(cam_id);
        uv_norm(0) = (uv(0) - vK_[cam_id](0, 2)) / vK_[cam_id](0, 0);
        uv_norm(1) = (uv(1) - vK_[cam_id](1, 2)) / vK_[cam_id](1, 1);
        obs.uv_norm[cam_id] = uv_norm;
    }
}

void CamModel::RectifyImage(const int32_t cam_id, const cv::Mat& img_raw_ptr, cv::Mat* img_rectified_ptr)
{
    // Apply rectification
    cv::remap(img_raw_ptr, *img_rectified_ptr, rectify_map_[cam_id * 2], rectify_map_[cam_id * 2 + 1], cv::INTER_LINEAR);
}
