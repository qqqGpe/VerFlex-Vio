#ifndef __CAM_MODEL__
#define __CAM_MODEL__

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "parameter.h"
#include "sensor_data.h"

class CameraModel
{
   public:
    CameraModel() = default;
    ~CameraModel() {}

    CameraModel(const Param params)
    {
        camera_num_ = params.camera_num;
        image_size_ = cv::Size(params.img_width, params.img_height);

        Ric_0_ = params.Ric[0];
        tic_0_ = params.tic[0];
        Ric_1_ = params.Ric[1];
        tic_1_ = params.tic[1];

        Kl_origin_ = params.intrinsics[0];
        Kr_origin_ = params.intrinsics[1];
        D_l_ = params.distortion[0];
        D_r_ = params.distortion[1];

        R_rl_ = Ric_0_.transpose() * Ric_1_;
        p_rl_ = Ric_0_.transpose() * (tic_1_ - tic_0_);
        baseline_ = p_rl_.norm();

        CalculateUndistortRectifyMap(Kl_origin_, D_l_, Kl_undistort_, rectify_map1_left, rectify_map2_left);
        CalculateUndistortRectifyMap(Kr_origin_, D_r_, Kr_undistort_, rectify_map1_right, rectify_map2_right);
    }

    void CalculateUndistortRectifyMap(const Eigen::Matrix3d& K, const Eigen::VectorXd& D, Eigen::Matrix3d& K_undistort, cv::Mat& map1, cv::Mat& map2);

    // For debug
    void SetCameraIntrinsicMatrix(const std::vector<double>& intrinsic_coeff);

    // For debug
    void set_camera_distort_coeff(const std::vector<double>& distort_coeff);

    Eigen::Vector2d project_left(Eigen::Vector3d p3d_norm);

    Eigen::Vector2d project_right(Eigen::Vector3d p3d_norm);

    Eigen::Vector3d back_project(Eigen::Vector2d uv_2d);

    void back_project_stereo(CameraObs& obs);

    void RectifyStereoImages(const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& rectified_left, cv::Mat& rectified_right);

    double baseline() const { return baseline_; }

    Eigen::Matrix3d Ric_l() const { return Ric_0_; }

    Eigen::Vector3d tic_l() const { return tic_0_; }

    Eigen::Matrix3d Ric_r() const { return Ric_1_; }

    Eigen::Vector3d tic_r() const { return tic_1_; }

    Eigen::Matrix3d K_l() const { return Kl_undistort_; }

    Eigen::Matrix3d K_r() const { return Kr_undistort_; }

    Eigen::Matrix3d R_rl() const { return R_rl_; }

    Eigen::Vector3d p_rl() const { return p_rl_; }

    int camera_num() const { return camera_num_; }

   private:
    int camera_num_ = 2;
    cv::Size image_size_;
    Eigen::Matrix3d Ric_0_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic_0_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Ric_1_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic_1_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Kl_origin_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Kr_origin_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Kl_undistort_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Kr_undistort_ = Eigen::Matrix3d::Identity();
    Eigen::VectorXd D_l_;
    Eigen::VectorXd D_r_;
    cv::Mat rectify_map1_left, rectify_map2_left;
    cv::Mat rectify_map1_right, rectify_map2_right;

    Eigen::Matrix3d R_rl_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_rl_ = Eigen::Vector3d::Zero();
    double baseline_ = 0.f;
};
#endif