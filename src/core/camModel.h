#pragma once

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "parameter.h"
#include "sensor_data.h"

enum CameraId
{
    LEFT_CAM = 0,
    RIGHT_CAM = 1,
    MAX_CAM_NUM
};

enum CamType
{
    MONO = 1,
    STEREO = 2
};

class CamModel
{
public:
    static CamModel& getInstance()
    {
        static CamModel* instance = new CamModel();
        return *instance;
    }

    CamModel() = default;
    CamModel(CamModel const&) = delete;
    CamModel& operator=(CamModel const&) = delete;

    void CalculateUndistortRectifyMap(const Eigen::Matrix3d& K, const Eigen::VectorXd& D, Eigen::Matrix3d& K_undistort, cv::Mat& map1, cv::Mat& map2);

    void SetCameraIntrinsicMatrix(const std::vector<double>& intrinsic_coeff); // For debug

    void back_project(CameraObs& obs) const;

    Eigen::Vector2d project(const uint32_t cam_id, const Eigen::Vector3d& p3d_norm) const;

    void RectifyImage(const int32_t cam_id, const cv::Mat& img_raw_ptr, cv::Mat* img_rectified_ptr);    // Convert raw image to rectified image

    void Init(const Param params);

    double getBaseline() const
    {
        return p_rl().norm();
    }

    Eigen::Matrix3d Ric(const uint32_t cam_id) const
    {
        if (cam_id >= vD_.size())
        {
            throw std::out_of_range("Camera ID out of range for Ric.");
        }
        return vRic_[cam_id];
    }

    Eigen::Vector3d Tic(const uint32_t cam_id) const
    {
        if (cam_id >= vD_.size())
        {
            throw std::out_of_range("Camera ID out of range for Tic.");
        }
        return vTic_[cam_id];
    }

    Eigen::VectorXd getDistortParam(const uint32_t cam_id) const
    {
        if (cam_id >= vD_.size())
        {
            throw std::out_of_range("Camera ID out of range for distortion parameters.");
        }
        return vD_[cam_id];
    }

    Eigen::Matrix3d K(const uint32_t cam_id) const
    {
        if (cam_id >= vD_.size())
        {
            throw std::out_of_range("Camera ID out of range for K.");
        }
        return vK_[cam_id];
    }

    Eigen::Vector3d p_rl() const
    {
        if (camera_num_ < 2)
        {
            throw std::runtime_error("Not enough cameras to compute R_rl.");
        }
        Eigen::Matrix3d Ric_l = vRic_[0];
        Eigen::Vector3d pic_l = vTic_[0];
        Eigen::Vector3d pic_r = vTic_[1];
        return Ric_l.transpose() * (pic_r - pic_l);
    }

    Eigen::Matrix3d R_rl() const
    {
        if (camera_num_ < 2)
        {
            throw std::runtime_error("Not enough cameras to compute R_rl.");
        }
        Eigen::Matrix3d Ric_l = vRic_[0];
        Eigen::Matrix3d Ric_r = vRic_[1];
        return Ric_l.transpose() * Ric_r;
    }

    int32_t camera_num() const { return camera_num_; }

private:
    int32_t camera_num_ = 0;
    cv::Size image_size_;
    std::vector<Eigen::Matrix3d> vRic_;
    std::vector<Eigen::Vector3d> vTic_;
    std::vector<Eigen::VectorXd> vD_;
    std::vector<Eigen::Matrix3d> vK_;
    std::vector<cv::Mat> rectify_map_;
};