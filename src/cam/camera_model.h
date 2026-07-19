#pragma once

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <memory>
#include <stdexcept>

#include "parameter.h"
#include "sensorType.h"

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

// Abstract base for camera projection models. Pinhole (Brown-Conrady) and
// Fisheye (Kannala-Brandt equidistant) subclass it. Held as a singleton so the
// 40+ getInstance() call sites are unchanged; Init() is now a static factory
// that picks the subclass from params.camera_model.
class CamModel
{
  public:
    virtual ~CamModel() = default;

    // Singleton accessor. Init(params) must be called once beforehand; throws
    // if used before initialization.
    static CamModel &getInstance();

    // Factory: creates the concrete subclass from params.camera_model and
    // initializes it. Replaces the old getInstance().Init(params).
    static void Init(const Parameter &params);

    // --- Projection model interface (subclass-specific) ---
    virtual void compute_distort_jacobian(const int &cam_id, const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn) = 0;
    virtual void back_project_undistort(CameraObs &obs) const = 0;
    virtual Eigen::Vector2d project_distort(const uint32_t cam_id, const Eigen::Vector3d &p3d) const = 0;

    // Both models now skip image rectification: fisheye projects via KB directly
    // on the raw image, pinhole keeps raw K/D. Kept as a plain clone so the
    // existing call sites in vioManager are unchanged.
    void RectifyImage(const int32_t cam_id, const cv::Mat &img_raw_ptr, cv::Mat *img_rectified_ptr);

    // --- Shared geometry accessors (model-independent) ---
    double getBaseline() const { return plr().norm(); }

    Eigen::Matrix3d Ric(const uint32_t cam_id) const
    {
        if (cam_id >= vRic_.size())
            throw std::out_of_range("Camera ID out of range for Ric.");
        return vRic_[cam_id];
    }

    Eigen::Vector3d Tic(const uint32_t cam_id) const
    {
        if (cam_id >= vTic_.size())
            throw std::out_of_range("Camera ID out of range for Tic.");
        return vTic_[cam_id];
    }

    Eigen::VectorXd getDistortParam(const uint32_t cam_id) const
    {
        if (cam_id >= vD_.size())
            throw std::out_of_range("Camera ID out of range for distortion parameters.");
        return vD_[cam_id];
    }

    Eigen::Matrix3d K(const uint32_t cam_id) const
    {
        if (cam_id >= vK_.size())
            throw std::out_of_range("Camera ID out of range for K.");
        return vK_[cam_id];
    }

    Eigen::Vector3d plr() const
    {
        if (camera_num_ < 2)
            throw std::runtime_error("Not enough cameras to compute plr.");
        return vRic_[0].transpose() * (vTic_[1] - vTic_[0]);
    }

    Eigen::Matrix3d Rlr() const
    {
        if (camera_num_ < 2)
            throw std::runtime_error("Not enough cameras to compute Rlr.");
        return vRic_[0].transpose() * vRic_[1];
    }

    int32_t camera_num() const { return camera_num_; }

  protected:
    CamModel() = default;

    // Per-subclass initialization (invoked by the Init factory after construction).
    virtual void initImpl(const Parameter &params) = 0;

    // Shared init: fills camera_num_, image_size_, vRic_, vTic_. Subclasses call
    // this first, then fill their own vK_/vD_.
    void initCommon(const Parameter &params);

    int32_t camera_num_ = 0;
    cv::Size image_size_;
    std::vector<Eigen::Matrix3d> vRic_;
    std::vector<Eigen::Vector3d> vTic_;
    std::vector<Eigen::VectorXd> vD_;
    std::vector<Eigen::Matrix3d> vK_;

  private:
    // Holds the concrete subclass instance; function-local static avoids
    // static-initialization-order issues.
    static std::unique_ptr<CamModel> &storage_();
};
