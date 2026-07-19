#pragma once

#include "camera_model.h"

// Fisheye with the Kannala-Brandt equidistant model (the same model cv::fisheye
// uses). Projects directly on the raw fisheye image — no pre-rectification.
// Distortion params vD_ = [k1, k2, k3, k4] (radial only).
class Fisheye : public CamModel
{
  public:
    void compute_distort_jacobian(const int &cam_id, const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn) override;

    void back_project_undistort(CameraObs &obs) const override;

    Eigen::Vector2d project_distort(const uint32_t cam_id, const Eigen::Vector3d &p3d) const override;

  protected:
    void initImpl(const Parameter &params) override;
};
