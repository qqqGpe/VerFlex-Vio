#pragma once

#include "camera_model.h"

// Pinhole + Brown-Conrady radial/tangential distortion. Raw K/D are kept (no
// image rectification); distortion is applied analytically per feature.
class Pinhole : public CamModel
{
  public:
    void compute_distort_jacobian(const int &cam_id, const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn) override;

    void back_project_undistort(CameraObs &obs) const override;

    Eigen::Vector2d project_distort(const uint32_t cam_id, const Eigen::Vector3d &p3d) const override;

  protected:
    void initImpl(const Parameter &params) override;
};
