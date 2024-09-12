#ifndef __CAM_MODEL__
#define __CAM_MODEL__

#include "parameter.h"
#include <Eigen/Core>

enum class CameraType {
    NO_TYPE = 0,
    PINHOLE,
    FISHEYE,
};

class CameraModel {
public:
    CameraModel() = default;
    ~CameraModel() { }

    void set_camera_intrin_matrix(const std::vector<double>& intrinsic_coeff) {
        _K.setIdentity();
        _K(0, 0) = intrinsic_coeff[0];
        _K(1, 1) = intrinsic_coeff[1];
        _K(0, 2) = intrinsic_coeff[2];
        _K(1, 2) = intrinsic_coeff[3];
    }

    void set_camera_distort_coeff(const std::vector<double>& distort_coeff) {
        if (distort_coeff.empty()) {
            return;
        }
        _distort_coeff = Eigen::VectorXd::Zero(distort_coeff.size());
        for (size_t i = 0; i < distort_coeff.size(); i++) {
            _distort_coeff(i) = distort_coeff[i];
        }
    }

    CameraModel(const CameraType &type, const Param params)
    {
        _type = type;
        _Ric_init = params.Ric[0];
        _tic_init = params.tic[0];
        set_camera_intrin_matrix(params.intrinsic_cam_0);
        set_camera_distort_coeff(params.distortion_cam_0);
    }

    Eigen::Vector2d project(Eigen::Vector3d p3d_norm);

    Eigen::Vector3d back_project(Eigen::Vector2d uv_2d)
    {
        Eigen::Vector3d feat_norm((uv_2d.x() - _K(0, 2)) / _K(0, 0), (uv_2d.y() - _K(1, 2)) / _K(1, 1), 1.0);
        return feat_norm;
    }

    Eigen::Matrix3d intrinsic() { return _K; }

    Eigen::Matrix3d Ric() { return _Ric_init; }

    Eigen::Vector3d tic() { return _tic_init; }

private:
    CameraType _type = CameraType::NO_TYPE;
    Eigen::Matrix3d _K;
    Eigen::VectorXd _distort_coeff;
    Eigen::Matrix3d _Ric_init;
    Eigen::Vector3d _tic_init;
};
#endif