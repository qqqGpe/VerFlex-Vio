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

    CameraModel(const CameraType &type, const std::vector<double>& intrinsic_coeff, const std::vector<double>& distort_coeff)
    {
        _type = type;
        _distort_coeff = distort_coeff;
        _K(0, 0) = intrinsic_coeff[0];
        _K(1, 1) = intrinsic_coeff[1];
        _K(0, 2) = intrinsic_coeff[2];
        _K(1, 2) = intrinsic_coeff[3];
        _K(2, 2) = 1.0;
    }
    Eigen::Vector2d project(Eigen::Vector3d p3d_norm);
    Eigen::Vector3d back_project(Eigen::Vector2d uv_2d);
    Eigen::Matrix3d intrinsic() { return _K; }

private:
    CameraType _type = CameraType::NO_TYPE;
    Eigen::Matrix3d _K;
    std::vector<double> _distort_coeff;
};
#endif