#ifndef __CAM_MODEL__
#define __CAM_MODEL__

#include "sensor_data.h"
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

    void set_camera_intrin_matrix(const std::vector<double>& intrinsic_coeff, Eigen::Matrix3d& K)
    {
        K.setIdentity();
        K(0, 0) = intrinsic_coeff[0];
        K(1, 1) = intrinsic_coeff[1];
        K(0, 2) = intrinsic_coeff[2];
        K(1, 2) = intrinsic_coeff[3];
    }

    void set_camera_distort_coeff(const std::vector<double>& distort_coeff, Eigen::VectorXd& param)
    {
        if (distort_coeff.empty()) {
            return;
        }
        param = Eigen::VectorXd::Zero(distort_coeff.size());
        for (size_t i = 0; i < distort_coeff.size(); i++) {
            param(i) = distort_coeff[i];
        }
    }

    CameraModel(const CameraType &type, const Param params)
    {
        _type = type;

        Ric_0_ = params.Ric[0];
        tic_0_ = params.tic[0];
        set_camera_intrin_matrix(params.intrinsic_cam_0, K_l_);
        set_camera_intrin_matrix(params.intrinsic_cam_1, K_r_);

        Ric_1_ = params.Ric[1];
        tic_1_ = params.tic[1];
        set_camera_distort_coeff(params.distortion_cam_0, distort_param_l);
        set_camera_distort_coeff(params.distortion_cam_1, distort_param_r);

        R_rl_ = Ric_0_.transpose() * Ric_1_;
        t_rl_ = Ric_0_.transpose() * (tic_1_ - tic_0_);
        baseline_ = t_rl_.norm();
    }

    Eigen::Vector2d project_left(Eigen::Vector3d p3d_norm)
    {
        Eigen::Vector2d feature_norm(K_l_(0, 0) * p3d_norm.x() / p3d_norm.z() + K_l_(0, 2),
                                     K_l_(1, 1) * p3d_norm.y() / p3d_norm.z() + K_l_(1, 2));
        return feature_norm;
    }

    Eigen::Vector2d project_right(Eigen::Vector3d p3d_norm)
    {
        Eigen::Vector2d feature_norm(K_r_(0, 0) * p3d_norm.x() / p3d_norm.z() + K_r_(0, 2),
                                     K_r_(1, 1) * p3d_norm.y() / p3d_norm.z() + K_r_(1, 2));
        return feature_norm;
    }

    Eigen::Vector3d back_project(Eigen::Vector2d uv_2d)
    {
        Eigen::Vector3d feat_norm((uv_2d.x() - K_l_(0, 2)) / K_l_(0, 0), (uv_2d.y() - K_l_(1, 2)) / K_l_(1, 1), 1.0);
        return feat_norm;
    }

    void back_project_stereo(CameraObs &obs)
    {
        Eigen::Vector3d feat_norm_left((obs.u - K_l_(0, 2)) / K_l_(0, 0), (obs.v - K_l_(1, 2)) / K_l_(1, 1), 1.0);
        Eigen::Vector3d feat_norm_right((obs.ur - K_r_(0, 2)) / K_r_(0, 0), (obs.vr - K_r_(1, 2)) / K_r_(1, 1), 1.0);

        obs.u_norm = feat_norm_left.x();
        obs.v_norm = feat_norm_left.y();
        obs.ur_norm = feat_norm_right.x();
        obs.vr_norm = feat_norm_right.y();
    }

    double baseline() const { return baseline_; }

    Eigen::Matrix3d Ric_l() { return Ric_0_; }

    Eigen::Vector3d tic_l() { return tic_0_; }

    Eigen::Matrix3d Ric_r() { return Ric_1_; }

    Eigen::Vector3d tic_r() { return tic_1_; }

    Eigen::Matrix3d K_l() { return K_l_; }

    Eigen::Matrix3d K_r() { return K_r_; }

    Eigen::Matrix3d R_rl() { return R_rl_; }

private:
    CameraType _type = CameraType::NO_TYPE;
    Eigen::VectorXd distort_param_l;
    Eigen::VectorXd distort_param_r;
    Eigen::Matrix3d Ric_0_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic_0_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Ric_1_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic_1_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d K_l_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d K_r_ = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d R_rl_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_rl_ = Eigen::Vector3d::Zero();
    double baseline_ = 0.f;
};
#endif