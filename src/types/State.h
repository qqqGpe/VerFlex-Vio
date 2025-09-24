#ifndef __VIO_BACKEND_STATE__
#define __VIO_BACKEND_STATE__
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <memory>

#include "ImuState.h"
#include "Pose.h"
#include "Scalar.h"
#include "parameter.h"

struct CameraPose
{
    Eigen::Matrix3d Rwi;
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d pwi;
    Eigen::Vector3d pwc;

    Eigen::Matrix3d Rwi_fej;
    Eigen::Matrix3d Rwc_fej;
    Eigen::Vector3d pwi_fej;
    Eigen::Vector3d pwc_fej;
};

class State
{
   public:
    explicit State(const Param& param) : enable_estimate_ric_(param.estimate_ric), enable_estimate_td_visual_(param.estimate_td_visual)
    {
        _imu_state = std::make_shared<ImuState>();
        qic_ = std::make_shared<Quat>();
        tic_ = std::make_shared<Vec>();

        _imu_state->set_local_id(_dim);
        _variables.push_back(_imu_state->q());
        _dim += _imu_state->q()->size();
        _variables.push_back(_imu_state->p());
        _dim += _imu_state->p()->size();
        _variables.push_back(_imu_state->v());
        _dim += _imu_state->v()->size();
        _variables.push_back(_imu_state->bg());
        _dim += _imu_state->bg()->size();
        _variables.push_back(_imu_state->ba());
        _dim += _imu_state->ba()->size();

        if (enable_estimate_ric_)
        {
            qic_->set_local_id(_dim);
            _variables.push_back(qic_);
            _dim += qic_->size();
        }

        if (enable_estimate_td_visual_)
        {
            td_visual_ = std::make_shared<Scalar>();
            td_visual_->set_local_id(_dim);
            _variables.push_back(td_visual_);
            _dim += td_visual_->size();
        }

        SetCovariance(Eigen::MatrixXd::Identity(_dim, _dim));             // initialize covariance;
        SetSqrtPt(Eigen::MatrixXd::Identity(_dim, _dim));                 // initialize sqrt-root covariance;
        SetCamExtrinsic(Eigen::Quaterniond(param.Ric[0]), param.tic[0]);  // initialize camera extrinsic
    }
    ~State() {}

    double ts_sec() const { return _imu_state->ts(); }

    uint32_t dim() const { return _dim; }

    Eigen::MatrixXd Covariance() const { return _covariance; }

    Eigen::MatrixXd Sqrt_Pt() const { return sqrt_Pt_; }

    std::shared_ptr<ImuState> getImuState() const { return _imu_state; }

    Quat qic() const { return *qic_; }

    Scalar td_visual() const { return *td_visual_; }

    std::map<double, CameraPose> AccessClonePoseBuffer() const
    {
        std::map<double, CameraPose> camera_clone_poses;
        for (auto it = _clone_pose.begin(); it != _clone_pose.end(); it++)
        {
            CameraPose camera_pose;
            camera_pose.Rwi = it->second->quat().normalized().toRotationMatrix();
            camera_pose.pwi = it->second->p();
            Eigen::Matrix3d R_CtoI = qic_->q().toRotationMatrix();
            Eigen::Vector3d p_CinI = tic_->vec();
            camera_pose.Rwc = camera_pose.Rwi * R_CtoI;
            camera_pose.pwc = camera_pose.pwi + camera_pose.Rwi * p_CinI;
            camera_clone_poses.insert(std::make_pair(it->first, camera_pose));
        }
        return camera_clone_poses;
    }

    void set_ts_sec(double ts_sec) { _imu_state->set_ts(ts_sec); }

    void SetCamExtrinsic(Eigen::Quaterniond qic, Eigen::Vector3d tic)
    {
        qic_->set_value(qic.normalized().coeffs());
        tic_->set_value(tic);
    }

    void SetCovariance(const Eigen::MatrixXd& covariance_new)
    {
        _covariance = covariance_new;
        _imu_state->set_covariance(covariance_new.block(_imu_state->id(), _imu_state->id(), _imu_state->size(), _imu_state->size()));
    }

    void SetSqrtPt(const Eigen::MatrixXd& sqrt_Pt_new)
    {
        sqrt_Pt_.noalias() = sqrt_Pt_new;
        _imu_state->set_sqrt_Pt(sqrt_Pt_new.block(_imu_state->id(), _imu_state->id(), _imu_state->size(), _imu_state->size()));
    }

    void reset()
    {
        // Reset imu state
        _imu_state->reset();

        // Reset td_visual
        if (enable_estimate_td_visual_)
        {
            td_visual_->reset();
        }

        // Clear all clone poses
        for (auto it = _clone_pose.begin(); it != _clone_pose.end();)
        {
            _variables.erase(std::remove(_variables.begin(), _variables.end(), it->second), _variables.end());
            _dim = _dim - it->second->size();
            it = _clone_pose.erase(it);
        }

        // Reset covariance and sqrt_Pt
        SetCovariance(Eigen::MatrixXd::Identity(_dim, _dim));
        SetSqrtPt(Eigen::MatrixXd::Identity(_dim, _dim));
    }

    int _dim = 0;

    std::shared_ptr<ImuState> _imu_state;
    std::shared_ptr<Quat> qic_;
    std::shared_ptr<Vec> tic_;
    std::shared_ptr<Scalar> td_visual_;

    std::map<double, std::shared_ptr<Pose>> _clone_pose;
    std::vector<std::shared_ptr<Type>> _variables;

    Eigen::MatrixXd _covariance;
    Eigen::MatrixXd sqrt_Pt_;

    int32_t enable_estimate_ric_ = 0;
    int32_t enable_estimate_td_visual_ = 0;
};

#endif