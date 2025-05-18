#ifndef __VIO_BACKEND_STATE__
#define __VIO_BACKEND_STATE__
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
#include <memory>
#include "Imu_state.h"
#include "Pose.h"

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
    State()
    {
        _imu_state = std::make_shared<ImuState>();
        _Tic = std::make_shared<Pose>();

        // initialize local id
        _dim = 0;
        // _imu_state->set_local_id(_dim);
        // _variables.push_back(_imu_state);
        // _dim += _imu_state->size();
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

        if (_do_calibration_update)
        {
            _Tic->set_local_id(_dim);
            _variables.push_back(_Tic);
            _dim += _Tic->size();
        }

        // initialize state covariance
        SetCovariance(Eigen::MatrixXd::Identity(_dim, _dim)); // initialize covariance size;
        SetSqrtPt(Eigen::MatrixXd::Identity(_dim, _dim)); // initialize covariance size;
    }
    ~State() {}

    void set_ts_sec(double ts_sec) { _imu_state->set_ts(ts_sec); }  // for debug

    void set_extrinsic(Eigen::Quaterniond qic, Eigen::Vector3d tic)
    {
        Eigen::VectorXd extrin_vector = Eigen::VectorXd::Zero(7);
        extrin_vector << qic.coeffs(), tic;
        _Tic->set_value(extrin_vector);
    }

    double ts_sec() { return _imu_state->ts(); }

    std::map<double, CameraPose> AccessClonePoseBuffer() const
    {
        std::map<double, CameraPose> camera_clone_poses;
        for (auto it = _clone_pose.begin(); it != _clone_pose.end(); it++)
        {
            CameraPose camera_pose;
            camera_pose.Rwi = it->second->quat().normalized().toRotationMatrix();
            camera_pose.pwi = it->second->p();
            Eigen::Matrix3d R_CtoI = _Tic->quat().normalized().toRotationMatrix();
            Eigen::Vector3d p_CinI = _Tic->p();
            camera_pose.Rwc = camera_pose.Rwi * R_CtoI;
            camera_pose.pwc = camera_pose.pwi + camera_pose.Rwi * p_CinI;
            camera_clone_poses.insert(std::make_pair(it->first, camera_pose));
        }
        return camera_clone_poses;
    }

    Eigen::MatrixXd Covariance() { return _covariance; }

    void SetCovariance(const Eigen::MatrixXd& covariance_new)
    {
        _covariance = covariance_new;
        _imu_state->set_covariance(covariance_new.block(_imu_state->id(), _imu_state->id(), _imu_state->size(), _imu_state->size()));
    }

    Eigen::MatrixXd Sqrt_Pt() { return sqrt_Pt_; }

    void SetSqrtPt(const Eigen::MatrixXd& sqrt_Pt_new)
    {
        assert(sqrt_Pt_new.cols() == _dim);
        sqrt_Pt_.noalias() = sqrt_Pt_new;
        _imu_state->set_sqrt_Pt(sqrt_Pt_new.block(_imu_state->id(), _imu_state->id(), _imu_state->size(), _imu_state->size()));
    }

    ImuState getImuState() { return *_imu_state; }

    void reset()
    {
        _imu_state->reset();

        for (auto it = _clone_pose.begin(); it != _clone_pose.end(); it++)
        {
            _variables.erase(std::remove(_variables.begin(), _variables.end(), it->second), _variables.end());
            _dim = _dim - it->second->size();
        }
        _clone_pose.clear();
        SetCovariance(Eigen::MatrixXd::Identity(_dim, _dim));
    }

    int _dim = 0;
    std::shared_ptr<ImuState> _imu_state;
    std::map<double, std::shared_ptr<Pose>> _clone_pose;
    std::shared_ptr<Pose> _Tic;  // R_CtoI, p_CinI
    std::vector<std::shared_ptr<Type>> _variables;
    Eigen::MatrixXd _covariance;
    Eigen::MatrixXd sqrt_Pt_;
    int32_t _do_calibration_update = 0;
};

#endif