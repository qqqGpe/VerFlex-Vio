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
    explicit CameraPose(const int cam_num)
    {
        Rwc.resize(cam_num);
        Rwc_fej.resize(cam_num);
        pwc.resize(cam_num);
        pwc_fej.resize(cam_num);
    }
    std::vector<Eigen::Matrix3d> Rwc, Rwc_fej;
    std::vector<Eigen::Vector3d> pwc, pwc_fej;
    Eigen::Matrix3d Rwi;
    Eigen::Vector3d pwi;
    Eigen::Matrix3d Rwi_fej;
    Eigen::Vector3d pwi_fej;
};

class State
{
   public:
    explicit State(const Param& param)
    {
        _param = param;
        enable_estimate_ric_ = param.estimate_ric;
        enable_estimate_td_visual_ = param.estimate_td_visual;

        _imu_state = std::make_shared<ImuState>();
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

        _vQic.resize(param.camera_num);
        _vPic.resize(param.camera_num);
        for (int i = 0; i < param.camera_num; i++)
        {
            _vQic[i] = std::make_shared<Quat>();
            _vPic[i] = std::make_shared<Vec>();
            InitCamExtrinsic(i, Eigen::Quaterniond(param.Ric[i]), param.tic[i]);
            if (enable_estimate_ric_)
            {
                _vQic[i]->set_local_id(_dim);
                _variables.push_back(_vQic[i]);
                _dim += _vQic[i]->size();

                _vPic[i]->set_local_id(_dim);
                _variables.push_back(_vPic[i]);
                _dim += _vPic[i]->size();
            }
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
    }
    ~State() {}

    double ts_sec() const { return _imu_state->ts(); }

    uint32_t dim() const { return _dim; }

    bool enableEstimateRic() const { return enable_estimate_ric_; }

    bool enableEstimateTdVisual() const { return enable_estimate_td_visual_; }

    uint8_t CameraNum() const { return _param.camera_num; }

    Eigen::MatrixXd Covariance() const { return _covariance; }

    Eigen::MatrixXd Sqrt_Pt() const { return sqrt_Pt_; }

    std::shared_ptr<ImuState> getImuState() const { return _imu_state; }

    std::shared_ptr<Quat> mutable_Qic(const int cam_id) const { return _vQic[cam_id]; }

    const Eigen::Quaterniond Qic(const int cam_id) const { return _vQic[cam_id]->q(); }

    std::shared_ptr<Vec> mutable_Pic(const int cam_id) const { return _vPic[cam_id]; }

    const Eigen::Vector3d Pic(const int cam_id) const { return _vPic[cam_id]->vec(); }

    Eigen::Quaterniond Qwi() const { return _imu_state->q()->q(); }

    Eigen::Vector3d Pwi() const { return _imu_state->p()->vec(); }

    Eigen::Vector3d Vwi() const { return _imu_state->v()->vec(); }

    Eigen::Vector3d Bg() const { return _imu_state->bg()->vec(); }

    Eigen::Vector3d Ba() const { return _imu_state->ba()->vec(); }

    Scalar td_visual() const { return *td_visual_; }

    void AccessClonePoseBuffer(std::map<double, CameraPose>& camera_clone_poses) const
    {
        for (auto it = _clone_pose.begin(); it != _clone_pose.end(); it++)
        {
            CameraPose camera_pose(_param.camera_num);
            camera_pose.Rwi = it->second->quat().normalized().toRotationMatrix();
            camera_pose.pwi = it->second->p();
            for (int i_cam = 0; i_cam < _param.camera_num; i_cam++)
            {
                Eigen::Matrix3d R_CtoI = _vQic[i_cam]->q().toRotationMatrix();
                Eigen::Vector3d p_CinI = _vPic[i_cam]->vec();
                camera_pose.Rwc[i_cam] = camera_pose.Rwi * R_CtoI;
                camera_pose.pwc[i_cam] = camera_pose.pwi + camera_pose.Rwi * p_CinI;
            }
            camera_clone_poses.try_emplace(it->first, camera_pose);
        }
    }

    void set_ts_sec(double ts_sec) { _imu_state->set_ts(ts_sec); }

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

    void InitCamExtrinsic(const int cam_id, const Eigen::Quaterniond& qic, const Eigen::Vector3d& tic)
    {
        _vQic[cam_id]->set_value(qic.normalized().coeffs());
        _vPic[cam_id]->set_value(tic);
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

    uint32_t _dim = 0;
    Param _param;
    std::shared_ptr<ImuState> _imu_state;
    std::shared_ptr<Scalar> td_visual_;
    std::map<double, std::shared_ptr<Pose>> _clone_pose;
    std::vector<std::shared_ptr<Type>> _variables;
    Eigen::MatrixXd _covariance;
    Eigen::MatrixXd sqrt_Pt_;

   private:
    std::vector<std::shared_ptr<Quat>> _vQic;
    std::vector<std::shared_ptr<Vec>> _vPic;
    bool enable_estimate_ric_ = false;
    bool enable_estimate_td_visual_ = false;
};

#endif