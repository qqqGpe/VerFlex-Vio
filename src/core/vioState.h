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
#include "sensorType.h"

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

struct SlamFeature
{
    uint32_t _id = -1;
    std::shared_ptr<Vec> _state_ptr;
    Feature* _info;
};

class State
{
   public:
    State(const Param& param);
    virtual ~State() = default;

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

    void AccessClonePoseBuffer(std::map<double, CameraPose>& camera_clone_poses) const;

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

    void reset();

    bool IsOldSlamFeature(const uint32_t feature_id)
    {
        return _slam_features.find(feature_id) != _slam_features.end();
    }

    void insert_after(const std::shared_ptr<Type>& pose_to_insert_after, const std::shared_ptr<Type>& new_variable);

    void UpdateSlamFeatureAfterVisualUpdate();

    std::map<uint32_t, SlamFeature>& mutable_slam_features() { return _slam_features; }

    const std::map<uint32_t, SlamFeature> slam_features() const { return _slam_features; }

    std::map<double, std::shared_ptr<Pose>>& mutable_clone_poses() { return _clone_pose; }

    const std::map<double, std::shared_ptr<Pose>> clone_poses() const { return _clone_pose; }

    Eigen::MatrixXd& mutable_covariance() { return _covariance; }

    const Eigen::MatrixXd& covariance() const { return _covariance; }

    bool AugumentSlamFeature(Feature* feature,
                             const Eigen::MatrixXd& Hf,
                             const Eigen::MatrixXd& Hx,
                             const std::vector<std::shared_ptr<Type>>& Hx_order,
                             const Eigen::MatrixXd& R,
                             std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx);

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
    std::map<uint32_t, SlamFeature> _slam_features;
    bool enable_estimate_ric_ = false;
    bool enable_estimate_td_visual_ = false;
};

#endif