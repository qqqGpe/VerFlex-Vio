#ifndef __VISUAL_MANAGER__
#define __VISUAL_MANAGER__

#include <map>
#include <vector>
#include <mutex>
#include <queue>
#include "ImuState.h"
#include "camModel.h"
#include "vioFrontend.h"
#include "parameter.h"
#include "sensorType.h"
#include "vioState.h"
#include "solver.h"
#include "eskf_solver.h"
#include "sqrt_eskf_solver.h"

namespace {
    constexpr int kMaxImageBufferSize = 10000;
}

class VisualManager
{
   public:
    static constexpr uint32_t kMaxFeatureForUpdate = 40;
    static constexpr uint32_t kMinFeatureForUpdate = 8;

    VisualManager() = default;
    VisualManager(std::shared_ptr<ros::NodeHandle>& nh,
                  const Param& params,
                  std::shared_ptr<State>& state,
                  std::shared_ptr<MsckfSolverBase> solver = nullptr);

    ~VisualManager()
    {
        for (int i = 0; i < param_.max_feat_n; i++)
        {
            delete feature_base_[i];
        }
    }

    void UpdateFeatureStatistic(const double timestamp, std::pair<double, std::vector<CameraObs>> feature_observes);

    bool VisualUpdate();

    KeyFrameStatus MaybeSetKeyframe(std::shared_ptr<State> _state, std::vector<Feature*> feats);

    void UpdateFeatureBase(const double timestamp);

    void ResetFeatureBase();

    void InitFeatureBase(std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_feature_triangulated);

    void ClearOldFeatureObs(const double timestamp_to_drop);

    void FeatureTriangulation(const std::map<double, CameraPose> camera_pose_buffer, std::vector<Feature*>& feats);

    bool StereoTriangulation(CameraObs& cam_obs, Eigen::Vector3d& pcf) const;

    bool PnpRansac(Eigen::Matrix3d& R_12,
                   Eigen::Vector3d& p_12,
                   std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_obs_triangulated) const;

    bool least_square_triangulation(const std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool StereoLeastSqureTriangulation(CameraObs& cam_obs, Eigen::Vector3d& pcf) const;

    bool GaussianNewtonOptimization(const std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool ConstructFeatureJacobianFull(std::vector<Feature*> feats, Eigen::MatrixXd& Hx_full, Eigen::VectorXd& res);

    bool SingleFeatureJacobian(Feature* feat, std::unordered_map<std::shared_ptr<Type>, size_t> map_hx, const int total_hx, Eigen::MatrixXd& Hx_single);

    bool PnpRansacToRejectOutliers(std::vector<Feature*> feats);

    void CalculateMaxFeatureParallex(std::vector<Feature*>& feats);

    void SelectMsckfFeatures(const std::vector<Feature*> feats, std::vector<Feature*>& feat_msckf);

    void set_state(std::shared_ptr<State> state) { _state = state; }  // only for debug

    void FeedImages(const std::pair<double, std::vector<cv::Mat>> input);

    bool MsckfFeatureUpdate(std::vector<Feature*> feats);

    void ClearExpiredMeasurements(const double timestamp);

    static double calcVisualObsParallex(const std::unordered_map<uint32_t, CameraObs>& visual_obs_a,
                                        const std::unordered_map<uint32_t, CameraObs>& visual_obs_b);

    static std::map<uint32_t, CameraObs> covisibleFeatures(const std::unordered_map<uint32_t, CameraObs>& visual_obs_a,
                                                           const std::unordered_map<uint32_t, CameraObs>& visual_obs_b);

    KeyFrameStatus GetKeyframeState() { return *_keyframe; }

    void SetKeyframeState(const KeyFrameStatus state) { *_keyframe = state; }

    std::vector<Feature*> GetFeatureBase() { return feature_base_; }

    void reset();

    uint32_t max_clone_pose_ = 6;
    uint32_t feature_mapping_success_ = 0;
    uint32_t feature_mapping_in_ = 0;

    std::queue<std::pair<double, std::vector<cv::Mat>>> _input_image_buffer;
    std::queue<std::pair<double, std::vector<CameraObs>>> feature_obs_buffer;
    mutable std::mutex input_image_buffer_mutex_; // Mutex for _input_image_buffer

    std::map<double, std::vector<cv::Mat>> stored_images_;

    std::vector<Feature*> feature_base_;
    std::vector<Feature*> feature_tracked_;
    std::vector<Feature*> feature_lost_;
    std::vector<CameraObs> feature_new_;
    std::vector<Feature*> feat_msckf_;
    std::vector<Feature*> feat_slam_old_;
    std::vector<Feature*> feat_slam_new_;

    std::shared_ptr<KeyFrameStatus> _keyframe;

    std::shared_ptr<VioFrontend> vio_frontend;

    friend VioFrontend;

   protected:
    Param param_;
    std::shared_ptr<ros::NodeHandle> nh_;
    std::shared_ptr<State> _state;
    std::unordered_map<std::shared_ptr<Type>, size_t> _map_hx;
    std::vector<std::shared_ptr<Type>> _Hx_order;
    std::shared_ptr<MsckfSolverBase> solver_;
};

#endif