#ifndef __VISUAL_MANAGER__
#define __VISUAL_MANAGER__

#include <map>
#include <vector>
#include "Imu_state.h"
#include "camModel.h"
#include "frontend.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
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
   static constexpr uint32_t kMinFeatureForUpdate = 10;

    VisualManager() = default;
    ~VisualManager()
    {
        for (int i = 0; i < max_feat_n_; i++)
        {
            delete feature_base_[i];
        }
    }

    VisualManager(const Param& paramters,
                  std::shared_ptr<State>& state,
                  std::shared_ptr<MsckfSolverBase> solver = nullptr)
    {
        _state = state;
        param_ = paramters;
        _keyframe = std::make_shared<KeyFrameStatus>(KeyFrameStatus::kNone);
        vio_frontend = std::make_shared<VioFrontend>(paramters, _keyframe);
        max_clone_pose_ = paramters.max_clone_pose;
        max_feat_n_ = paramters.max_feat_n;
        solver_ = solver;
        for (int i = 0; i < max_feat_n_; i++)
        {
            Feature* feat = new Feature();
            feature_base_.push_back(feat);
        }
    }

    void UpdateFeature(std::pair<double, std::vector<CameraObs>> feature_observes);

    bool VisualUpdate();

    KeyFrameStatus CheckKeyframe(std::shared_ptr<State> _state, std::vector<Feature*> feats);

    void UpdateFeatureBase();

    void ResetFeatureBase();

    void InitFeatureBase(std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_feature_triangulated);

    void DropFeatureObsrvs(const double timestamp_to_drop);

    void FeatureTriangulation(std::vector<Feature*>& feats, std::map<double, CameraPose> camera_pose_buffer);

    bool StereoTriangulation(CameraObs& cam_obs, Eigen::Vector3d& pcf) const;

    bool PnpRansac(Eigen::Matrix3d& R_12,
                   Eigen::Vector3d& p_12,
                   std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_obs_triangulated) const;

    bool least_square_triangulation(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool StereoLeastSqureTriangulation(CameraObs& cam_obs, Eigen::Vector3d& pcf) const;

    bool GaussianNewtonOptimization(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool ConstructFeatureJacobianFull(std::vector<Feature*> feats, Eigen::MatrixXd& Hx_full, Eigen::VectorXd& res);

    bool SingleFeatureJacobian(Feature* feat, std::unordered_map<std::shared_ptr<Type>, size_t> map_hx, int total_hx, Eigen::MatrixXd& Hx_single);

    bool PnpRansacToRejectOutliers(std::vector<Feature*> feats);

    void CalculateFeatureParallex(std::vector<Feature*>& feats);

    std::vector<Feature*> SelectMsckfFeatures(const std::vector<Feature*> feats);

    void set_state(std::shared_ptr<State> state) { _state = state; }  // only for debug

    void FeedImages(const std::pair<double, std::pair<cv::Mat, cv::Mat>> input);

    static double calcVisualObsParallex(const std::unordered_map<uint32_t, CameraObs>& visual_obs_a,
                                        const std::unordered_map<uint32_t, CameraObs>& visual_obs_b);

    static std::map<uint32_t, CameraObs> covisibleFeatures(const std::unordered_map<uint32_t, CameraObs>& visual_obs_a,
                                                           const std::unordered_map<uint32_t, CameraObs>& visual_obs_b);

    KeyFrameStatus GetKeyframeState() { return *_keyframe; }

    void SetKeyframeState(const KeyFrameStatus state) { *_keyframe = state; }

    std::vector<Feature*> GetFeatureBase() { return feature_base_; }

    void reset();

    uint32_t max_clone_pose_ = 6;
    uint32_t max_feat_n_ = 0;
    uint32_t feature_mapping_success_ = 0;
    uint32_t feature_mapping_in_ = 0;

    std::queue<std::pair<double, std::pair<cv::Mat, cv::Mat>>> _input_image_buffer;
    std::queue<std::pair<double, std::vector<CameraObs>>> feature_obs_buffer;

    std::map<double, std::pair<cv::Mat, cv::Mat>> stored_images_;

    std::vector<Feature*> feature_base_;
    std::vector<Feature*> feature_tracked_;
    std::vector<Feature*> feature_lost_;
    std::vector<CameraObs> feature_new_;
    std::vector<Feature*> feat_msckf_;

    boost::posix_time::ptime visual_rT, visual_rT1, visual_rT2, visual_rT3, visual_rT4;

    std::shared_ptr<KeyFrameStatus> _keyframe;

    std::shared_ptr<VioFrontend> vio_frontend;

    friend VioFrontend;

   protected:
    Param param_;
    std::shared_ptr<State> _state;
    std::unordered_map<std::shared_ptr<Type>, size_t> _map_hx;
    std::vector<std::shared_ptr<Type>> _Hx_order;
    int _origin_feature_tracked = 0.f;
    std::shared_ptr<MsckfSolverBase> solver_;
};

#endif