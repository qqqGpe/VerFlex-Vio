#ifndef __VISUAL_MANAGER__
#define __VISUAL_MANAGER__

#include <map>
#include <vector>
#include "Imu_state.h"
#include "cameraModel.h"
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
    VisualManager() = default;

    ~VisualManager()
    {
        for (int i = 0; i < _max_feat_n; i++)
        {
            delete _feature_base[i];
        }
    }

    VisualManager(const Param& paramters,
                  std::shared_ptr<State>& state,
                  std::shared_ptr<CameraModel>& camera_model,
                  std::shared_ptr<MsckfSolverBase> solver = nullptr)
    {
        _state = state;
        _camera_model = camera_model;
        _keyframe = std::make_shared<KeyFrameStatus>(KeyFrameStatus::kNone);
        vio_frontend = std::make_shared<VioFrontend>(paramters, camera_model, _keyframe);
        _max_clone_pose = paramters.max_clone_pose;
        _max_feat_n = paramters.max_feat_n;
        solver_ = solver;
        for (int i = 0; i < _max_feat_n; i++)
        {
            Feature* feat = new Feature();
            _feature_base.push_back(feat);
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

    bool StereoTriangulation(const std::shared_ptr<CameraModel> camera_model, CameraObs& cam_obs, Eigen::Vector3d& pcf) const;

    bool PnpRansac(const std::shared_ptr<CameraModel> camera_model,
                   std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_obs_triangulated,
                   Eigen::Matrix3d& R_21, Eigen::Vector3d& p_21) const;

    bool least_square_triangulation(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool StereoLeastSqureTriangulation(const std::shared_ptr<CameraModel> camera_model, CameraObs& cam_obs, Eigen::Vector3d& pcf) const;

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

    std::vector<Feature*> GetFeatureBase() { return _feature_base; }

    void reset();

    uint32_t _max_clone_pose = 6;
    uint32_t _max_feat_n = 0;
    uint32_t _feature_mapping_success = 0;
    uint32_t _feature_mapping_in = 0;
    uint32_t _max_visual_feat_to_use = 100;

    std::queue<std::pair<double, std::pair<cv::Mat, cv::Mat>>> _input_image_buffer;
    std::queue<std::pair<double, std::vector<CameraObs>>> feature_obs_buffer;

    std::map<double, std::pair<cv::Mat, cv::Mat>> stored_images_;

    std::vector<Feature*> _feature_base;
    std::vector<Feature*> _feature_tracked;
    std::vector<Feature*> _feature_lost;
    std::vector<Feature*> _feature_new_base;
    std::vector<CameraObs> _feature_new;

    boost::posix_time::ptime visual_rT, visual_rT1, visual_rT2, visual_rT3, visual_rT4;

    std::shared_ptr<KeyFrameStatus> _keyframe;

    std::shared_ptr<VioFrontend> vio_frontend;

    friend VioFrontend;

   protected:
    std::shared_ptr<State> _state;
    std::shared_ptr<CameraModel> _camera_model;
    std::unordered_map<std::shared_ptr<Type>, size_t> _map_hx;
    std::vector<std::shared_ptr<Type>> _Hx_order;
    int _origin_feature_tracked = 0.f;
    std::shared_ptr<MsckfSolverBase> solver_;
};

#endif