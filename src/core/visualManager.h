#ifndef __VISUAL_MANAGER__
#define __VISUAL_MANAGER__

#include "Imu_state.h"
#include "cameraModel.h"
#include "frontend.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include <map>
#include <vector>

namespace {
constexpr int kMaxImageBufferSize = 10000;
}

class VisualManager {
public:
    VisualManager() = default;
    ~VisualManager()
    {
        for (int i = 0; i < _max_feat_n; i++) {
            delete _feature_base[i];
        }
    }

    VisualManager(const Param& paramters, std::shared_ptr<State>& state, std::shared_ptr<CameraModel>& camera_model)
    {
        _state = state;
        _camera_model = camera_model;
        vio_frontend = std::make_shared<VioFrontend>(paramters, camera_model, &_keyframe);

        _max_clone_pose = paramters.max_clone_pose;
        _max_feat_n = paramters.max_feat_n;
        for (int i = 0; i < _max_feat_n; i++) {
            Feature* feat = new Feature();
            _feature_base.push_back(feat);
        }
    }

    void update_feature(std::pair<double, std::vector<cam_obs_t>> feature_observes);

    void visual_update();

    void reset_keyframe() { _keyframe = keyframe_flag_e::not_keyframe; }

    keyframe_flag_e decide_keyframe(std::shared_ptr<State> _state, std::vector<Feature*> feats);

    void update_feature_base();

    void drop_feature_obs(const double timestamp_to_drop);

    void feature_triangulation(std::vector<Feature*> feats, std::map<double, CameraPose> camera_pose_buffer);

    bool least_square_triangulation(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool gaussian_newton_optimization(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool construct_feature_jocabian_full(std::vector<Feature*> feats, Eigen::MatrixXd& Hx_full, Eigen::VectorXd& res);

    Eigen::MatrixXd get_single_feature_jacobian(Feature* feat, std::unordered_map<std::shared_ptr<Type>, size_t> map_hx, int total_hx);

    bool pnp_ransac_to_reject_outliers(std::vector<Feature*> feats);

    void set_state(std::shared_ptr<State> state) { _state = state; } // for debug

    void feed_image(const std::pair<double, cv::Mat> input_data)
    {
        while (_input_image_buffer.size() > kMaxImageBufferSize) {
            _input_image_buffer.pop();
        }
        _input_image_buffer.push(input_data);
    }

    uint32_t _max_clone_pose = 6;
    uint32_t _max_feat_n = 0;
    uint32_t _feature_mapping_success = 0;
    uint32_t _feature_mapping_in = 0;

    std::queue<std::pair<double, cv::Mat>> _input_image_buffer;
    std::queue<std::pair<double, std::vector<cam_obs_t>>> feature_obs_buffer;

    std::vector<Feature*> _feature_base;
    std::vector<Feature*> _feature_tracked;
    std::vector<Feature*> _feature_lost;
    std::vector<cam_obs_t> _feature_new;

    boost::posix_time::ptime visual_rT, visual_rT1, visual_rT2, visual_rT3, visual_rT4;

    keyframe_flag_e _keyframe = keyframe_flag_e::not_keyframe;

    std::shared_ptr<VioFrontend> vio_frontend;

    friend VioFrontend;

protected:

    std::shared_ptr<State> _state;
    std::shared_ptr<CameraModel> _camera_model;
    std::unordered_map<std::shared_ptr<Type>, size_t> _map_hx;
    std::vector<std::shared_ptr<Type>> _Hx_order;
};

#endif