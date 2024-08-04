#ifndef __VISUAL_MANAGER__
#define __VISUAL_MANAGER__

#include "Imu_state.h"
#include "state.h"
#include "sensor_data.h"
#include "parameter.h"
#include "camera_model.h"
#include "frontend.h"
#include <map>
#include <vector>

namespace {
    constexpr int kMaxImageBufferSize = 1000;
}

enum keyframe_flag_e {
    not_keyframe = 0,
    large_parallex_flag,
    feat_lost_too_much
};

class VisualManager
{
public:
    VisualManager() = default;
    ~VisualManager(){
        for (int i = 0; i < _max_feat_n; i++) {
            delete _feature_base[i];
        }
    }

    VisualManager(const Param &paramters, std::shared_ptr<State> &state, std::shared_ptr<CameraModel> &camera_model){
        _state = state;
        _camera_model = camera_model;
        vio_frontend = std::make_shared<VioFrontend>(paramters, camera_model, &keyframe);

        _max_clone_pose = paramters.max_clone_pose;
        _max_feat_n = paramters.max_feat_n;
        for (int i = 0; i < _max_feat_n; i++) {
            Feature *feat = new Feature();
            _feature_base.push_back(feat);
        }
    }

    void update_feature(std::pair<double, std::vector<cam_obs_t>> feature_observes);

    void update();

    keyframe_flag_e decide_keyframe(std::shared_ptr<State> _state, std::vector<Feature*> feats);

    void feature_triangulation(std::vector<Feature*> feats, std::map<double, CameraPose> camera_pose_buffer);

    bool least_square_triangulation(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool gaussian_newton_optimization(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat);

    bool construct_feature_jocabian_full(std::vector<Feature*> feats, Eigen::MatrixXd& Hx_full, Eigen::VectorXd &res);

    Eigen::MatrixXd get_single_feature_jacobian(Feature* feat, std::unordered_map<std::shared_ptr<Type>, size_t> map_hx, int total_hx);

    void pnp_ransac_to_reject_outliers(std::vector<Feature* > feats);

    void set_state(std::shared_ptr<State> state) { _state = state; }    // for debug

    void feed_image(const std::pair<double, cv::Mat> input_data)
    {
        while (_input_image_buffer.size() > kMaxImageBufferSize) {
            _input_image_buffer.pop();
        }
        _input_image_buffer.push(input_data);
    }

    uint32_t _max_clone_pose = 6;
    uint32_t _max_feat_n = 0;
    std::queue<std::pair<double, cv::Mat>> _input_image_buffer;
    std::queue<std::pair<double, std::vector<cam_obs_t>>> feature_obs_buffer;
    std::vector<Feature* > _feature_base;
    std::vector<Feature* > _feature_tracked;
    std::vector<Feature* > _feature_lost;
    std::vector<Feature* > _feature_new;

    std::shared_ptr<VioFrontend> vio_frontend;

    friend VioFrontend;

protected:
    std::atomic<int> keyframe = 0;
    std::shared_ptr<State> _state;
    std::shared_ptr<CameraModel> _camera_model;
    std::unordered_map<std::shared_ptr<Type>, size_t> _map_hx;
    std::vector<std::shared_ptr<Type>> _Hx_order;
};


#endif