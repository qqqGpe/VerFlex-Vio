#ifndef __VISUAL_MANAGER__
#define __VISUAL_MANAGER__

#include "Imu_state.h"
#include "state.h"
#include "sensor_data.h"
#include "parameter.h"
#include "camera_model.h"
#include <map>
#include <vector>

struct cam_obs_t
{
    double u, v;
    double u_norm, v_norm;
} ;

struct Feature {
    Feature()
    {
        _id = -1;
        _valid = false;
        _is_triangulated = false;
        _visual_obs_buffer.clear();
    }
    int _id = -1;
    bool _valid = false;
    Eigen::Vector3d _pwf;
    bool _is_triangulated = false;
    std::map<double, cam_obs_t> _visual_obs_buffer; // <ts_sec, obs>
};

class VisualManager
{
public:
    VisualManager() = default;
    ~VisualManager(){}

    VisualManager(const Param &paramters, std::shared_ptr<State> &state, std::shared_ptr<CameraModel> &camera_model){
        _state = state;
        _camera_model = camera_model;
    }

    void feed_visual_measurement(const FeatureData &data);

    void feature_triangulation(std::vector<Feature* > feats, std::map<double, CameraPose> camera_pose_buffer);

    bool least_square_triangulation(std::map<double, CameraPose> &clone_pose_buffer, Feature* feat);

    bool gaussian_newton_optimization(std::map<double, CameraPose> &clone_pose_buffer, Feature* feat);

private:
    std::shared_ptr<State> _state;
    std::shared_ptr<CameraModel> _camera_model;

    std::vector<Feature* > _feature_base;
    std::vector<Feature* > _feature_tracked;
    std::vector<Feature* > _feature_lost;
    std::vector<Feature* > _feature_new;
};


#endif