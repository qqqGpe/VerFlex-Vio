#include <Eigen/Eigen>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include <glog/logging.h>
#include <memory>
#include <unistd.h>

#include "cameraModel.h"
#include "parameter.h"
#include "sensor_data.h"

// #include "datatypes.h"
#include <ros/node_handle.h>
#include <sensor_msgs/PointCloud.h>

#ifndef __FRONTEND__
#define __FRONTEND__

class VioFrontend {
public:
    enum status_t {
        STATUS_OK = 0,
        STATUS_NO_INPUT_DATA,
        STATUS_ERROR
    };

    VioFrontend(const Param parameters, std::shared_ptr<CameraModel>& camera_model, keyframe_flag_e *keyframe)
    {
        _width = parameters.img_width;
        _height = parameters.img_height;
        _max_feat_n = parameters.max_feat_n;
        _grid_w = parameters.grid_w;
        _grid_h = parameters.grid_h;

        _keyframe = keyframe;
        _camera_model = camera_model;

        ref_feat_to_track.resize(_max_feat_n, cam_obs_t());
    }

    bool inBorder(int x, int y);

    status_t run();

    bool track(const std::pair<double, cv::Mat>& input_image, std::pair<double, std::vector<cam_obs_t>>& feature_observes);

    // void publish_features(const frontend_frame_t &frame);

    // keyframe_flag_e decide_keyframe(const std::vector<FeatObs> &ref_feat_to_track, const std::vector<FeatObs> &cur_feat_to_track);

    status_t outlier_rejection(const std::vector<cam_obs_t>& obs_prev, const std::vector<cam_obs_t>& obs_curr, std::vector<uchar>& inliers);

    bool is_first_frame = true;
    std::pair<double, cv::Mat> ref_frame; // (ts_sec, image)
    std::pair<double, cv::Mat> cur_frame;
    std::vector<cam_obs_t> ref_feat_to_track; // valid, (x, y)

private:
    uint32_t frame_id = 0;
    uint32_t feat_id = 0;
    uint32_t _width, _height;
    uint32_t _max_feat_n;
    uint32_t _grid_w, _grid_h;
    keyframe_flag_e *_keyframe;
    std::shared_ptr<CameraModel> _camera_model;
    boost::posix_time::ptime frontend_rT, frontend_rT1, frontend_rT2, frontend_rT3, frontend_rT4;
};

#endif