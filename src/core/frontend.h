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

    VioFrontend(const Param parameters, std::shared_ptr<CameraModel>& camera_model, KeyFrameType *keyframe)
    {
        width_ = parameters.img_width;
        height_ = parameters.img_height;
        _max_feat_n = parameters.max_feat_n;
        grid_w_ = parameters.grid_w;
        grid_h_ = parameters.grid_h;

        _keyframe = keyframe;
        _camera_model = camera_model;

        ref_feat_to_track_.resize(_max_feat_n, CameraObs());
    }

    bool InBorder(int x, int y);

    status_t run();

    bool TrackMonocular(const std::pair<double, std::pair<cv::Mat, cv::Mat>>& input_image, std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool TrackStereo(const std::pair<double, std::pair<cv::Mat, cv::Mat>>& input_image, std::pair<double, std::vector<CameraObs>>& feature_observes);

    std::vector<bool> TrackFeatures(const cv::Mat image_left, const cv::Mat image_right, const std::vector<cv::Point2f> pts_to_track, std::vector<cv::Point2f>& pts_tracked);

    status_t OutlierRejection(const std::vector<CameraObs>& obs_prev, const std::vector<CameraObs>& obs_curr, std::vector<uchar>& inliers);

    std::pair<double, cv::Mat> ref_frame; // (ts_sec, image)
    std::pair<double, cv::Mat> cur_frame;
    std::vector<CameraObs> ref_feat_to_track_; // valid, (x, y)

private:
    bool is_first_frame_ = true;
    uint32_t frame_id = 0;
    uint32_t global_feature_id_ = 0;
    uint32_t width_, height_;
    uint32_t _max_feat_n;
    uint32_t grid_w_, grid_h_;
    KeyFrameType *_keyframe;
    std::shared_ptr<CameraModel> _camera_model;
    boost::posix_time::ptime frontend_rT, frontend_rT1, frontend_rT2, frontend_rT3, frontend_rT4;
};

#endif