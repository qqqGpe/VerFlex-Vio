#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <unistd.h>
#include <Eigen/Eigen>
#include <deque>
#include <memory>

#include "camModel.h"
#include "parameter.h"
#include "sensor_data.h"

// #include "datatypes.h"
#include <ros/node_handle.h>
#include <sensor_msgs/PointCloud.h>

#ifndef __FRONTEND__
#define __FRONTEND__

class VioFrontend
{
   public:
    enum status_t
    {
        STATUS_OK = 0,
        STATUS_NO_INPUT_DATA,
        STATUS_ERROR
    };

    VioFrontend(const Param params, std::shared_ptr<KeyFrameStatus> keyframe)
    {
        width_ = params.img_width;
        height_ = params.img_height;
        max_feat_n_ = params.max_feat_n;
        grid_w_ = params.grid_w;
        grid_h_ = params.grid_h;
        _keyframe = keyframe;
        do_prediction_ = params.frontend_prediction;
        ref_features_to_track_.resize(max_feat_n_, CameraObs());
    }

    bool InBorder(int x, int y);

    bool TrackMonocular(const std::pair<double, std::vector<cv::Mat>>& input_image,
                        const Eigen::Matrix3d Rwc,
                        const bool do_prediction_flag,
                        std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool TrackStereo(const std::pair<double, std::vector<cv::Mat>>& input_image,
                     const Eigen::Matrix3d Rwc,
                     const bool do_prediction_flag,
                     std::pair<double, std::vector<CameraObs>>& feature_observes);

    std::pair<double, cv::Mat> getImageWithFeatures() const
    {
        return image_with_features_;
    }

    std::vector<uint8_t> TrackFeatures(const cv::Mat image_left,
                                       const cv::Mat image_right,
                                       const Eigen::Matrix3d Rwi,
                                       const Eigen::Matrix3d Rwj,
                                       const bool is_stereo_tracking,
                                       const bool do_prediction_flag,
                                       const std::vector<cv::Point2f> pts_to_track,
                                       std::vector<cv::Point2f>& pts_tracked);

    status_t MonoCheckEpipolarLine(const std::vector<CameraObs>& obs_prev, const std::vector<CameraObs>& obs_curr, std::vector<uint8_t>& inliers);

    std::pair<double, cv::Mat> ref_frame;  // (ts_sec, image)
    std::pair<double, cv::Mat> cur_frame;
    std::vector<CameraObs> ref_features_to_track_;  // valid, (x, y)
    Eigen::Matrix3d R_ref = Eigen::Matrix3d::Identity();  // Rotation of the reference frame

   private:
    bool is_first_frame_ = true;
    bool do_prediction_ = false;
    uint32_t frame_id = 0;
    uint32_t global_feature_id_ = 0;
    uint32_t width_, height_;
    uint32_t max_feat_n_;
    uint32_t grid_w_, grid_h_;
    std::shared_ptr<KeyFrameStatus> _keyframe;
    std::pair<double, cv::Mat> image_with_features_ = {-1, cv::Mat()};  // (ts_sec, image)
    boost::posix_time::ptime frontend_rT, frontend_rT1, frontend_rT2, frontend_rT3, frontend_rT4;
};

#endif