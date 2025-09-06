#include <Eigen/Eigen>
#include <cv_bridge/cv_bridge.h>
#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <unistd.h>
#include <vio/nnFeatures.h>
#include <ros/node_handle.h>
#include <sensor_msgs/PointCloud.h>

#include "camModel.h"
#include "parameter.h"
#include "sensorType.h"

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

    VioFrontend(std::shared_ptr<ros::NodeHandle> &nh, const Param params, std::shared_ptr<KeyFrameStatus> keyframe);

    bool InBorder(int x, int y);

    bool TrackMonocular(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                        const bool do_prediction_flag, std::pair<double, std::vector<CameraObs>> &feature_observes);

    bool TrackStereo(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                     const bool do_prediction_flag, std::pair<double, std::vector<CameraObs>> &feature_observes);

    static cv::Mat ComputeCensusTransform(const cv::Mat &input);

    std::pair<double, cv::Mat> getImageWithFeatures() const { return image_with_features_; }

    std::vector<uint8_t> TrackFeatures(const cv::Mat image_left, const cv::Mat image_right, const Eigen::Matrix3d Rwi,
                                       const Eigen::Matrix3d Rwj, const bool is_stereo_tracking,
                                       const bool do_prediction_flag, const bool use_census_transform,
                                       const std::vector<cv::Point2f> pts_to_track,
                                       std::vector<cv::Point2f> &pts_tracked);

    std::vector<uint8_t> TrackNNFeatures(const cv::Mat image_left, const cv::Mat image_right,
                                         const std::vector<CameraObs> &ref_features_to_track,
                                         const std::map<uint32_t, cv::Mat> ref_feature_descriptors_map_,
                                         std::vector<cv::Point2f> &pts_tracked);

    status_t MonoCheckEpipolarLine(const std::vector<CameraObs> &obs_prev, const std::vector<CameraObs> &obs_curr,
                                   std::vector<uint8_t> &inliers);

    std::pair<double, cv::Mat> ref_frame;                           // (ts_sec, image)
    std::pair<double, cv::Mat> cur_frame;                           // (ts_sec, image)
    std::vector<CameraObs> ref_features_to_track_;                  // valid, (x, y)
    std::map<uint32_t, cv::Mat> ref_feature_descriptors_map_;       // For nn feature tracking
    Eigen::Matrix3d R_ref = Eigen::Matrix3d::Identity();            // Rotation of the reference frame

  private:
    static uint8_t ComputeCensusByte(const cv::Mat &image, int x, int y);

    bool ExtractFeatures(const cv::Mat& image, std::vector<cv::Point2f>& keypoints, std::vector<cv::Mat>& descriptors);

    std::shared_ptr<ros::NodeHandle> nh_;
    ros::ServiceClient client_;
    bool use_nn_feature_ = false;
    bool use_census_transform_ = true;
    bool is_first_frame_ = true;
    bool do_prediction_ = false;
    uint32_t frame_id = 0;
    uint32_t global_feature_id_ = 0;
    uint32_t width_, height_;
    uint32_t max_feat_n_;
    uint32_t grid_w_, grid_h_;
    std::shared_ptr<KeyFrameStatus> _keyframe;
    std::pair<double, cv::Mat> image_with_features_ = {-1, cv::Mat()}; // (ts_sec, image)
};

#endif