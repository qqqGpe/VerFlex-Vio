/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-06-23 02:03:20
 * @LastEditors: pengen.gao gaope.hb@gmail.com
 * @LastEditTime: 2025-09-24 00:17:49
 * @FilePath: /catkin_ws/src/vio_backend/src/core/vioFrontend.h
 */
#include "camModel.h"
#include "parameter.h"
#include "sensorType.h"
#include <Eigen/Eigen>
#include <fmt/format.h>
#include <glog/logging.h>
#include <unistd.h>

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

    VioFrontend(const Parameter &params)
    {
        grid_w_ = params.grid_w;
        grid_h_ = params.grid_h;
        width_ = params.img_width;
        height_ = params.img_height;
        max_feat_n_ = params.max_feat_n;
        use_census_transform_ = params.use_census_transform;
        do_prediction_ = params.do_prediction;
        do_warp_klt_ = params.do_warp_klt;
    }

    bool TrackMonocular(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                        std::pair<double, std::vector<CameraObs>> &feature_observes);

    bool TrackStereo(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                     std::pair<double, std::vector<CameraObs>> &feature_observes);

    std::pair<double, cv::Mat> getImageWithFeatures() const { return image_with_features_; }

    KeyFrameStatus getKeyframeStatus() const { return keyframe_; }
    void setKeyframeStatus(KeyFrameStatus status) { keyframe_ = status; }

    std::vector<uint8_t> TrackFeatures(const cv::Mat image_left, const cv::Mat image_right, const Eigen::Matrix3d Rwi, const Eigen::Matrix3d Rwj,
                                       const bool is_stereo_tracking, const bool use_census_transform, const std::vector<cv::Point2f> pts_to_track,
                                       std::vector<cv::Point2f> &pts_tracked, const int warp_cam_id = LEFT_CAM);

  private:
    bool InBorder(int x, int y);

    // Perform 4-step circular tracking of existing features:
    // prev_left -> cur_left -> cur_right -> prev_right (validation)
    std::unordered_map<uint32_t, CameraObs> CircularTrackFeatures(double ts_sec, const cv::Mat &prev_image_left, const cv::Mat &cur_image_left,
                                                                  const cv::Mat &cur_image_right, const cv::Mat &prev_image_right,
                                                                  const Eigen::Matrix3d &Rwc,
                                                                  const std::unordered_map<uint32_t, CameraObs> &feature_umap,
                                                                  std::vector<uint32_t> obs_ids, const std::vector<cv::Point2f> &prev_left_pts);

    // Populate occupied grid with tracked features; resolve per-cell conflicts.
    // Candidates must be valid CameraObs pointers; losers are marked invalid.
    void ResolveGridConflicts(std::vector<CameraObs *> &candidates, int h_step, int w_step, std::vector<std::vector<uint8_t>> &occupied_mat);

    // Detect new stereo-verified Harris features and insert into grid/map
    void AddNewStereoFeatures(const cv::Mat &cur_image_left, const cv::Mat &cur_image_right, double ts_sec, int h_step, int w_step,
                              std::vector<std::vector<uint8_t>> &occupied_mat, std::unordered_map<uint32_t, CameraObs> &cur_feature_obs_umap);

    // Remove invalid features, back-project surviving ones, and populate output
    void FinalizeObservations(std::unordered_map<uint32_t, CameraObs> &cur_feature_obs_umap, double ts_sec,
                              std::pair<double, std::vector<CameraObs>> &feature_observes);

    // Render tracked feature points onto the left image and cache the result
    void VisualizeFeatureTracking(const cv::Mat &image, const std::pair<double, std::vector<CameraObs>> &feature_observes);

    // LK-track existing mono features, apply epipolar RANSAC, resolve per-cell conflicts
    // Populates occupied_mat for cells that have surviving features
    void MonoTrackAndFilter(std::vector<CameraObs> &cur_features, const cv::Mat &cur_image, const Eigen::Matrix3d &Rwc, int h_step, int w_step,
                            std::vector<std::vector<uint8_t>> &occupied_mat);

    // Detect new Harris features and fill empty grid cells up to max_feat_n_
    void AddNewMonoFeatures(const cv::Mat &image, int h_step, int w_step, std::vector<std::vector<uint8_t>> &occupied_mat,
                            std::vector<CameraObs> &cur_features);

    status_t MonoCheckEpipolarLine(const std::vector<CameraObs> &obs_prev, const std::vector<CameraObs> &obs_curr, std::vector<uint8_t> &inliers);

    bool do_prediction_ = false;
    bool do_warp_klt_ = false;
    bool use_census_transform_ = true;

    uint32_t frame_id = 0;
    uint32_t global_feature_id_ = 0;
    uint32_t width_, height_;
    uint32_t max_feat_n_;
    uint32_t grid_w_, grid_h_;
    KeyFrameStatus keyframe_ = KeyFrameStatus::kNone;
    std::pair<double, cv::Mat> image_with_features_ = {-1, cv::Mat()}; // (ts_sec, image)
    std::pair<double, std::vector<cv::Mat>> prev_images_ = {-1.0f, {}};
    std::pair<double, std::vector<CameraObs>> previous_observations_ = {-1.0f, {}};
    Eigen::Matrix3d R_ref = Eigen::Matrix3d::Identity(); // Rotation of the reference frame
};

#endif