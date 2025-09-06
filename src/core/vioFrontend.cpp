#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Header.h>

#include "camModel.h"
#include "vioFrontend.h"
#include "opencv2/core/mat.hpp"
#include "sensor_data.h"
#include "utils.h"

namespace
{
constexpr double kPixelErrorThreshold = 1.0;
constexpr double kCircularTrackPixelErrorThres = 1.0;
} // namespace

double CalcPixelDistance(const cv::Point2d &pt1, const cv::Point2d &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool VioFrontend::InBorder(int x, int y)
{
    return (x >= 0 && x < width_) && (y >= 0 && y < height_);
}

VioFrontend::VioFrontend(std::shared_ptr<ros::NodeHandle>& nh, const Param params, std::shared_ptr<KeyFrameStatus> keyframe)
{
    nh_ = nh;
    _keyframe = keyframe;
    grid_w_ = params.grid_w;
    grid_h_ = params.grid_h;
    width_ = params.img_width;
    height_ = params.img_height;
    max_feat_n_ = params.max_feat_n;
    use_nn_feature_ = params.use_nn_feature;
    use_census_transform_ = params.use_census_transform;
    do_prediction_ = params.frontend_prediction;
    ref_features_to_track_.resize(max_feat_n_, CameraObs());
    if (use_nn_feature_)
    {
        client_ = nh_->serviceClient<vio::nnFeatures>("/extract_features");
        if (client_.exists())
        {
            LOG(INFO) << "SuperPoint service connected successfully";
        }
        else
        {
            LOG(ERROR) << "nn Feature service not available";
            use_nn_feature_ = false;
            exit(1);
        }
    }
}

void HomographyRansac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr,
                      std::vector<uchar> *inliers)
{
    assert(points_prev.size() == points_curr.size());
    *inliers = std::vector<uchar>(points_prev.size(), 0);
    cv::findHomography(points_prev, points_curr, *inliers, cv::RANSAC, 3);
}

void EpipolarRansac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr,
                    std::vector<uchar> *inliers)
{
    constexpr double kEssentialMatrixProb = 0.95;
    constexpr uint32_t kMinRequiredPointsForEpipolarRansac = 8;
    constexpr double kEssentialThres = 1.0; // 1 pixel threshold for essential matrix

    assert(points_prev.size() == points_curr.size());
    *inliers = std::vector<uchar>(points_prev.size(), 0);

    if (points_prev.size() < kMinRequiredPointsForEpipolarRansac ||
        points_curr.size() < kMinRequiredPointsForEpipolarRansac)
    {
        LOG(ERROR) << "Not enough points for epipolar RANSAC";
        return;
    }

    Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
    cv::Mat K_mat;
    cv::eigen2cv(K, K_mat);
    cv::Mat essential_matrix =
        cv::findEssentialMat(points_prev, points_curr, K_mat, cv::RANSAC, kEssentialMatrixProb, kEssentialThres);

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> essential_mat;
    cv::cv2eigen(essential_matrix, essential_mat);
    for (int i = 0; i < points_prev.size(); i++)
    {
        Eigen::Vector3d p_prev(points_prev[i].x, points_prev[i].y, 1.0);
        Eigen::Vector3d p_curr(points_curr[i].x, points_curr[i].y, 1.0);
        p_prev = K.inverse() * p_prev.eval();
        p_curr = K.inverse() * p_curr.eval();
        double dist = p_prev.transpose() * essential_mat * p_curr;
        if (dist < kEssentialThres)
        {
            (*inliers)[i] = 1;
        }
    }
}

uint8_t VioFrontend::ComputeCensusByte(const cv::Mat &image, int x, int y)
{
    uint8_t census = 0;
    const int center = image.at<uint8_t>(y, x);

    // Compare center pixel with its 8 neighbors in a 3x3 window
    int bit = 0;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0)
            {
                continue; // Skip center pixel
            }

            int nx = x + dx;
            int ny = y + dy;

            // Check if neighbor is within image bounds
            if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows)
            {
                // Set bit to 1 if neighbor pixel is darker than center
                if (image.at<uint8_t>(ny, nx) < center)
                {
                    census |= (1 << bit);
                }
            }
            bit++;
        }
    }
    return census;
}

cv::Mat VioFrontend::ComputeCensusTransform(const cv::Mat &input)
{
    // Ensure input is grayscale
    CV_Assert(input.type() == CV_8UC1);

    // Initialize output census image
    cv::Mat census = cv::Mat::zeros(input.size(), CV_8UC1);

    // Process all pixels except border
    // Since we need 3x3 window, skip first and last row/column
    for (int y = 1; y < input.rows - 1; y++)
    {
        for (int x = 1; x < input.cols - 1; x++)
        {
            census.at<uint8_t>(y, x) = ComputeCensusByte(input, x, y);
        }
    }

    return census;
}

VioFrontend::status_t VioFrontend::MonoCheckEpipolarLine(const std::vector<CameraObs> &obs_prev,
                                                         const std::vector<CameraObs> &obs_curr,
                                                         std::vector<uint8_t> &inliers)
{
    constexpr uint32_t kMinObsSizeForEpipolarCheck = 10;
    if (obs_prev.size() != obs_curr.size())
    {
        LOG(ERROR) << "obs_prev.size() != obs_curr.size()";
        return STATUS_ERROR;
    }

    if (obs_prev.size() < kMinObsSizeForEpipolarCheck)
    {
        LOG(WARNING) << "Not enough observations for epipolar check: " << obs_prev.size();
        inliers.assign(obs_prev.size(), 0);
        return STATUS_OK;
    }

    std::vector<cv::Point2f> points_prev, points_curr;
    for (size_t i = 0; i < obs_prev.size(); i++)
    {
        if (obs_prev[i].valid && obs_curr[i].valid)
        {
            points_prev.emplace_back(obs_prev[i].uv.at(LEFT_CAM).x(), obs_prev[i].uv.at(LEFT_CAM).y());
            points_curr.emplace_back(obs_curr[i].uv.at(LEFT_CAM).x(), obs_curr[i].uv.at(LEFT_CAM).y());
        }
    }
    std::vector<uchar> inliers_epipolar;
    EpipolarRansac(points_prev, points_curr, &inliers_epipolar);
    inliers = inliers_epipolar;
    return STATUS_OK;
}

std::vector<uint8_t> VioFrontend::TrackNNFeatures(const cv::Mat image_left, const cv::Mat image_right,
                                                  const std::vector<CameraObs> &ref_features_to_track,
                                                  const std::map<uint32_t, cv::Mat> ref_feature_descriptors_map_,
                                                  std::vector<cv::Point2f> &pts_tracked)
{
    std::vector<uint8_t> status;
    std::vector<cv::Point2f> cur_feature_extracted;;
    std::vector<cv::Mat> cur_feature_descriptors;
    ExtractFeatures(image_right, cur_feature_extracted, cur_feature_descriptors);

    pts_tracked.clear();
    if (ref_features_to_track.empty() || ref_feature_descriptors_map_.empty() || cur_feature_extracted.empty() || cur_feature_descriptors.empty())
    {
        // 没有可用的特征或描述子，直接返回
        status.resize(ref_features_to_track.size(), 0);
        return status;
    }

    // 构建参考帧的描述子矩阵
    std::vector<cv::Mat> ref_descriptors_vec;
    std::vector<uint32_t> ref_feat_ids;
    for (const auto& obs : ref_features_to_track)
    {
        if (!obs.valid) {
            ref_descriptors_vec.push_back(cv::Mat()); // 占位
            ref_feat_ids.push_back(obs.feat_id);
            continue;
        }
        auto it = ref_feature_descriptors_map_.find(obs.feat_id);
        if (it != ref_feature_descriptors_map_.end())
        {
            ref_descriptors_vec.push_back(it->second);
        }
        else
        {
            ref_descriptors_vec.push_back(cv::Mat());
        }
        ref_feat_ids.push_back(obs.feat_id);
    }

    // 将描述子vector转为cv::Mat
    cv::Mat ref_descriptors, cur_descriptors;
    if (!ref_descriptors_vec.empty() && !ref_descriptors_vec[0].empty())
    {
        cv::vconcat(ref_descriptors_vec, ref_descriptors);
    }

    if (!cur_feature_descriptors.empty() && !cur_feature_descriptors[0].empty())
    {
        cv::vconcat(cur_feature_descriptors, cur_descriptors);
    }

    if (ref_descriptors.empty() || cur_descriptors.empty())
    {
        status.resize(ref_features_to_track.size(), 0);
        return status;
    }

    // 使用BFMatcher进行匹配
    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(ref_descriptors, cur_descriptors, matches);

    // 初始化status为0
    status.resize(ref_features_to_track.size(), 0);
    pts_tracked.resize(ref_features_to_track.size(), cv::Point2f(-1, -1));

    // 记录每个ref index的最佳匹配
    for (const auto& match : matches)
    {
        int ref_idx = match.queryIdx;
        int cur_idx = match.trainIdx;
        if (ref_idx >= 0 && ref_idx < ref_features_to_track.size() && cur_idx >= 0 && cur_idx < cur_feature_extracted.size())
        {
            if (ref_features_to_track[ref_idx].valid)
            {
                status[ref_idx] = 1;
                pts_tracked[ref_idx] = cur_feature_extracted[cur_idx];
            }
        }
    }

    return status;
}

std::vector<uint8_t> VioFrontend::TrackFeatures(const cv::Mat image_left, const cv::Mat image_right,
                                                const Eigen::Matrix3d Rwi, const Eigen::Matrix3d Rwj,
                                                const bool is_stereo_tracking, const bool do_prediction_flag,
                                                const bool use_census_transform,
                                                const std::vector<cv::Point2f> pts_to_track,
                                                std::vector<cv::Point2f> &pts_tracked)
{
    constexpr double kMaxAllowedRelativePoseAngle = 10; // In degrees

    std::vector<uint8_t> status;
    std::vector<uchar> forward_status, backward_status;
    std::vector<cv::Point2f> reverse_pts;
    std::vector<float> err;

    cv::Mat image_left_gray, image_right_gray;
    if (use_census_transform)
    {
        image_left_gray = ComputeCensusTransform(image_left);
        image_right_gray = ComputeCensusTransform(image_right);
    }
    else
    {
        image_left_gray = image_left.clone();
        image_right_gray = image_right.clone();
    }

    if (pts_to_track.empty())
    {
        return status;
    }

    if (do_prediction_flag && !is_stereo_tracking)
    {
        Eigen::Matrix3d Rij = Rwi.transpose() * Rwj;
        Eigen::AngleAxisd angle_axis(Rij);
        if (angle_axis.angle() > kMaxAllowedRelativePoseAngle / 180.0 * M_PI)
        {
            // std::cout << "Warning: Relative pose angle is too large: " << angle_axis.angle() * 180.0 / M_PI << "
            // degrees." << std::endl;
            Rij.setIdentity();
        }

        Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
        Eigen::Matrix3d H_eigen = K * Rij * K.inverse();
        cv::Mat H_cv;
        cv::eigen2cv(H_eigen, H_cv);
        cv::Mat image_right_warped;
        cv::warpPerspective(image_right_gray, image_right_warped, H_cv, image_right_gray.size(), cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT);

        // forward tracking
        cv::calcOpticalFlowPyrLK(image_left_gray, image_right_warped, pts_to_track, pts_tracked, forward_status, err,
                                 cv::Size(21, 21), 4,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

        // backward tracking
        reverse_pts = pts_tracked;
        cv::calcOpticalFlowPyrLK(
            image_right_warped, image_left_gray, pts_tracked, reverse_pts, backward_status, err, cv::Size(21, 21), 4,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

        cv::perspectiveTransform(pts_tracked, pts_tracked, H_cv.inv()); // Remember to devide point.z
    }
    else
    {
        // forward tracking
        cv::calcOpticalFlowPyrLK(image_left_gray, image_right_gray, pts_to_track, pts_tracked, forward_status, err);

        // backward tracking
        reverse_pts = pts_tracked;
        cv::calcOpticalFlowPyrLK(
            image_right_gray, image_left_gray, pts_tracked, reverse_pts, backward_status, err, cv::Size(21, 21), 4,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    }

    // double check if the tracked point is out of range
    for (int i = 0; i < forward_status.size(); i++)
    {
        if (forward_status[i] && backward_status[i] &&
            CalcPixelDistance(pts_to_track[i], reverse_pts[i]) < kPixelErrorThreshold &&
            InBorder(pts_tracked[i].x, pts_tracked[i].y))
        {
            status.push_back(true);
        }
        else
        {
            status.push_back(false);
        }
    }
    return status;
}

bool VioFrontend::TrackStereo(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                              const bool do_prediction_flag,
                              std::pair<double, std::vector<CameraObs>> &feature_observes)
{
    static bool is_first_entry = true;
    double ts_sec = input_image.first;
    static std::pair<double, std::vector<cv::Mat>> prev_images = {-1, {cv::Mat(), cv::Mat()}};
    static std::pair<double, std::vector<CameraObs>> previous_observations;

    cv::Mat cur_image_left = input_image.second[LEFT_CAM];
    cv::Mat cur_image_right = input_image.second[RIGHT_CAM];
    cv::Mat prev_image_left = prev_images.second[LEFT_CAM];
    cv::Mat prev_image_right = prev_images.second[RIGHT_CAM];

    const int h_step = height_ / grid_h_;
    const int w_step = width_ / grid_w_;
    std::vector<std::vector<bool>> occupied_grid(grid_h_ + 1, std::vector<bool>(grid_w_ + 1, false));
    std::vector<std::vector<int>> occupied_grid_feature_id(grid_h_ + 1, std::vector<int>(grid_w_ + 1, -1));

    std::vector<cv::Point2f> prev_left_pts;
    std::vector<cv::Point2d> prev_right_pts;
    std::vector<cv::Point2d> cur_left_pts;
    std::vector<cv::Point2d> cur_right_pts;

    std::vector<uint32_t> obs_ids;
    std::unordered_map<uint32_t, CameraObs> feature_umap;
    std::unordered_map<uint32_t, CameraObs> cur_feature_obs_umap; // {feature_id, CameraObs}

    if (prev_images.first > 0)
    {
        for (int i = 0; i < previous_observations.second.size(); i++)
        {
            CameraObs obs = previous_observations.second[i];
            if (!obs.valid)
            {
                continue;
            }
            feature_umap.emplace(obs.feat_id, obs);
            obs_ids.push_back(obs.feat_id);
            prev_left_pts.emplace_back(obs.uv[LEFT_CAM].x(), obs.uv[LEFT_CAM].y());
        }

        // step1: track previous left-cam visual points to current left-cam visual points
        std::vector<cv::Point2f> prev_l_to_cur_l_tracked;
        std::unordered_map<uint32_t, cv::Point2d> prev_l_to_cur_l_tracked_umap;
        auto pre_left_to_cur_left_status =
            TrackFeatures(prev_image_left, cur_image_left, R_ref, Rwc, false, do_prediction_flag, use_census_transform_,
                          prev_left_pts, prev_l_to_cur_l_tracked);
        int idx = 0;
        std::vector<uint32_t>::iterator id_iter = obs_ids.begin();
        for (auto it = prev_l_to_cur_l_tracked.begin(); it != prev_l_to_cur_l_tracked.end(); idx++)
        {
            if (pre_left_to_cur_left_status[idx] == false)
            {
                it = prev_l_to_cur_l_tracked.erase(it);
                id_iter = obs_ids.erase(id_iter);
                continue;
            }
            prev_l_to_cur_l_tracked_umap.emplace(*id_iter, *it);
            ++it;
            ++id_iter;
        }

        // step2: track current left-cam visual points to current right-cam visual points
        std::vector<cv::Point2f> cur_l_to_cur_r_tracked;
        std::unordered_map<uint32_t, cv::Point2d> cur_l_to_cur_r_tracked_umap;
        Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
        auto cur_left_to_cur_right_status =
            TrackFeatures(cur_image_left, cur_image_right, I3x3, I3x3, true, false, use_census_transform_,
                          prev_l_to_cur_l_tracked, cur_l_to_cur_r_tracked);
        idx = 0;
        id_iter = obs_ids.begin();
        for (auto it = cur_l_to_cur_r_tracked.begin(); it != cur_l_to_cur_r_tracked.end(); idx++)
        {
            if (cur_left_to_cur_right_status[idx] == false)
            {
                it = cur_l_to_cur_r_tracked.erase(it);
                id_iter = obs_ids.erase(id_iter);
                continue;
            }
            cur_l_to_cur_r_tracked_umap.emplace(*id_iter, *it);
            ++it;
            ++id_iter;
        }

        // step3: track current right-cam visual points to previous right-cam visual points
        std::vector<cv::Point2f> cur_r_to_pre_r_pts_tracked;
        std::unordered_map<uint32_t, cv::Point2d> cur_r_to_pre_r_pts_tracked_umap;
        Eigen::Matrix3d Rwr_cur = Rwc * CamModel::getInstance().Rlr();
        Eigen::Matrix3d Rwr_prev = R_ref * CamModel::getInstance().Rlr();
        auto cur_right_to_prev_right_status =
            TrackFeatures(cur_image_right, prev_image_right, Rwr_cur, Rwr_prev, false, do_prediction_flag,
                          use_census_transform_, cur_l_to_cur_r_tracked, cur_r_to_pre_r_pts_tracked);
        idx = 0;
        id_iter = obs_ids.begin();
        for (auto it = cur_r_to_pre_r_pts_tracked.begin(); it != cur_r_to_pre_r_pts_tracked.end(); idx++)
        {
            if (cur_right_to_prev_right_status[idx] == false)
            {
                it = cur_r_to_pre_r_pts_tracked.erase(it);
                id_iter = obs_ids.erase(id_iter);
                continue;
            }
            cur_r_to_pre_r_pts_tracked_umap.emplace(*id_iter, *it);
            ++it;
            ++id_iter;
        }

        // step4: compare circular-tracked visual points with previous right-cam visual points
        for (int i = 0; i < obs_ids.size(); i++)
        {
            uint32_t feature_id = obs_ids[i];
            cv::Point2d circular_tracked_point = cur_r_to_pre_r_pts_tracked[i];
            CameraObs obs = feature_umap[feature_id];
            cv::Point2d prev_obs_right(obs.uv[RIGHT_CAM].x(), obs.uv[RIGHT_CAM].y());
            if (CalcPixelDistance(prev_obs_right, circular_tracked_point) < kCircularTrackPixelErrorThres)
            {
                cv::Point2d cur_obs_l = prev_l_to_cur_l_tracked_umap[feature_id];
                cv::Point2d cur_obs_r = cur_l_to_cur_r_tracked_umap[feature_id];
                CameraObs cur_obs(ts_sec, cur_obs_l.x, cur_obs_l.y, cur_obs_r.x, cur_obs_r.y);
                cur_obs.feat_id = feature_id;
                cur_feature_obs_umap.emplace(obs.feat_id, cur_obs);
            }
        }
    }

    // Add current feature observations to grid map
    for (auto &[feature_id, obs] : cur_feature_obs_umap)
    {
        int h = obs.uv[LEFT_CAM].y() / h_step;
        int w = obs.uv[LEFT_CAM].x() / w_step;
        if (occupied_grid[h][w] == false)
        {
            occupied_grid[h][w] = true;
            occupied_grid_feature_id[h][w] = feature_id;
            obs.valid = true;
            obs.obs_times_n++;
        }
        else
        {
            uint32_t obs_candidate_feature_id = occupied_grid_feature_id[h][w];
            CameraObs &obs_candidate = feature_umap[obs_candidate_feature_id];
            if (obs.obs_times_n < obs_candidate.obs_times_n)
            {
                obs.setInvalid();
            }
            else
            {
                obs_candidate.setInvalid();
                occupied_grid_feature_id[h][w] = feature_id;
                obs.valid = true;
                obs.obs_times_n++;
            }
        }
    }

    // Add new features to current observations if keyframe or first frame
    bool switch_keyframe = (*_keyframe != KeyFrameStatus::kNone) || is_first_entry;
    if (switch_keyframe)
    {
        std::vector<cv::Point2f> harris_new, harris_tracked;
        std::vector<CameraObs> cur_stereo_ok_features;
        cv::goodFeaturesToTrack(cur_image_left, harris_new, max_feat_n_, 0.01, 20);
        Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
        auto status = TrackFeatures(cur_image_left, cur_image_right, I3x3, I3x3, true, false, use_census_transform_,
                                    harris_new, harris_tracked);
        for (int i = 0; i < status.size(); i++)
        {
            if (status[i] == true)
            {
                CameraObs obs(ts_sec, harris_new[i].x, harris_new[i].y, harris_tracked[i].x, harris_tracked[i].y);
                cur_stereo_ok_features.emplace_back(obs);
            }
        }

        for (auto &feature_new : cur_stereo_ok_features)
        {
            int h = feature_new.uv[LEFT_CAM].y() / h_step;
            int w = feature_new.uv[LEFT_CAM].x() / w_step;
            if (occupied_grid[h][w] == false)
            {
                feature_new.valid = true;
                feature_new.feat_id = global_feature_id_++;
                feature_new.obs_times_n = 1;
                occupied_grid[h][w] = true;
                occupied_grid_feature_id[h][w] = feature_new.feat_id;
                cur_feature_obs_umap.insert_or_assign(feature_new.feat_id, feature_new);
            }
        }
    }

    // Clear invalid features and calculate normalized coordinates
    feature_observes.first = ts_sec;
    feature_observes.second.clear();
    for (auto it = cur_feature_obs_umap.begin(); it != cur_feature_obs_umap.end();)
    {
        if (!it->second.valid)
        {
            it = cur_feature_obs_umap.erase(it);
        }
        else
        {
            CamModel::getInstance().back_project(it->second);
            feature_observes.second.push_back(it->second);
            ++it;
        }
    }

    if (switch_keyframe)
    {
        is_first_entry = false;
        prev_images = input_image;
        previous_observations = feature_observes;
        R_ref = Rwc;
    }
    cv::Mat image_to_show;
    Utils::visualize_feature_tracking_results(input_image.second[LEFT_CAM].clone(), feature_observes, 1,
                                              &image_to_show);
    image_with_features_ = std::make_pair(ts_sec, image_to_show);
    return true;
}

bool VioFrontend::TrackMonocular(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                                 const bool do_prediction_flag,
                                 std::pair<double, std::vector<CameraObs>> &feature_observes)
{
    const double ts_sec = input_image.first;
    cur_frame = std::make_pair(ts_sec, input_image.second[0].clone());
    std::vector<CameraObs> cur_features_to_track = ref_features_to_track_;
    const int h_step = height_ / grid_h_;
    const int w_step = width_ / grid_w_;
    std::vector<std::vector<uint8_t>> occupied_mat(grid_h_ + 1, std::vector<uint8_t>(grid_w_ + 1, 0));

    // For nn feature tracking
    std::map<uint32_t, cv::Mat> cur_feature_descriptors_map;
    if (use_nn_feature_)
    {
        cur_feature_descriptors_map = ref_feature_descriptors_map_;
    }

    if (!is_first_frame_)
    {
        if (cur_frame.first <= ref_frame.first)
        {
            LOG(WARNING) << "invalid image timestamp!";
            return STATUS_ERROR;
        }

        std::vector<cv::Point2f> prev_pts, curr_pts;
        std::vector<int> feat_idx;

        for (int idx = 0; idx < ref_features_to_track_.size(); idx++)
        {
            if (ref_features_to_track_[idx].valid)
            {
                feat_idx.push_back(idx);
                prev_pts.emplace_back(ref_features_to_track_[idx].uv[LEFT_CAM].x(),
                                      ref_features_to_track_[idx].uv[LEFT_CAM].y());
            }
        }

        // Circular track
        std::vector<uint8_t> status;
        if (use_nn_feature_)
        {
            status = TrackNNFeatures(ref_frame.second, cur_frame.second, ref_features_to_track_,
                                     ref_feature_descriptors_map_, curr_pts);

        }
        else
        {
            status = TrackFeatures(ref_frame.second, cur_frame.second, R_ref, Rwc, false, do_prediction_flag,
                                   use_census_transform_, prev_pts, curr_pts);
        }

        for (int i = 0; i < status.size(); i++)
        {
            const int idx = feat_idx[i];
            if (status[i])
            {
                cur_features_to_track[idx].uv[LEFT_CAM] = Eigen::Vector2d(curr_pts[i].x, curr_pts[i].y);
                cur_features_to_track[idx].obs_times_n++;
            }
            else
            {
                cur_features_to_track[idx].setInvalid();
            }
        }

        // Epipolar RANSAC to reject outliers
        std::vector<uint8_t> inliers;
        MonoCheckEpipolarLine(ref_features_to_track_, cur_features_to_track, inliers);
        for (int i = 0; i < ref_features_to_track_.size(); i++)
        {
            if (inliers[i] == 0)
            {
                cur_features_to_track[i].setInvalid();
            }
        }

        // Remove Conflicting Observations
        std::vector<std::vector<CameraObs *>> occupied_feat(grid_h_ + 1, std::vector<CameraObs *>(grid_w_ + 1, nullptr));
        for (size_t i = 0; i < cur_features_to_track.size(); i++)
        {
            if (!cur_features_to_track[i].valid)
            {
                continue;
            }
            const int32_t h = cur_features_to_track[i].uv[LEFT_CAM].y() / h_step;
            const int32_t w = cur_features_to_track[i].uv[LEFT_CAM].x() / w_step;
            if (occupied_mat[h][w] == 0)
            {
                occupied_mat[h][w] = 1;
                occupied_feat[h][w] = &(cur_features_to_track[i]);
            }
            else
            {
                // Delete features with less obs times
                if (cur_features_to_track[i].obs_times_n > occupied_feat[h][w]->obs_times_n)
                {
                    occupied_feat[h][w]->setInvalid();
                    occupied_feat[h][w] = &(cur_features_to_track[i]);
                }
                else
                {
                    cur_features_to_track[i].setInvalid();
                }
            }
        }
    }

    // Delate invalid features
    for (auto it = cur_features_to_track.begin(); it != cur_features_to_track.end();)
    {
        if (!it->valid)
        {
            uint32_t feat_id_erased = it->feat_id;
            it = cur_features_to_track.erase(it);
            if (use_nn_feature_)
            {
                cur_feature_descriptors_map.erase(feat_id_erased);
            }
            continue;
        }
        it = std::next(it);
    }

    // Add new features if keyframe
    if (*_keyframe != KeyFrameStatus::kNone || is_first_frame_)
    {
        std::deque<CameraObs> new_features;
        std::deque<cv::Mat> new_feature_descriptors;
        std::vector<cv::Point2f> candidate_keypoints;
        std::vector<cv::Mat> candidate_feature_descriptors;
        ExtractFeatures(cur_frame.second, candidate_keypoints, candidate_feature_descriptors);
        // cv::goodFeaturesToTrack(cur_frame.second, candidate_keypoints, max_feat_n_, 0.01, 30);
        for (int i = 0; i < candidate_keypoints.size(); i++)
        {
            int h = candidate_keypoints[i].y / h_step;
            int w = candidate_keypoints[i].x / w_step;
            if (occupied_mat[h][w] == 0)
            {
                CameraObs feature;
                feature.feat_id = global_feature_id_++;
                feature.obs_times_n = 1;
                feature.uv[LEFT_CAM] = Eigen::Vector2d(candidate_keypoints[i].x, candidate_keypoints[i].y);
                new_features.push_back(feature);
                occupied_mat[h][w] = 1;

                if (use_nn_feature_ && !candidate_feature_descriptors.empty())
                {
                    new_feature_descriptors.push_back(candidate_feature_descriptors[i]);
                }
            }
        }

        while (!new_features.empty() && cur_features_to_track.size() < max_feat_n_)
        {
            const uint32_t feat_new_id = new_features.front().feat_id;
            cur_features_to_track.push_back(new_features.front());
            cur_features_to_track.back().valid = true;
            new_features.pop_front();

            if (use_nn_feature_ && !new_feature_descriptors.empty())
            {
                cur_feature_descriptors_map.insert_or_assign(feat_new_id, new_feature_descriptors.front());
                new_feature_descriptors.pop_front();
            }
        }

        ref_frame = cur_frame;
        ref_features_to_track_ = cur_features_to_track;
        ref_feature_descriptors_map_ = cur_feature_descriptors_map;
        R_ref = Rwc;
    }

    // Back project from pixel coordinates to normalized coordinates
    for (auto &obs : cur_features_to_track)
    {
        obs.ts_sec = ts_sec;
        CamModel::getInstance().back_project(obs);
    }

    feature_observes = std::make_pair(ts_sec, cur_features_to_track);
    cv::Mat image_to_show;
    // Utils::visualize_feature_tracking_results(input_image.second[LEFT_CAM].clone(), feature_observes, 1,
    //                                           &image_to_show);
    image_with_features_ = std::make_pair(ts_sec, image_to_show);
    is_first_frame_ = false;
    return true;
}

bool VioFrontend::ExtractFeatures(const cv::Mat& image, std::vector<cv::Point2f>& keypoints, std::vector<cv::Mat>& descriptors)
{
    keypoints.clear();
    descriptors.clear();

    if (use_nn_feature_)
    {
        vio::nnFeatures srv;
        srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
        if (client_.call(srv))
        {
            std::cout << "num_keypoints: " << srv.response.num_keypoints << std::endl;
            std::cout << "descriptors size: " << srv.response.descriptors.size() << std::endl;
            for (size_t i = 0; i < srv.response.num_keypoints; i++)
            {
                keypoints.emplace_back(srv.response.keypoints[2 * i], srv.response.keypoints[2 * i + 1]);
                descriptors.emplace_back(cv::Mat(1, 256, CV_32F, &srv.response.descriptors.data()[i * 256]).clone());
            }
            return true;
        }
        else
        {
            LOG(ERROR) << "Failed to call nnFeatures service";
            return false;
        }
    }
    else
    {
        cv::goodFeaturesToTrack(cur_frame.second, keypoints, max_feat_n_, 0.01, 30);
        return true;
    }
}