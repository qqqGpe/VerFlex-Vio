#include "vioFrontend.h"
#include "camModel.h"
#include "opencv2/core/mat.hpp"
#include "sensorType.h"
#include "utils.h"
#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace
{
constexpr double kPixelErrorThreshold = 1.0;
constexpr double kCircularTrackPixelErrorThres = 1.0;
} // namespace

static double CalcPixelDistance(const cv::Point2d &pt1, const cv::Point2d &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool VioFrontend::InBorder(int x, int y)
{
    if (x < 0 || x >= width_ || y < 0 || y >= height_)
    {
        return false;
    }
    else
    {
        return true;
    }
}

static void HomographyRansac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr, std::vector<uchar> &inliers)
{
    assert(points_prev.size() == points_curr.size());
    inliers = std::vector<uchar>(points_prev.size(), 0);
    cv::findHomography(points_prev, points_curr, inliers, cv::RANSAC, 3);
}

static void EpipolarRansac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr, std::vector<uchar> &inliers)
{
    constexpr double kEssentialMatrixProb = 0.95;
    constexpr uint32_t kMinRequiredPointsForEpipolarRansac = 8;
    constexpr double kEssentialThres = 1.0; // 1 pixel threshold for essential matrix

    assert(points_prev.size() == points_curr.size());
    inliers = std::vector<uchar>(points_prev.size(), 0);

    if (points_prev.size() < kMinRequiredPointsForEpipolarRansac || points_curr.size() < kMinRequiredPointsForEpipolarRansac)
    {
        LOG_ERROR("Not enough points for epipolar RANSAC");
        return;
    }

    Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
    cv::Mat K_mat;
    cv::eigen2cv(K, K_mat);
    cv::Mat essential_matrix = cv::findEssentialMat(points_prev, points_curr, K_mat, cv::RANSAC, kEssentialMatrixProb, kEssentialThres);

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
            inliers[i] = 1;
        }
    }
}

static uint8_t ComputeCensusByte(const cv::Mat &image, int x, int y)
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

static cv::Mat ComputeCensusTransform(const cv::Mat &input)
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

VioFrontend::status_t VioFrontend::MonoCheckEpipolarLine(const std::vector<CameraObs> &obs_prev, const std::vector<CameraObs> &obs_curr,
                                                         std::vector<uint8_t> &inliers)
{
    constexpr uint32_t kMinObsSizeForEpipolarCheck = 10;
    if (obs_prev.size() != obs_curr.size())
    {
        LOG_ERROR("obs_prev.size() != obs_curr.size()");
        return STATUS_ERROR;
    }

    if (obs_prev.size() < kMinObsSizeForEpipolarCheck)
    {
        LOG_WARN("Not enough observations for epipolar check: {}", obs_prev.size());
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
    EpipolarRansac(points_prev, points_curr, inliers_epipolar);
    inliers = inliers_epipolar;
    return STATUS_OK;
}

std::vector<uint8_t> VioFrontend::TrackFeatures(const cv::Mat image_left, const cv::Mat image_right, const Eigen::Matrix3d Rwi,
                                                const Eigen::Matrix3d Rwj, const bool is_stereo_tracking, const bool use_census_transform,
                                                const std::vector<cv::Point2f> pts_to_track, std::vector<cv::Point2f> &pts_tracked)
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
        LOG_WARN("No points to track");
        return status;
    }

    if (do_warp_klt_ && !is_stereo_tracking)
    {
        Eigen::Matrix3d Rij = Rwi.transpose() * Rwj;
        Eigen::AngleAxisd angle_axis(Rij);
        if (angle_axis.angle() > kMaxAllowedRelativePoseAngle / 180.0 * M_PI)
        {
            Rij.setIdentity();
        }

        Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
        Eigen::Matrix3d H_eigen = K * Rij * K.inverse();
        cv::Mat H_cv;
        cv::eigen2cv(H_eigen, H_cv);
        cv::Mat image_right_warped;
        cv::warpPerspective(image_right_gray, image_right_warped, H_cv, image_right_gray.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        // forward tracking
        cv::calcOpticalFlowPyrLK(image_left_gray, image_right_warped, pts_to_track, pts_tracked, forward_status, err, cv::Size(21, 21), 4,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

        // backward tracking
        reverse_pts = pts_tracked;
        cv::calcOpticalFlowPyrLK(image_right_warped, image_left_gray, pts_tracked, reverse_pts, backward_status, err, cv::Size(21, 21), 4,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

        cv::perspectiveTransform(pts_tracked, pts_tracked, H_cv.inv()); // Remember to devide point.z
    }
    else
    {
        // forward tracking
        cv::calcOpticalFlowPyrLK(image_left_gray, image_right_gray, pts_to_track, pts_tracked, forward_status, err);

        // backward tracking
        reverse_pts = pts_tracked;
        cv::calcOpticalFlowPyrLK(image_right_gray, image_left_gray, pts_tracked, reverse_pts, backward_status, err, cv::Size(21, 21), 4,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    }

    // double check if the tracked point is out of range
    for (int i = 0; i < forward_status.size(); i++)
    {
        if (forward_status[i] && backward_status[i] && CalcPixelDistance(pts_to_track[i], reverse_pts[i]) < kPixelErrorThreshold &&
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

std::unordered_map<uint32_t, CameraObs> VioFrontend::CircularTrackFeatures(double ts_sec, const cv::Mat &prev_image_left,
                                                                           const cv::Mat &cur_image_left, const cv::Mat &cur_image_right,
                                                                           const cv::Mat &prev_image_right, const Eigen::Matrix3d &Rwc,
                                                                           const std::unordered_map<uint32_t, CameraObs> &feature_umap,
                                                                           std::vector<uint32_t> obs_ids,
                                                                           const std::vector<cv::Point2f> &prev_left_pts)
{
    std::unordered_map<uint32_t, CameraObs> cur_feature_obs_umap;
    std::vector<cv::Point2f> prev_l_to_cur_l_tracked;
    std::vector<cv::Point2f> cur_l_to_cur_r_tracked;
    std::vector<cv::Point2f> cur_r_to_pre_r_pts_tracked;
    std::unordered_map<uint32_t, cv::Point2d> prev_l_to_cur_l_tracked_umap;
    std::unordered_map<uint32_t, cv::Point2d> cur_l_to_cur_r_tracked_umap;

    // step1: track previous left-cam visual points to current left-cam visual points
    auto pre_left_to_cur_left_status =
        TrackFeatures(prev_image_left, cur_image_left, R_ref, Rwc, true, use_census_transform_, prev_left_pts, prev_l_to_cur_l_tracked);
    int idx = 0;
    auto id_iter = obs_ids.begin();
    for (auto it = prev_l_to_cur_l_tracked.begin(); it != prev_l_to_cur_l_tracked.end(); idx++)
    {
        if (!pre_left_to_cur_left_status[idx])
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
    Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
    auto cur_left_to_cur_right_status =
        TrackFeatures(cur_image_left, cur_image_right, I3x3, I3x3, true, use_census_transform_, prev_l_to_cur_l_tracked, cur_l_to_cur_r_tracked);
    idx = 0;
    id_iter = obs_ids.begin();
    for (auto it = cur_l_to_cur_r_tracked.begin(); it != cur_l_to_cur_r_tracked.end(); idx++)
    {
        if (!cur_left_to_cur_right_status[idx])
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
    Eigen::Matrix3d Rwr_cur = Rwc * CamModel::getInstance().Rlr();
    Eigen::Matrix3d Rwr_prev = R_ref * CamModel::getInstance().Rlr();
    auto cur_right_to_prev_right_status = TrackFeatures(cur_image_right, prev_image_right, Rwr_cur, Rwr_prev, true, use_census_transform_,
                                                        cur_l_to_cur_r_tracked, cur_r_to_pre_r_pts_tracked);
    idx = 0;
    id_iter = obs_ids.begin();
    for (auto it = cur_r_to_pre_r_pts_tracked.begin(); it != cur_r_to_pre_r_pts_tracked.end(); idx++)
    {
        if (!cur_right_to_prev_right_status[idx])
        {
            it = cur_r_to_pre_r_pts_tracked.erase(it);
            id_iter = obs_ids.erase(id_iter);
            continue;
        }
        ++it;
        ++id_iter;
    }

    // step4: validate by comparing circular-tracked right points with previous right observations
    for (int i = 0; i < (int)obs_ids.size(); i++)
    {
        uint32_t feature_id = obs_ids[i];
        const CameraObs &prev_obs = feature_umap.at(feature_id);
        cv::Point2d prev_obs_right(prev_obs.uv.at(RIGHT_CAM).x(), prev_obs.uv.at(RIGHT_CAM).y());
        if (CalcPixelDistance(prev_obs_right, cur_r_to_pre_r_pts_tracked[i]) < kCircularTrackPixelErrorThres)
        {
            cv::Point2d cur_obs_l = prev_l_to_cur_l_tracked_umap[feature_id];
            cv::Point2d cur_obs_r = cur_l_to_cur_r_tracked_umap[feature_id];
            CameraObs cur_obs(ts_sec, cur_obs_l.x, cur_obs_l.y, cur_obs_r.x, cur_obs_r.y);
            cur_obs.feat_id = feature_id;
            cur_obs.valid = true;
            cur_obs.obs_times_n = prev_obs.obs_times_n + 1;
            cur_feature_obs_umap.emplace(prev_obs.feat_id, cur_obs);
        }
    }

    return cur_feature_obs_umap;
}

void VioFrontend::ResolveGridConflicts(std::vector<CameraObs *> &candidates, int h_step, int w_step, std::vector<std::vector<uint8_t>> &occupied_mat)
{
    std::vector<std::vector<CameraObs *>> occupied_feat(grid_h_ + 1, std::vector<CameraObs *>(grid_w_ + 1, nullptr));
    for (CameraObs *obs : candidates)
    {
        if (!obs->valid)
            continue;
        const int h = obs->uv.at(LEFT_CAM).y() / h_step;
        const int w = obs->uv.at(LEFT_CAM).x() / w_step;
        if (occupied_mat[h][w] == 0)
        {
            occupied_mat[h][w] = 1;
            occupied_feat[h][w] = obs;
        }
        else
        {
            if (obs->obs_times_n > occupied_feat[h][w]->obs_times_n)
            {
                occupied_feat[h][w]->setInvalid();
                occupied_feat[h][w] = obs;
            }
            else
            {
                obs->setInvalid();
            }
        }
    }
}

void VioFrontend::AddNewStereoFeatures(const cv::Mat &cur_image_left, const cv::Mat &cur_image_right, double ts_sec, int h_step, int w_step,
                                       std::vector<std::vector<uint8_t>> &occupied_mat, std::unordered_map<uint32_t, CameraObs> &cur_feature_obs_umap)
{
    std::vector<cv::Point2f> harris_new, harris_tracked;
    cv::goodFeaturesToTrack(cur_image_left, harris_new, max_feat_n_, 0.01, 20);
    Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
    auto status = TrackFeatures(cur_image_left, cur_image_right, I3x3, I3x3, true, use_census_transform_, harris_new, harris_tracked);

    for (int i = 0; i < (int)status.size(); i++)
    {
        if (!status[i])
        {
            continue;
        }
        int h = harris_new[i].y / h_step;
        int w = harris_new[i].x / w_step;
        if (occupied_mat[h][w])
        {
            continue;
        }
        CameraObs obs(ts_sec, harris_new[i].x, harris_new[i].y, harris_tracked[i].x, harris_tracked[i].y);
        obs.valid = true;
        obs.feat_id = global_feature_id_++;
        obs.obs_times_n = 1;
        occupied_mat[h][w] = 1;
        cur_feature_obs_umap.insert_or_assign(obs.feat_id, obs);
    }
}

/// @brief Finalizes feature observations by removing invalid features and back-projecting valid ones
/// @param cur_feature_obs_umap Map of current feature observations indexed by feature ID
/// @param ts_sec Timestamp in seconds
/// @param feature_observes Output pair containing timestamp and finalized observations
void VioFrontend::FinalizeObservations(std::unordered_map<uint32_t, CameraObs> &cur_feature_obs_umap, double ts_sec,
                                       std::pair<double, std::vector<CameraObs>> &feature_observes)
{
    feature_observes.first = ts_sec;
    feature_observes.second.clear();
    for (auto it = cur_feature_obs_umap.begin(); it != cur_feature_obs_umap.end();)
    {
        CameraObs &obs = it->second;
        if (!obs.valid)
        {
            it = cur_feature_obs_umap.erase(it);
        }
        else
        {
            CamModel::getInstance().back_project_undistort(obs);
            feature_observes.second.push_back(obs);
            ++it;
        }
    }
}

/// @brief Visualizes tracked features on the image with their feature IDs
/// @param image Input image to draw features on
/// @param feature_observes Pair containing timestamp and feature observations for visualization
void VioFrontend::VisualizeFeatureTracking(const cv::Mat &image, const std::pair<double, std::vector<CameraObs>> &feature_observes)
{
    cv::Mat image_to_show;
    std::vector<std::pair<int32_t, cv::Point2f>> feature_with_ids;
    for (const auto &obs : feature_observes.second)
    {
        if (obs.valid)
        {
            feature_with_ids.emplace_back(obs.feat_id, cv::Point2f(obs.uv.at(LEFT_CAM).x(), obs.uv.at(LEFT_CAM).y()));
        }
    }
    utils::visualize_feature_tracking_results(image.clone(), feature_with_ids, 1, &image_to_show);
    image_with_features_ = std::make_pair(feature_observes.first, image_to_show);
}

/// @brief Performs stereo feature tracking between consecutive frames
/// @param input_image Pair containing timestamp and stereo image pair (left and right)
/// @param Rwc Rotation matrix from world to camera frame
/// @param do_warp_flag Flag to enable perspective warp for feature tracking
/// @param feature_observes Output pair containing timestamp and tracked feature observations
/// @return True if tracking was successful
bool VioFrontend::TrackStereo(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                              std::pair<double, std::vector<CameraObs>> &feature_observes)
{
    static bool is_first_entry = true;

    const double ts_sec = input_image.first;
    const cv::Mat &cur_image_left = input_image.second[LEFT_CAM];
    const cv::Mat &cur_image_right = input_image.second[RIGHT_CAM];

    const int h_step = height_ / grid_h_;
    const int w_step = width_ / grid_w_;
    std::vector<std::vector<uint8_t>> occupied_mat(grid_h_ + 1, std::vector<uint8_t>(grid_w_ + 1, 0));

    std::unordered_map<uint32_t, CameraObs> feature_umap;
    std::unordered_map<uint32_t, CameraObs> cur_feature_obs_umap;

    if (prev_images_.first > 0)
    {
        std::vector<uint32_t> obs_ids;
        std::vector<cv::Point2f> prev_left_pts;
        for (const auto &obs : previous_observations_.second)
        {
            if (!obs.valid)
                continue;
            feature_umap.emplace(obs.feat_id, obs);
            obs_ids.push_back(obs.feat_id);
            prev_left_pts.emplace_back(obs.uv.at(LEFT_CAM).x(), obs.uv.at(LEFT_CAM).y());
        }

        cur_feature_obs_umap = CircularTrackFeatures(ts_sec, prev_images_.second[LEFT_CAM], cur_image_left, cur_image_right,
                                                     prev_images_.second[RIGHT_CAM], Rwc, feature_umap, obs_ids, prev_left_pts);

        std::vector<CameraObs *> candidates;
        for (auto &[id, obs] : cur_feature_obs_umap)
            candidates.push_back(&obs);
        ResolveGridConflicts(candidates, h_step, w_step, occupied_mat);
    }

    const bool switch_keyframe = (keyframe_ != KeyFrameStatus::kNone) || is_first_entry;
    if (switch_keyframe)
    {
        AddNewStereoFeatures(cur_image_left, cur_image_right, ts_sec, h_step, w_step, occupied_mat, cur_feature_obs_umap);
    }

    FinalizeObservations(cur_feature_obs_umap, ts_sec, feature_observes);

    if (switch_keyframe)
    {
        is_first_entry = false;
        prev_images_ = input_image;
        previous_observations_ = feature_observes;
        R_ref = Rwc;
    }

    VisualizeFeatureTracking(cur_image_left, feature_observes);

    return true;
}

/// @brief Performs monocular feature tracking between consecutive frames
void VioFrontend::MonoTrackAndFilter(std::vector<CameraObs> &cur_features, const cv::Mat &cur_image, const Eigen::Matrix3d &Rwc, int h_step,
                                     int w_step, std::vector<std::vector<uint8_t>> &occupied_mat)
{
    // Build input point list from valid reference features
    std::vector<cv::Point2f> prev_pts, curr_pts;
    std::vector<int> feat_idx;
    for (int idx = 0; idx < (int)previous_observations_.second.size(); idx++)
    {
        if (previous_observations_.second[idx].valid)
        {
            feat_idx.push_back(idx);
            prev_pts.emplace_back(previous_observations_.second[idx].uv.at(LEFT_CAM).x(), previous_observations_.second[idx].uv.at(LEFT_CAM).y());
        }
    }

    // LK optical flow tracking
    auto status = TrackFeatures(prev_images_.second[LEFT_CAM], cur_image, R_ref, Rwc, false, use_census_transform_, prev_pts, curr_pts);
    for (int i = 0; i < (int)status.size(); i++)
    {
        const int idx = feat_idx[i];
        if (status[i])
        {
            cur_features[idx].uv[LEFT_CAM] = Eigen::Vector2d(curr_pts[i].x, curr_pts[i].y);
            cur_features[idx].obs_times_n++;
        }
        else
        {
            cur_features[idx].setInvalid();
        }
    }

    // Epipolar RANSAC outlier rejection
    std::vector<uint8_t> inliers;
    MonoCheckEpipolarLine(previous_observations_.second, cur_features, inliers);
    for (int i = 0; i < (int)previous_observations_.second.size(); i++)
    {
        if (inliers[i] == 0)
            cur_features[i].setInvalid();
    }

    // Resolve per-cell conflicts: keep the feature with more observations
    std::vector<CameraObs *> candidates;
    for (auto &obs : cur_features)
        candidates.push_back(&obs);
    ResolveGridConflicts(candidates, h_step, w_step, occupied_mat);
}

void VioFrontend::AddNewMonoFeatures(const cv::Mat &image, int h_step, int w_step, std::vector<std::vector<uint8_t>> &occupied_mat,
                                     std::vector<CameraObs> &cur_features)
{
    std::vector<cv::Point2f> candidate_keypoints;
    cv::goodFeaturesToTrack(image, candidate_keypoints, max_feat_n_, 0.01, 30);

    std::deque<CameraObs> new_features;
    for (const auto &kp : candidate_keypoints)
    {
        int h = kp.y / h_step;
        int w = kp.x / w_step;
        if (occupied_mat[h][w] == 0)
        {
            CameraObs feature;
            feature.feat_id = global_feature_id_++;
            feature.obs_times_n = 1;
            feature.uv[LEFT_CAM] = Eigen::Vector2d(kp.x, kp.y);
            new_features.push_back(feature);
            occupied_mat[h][w] = 1;
        }
    }

    while (!new_features.empty() && cur_features.size() < max_feat_n_)
    {
        cur_features.push_back(new_features.front());
        cur_features.back().valid = true;
        new_features.pop_front();
    }
}

/// @param input_image Pair containing timestamp and monocular image
/// @param Rwc Rotation matrix from world to camera frame
/// @param do_warp_flag Flag to enable perspective warp for feature tracking
/// @param feature_observes Output pair containing timestamp and tracked feature observations
/// @return True if tracking was successful
bool VioFrontend::TrackMonocular(const std::pair<double, std::vector<cv::Mat>> &input_image, const Eigen::Matrix3d Rwc,
                                 std::pair<double, std::vector<CameraObs>> &feature_observes)
{
    const double ts_sec = input_image.first;
    const cv::Mat cur_image = input_image.second[LEFT_CAM].clone();
    std::vector<CameraObs> cur_features = previous_observations_.second;
    const int h_step = height_ / grid_h_;
    const int w_step = width_ / grid_w_;
    std::vector<std::vector<uint8_t>> occupied_mat(grid_h_ + 1, std::vector<uint8_t>(grid_w_ + 1, 0));

    if (prev_images_.first >= 0)
    {
        if (ts_sec <= prev_images_.first)
        {
            LOG_WARN("invalid image timestamp!");
            return STATUS_ERROR;
        }
        MonoTrackAndFilter(cur_features, cur_image, Rwc, h_step, w_step, occupied_mat);
    }

    // Remove invalid features
    cur_features.erase(std::remove_if(cur_features.begin(), cur_features.end(), [](const CameraObs &obs) { return !obs.valid; }), cur_features.end());

    if (keyframe_ != KeyFrameStatus::kNone || prev_images_.first < 0)
    {
        AddNewMonoFeatures(cur_image, h_step, w_step, occupied_mat, cur_features);
        // Save pixel-coord features and image before back-projection
        prev_images_ = {ts_sec, {cur_image}};
        previous_observations_ = {ts_sec, cur_features};
        R_ref = Rwc;
    }

    // Back-project from pixel coordinates to normalized coordinates
    for (auto &obs : cur_features)
    {
        obs.ts_sec = ts_sec;
        CamModel::getInstance().back_project_undistort(obs);
    }

    feature_observes = std::make_pair(ts_sec, cur_features);

    VisualizeFeatureTracking(input_image.second[LEFT_CAM], feature_observes);

    return true;
}