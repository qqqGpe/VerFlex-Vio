#include <vector>

#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "camModel.h"
#include "frontend.h"
#include "opencv2/core/mat.hpp"
#include "sensor_data.h"
#include "utils.h"

namespace
{
constexpr double kPixelErrorThreshold = 1.0;
constexpr double kCircularTrackPixelErrorThres = 1.0;
}  // namespace

double CalcPixelDistance(const cv::Point2d& pt1, const cv::Point2d& pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool VioFrontend::InBorder(int x, int y)
{
    return (x >= 0 && x < width_) && (y >= 0 && y < height_);
}

void HomographyRansac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr, std::vector<uchar>* inliers)
{
    assert(points_prev.size() == points_curr.size());
    *inliers = std::vector<uchar>(points_prev.size(), 0);
    cv::findHomography(points_prev, points_curr, *inliers, cv::RANSAC, 3);
}

void EpipolarRansac(const std::vector<cv::Point2f> points_prev,
                    const std::vector<cv::Point2f> points_curr,
                    std::vector<uchar>* inliers)
{
    constexpr double kEssentialMatrixProb = 0.95;
    constexpr double kEssentialThres = 1.0; // 1 pixel threshold for essential matrix

    assert(points_prev.size() == points_curr.size());
    *inliers = std::vector<uchar>(points_prev.size(), 0);

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
            (*inliers)[i] = 1;
        }
    }
}

VioFrontend::status_t VioFrontend::MonoCheckEpipolarLine(const std::vector<CameraObs>& obs_prev,
                                                         const std::vector<CameraObs>& obs_curr,
                                                         std::vector<uint8_t>& inliers)
{
    if (obs_prev.size() != obs_curr.size())
    {
        LOG(ERROR) << "obs_prev.size() != obs_curr.size()";
        return STATUS_ERROR;
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

std::vector<uint8_t> VioFrontend::TrackFeatures(const cv::Mat image_left,
                                                const cv::Mat image_right,
                                                const std::vector<cv::Point2f> pts_to_track,
                                                std::vector<cv::Point2f>& pts_tracked)
{
    std::vector<uint8_t> status;
    std::vector<uchar> forward_status, backward_status;
    std::vector<float> err;

    if (pts_to_track.empty())
    {
        return status;
    }

    // forward tracking
    cv::calcOpticalFlowPyrLK(image_left, image_right, pts_to_track, pts_tracked, forward_status, err);

    // backward tracking
    std::vector<cv::Point2f> reverse_pts = pts_tracked;
    cv::calcOpticalFlowPyrLK(image_right, image_left, pts_tracked, reverse_pts, backward_status, err, cv::Size(21, 21), 4,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

    // double check if the tracked point is out of range
    for (int i = 0; i < forward_status.size(); i++)
    {
        if (forward_status[i] && backward_status[i] && (CalcPixelDistance(pts_to_track[i], reverse_pts[i]) < kPixelErrorThreshold) &&
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

bool VioFrontend::TrackStereo(const std::pair<double, std::vector<cv::Mat>>& input_image, std::pair<double, std::vector<CameraObs>>& feature_observes)
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
    std::unordered_map<uint32_t, CameraObs> cur_feature_obs_umap;  // {feature_id, CameraObs}

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
        auto pre_left_to_cur_left_status = TrackFeatures(prev_image_left, cur_image_left, prev_left_pts, prev_l_to_cur_l_tracked);
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
        auto cur_left_to_cur_right_status = TrackFeatures(cur_image_left, cur_image_right, prev_l_to_cur_l_tracked, cur_l_to_cur_r_tracked);
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
        auto cur_right_to_prev_right_status = TrackFeatures(cur_image_right, prev_image_right, cur_l_to_cur_r_tracked, cur_r_to_pre_r_pts_tracked);
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
    for (auto& [feature_id, obs] : cur_feature_obs_umap)
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
            CameraObs& obs_candidate = feature_umap[obs_candidate_feature_id];
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
        auto status = TrackFeatures(cur_image_left, cur_image_right, harris_new, harris_tracked);
        for (int i = 0; i < status.size(); i++)
        {
            if (status[i] == true)
            {
                CameraObs obs(ts_sec, harris_new[i].x, harris_new[i].y, harris_tracked[i].x, harris_tracked[i].y);
                cur_stereo_ok_features.emplace_back(obs);
            }
        }

        for (auto& feature_new : cur_stereo_ok_features)
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
    }

    // Utils::visualize_feature_tracking_results(input_image.second.first.clone(), feature_observes);
    return true;
}

bool VioFrontend::TrackMonocular(const std::pair<double, std::vector<cv::Mat>>& input_image,
                                 std::pair<double, std::vector<CameraObs>>& feature_observes)
{
    const double ts_sec = input_image.first;
    cur_frame = std::make_pair(ts_sec, input_image.second[0].clone());
    std::vector<CameraObs> cur_features_to_track = ref_features_to_track_;

    const int h_step = height_ / grid_h_;
    const int w_step = width_ / grid_w_;
    std::vector<std::vector<uint8_t>> occupied_mat(grid_h_ + 1, std::vector<uint8_t>(grid_w_ + 1, 0));

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
                prev_pts.emplace_back(ref_features_to_track_[idx].uv[LEFT_CAM].x(), ref_features_to_track_[idx].uv[LEFT_CAM].y());
            }
        }

        // Circular track
        std::vector<uint8_t> status = TrackFeatures(ref_frame.second, cur_frame.second, prev_pts, curr_pts);

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
        std::vector<std::vector<CameraObs*>> occupied_feat(grid_h_ + 1, std::vector<CameraObs*>(grid_w_ + 1, nullptr));
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
            it = cur_features_to_track.erase(it);
            continue;
        }
        it = std::next(it);
    }

    // Add new features if keyframe
    if (*_keyframe != KeyFrameStatus::kNone || is_first_frame_)
    {
        std::deque<CameraObs> new_features;
        std::vector<cv::Point2f> good_points;
        cv::goodFeaturesToTrack(cur_frame.second, good_points, max_feat_n_, 0.01, 30);
        for (auto& feat_new : good_points)
        {
            int h = feat_new.y / h_step;
            int w = feat_new.x / w_step;
            if (occupied_mat[h][w] == 0)
            {
                CameraObs feature;
                feature.feat_id = global_feature_id_++;
                feature.obs_times_n = 1;
                feature.uv[LEFT_CAM] = Eigen::Vector2d(feat_new.x, feat_new.y);
                new_features.push_back(feature);
                occupied_mat[h][w] = 1;
            }
        }

        while (!new_features.empty() && cur_features_to_track.size() < max_feat_n_)
        {
            cur_features_to_track.push_back(new_features.front());
            cur_features_to_track.back().valid = true;
            new_features.pop_front();
        }

        ref_frame = cur_frame;
        ref_features_to_track_ = cur_features_to_track;
    }

    // Back project features
    for (auto& obs : cur_features_to_track)
    {
        obs.ts_sec = ts_sec;
        CamModel::getInstance().back_project(obs);
    }

    feature_observes = std::make_pair(ts_sec, cur_features_to_track);
    // Utils::visualize_feature_tracking_results(input_image.second.first.clone(), feature_observes);
    is_first_frame_ = false;
    return true;
}