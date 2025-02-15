#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "frontend.h"
#include "opencv2/core/mat.hpp"
#include "sensor_data.h"
#include "utils.h"

namespace {
    constexpr double kPixelErrorThreshold = 2.0;
    constexpr double kCircularTrackPixelErrorThres = 2.0;
}

double Distance(const cv::Point2d& pt1, const cv::Point2d& pt2)
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

void EpipolarRansac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr, const Eigen::Matrix3d K, std::vector<uchar>* inliers)
{
    constexpr double kEssentialMatrixProb = 0.95;
    constexpr double kEssentialThres = 1.0;
    assert(points_prev.size() == points_curr.size());
    *inliers = std::vector<uchar>(points_prev.size(), 0);

    cv::Mat K_mat;
    cv::eigen2cv(K, K_mat);
    cv::Mat essential_matrix = cv::findEssentialMat(points_prev, points_curr, K_mat, cv::RANSAC, kEssentialMatrixProb, kEssentialThres);

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> essential_mat;
    cv::cv2eigen(essential_matrix, essential_mat);
    for (int i = 0; i < points_prev.size(); i++) {
        Eigen::Vector3d p_prev(points_prev[i].x, points_prev[i].y, 1.0);
        Eigen::Vector3d p_curr(points_curr[i].x, points_curr[i].y, 1.0);
        p_prev = K.inverse() * p_prev.eval();
        p_curr = K.inverse() * p_curr.eval();
        double dist = p_prev.transpose() * essential_mat * p_curr;
        if (dist < kEssentialThres) {
            (*inliers)[i] = 1;
        }
    }
}

VioFrontend::status_t VioFrontend::OutlierRejection(const std::vector<CameraObs>& obs_prev, const std::vector<CameraObs>& obs_curr, std::vector<uchar>& inliers)
{
    if (obs_prev.size() != obs_curr.size()) {
        LOG(ERROR) << "obs_prev.size() != obs_curr.size()";
        return STATUS_ERROR;
    }

    size_t size = obs_prev.size();
    std::vector<cv::Point2f> points_prev, points_curr;
    for (int i = 0; i < size; i++) {
        if (obs_prev[i].valid && obs_prev[i].valid) {
            points_prev.emplace_back(obs_prev[i].u, obs_prev[i].v);
            points_curr.emplace_back(obs_curr[i].u, obs_curr[i].v);
        }
    }

    // std::vector<uchar> inliers_homography;
    std::vector<uchar> inliers_epipolar;
    // std::thread r1 = std::thread(HomographyRansac, points_prev, points_curr, &inliers_homography);
    // std::thread r2 = std::thread(EpipolarRansac, points_prev, points_curr, _camera_model->intrinsic(), &inliers_epipolar);
    // r1.join();
    // r2.join();

    EpipolarRansac(points_prev, points_curr, _camera_model->K_l(), &inliers_epipolar);

    // int sum_h = 0;
    // int sum_e = 0;
    // for (int i = 0; i < inliers_epipolar.size(); i++) {
    //     sum_h += inliers_homography[i];
    //     sum_e += inliers_epipolar[i];
    // }

    // inliers = (sum_h > sum_e) ? inliers_homography : inliers_epipolar;
    // std::cout << cv::format("sum_h: %d, sum_e: %d\n", sum_h, sum_e);
    inliers = inliers_epipolar;
    return STATUS_OK;
}

std::vector<bool> VioFrontend::TrackFeatures(const cv::Mat image_left, const cv::Mat image_right, const std::vector<cv::Point2f> pts_to_track, std::vector<cv::Point2f>& pts_tracked)
{
    assert(!pts_to_track.empty());
    std::vector<uchar> forward_status, backward_status;
    std::vector<float> err;

    // forward tracking
    cv::calcOpticalFlowPyrLK(image_left, image_right, pts_to_track,
        pts_tracked, forward_status, err);

    // backward tracking
    std::vector<cv::Point2f> reverse_pts = pts_tracked;
    cv::calcOpticalFlowPyrLK(
        image_right, image_left, pts_tracked, reverse_pts,
        backward_status, err, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
            0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // double check if the tracked point is out of range
    std::vector<bool> status;
    for (int i = 0; i < forward_status.size(); i++) {
        if (forward_status[i] && backward_status[i] && (Distance(pts_to_track[i], reverse_pts[i]) < kPixelErrorThreshold) && InBorder(pts_tracked[i].x, pts_tracked[i].y)) {
            status.push_back(true);
        } else {
            status.push_back(false);
        }
    }
    return status;
}

bool VioFrontend::TrackStereo(const std::pair<double, std::pair<cv::Mat, cv::Mat>>& input_image, std::pair<double, std::vector<CameraObs>>& feature_observes)
{
    static bool is_first_entry = true;
    double ts_sec = input_image.first;
    static std::pair<double, std::pair<cv::Mat, cv::Mat>> prev_images = {-1, {cv::Mat(), cv::Mat()}};
    static std::pair<double, std::vector<CameraObs>> previous_observations;

    cv::Mat cur_image_left = input_image.second.first;
    cv::Mat cur_image_right = input_image.second.second;
    cv::Mat prev_image_left = prev_images.second.first;
    cv::Mat prev_image_right = prev_images.second.second;

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
    std::unordered_map<uint32_t, CameraObs> cur_feature_obs_umap;   // {feature_id, CameraObs}

    if (prev_images.first > 0) {
        for (int i = 0; i < previous_observations.second.size(); i++) {
            CameraObs obs = previous_observations.second[i];
            if (!obs.valid) {
                continue;
            }
            feature_umap.insert({ obs.feat_id, obs });
            obs_ids.push_back(obs.feat_id);
            prev_left_pts.push_back({ obs.u, obs.v });
        }

        // step1: track previous left-cam visual points to current left-cam visual points
        std::vector<cv::Point2f> prev_l_to_cur_l_tracked;
        std::unordered_map<uint32_t, cv::Point2d> prev_l_to_cur_l_tracked_umap;
        auto pre_left_to_cur_left_status = TrackFeatures(prev_image_left, cur_image_left, prev_left_pts, prev_l_to_cur_l_tracked);
        int idx = 0;
        std::vector<uint32_t>::iterator id_iter = obs_ids.begin();
        for (auto it = prev_l_to_cur_l_tracked.begin(); it != prev_l_to_cur_l_tracked.end(); idx++) {
            if (pre_left_to_cur_left_status[idx] == false) {
                it = prev_l_to_cur_l_tracked.erase(it);
                id_iter = obs_ids.erase(id_iter);
                continue;
            }
            prev_l_to_cur_l_tracked_umap.insert({ *id_iter, *it });
            ++it;
            ++id_iter;
        }

        // step2: track current left-cam visual points to current right-cam visual points
        std::vector<cv::Point2f> cur_l_to_cur_r_tracked;
        std::unordered_map<uint32_t, cv::Point2d> cur_l_to_cur_r_tracked_umap;
        auto cur_left_to_cur_right_status = TrackFeatures(cur_image_left, cur_image_right, prev_l_to_cur_l_tracked, cur_l_to_cur_r_tracked);
        idx = 0;
        id_iter = obs_ids.begin();
        for (auto it = cur_l_to_cur_r_tracked.begin(); it != cur_l_to_cur_r_tracked.end(); idx++) {
            if (cur_left_to_cur_right_status[idx] == false) {
                it = cur_l_to_cur_r_tracked.erase(it);
                id_iter = obs_ids.erase(id_iter);
                continue;
            }
            cur_l_to_cur_r_tracked_umap.insert({ *id_iter, *it });
            ++it;
            ++id_iter;
        }

        // step3: track current right-cam visual points to previous right-cam visual points
        std::vector<cv::Point2f> cur_r_to_pre_r_pts_tracked;
        std::unordered_map<uint32_t, cv::Point2d> cur_r_to_pre_r_pts_tracked_umap;
        auto cur_right_to_prev_right_status = TrackFeatures(cur_image_right, prev_image_right, cur_l_to_cur_r_tracked, cur_r_to_pre_r_pts_tracked);
        idx = 0;
        id_iter = obs_ids.begin();
        for (auto it = cur_r_to_pre_r_pts_tracked.begin(); it != cur_r_to_pre_r_pts_tracked.end(); idx++) {
            if (cur_right_to_prev_right_status[idx] == false) {
                it = cur_r_to_pre_r_pts_tracked.erase(it);
                id_iter = obs_ids.erase(id_iter);
                continue;
            }
            cur_r_to_pre_r_pts_tracked_umap.insert({ *id_iter, *it });
            ++it;
            ++id_iter;
        }

        // step4: compare circular-tracked visual points with previous right-cam visual points
        for (int i = 0; i < obs_ids.size(); i++) {
            uint32_t feature_id = obs_ids[i];
            cv::Point2d circular_tracked_point = cur_r_to_pre_r_pts_tracked[i];
            CameraObs obs = feature_umap[feature_id];
            cv::Point2d prev_obs_right(obs.ur, obs.vr);
            if (Distance(prev_obs_right, circular_tracked_point) < kCircularTrackPixelErrorThres) {
                cv::Point2d cur_obs_l = prev_l_to_cur_l_tracked_umap[feature_id];
                cv::Point2d cur_obs_r = cur_l_to_cur_r_tracked_umap[feature_id];
                CameraObs cur_obs(ts_sec, cur_obs_l.x, cur_obs_l.y, cur_obs_r.x, cur_obs_r.y);
                cur_obs.feat_id = feature_id;
                cur_feature_obs_umap.insert({obs.feat_id, cur_obs});
            }
        }
    }

    // Add current feature observations to grid map
    for (auto& [feature_id, obs] : cur_feature_obs_umap) {
        int h = obs.v / h_step;
        int w = obs.u / w_step;
        if (occupied_grid[h][w] == false) {
            occupied_grid[h][w] = true;
            occupied_grid_feature_id[h][w] = feature_id;
            obs.valid = true;
            obs.obs_times_n++;
        }
        else {
            uint32_t obs_candidate_feature_id = occupied_grid_feature_id[h][w];
            CameraObs& obs_candidate = feature_umap[obs_candidate_feature_id];
            if (obs.obs_times_n < obs_candidate.obs_times_n) {
                obs.set_invalid();
            } else {
                obs_candidate.set_invalid();
                occupied_grid_feature_id[h][w] = feature_id;
                obs.valid = true;
                obs.obs_times_n++;
            }
        }
    }
    // Add new features to current observations if keyframe or first frame
    bool is_keyframe = ((*_keyframe) != KeyFrameType::not_keyframe) || is_first_entry;
    // bool is_keyframe = true; // always keyframe for debug
    if (is_keyframe) {
        std::vector<cv::Point2f> harris_features, harris_tmp;
        std::vector<CameraObs> cur_stereo_ok_features;
        cv::goodFeaturesToTrack(cur_image_left, harris_features, _max_feat_n, 0.01, 30);
        auto status = TrackFeatures(cur_image_left, cur_image_right, harris_features, harris_tmp);
        for (int i = 0; i < status.size(); i++) {
            if (status[i] == true) {
                cur_stereo_ok_features.emplace_back(ts_sec, harris_features[i].x, harris_features[i].y, harris_tmp[i].x, harris_tmp[i].y);
            }
        }

        for (auto& feature_new : cur_stereo_ok_features) {
            int h = feature_new.v / h_step;
            int w = feature_new.u / w_step;
            if (occupied_grid[h][w] == false) {
                feature_new.valid = true;
                feature_new.feat_id = global_feature_id_++;
                feature_new.obs_times_n = 1;
                occupied_grid[h][w] = true;
                occupied_grid_feature_id[h][w] = feature_new.feat_id;
                cur_feature_obs_umap.insert({ feature_new.feat_id, feature_new });
            }
        }
    }

    // clear invalid features
    feature_observes.first = ts_sec;
    feature_observes.second.clear();
    for (auto it = cur_feature_obs_umap.begin(); it != cur_feature_obs_umap.end();) {
        if (it->second.valid == false) {
            it = cur_feature_obs_umap.erase(it);
        } else {
            feature_observes.second.push_back(it->second);
            ++it;
        }
    }

    if (is_keyframe) {
        is_first_entry = false;
        prev_images = input_image;
        previous_observations = feature_observes;
    }

    // Utils::visualize_feature_tracking_results(input_image.second.first.clone(), feature_observes);
    return true;
}

bool VioFrontend::TrackMonocular(const std::pair<double, std::pair<cv::Mat, cv::Mat>>& input_image, std::pair<double, std::vector<CameraObs>>& feature_observes)
{
    double ts_sec = input_image.first;
    cur_frame = std::make_pair(ts_sec, input_image.second.first.clone());
    std::vector<CameraObs> cur_feat_to_track = ref_feat_to_track_;

    int h_step = height_ / grid_h_;
    int w_step = width_ / grid_w_;
    std::vector<std::vector<bool>> occupied_mat(grid_h_ + 1, std::vector<bool>(grid_w_ + 1, false));

    if (!is_first_frame_) {

        if (cur_frame.first <= ref_frame.first) {
            LOG(WARNING) << "invalid image timestamp!";
            return STATUS_ERROR;
        }

        frontend_rT = boost::posix_time::microsec_clock::local_time();
        std::vector<cv::Point2f> prev_pts, curr_pts;
        std::vector<int> feat_idx;

        assert(ref_feat_to_track_.size() == _max_feat_n);

        for (int idx = 0; idx < ref_feat_to_track_.size(); idx++) {
            if (ref_feat_to_track_[idx].valid) {
                feat_idx.push_back(idx);
                prev_pts.emplace_back(ref_feat_to_track_[idx].u, ref_feat_to_track_[idx].v);
            }
        }
        std::vector<uchar> status, reverse_status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(ref_frame.second, cur_frame.second, prev_pts, curr_pts, status, err);

        // reverse check
        std::vector<cv::Point2f> reverse_pts = prev_pts;
        cv::calcOpticalFlowPyrLK(cur_frame.second, ref_frame.second, curr_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

        frontend_rT1 = boost::posix_time::microsec_clock::local_time();

        // double check and check if the tracked point is out of bound
        for (int i = 0; i < status.size(); i++) {
            int idx = feat_idx[i];
            if (status[i] && reverse_status[i] && (Distance(prev_pts[i], reverse_pts[i]) <= 1) && InBorder(curr_pts[i].x, curr_pts[i].y)) {
                cur_feat_to_track[idx].u = curr_pts[i].x;
                cur_feat_to_track[idx].v = curr_pts[i].y;
                cur_feat_to_track[idx].obs_times_n++;
            } else {
                cur_feat_to_track[idx].set_invalid();
            }
        }

        std::vector<uchar> inliers;
        OutlierRejection(ref_feat_to_track_, cur_feat_to_track, inliers);
        for (int i = 0; i < ref_feat_to_track_.size(); i++) {
            if (inliers[i] == 0) {
                cur_feat_to_track[i].set_invalid();
            }
        }

        frontend_rT2 = boost::posix_time::microsec_clock::local_time();

        std::vector<std::vector<CameraObs*>> occupied_feat(grid_h_ + 1, std::vector<CameraObs*>(grid_w_ + 1, nullptr));
        for (int i = 0; i < cur_feat_to_track.size(); i++) {
            if (!cur_feat_to_track[i].valid) {
                continue;
            }
            int h = cur_feat_to_track[i].v / h_step;
            int w = cur_feat_to_track[i].u / w_step;
            if (occupied_mat[h][w] == false) {
                occupied_mat[h][w] = true;
                occupied_feat[h][w] = &(cur_feat_to_track[i]);
            } else {
                // delete features with less obs times
                if (cur_feat_to_track[i].obs_times_n > occupied_feat[h][w]->obs_times_n) {
                    occupied_feat[h][w]->set_invalid();
                    occupied_feat[h][w] = &(cur_feat_to_track[i]);
                }
            }
        }
    }

    frontend_rT3 = boost::posix_time::microsec_clock::local_time();
    // add new features to ref_feat_to_track_
    // std::cout << cv::format("keyframe: %d", static_cast<int>(*_keyframe));
    if (*_keyframe != KeyFrameType::not_keyframe || is_first_frame_)
    { // debug always keyframe
    // if (1) { // debug: always keyframe
        std::deque<CameraObs> feats_new;
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(cur_frame.second, corners, _max_feat_n, 0.01, 30);
        for (auto& feat_new : corners)
        {
            int h = feat_new.y / h_step;
            int w = feat_new.x / w_step;
            if (occupied_mat[h][w] == false)
            {
                CameraObs feat;
                feat.feat_id = global_feature_id_++;
                feat.obs_times_n = 1;
                feat.u = feat_new.x;
                feat.v = feat_new.y;
                feats_new.push_back(feat);
                occupied_mat[h][w] = true;
            }
        }

        // add new features if ref_feat_to_track_[i] is invalid
        int new_feat_added_num = 0;
        for (int i = 0; i < cur_feat_to_track.size(); i++)
        {
            if (cur_feat_to_track[i].valid == false && !feats_new.empty())
            {
                cur_feat_to_track[i] = feats_new.front();
                cur_feat_to_track[i].valid = true;
                feats_new.pop_front();
                new_feat_added_num++;
            }
        }
        ref_frame = cur_frame;
        ref_feat_to_track_ = cur_feat_to_track;
    }

    frontend_rT4 = boost::posix_time::microsec_clock::local_time();

    // double track_duration = (frontend_rT1 - frontend_rT).total_microseconds() * 1e-6;
    // double outlier_rejection_duration = (frontend_rT2 - frontend_rT1).total_microseconds() * 1e-6;
    // double add_feat_duration = (frontend_rT4 - frontend_rT3).total_microseconds() * 1e-6;
    // LOG(INFO) << cv::format("tracking duration: %f, outlier rejection duration: %f, add_feat_duration: %f\n",
    //                         track_duration, outlier_rejection_duration, add_feat_duration);

    // assign timestamp to each feature obs
    for (auto & obs : cur_feat_to_track)
    {
        obs.ts_sec = ts_sec;
    }

    feature_observes = { ts_sec, cur_feat_to_track };

    Utils::visualize_feature_tracking_results(input_image.second.first.clone(), feature_observes);
    is_first_frame_ = false;
    // *_keyframe = KeyFrameType::not_keyframe;
    return true;
}