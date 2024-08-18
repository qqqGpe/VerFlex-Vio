#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "frontend.h"
#include "geometry_msgs/Point32.h"
#include "sensor_msgs/ChannelFloat32.h"
#include "sensor_msgs/PointCloud.h"
#include "utils.h"

double distance(cv::Point2f& pt1, cv::Point2f& pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool VioFrontend::inBorder(int x, int y)
{
    return (x > 0 && x < _width) && (y > 0 && y < _height);
}

// void VioFrontend::publish_features(const frontend_frame_t& frame)
// {
//     sensor_msgs::PointCloud feats;
//     feats.header = frame.header;
//     feats.channels.resize(4, sensor_msgs::ChannelFloat32());
//     feats.channels[0].name = "valid";
//     feats.channels[1].name = "feature_id";
//     feats.channels[2].name = "observsation_times";
//     feats.channels[3].name = "keyframe";
//     assert(frame.feat_obs_list.size() == _max_feat_n);
//     for (auto& feat : frame.feat_obs_list) {
//         geometry_msgs::Point32 point;
//         if (feat.valid) {
//             feats.channels[0].values.push_back(feat.valid); /*特征点是否有效*/
//             feats.channels[1].values.push_back(feat.feat_id); /*feature_id*/
//             feats.channels[2].values.push_back(feat.obs_times_n); /*被观测到的次数*/
//             feats.channels[3].values.push_back(frame.keyframe_flag); /*是否是关键帧*/
//             point.x = feat.obs.x();
//             point.y = feat.obs.y();
//             feats.points.push_back(point);
//         } else {
//             feats.channels[0].values.push_back(0);
//             feats.channels[1].values.push_back(-1);
//             feats.channels[2].values.push_back(0);
//             feats.channels[3].values.push_back(frame.keyframe_flag);
//             feats.points.push_back(point);
//         }
//     }
//     feat_pub.publish(feats);
// }

void homography_ransac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr, std::vector<uchar>* inliers)
{
    assert(points_prev.size() == points_curr.size());
    *inliers = std::vector<uchar>(points_prev.size(), 0);
    cv::findHomography(points_prev, points_curr, *inliers, cv::RANSAC, 3);
}

void epipolar_ransac(const std::vector<cv::Point2f> points_prev, const std::vector<cv::Point2f> points_curr, const Eigen::Matrix3d K, std::vector<uchar>* inliers)
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

VioFrontend::status_t VioFrontend::outlier_rejection(const std::vector<cam_obs_t>& obs_prev, const std::vector<cam_obs_t>& obs_curr, std::vector<uchar>& inliers)
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
    // std::thread r1 = std::thread(homography_ransac, points_prev, points_curr, &inliers_homography);
    // std::thread r2 = std::thread(epipolar_ransac, points_prev, points_curr, _camera_model->intrinsic(), &inliers_epipolar);
    // r1.join();
    // r2.join();

    epipolar_ransac(points_prev, points_curr, _camera_model->intrinsic(), &inliers_epipolar);

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

bool VioFrontend::track(const std::pair<double, cv::Mat>& input_image, std::pair<double, std::vector<cam_obs_t>>& feature_observes)
{
    double ts_sec = input_image.first;
    cur_frame = std::make_pair(ts_sec, input_image.second.clone());
    std::vector<cam_obs_t> cur_feat_to_track = ref_feat_to_track;

    int h_step = _height / _grid_h;
    int w_step = _width / _grid_w;
    std::vector<std::vector<bool>> occupied_mat(_grid_h + 1, std::vector<bool>(_grid_w + 1, false));

    if (!is_first_frame) {

        if (cur_frame.first <= ref_frame.first) {
            LOG(WARNING) << "invalid image timestamp!";
            return STATUS_ERROR;
        }

        frontend_rT = boost::posix_time::microsec_clock::local_time();
        std::vector<cv::Point2f> prev_pts, curr_pts;
        std::vector<int> feat_idx;

        assert(ref_feat_to_track.size() == _max_feat_n);

        for (int idx = 0; idx < ref_feat_to_track.size(); idx++) {
            if (ref_feat_to_track[idx].valid) {
                feat_idx.push_back(idx);
                prev_pts.emplace_back(ref_feat_to_track[idx].u, ref_feat_to_track[idx].v);
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
            if (status[i] && reverse_status[i] && (distance(prev_pts[i], reverse_pts[i]) <= 1) && inBorder(curr_pts[i].x, curr_pts[i].y)) {
                cur_feat_to_track[idx].u = curr_pts[i].x;
                cur_feat_to_track[idx].v = curr_pts[i].y;
                cur_feat_to_track[idx].obs_times_n++;
            } else {
                cur_feat_to_track[idx].set_invalid();
            }
        }

        std::vector<uchar> inliers;
        outlier_rejection(ref_feat_to_track, cur_feat_to_track, inliers);
        for (int i = 0; i < ref_feat_to_track.size(); i++) {
            if (inliers[i] == 0) {
                cur_feat_to_track[i].set_invalid();
            }
        }

        frontend_rT2 = boost::posix_time::microsec_clock::local_time();

        std::vector<std::vector<cam_obs_t*>> occupied_feat(_grid_h + 1, std::vector<cam_obs_t*>(_grid_w + 1, nullptr));
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
    // add new features to ref_feat_to_track
    // if (*_keyframe != keyframe_flag_e::not_keyframe || is_first_frame) { // debug always keyframe
    if (1) { // debug: always keyframe
        std::deque<cam_obs_t> feats_new;
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(cur_frame.second, corners, _max_feat_n, 0.01, 30);
        for (auto& feat_new : corners) {
            int h = feat_new.y / h_step;
            int w = feat_new.x / w_step;
            if (occupied_mat[h][w] == false) {
                cam_obs_t feat;
                feat.feat_id = feat_id++;
                feat.obs_times_n = 1;
                feat.u = feat_new.x;
                feat.v = feat_new.y;
                feats_new.push_back(feat);
                occupied_mat[h][w] = true;
            }
        }

        // add new features if ref_feat_to_track[i] is invalid
        int new_feat_added_num = 0;
        for (int i = 0; i < cur_feat_to_track.size(); i++) {
            if (cur_feat_to_track[i].valid == false && !feats_new.empty()) {
                cur_feat_to_track[i] = feats_new.front();
                cur_feat_to_track[i].valid = true;
                feats_new.pop_front();
                new_feat_added_num++;
            }
        }
        ref_frame = cur_frame;
        ref_feat_to_track = cur_feat_to_track;
    }

    frontend_rT4 = boost::posix_time::microsec_clock::local_time();

    double track_duration = (frontend_rT1 - frontend_rT).total_microseconds() * 1e-6;
    double outlier_rejection_duration = (frontend_rT2 - frontend_rT1).total_microseconds() * 1e-6;
    double add_feat_duration = (frontend_rT4 - frontend_rT3).total_microseconds() * 1e-6;
    // LOG(INFO) << cv::format("tracking duration: %f, outlier rejection duration: %f, add_feat_duration: %f\n",
    //                         track_duration, outlier_rejection_duration, add_feat_duration);

    // assign timestamp to each feature obs
    for (auto & obs : cur_feat_to_track)
    {
        obs.ts_sec = ts_sec;
    }

    feature_observes = { ts_sec, cur_feat_to_track };

    Utils::visualize_feature_tracking_results(input_image.second.clone(), feature_observes);
    is_first_frame = false;
    // *_keyframe = keyframe_flag_e::not_keyframe;
    return true;
}