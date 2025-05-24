#include "sfm.h"
#include <optional>

bool Sfm::MaybeAddSfmKeyframes(const std::pair<double, std::vector<CameraObs>>& current_feature_observe)
{
    if (feature_observes.second.size() < kMinRequiredFeaturesPerFrame)
    {
        return false;
    }

    if (all_feature_observes_.empty())
    {
        all_feature_observes_.insert(current_feature_observe);
        latest_keyframe_observe_ = current_feature_observe;
        return true;
    }
    else
    {
        std::unordered_map<uint32_t, CameraObs> latest_keyframe_observe_umap;
        std::unordered_map<uint32_t, CameraObs> current_keyframe_observe_umap;
        for (auto& obs : latest_keyframe_observe_.second)
        {
            latest_keyframe_observe_umap.insert({obs.feat_id, obs});
        }
        for (auto& obs : current_feature_observe.second)
        {
            current_keyframe_observe_umap.insert({obs.feat_id, obs});
        }

        double pixel_parallex = VisualManager::calcVisualObsParallex(latest_keyframe_observe_umap, current_keyframe_observe_umap);
        if (pixel_parallex >= kMinPixelParallexBetweenKeyframes)
        {
            all_feature_observes_.insert(current_feature_observe);
            latest_keyframe_observe_ = current_feature_observe;
            return true;
        }
        else if (abs(current_feature_observe.first - latest_keyframe_observe_.first) > kMaxTimeIntervalBetweenKeyframes)
        {
            LOG(INFO) << "Platform moves too slow, can not perform dynamic-initialization";
            ResetSfm();
            return false;
        }
    }

    return false;
}

bool Sfm::calcRelativePose(const std::vector<CameraObs>& obs_a,
                           const std::vector<CameraObs>& obs_b,
                           Eigen::Matrix3d& R_relative,
                           Eigen::Vector3d& t_relative)
{
    constexpr double kMaxPixelErrorForCalcEssentialMat = 1.0f;
    constexpr uint32_t kMinRequiredFeaturesForRelativePose = 15;
    constexpr uint32_t kMinInliersForRelativePose = 12;
    if (obs_a.size() < kMinRequiredFeaturesForRelativePose || obs_b.size() < kMinRequiredFeaturesForRelativePose)
    {
        LOG(INFO) << "Not enough observations to calculate relative pose";
        return false;
    }

    std::unordered_map<uint32_t, CameraObs> obs_a_umap;
    std::unordered_map<uint32_t, CameraObs> obs_b_umap;
    for (auto& obs : obs_a)
    {
        obs_a_umap.insert({obs.feat_id, obs});
    }
    for (auto& obs : obs_b)
    {
        obs_b_umap.insert({obs.feat_id, obs});
    }

    std::vector<cv::Point2d> corresponding_points_a;
    std::vector<cv::Point2d> corresponding_points_b;
    for (auto& [point_id, obs] : obs_a_umap)
    {
        auto it = obs_b_umap.find(point_id);
        if (it != obs_b_umap.end())
        {
            corresponding_points_a.push_back(cv::Point2d(obs.u, obs.v));
            corresponding_points_b.push_back(cv::Point2d(it->second.u, it->second.v));
        }
    }

    cv::Mat R;
    cv::Mat t;
    cv::Mat mask;
    cv::Mat K = cv::eigen2cv(camera_model_->K_l());
    cv::Mat E = cv::findEssentialMat(corresponding_points_a, corresponding_points_b, K, cv::RANSAC, 0.999, kMaxPixelErrorForCalcEssentialMat, mask);
    int inlier_cnt = cv::recoverPose(E, corresponding_points_a, corresponding_points_b, K, R, t, mask); // relative pose is a to b
    if (inlier_cnt < kMinInliersForRelativePose)
    {
        LOG(INFO) << "Not enough inliers to calculate relative pose";
        return false;
    }

    Eigen::Matrix3d R_relative = cv::cv2eigen(R).transpose();
    Eigen::Vector3d t_relative = -R_relative * cv::cv2eigen(t);
    return true;
}

void Sfm::triangulateFramePoints(const std::vector<CameraObs>& obs_A, const std::vector<CameraObs>& obs_B, const Pose& pose_a, const Pose& pose_b)
{
    constexpr double kMaxDifferenceRatio = 0.15f;

    if (obs_A.size() < kMinRequiredFeaturesPerFrame || obs_B.size() < kMinRequiredFeaturesPerFrame)
    {
        LOG(INFO) << "Not enough observations to triangulate points";
        return;
    }

    std::unordered_map<uint32_t, CameraObs> obs_A_umap;
    std::unordered_map<uint32_t, CameraObs> obs_B_umap;
    for (auto& obs : obs_A)
    {
        obs_A_umap.insert({obs.feat_id, obs});
    }
    for (auto& obs : obs_B)
    {
        obs_B_umap.insert({obs.feat_id, obs});
    }

    for (auto& [point_id, obs_a] : obs_A_umap)
    {
        if (obs_B_umap.count(point_id) == 0)
        {
            continue;
        }
        CameraObs obs_b = obs_B_umap[point_id];
        Eigen::Vector2d obs_a_norm(obs_a.u_norm, obs_a.v_norm);
        Eigen::Vector2d obs_b_norm(obs_b.u_norm, obs_b.v_norm);
        Eigen::Vector3d point_3d = triangulatePoint(pose_a, pose_b, obs_a_norm, obs_b_norm);

        // Update sfm feature base
        if (all_features_.find(point_id) != all_features_.end())
        {
            Feature& feature = all_features_[point_id];
            if ((feature._pwf - point_3d).norm() / feature._pwf.norm() > kMaxDifferenceRatio)
            {
                all_features_.erase(point_id);
                continue;
            }
            feature._visual_obs_buffer.insert({obs_a.ts_sec, obs_a});
            feature._visual_obs_buffer.insert({obs_b.ts_sec, obs_b});
        }
        else
        {
            Feature feature;
            feature._id = point_id;
            feature._pwf = point_3d;
            feature._visual_obs_buffer.insert({obs_a.ts_sec, obs_a});
            feature._visual_obs_buffer.insert({obs_b.ts_sec, obs_b});
            all_features_.insert({point_id, feature});
        }
    }
}

Eigen::Vector3d Sfm::triangulatePoint(const Pose pose0, const Pose pose1, const Vector2d& point0, const Vector2d& point1)
{
    Eigen::Matrix3d Rwc0 = pose0.R();
    Eigen::Matrix3d Rwc1 = pose1.R();
    Eigen::Vector3d twc0 = pose0.t();
    Eigen::Vector3d twc1 = pose1.t();
    Eigen::Matrix4d T_wtoc0 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_wtoc1 = Eigen::Matrix4d::Identity();
    T_wtoc0.block<3, 3>(0, 0) = Rwc0.transpose();
    T_wtoc1.block<3, 1>(0, 3) = -Rwc0.transpose() * twc0;
    T_wtoc1.block<3, 3>(0, 0) = Rwc1.transpose();
    T_wtoc1.block<3, 1>(0, 3) = -Rwc1.transpose() * twc1;

    Matrix4d A = Matrix4d::Zero();
    A.row(0) = point0[0] * T_wtoc0.row(2) - T_wtoc0.row(0);
    A.row(1) = point0[1] * T_wtoc0.row(2) - T_wtoc0.row(1);
    A.row(2) = point1[0] * T_wtoc1.row(2) - T_wtoc1.row(0);
    A.row(3) = point1[1] * T_wtoc1.row(2) - T_wtoc1.row(1);
    Vector4d triangulated_point = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);

    return point_3d;
}

bool Sfm::solveFrameByPnp(const std::vector<CameraObs> current_obsv, Pose &current_pose)
{
    constexpr double kMaxPixelErrorForPnp = 1.0f;
    constexpr uint32_t kMinRequiredFeaturesForPnp = 15;
    if (current_obsv.size() < kMinRequiredFeaturesForPnp)
    {
        LOG(INFO) << "Not enough observations to solve frame by PnP";
        return false;
    }

    std::unordered_map<uint32_t, CameraObs> current_obsv_umap;
    for (auto& obs : current_obsv)
    {
        current_obsv_umap.insert({obs.feat_id, obs});
    }

    std::vector<cv::Point2d> corresponding_points;
    std::vector<cv::Point3d> corresponding_points_3d;
    for (auto& [point_id, obs] : current_obsv_umap)
    {
        if (all_features_.count(point_id) == 0)
        {
            continue;
        }
        Feature& feature = all_features_[point_id];
        corresponding_points.push_back(cv::Point2d(obs.u, obs.v));
        corresponding_points_3d.push_back(cv::Point3d(feature._pwf(0), feature._pwf(1), feature._pwf(2)));
    }

    cv::Mat rvec, tvec;
    cv::Mat K = cv::eigen2cv(camera_model_->K_l());
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    cv::solvePnP(corresponding_points_3d, corresponding_points, K, dist_coeffs, rvec, tvec);    // R_wtoc, p_winc

    Eigen::Matrix3d R = cv::cv2eigen(cv::Rodrigues(rvec)).transpose();
    Eigen::Vector3d t = -R * cv::cv2eigen(tvec);

    current_pose = Pose(R, t);
    return true;
}

double Sfm::findReferenceKeyframeTimestamp()
{
    const double oldest_keyframe_timestamp = all_feature_observes_.begin()->first;
    const std::vector<CameraObs> oldest_keyframe_observes = all_feature_observes_.begin()->second;
    std::unordered_map<uint32_t, CameraObs> oldest_keyframe_observe_umap;
    for (auto& obs : oldest_keyframe_observes)
    {
        oldest_keyframe_observe_umap.insert({obs.feat_id, obs});
    }

    double largest_pixel_parallex = std::numeric_limits<double>::min();
    double reference_keyframe_timestamp = all_feature_observes_.begin()->first;
    for (auto &[timestamp, observes] : all_feature_observes_)
    {
        std::unordered_map<uint32_t, CameraObs> current_keyframe_observe_umap;
        for (auto& obs : observes)
        {
            current_keyframe_observe_umap.insert({obs.feat_id, obs});
        }

        double pixel_parallex = VisualManager::calcVisualObsParallex(oldest_keyframe_observe_umap, current_keyframe_observe_umap);
        if (pixel_parallex >= largest_pixel_parallex)
        {
            largest_pixel_parallex = pixel_parallex;
            reference_keyframe_timestamp = timestamp;
        }
    }

    return reference_keyframe_timestamp;
}

bool Sfm::initSfmSolver()
{
    // Find reference keyframe having the largest visual parallex pixels with the oldest keyframe
    oldest_keyframe_timestamp_ = all_feature_observes_.begin()->first;
    const std::vector<CameraObs> oldest_keyframe_observes = all_feature_observes_.begin()->second;
    reference_keyframe_timestamp_ = findReferenceKeyframeTimestamp();

    // Calculate the up-to-scale relative pose between the oldest keyframe and the reference keyframe
    Eigen::Matrix3d R_rto0;
    Eigen::Vector3d p_rin0;
    std::vector<CameraObs> reference_keyframe_observes = all_feature_observes_[reference_keyframe_timestamp_];
    if (!calcRelativePose(oldest_keyframe_observes, reference_keyframe_observes, R_rto0, p_rin0))
    {
        LOG(INFO) << "Failed to calculate relative pose";
        return false;
    }
    Pose reference_keyframe_pose(R_rto0, p_rin0);
    Pose oldest_keyframe_pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    keyframe_poses_.insert({oldest_keyframe_timestamp_, oldest_keyframe_pose});
    keyframe_poses_.insert({reference_keyframe_timestamp_, reference_keyframe_pose});

    // Triangulate points between the oldest keyframe and the reference keyframe
    triangulateFramePoints(oldest_keyframe_observes, reference_keyframe_observes, oldest_keyframe_pose, reference_keyframe_pose);

    // Init other keyframe poses and triangulate all points
    Pose previous_pose;
    std::vector<CameraObs> previous_observes;
    for (auto it = all_feature_observes_.begin(); it != all_feature_observes_.end(); ++it)
    {
        if (it->first == oldest_keyframe_timestamp_ && it->first == reference_keyframe_timestamp)
        {
            previous_observes = it->second;
            previous_pose = keyframe_poses_[it->first];
            continue;
        }

        Pose current_pose;
        std::vector<CameraObs> current_observes = it->second;
        if (!solveFrameByPnp(current_observes, current_pose))
        {
            LOG(INFO) << "Failed to solve frame by PnP";
            return false;
        }
        keyframe_poses_.insert({it->first, current_pose});
        triangulateFramePoints(previous_observes, current_observes, previous_pose, current_pose);
        previous_observes = current_observes;
        previous_pose = current_pose;
    }
    assert(keyframe_poses_.size() == all_feature_observes_.size());

    // Clear features having less than 3 observations
    for (auto it = all_features_.begin(); it != all_features_.end();)
    {
        if (it->second._visual_obs_buffer.size() < kMinRequiredObservTimesPerFeature)
        {
            it = all_features_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Check if we have enough features for sfm
    if (all_features_.size() < kMinRequiredFeaturesForSfm)
    {
        LOG(INFO) << "Not enough features for sfm";
        return false;
    }

    return true;
}

bool Sfm::Optimization()
{
    if (!isReady())
    {
        LOG(INFO) << "Not enough keyframes for optimization";
        return false;
    }

    if (!initSfmSolver())
    {
        LOG(INFO) << "Failed to initialize sfm state";
        return false;
    }

    // TODO: Implement optimization logic here
    ceres::Problem problem;
    ceres::Manifold *quat_manifold = new ceres::QuaternionManifold();
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double qs[kRequiredKeyframesForSfm][4] = {0.f};
    double ps[kRequiredKeyframesForSfm][3] = {0.f};

    auto indexInMap = [](const std::map<double, Pose>& poses, const double timestamp) -> std::optional<int>
    {
        auto it = poses.find(timestamp);
        if (it != poses.end())
            return static_cast<int>(std::distance(poses.begin(), it));
        return std::nullopt;  // Not found
    };

    // Set parameter blocks for keyframe poses
    for (const auto& [timestamp, pose] : keyframe_poses_)
    {
        std::optional<int> idx = indexInMap(keyframe_poses_, timestamp);
        if (!idx.has_value())
        {
            LOG(INFO) << "Keyframe pose not found for timestamp: " << timestamp;
            return false;
        }
        Eigen::Quaterniond q(pose.R());
        Eigen::Vector3d p(pose.t());
        qs[idx][0] = q.w();
        qs[idx][1] = q.x();
        qs[idx][2] = q.y();
        qs[idx][3] = q.z();

        ps[idx][0] = p.x();
        ps[idx][1] = p.y();
        ps[idx][2] = p.z();

        problem.AddParameterBlock(qs[idx], 4, quat_manifold);
        problem.AddParameterBlock(ps[idx], 3);

        if (timestamp == oldest_keyframe_timestamp_)
        {
            problem.SetParameterBlockConstant(qs[i]);
            problem.SetParameterBlockConstant(ps[i]);
        }
        else if (timestamp == reference_keyframe_timestamp_)
        {
            problem.SetParameterBlockConstant(ps[i]);
        }
    }

    // Add residuals for each feature

}