#include "sfm.h"
#include <optional>

bool Sfm::MaybeAddSfmKeyframes(const std::pair<double, std::vector<CameraObs>>& current_feature_observe,
                               std::optional<std::pair<double, cv::Mat>> image)
{
    if (current_feature_observe.second.size() < kMinRequiredFeaturesPerFrame)
    {
        LOG(INFO) << cv::format("Not enough features for SFM keyframe: %d < %d", static_cast<int>(current_feature_observe.second.size()), kMinRequiredFeaturesPerFrame);
        return false;
    }

    if (all_feature_observes_.empty())
    {
        all_feature_observes_.insert(current_feature_observe);
        latest_keyframe_observe_ = current_feature_observe;
        if (image.has_value())
        {
            keyframe_images_.insert_or_assign(current_feature_observe.first, image->second);
        }
        LOG(INFO) << cv::format("First keyframe added, timestamp: %f, observations: %d", current_feature_observe.first,
                                static_cast<int>(current_feature_observe.second.size()));
        return true;
    }
    else
    {
        std::unordered_map<uint32_t, CameraObs> latest_keyframe_observe_umap;
        std::unordered_map<uint32_t, CameraObs> current_keyframe_observe_umap;
        for (auto &obs : latest_keyframe_observe_.second)
        {
            latest_keyframe_observe_umap.insert({obs.feat_id, obs});
        }
        for (auto &obs : current_feature_observe.second)
        {
            current_keyframe_observe_umap.emplace(obs.feat_id, obs);
        }

        double pixel_parallex = VisualManager::calcVisualObsParallex(latest_keyframe_observe_umap, current_keyframe_observe_umap);
        if (pixel_parallex >= kMinPixelParallexBetweenKeyframes)
        {
            all_feature_observes_.insert(current_feature_observe);
            latest_keyframe_observe_ = current_feature_observe;
            if (image.has_value())
            {
                keyframe_images_.insert_or_assign(current_feature_observe.first, image->second);
            }
            LOG(INFO) << cv::format("Keyframe added, timestamp: %f, pixel parallex: %f, feature observations: %d", current_feature_observe.first,
                                    pixel_parallex, static_cast<int>(current_feature_observe.second.size()));
            return true;
        }
        else if (abs(current_feature_observe.first - latest_keyframe_observe_.first) > kMaxTimeIntervalBetweenKeyframes)
        {
            LOG(INFO) << "Platform moves too slow, can not perform dynamic-initialization";
            return false;
        }
        else
        {
            return false;
        }
    }

    return false;
}

bool Sfm::isReady() const
{
        if (all_feature_observes_.size() > kRequiredKeyframesForSfm)
        {
            LOG(ERROR) << fmt::format("Too many keyframes: {:d}, expected: {:d}", all_feature_observes_.size(), kRequiredKeyframesForSfm);
            std::exit(EXIT_FAILURE);
        }
        else if (all_feature_observes_.size() < kRequiredKeyframesForSfm)
        {
            return false;
        }
        return all_feature_observes_.size() == kRequiredKeyframesForSfm;
}

bool Sfm::calcRelativePose(const std::vector<CameraObs> &obs_a,
                           const std::vector<CameraObs> &obs_b,
                           Eigen::Matrix3d &R_BtoA,
                           Eigen::Vector3d &p_BinA)
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
    for (auto &obs : obs_a)
    {
        obs_a_umap.insert({obs.feat_id, obs});
    }
    for (auto &obs : obs_b)
    {
        obs_b_umap.insert({obs.feat_id, obs});
    }

    std::vector<cv::Point2d> corresponding_points_a;
    std::vector<cv::Point2d> corresponding_points_b;
    for (auto &[point_id, obs] : obs_a_umap)
    {
        auto it = obs_b_umap.find(point_id);
        if (it != obs_b_umap.end())
        {
            corresponding_points_a.push_back(cv::Point2d(obs.uv.at(LEFT_CAM).x(), obs.uv.at(LEFT_CAM).y()));
            corresponding_points_b.push_back(cv::Point2d(it->second.uv.at(LEFT_CAM).x(), it->second.uv.at(LEFT_CAM).y()));
        }
    }

    cv::Mat R_cv;
    cv::Mat t_cv;
    cv::Mat mask;
    cv::Mat K_cv;
    Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
    cv::eigen2cv(K, K_cv);
    cv::Mat E = cv::findEssentialMat(corresponding_points_a, corresponding_points_b, K_cv, cv::RANSAC, 0.999, kMaxPixelErrorForCalcEssentialMat, mask);
    int inlier_cnt = cv::recoverPose(E, corresponding_points_a, corresponding_points_b, K_cv, R_cv, t_cv, mask); // relative pose is R_AtoB, p_AinB
    if (inlier_cnt < kMinInliersForRelativePose)
    {
        LOG(INFO) << "Not enough inliers to calculate relative pose";
        return false;
    }

    Eigen::Matrix3d R_AtoB;
    Eigen::Vector3d p_AinB;
    cv::cv2eigen(R_cv, R_AtoB);
    cv::cv2eigen(t_cv, p_AinB);
    R_BtoA = R_AtoB.transpose();
    p_BinA = R_BtoA * (-p_AinB);
    return true;
}

void Sfm::triangulateFramePoints(const std::vector<CameraObs> &obs_A, const std::vector<CameraObs> &obs_B, const Pose &pose_a, const Pose &pose_b)
{
    constexpr double kMaxDifferenceRatio = 0.15f;

    if (obs_A.size() < kMinRequiredFeaturesPerFrame || obs_B.size() < kMinRequiredFeaturesPerFrame)
    {
        LOG(INFO) << "Not enough observations to triangulate points";
        return;
    }

    std::unordered_map<uint32_t, CameraObs> obs_A_umap;
    std::unordered_map<uint32_t, CameraObs> obs_B_umap;
    for (auto &obs : obs_A)
    {
        obs_A_umap.emplace(obs.feat_id, obs);
    }
    for (auto &obs : obs_B)
    {
        obs_B_umap.emplace(obs.feat_id, obs);
    }

    for (auto &[point_id, obs_a] : obs_A_umap)
    {
        if (all_features_.find(point_id) != all_features_.end())
        {
            Feature &feature = all_features_[point_id];
            feature._visual_obs_buffer.insert_or_assign(obs_a.ts_sec, obs_a);
            if (obs_B_umap.count(point_id) != 0)
            {
                CameraObs obs_b = obs_B_umap[point_id];
                feature._visual_obs_buffer.insert_or_assign(obs_b.ts_sec, obs_b);
            }
            continue;
        }
        else
        {
            if (obs_B_umap.count(point_id) == 0)
            {
                continue;
            }

            CameraObs obs_b = obs_B_umap[point_id];
            Eigen::Vector2d obs_a_norm(obs_a.uv_norm.at(LEFT_CAM).x(), obs_a.uv_norm.at(LEFT_CAM).y());
            Eigen::Vector2d obs_b_norm(obs_b.uv_norm.at(LEFT_CAM).x(), obs_b.uv_norm.at(LEFT_CAM).y());
            Eigen::Vector3d point_3d = triangulatePoint(pose_a, pose_b, obs_a_norm, obs_b_norm);

            // Add new sfm feature
            Feature feature;
            feature._id = point_id;
            feature._pwf = point_3d;
            feature._is_triangulated = true;
            feature._visual_obs_buffer.emplace(obs_a.ts_sec, obs_a);
            feature._visual_obs_buffer.emplace(obs_b.ts_sec, obs_b);
            all_features_.emplace(point_id, feature);
        }
    }
}

Eigen::Vector3d Sfm::triangulatePoint(const Pose pose0, const Pose pose1, const Vector2d &point0, const Vector2d &point1)
{
    Eigen::Matrix3d Rwc0 = pose0.quat().toRotationMatrix();
    Eigen::Matrix3d Rwc1 = pose1.quat().toRotationMatrix();
    Eigen::Vector3d twc0 = pose0.p();
    Eigen::Vector3d twc1 = pose1.p();
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
    Eigen::Vector3d point_3d;
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);

    return point_3d;
}

Eigen::Matrix3d generateRandomSmallRotation() {

    auto random_angle = []() -> double
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 0.5);
        return dis(gen);
    };
    Eigen::Vector3d axis(0, 0, 0);
    double angle = random_angle();
    Eigen::AngleAxisd rotation(angle, axis);
    return rotation.toRotationMatrix();
}

bool Sfm::solveFrameByPnp(const std::vector<CameraObs> current_obsv, Pose &current_pose)
{
    constexpr double kMaxPixelErrorForPnp = 1.0f;
    constexpr uint32_t kMinRequiredFeaturesForPnp = 15;

    std::unordered_map<uint32_t, CameraObs> current_obsv_umap;
    for (auto &obs : current_obsv)
    {
        current_obsv_umap.emplace(obs.feat_id, obs);
    }

    std::vector<cv::Point2d> corresponding_points;
    std::vector<cv::Point3d> corresponding_points_3d;
    for (auto &[point_id, obs] : current_obsv_umap)
    {
        if (all_features_.count(point_id) == 0)
        {
            continue;
        }
        Feature &feature = all_features_[point_id];
        corresponding_points.push_back(cv::Point2d(obs.uv.at(LEFT_CAM).x(), obs.uv.at(LEFT_CAM).y()));
        corresponding_points_3d.push_back(cv::Point3d(feature._pwf(0), feature._pwf(1), feature._pwf(2)));
    }

    if (corresponding_points.size() < kMinRequiredFeaturesForPnp)
    {
        LOG(INFO) << "Not enough observations to solve frame by PnP";
        return false;
    }

    // For debug
    // std::cout << cv::format("origin obv: %zu", current_obsv.size()) << std::endl;
    std::cout << fmt::format("corresponding_point: {}", corresponding_points.size()) << std::endl;
    std::cout << fmt::format("corresponding_points_3d: {}", corresponding_points_3d.size()) << std::endl;
    cv::Mat K_cv;
    cv::Mat rvec, tvec;
    Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
    cv::eigen2cv(K, K_cv);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<int> inliers;
    // cv::solvePnP(corresponding_points_3d, corresponding_points, K, dist_coeffs, rvec, tvec); // R_wtoc, p_winc
    cv::solvePnPRansac(corresponding_points_3d, corresponding_points, K_cv, dist_coeffs, rvec, tvec, false, 100, kMaxPixelErrorForPnp, 0.9, inliers,
                       cv::SOLVEPNP_EPNP);  // R_wtoc, p_winc

    cv::Mat R_GtoC_cv;
    cv::Rodrigues(rvec, R_GtoC_cv);
    Eigen::Matrix3d R_GtoC;
    Eigen::Vector3d p_GinC;
    cv::cv2eigen(R_GtoC_cv, R_GtoC);
    cv::cv2eigen(tvec, p_GinC);
    // Eigen::Matrix3d R_CtoG = R_GtoC.transpose() * generateRandomSmallRotation(); // For debug
    Eigen::Matrix3d R_CtoG = R_GtoC.transpose();
    Eigen::Vector3d p_CinG = -R_CtoG * p_GinC;

    current_pose = Pose(R_CtoG, p_CinG);
    double current_timestamp = current_obsv.begin()->ts_sec;
    current_pose.set_ts(current_timestamp);
    return true;
}

double Sfm::findReferenceKeyframeTimestamp()
{
    const double oldest_keyframe_timestamp = all_feature_observes_.begin()->first;
    const std::vector<CameraObs> oldest_keyframe_observes = all_feature_observes_.begin()->second;
    std::unordered_map<uint32_t, CameraObs> oldest_keyframe_observe_umap;
    for (auto &obs : oldest_keyframe_observes)
    {
        oldest_keyframe_observe_umap.emplace(obs.feat_id, obs);
    }

    double largest_pixel_parallex = std::numeric_limits<double>::min();
    double reference_keyframe_timestamp = all_feature_observes_.begin()->first;
    for (auto &[timestamp, observes] : all_feature_observes_)
    {
        std::unordered_map<uint32_t, CameraObs> current_keyframe_observe_umap;
        for (auto &obs : observes)
        {
            current_keyframe_observe_umap.insert({obs.feat_id, obs});
        }

        double pixel_parallex = VisualManager::calcVisualObsParallex(oldest_keyframe_observe_umap, current_keyframe_observe_umap);
        std::map<uint32_t, CameraObs> covisible_features =
            VisualManager::covisibleFeatures(oldest_keyframe_observe_umap, current_keyframe_observe_umap);

        if (pixel_parallex >= largest_pixel_parallex && covisible_features.size() >= kMinRequiredFeaturesPerFrame)
        {
            largest_pixel_parallex = pixel_parallex;
            reference_keyframe_timestamp = timestamp;
        }
    }

    return reference_keyframe_timestamp;
}

void Sfm::showKeyframeImages() const
{
    if (keyframe_images_.empty())
    {
        LOG(INFO) << "No keyframe images to show";
        return;
    }

    const uint32_t kScale = 2;;
    const cv::Mat image_tmp = keyframe_images_.begin()->second.clone();
    const uint32_t width = image_tmp.cols / kScale;
    const uint32_t height = image_tmp.rows / kScale;
    cv::Mat Canvas(4 * height, 4 * width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Mat> keyframes;
    for (const auto &[timestamp, keyframe_image] : keyframe_images_)
    {
        cv::Mat resized_image;
        cv::resize(keyframe_image, resized_image, cv::Size(width, height));
        cv::cvtColor(resized_image, resized_image, cv::COLOR_GRAY2BGR); // Convert BGR to RGB for visualization
        const std::vector<CameraObs> observations = all_feature_observes_.at(timestamp);  // Can not use [] at const function
        for (const auto &obs : observations)
        {
            const uint32_t cb = (obs.feat_id * 13) % 255;
            const uint32_t cg = (obs.feat_id * 23) % 255;
            const uint32_t cr = (obs.feat_id * 33) % 255;
            cv::circle(resized_image, cv::Point(obs.uv.at(LEFT_CAM).x() / kScale, obs.uv.at(LEFT_CAM).y() / kScale), 3, cv::Scalar(cb, cg, cr), -1);
        }
        keyframes.push_back(resized_image);
        cv::imshow("Keyframe Image", resized_image);
        cv::waitKey(0);
    }
    for (uint32_t i = 0; i < keyframes.size(); ++i)
    {
        uint32_t row = i / 4;
        uint32_t col = i % 4;
        keyframes[i].copyTo(Canvas(cv::Rect(col * width, row * height, width, height)));
    }
    cv::imshow("Keyframe Images", Canvas);
    cv::waitKey(0);
}

void Sfm::ShowPoseWrtFirstFrame(const Pose current_pose)
{
    Pose first_pose = keyframe_poses_[oldest_keyframe_timestamp_];
    Eigen::Vector3d rpy = MathUtils::R2rpy(first_pose.quat().toRotationMatrix().transpose() * current_pose.quat().toRotationMatrix()) * RAD2DEG;
    std::cout << cv::format("Relative pose between oldest keyframe and current keyframe(%f): RPY: [%f, %f, %f], p: [%f, %f, %f]", current_pose.ts(),
                            rpy(0), rpy(1), rpy(2), current_pose.p().x(), current_pose.p().y(), current_pose.p().z())
              << std::endl;
}

bool Sfm::initSfmSolver()
{
    // Find reference keyframe having the largest visual parallex pixels with the oldest keyframe
    oldest_keyframe_timestamp_ = all_feature_observes_.begin()->first;
    const std::vector<CameraObs> oldest_keyframe_observes = all_feature_observes_.begin()->second;
    reference_keyframe_timestamp_ = findReferenceKeyframeTimestamp();
    std::cout << "oldest_keyframe_timestamp_: " << oldest_keyframe_timestamp_ << std::endl;
    std::cout << "reference_keyframe_timestamp_: " << reference_keyframe_timestamp_ << std::endl;

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
    keyframe_poses_.insert_or_assign(oldest_keyframe_timestamp_, oldest_keyframe_pose);
    keyframe_poses_.insert_or_assign(reference_keyframe_timestamp_, reference_keyframe_pose);

    // ShowPoseWrtFirstFrame(reference_keyframe_pose);  // For debug
    // Triangulate points between the oldest keyframe and the reference keyframe
    triangulateFramePoints(oldest_keyframe_observes, reference_keyframe_observes, oldest_keyframe_pose, reference_keyframe_pose);

    // Init other keyframe poses and triangulate all points
    Pose previous_pose;
    std::vector<CameraObs> previous_observes;
    for (auto it = all_feature_observes_.begin(); it != all_feature_observes_.end(); ++it)
    {
        if (it->first == oldest_keyframe_timestamp_ || it->first == reference_keyframe_timestamp_)
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
        keyframe_poses_.insert_or_assign(current_pose.ts(), current_pose);

        // // For debug: Show feature reprojection
        // ShowPoseWrtFirstFrame(keyframe_poses_[current_pose.ts()]); // For debug
        // cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
        // std::cout << "current timestamp: " << current_pose.ts() << std::endl;
        // for (auto feature : all_features_)
        // {
        //     uint32_t feature_id = feature.second._id;
        //     Eigen::Matrix3d R_CtoG = keyframe_poses_[current_pose.ts()].quat().toRotationMatrix();
        //     Eigen::Vector3d p_CinG = keyframe_poses_[current_pose.ts()].p();
        //     Eigen::Matrix3d R_GtoC = R_CtoG.transpose();
        //     Eigen::Vector3d p_GinC = R_CtoG.transpose() * (-p_CinG);
        //     Eigen::Vector3d pwf = feature.second._pwf;
        //     Eigen::Vector3d pcf = R_GtoC * pwf + p_GinC;
        //     Eigen::Vector3d uv3 = camera_model_->K_l() * (pcf / pcf(2));
        //     uint32_t cb = (feature_id * 13) % 255;
        //     uint32_t cg = (feature_id * 33) % 255;
        //     uint32_t cr = (feature_id * 47) % 255;
        //     cv::circle(image, cv::Point(uv3.x(), uv3.y()), 3, cv::Scalar(cb, cg, cr), -1);
        // }
        // cv::imshow("reproject reference Observations", image);
        // cv::waitKey(0);

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
        LOG(INFO) << cv::format("Not enough points: %d, for sfm, minimum required points: %d", static_cast<int>(all_features_.size()), kMinRequiredFeaturesForSfm);
        return false;
    }

    // showKeyframeImages();    // Debug: Show all images with features
    return true;
}

template <typename KeyType, typename ValueType>
std::optional<uint32_t> Sfm::indexInMap(const std::map<KeyType, ValueType> &map, const KeyType &key) const
{
    auto it = map.find(key);
    if (it != map.end())
    {
        return static_cast<uint32_t>(std::distance(map.begin(), it));
    }
    return std::nullopt; // Not found
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

    ceres::Problem problem;
    ceres::Manifold *quat_manifold = new ceres::EigenQuaternionManifold();
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    double qs[kRequiredKeyframesForSfm][4] = {0.f};
    double ps[kRequiredKeyframesForSfm][3] = {0.f};

    // Set parameter blocks for keyframe poses
    for (const auto &[timestamp, pose] : keyframe_poses_)
    {
        std::optional<uint32_t> pose_idx = indexInMap<double, Pose>(keyframe_poses_, timestamp);
        if (!pose_idx.has_value())
        {
            LOG(INFO) << "Keyframe pose not found for timestamp: " << timestamp;
            return false;
        }
        uint32_t idx = pose_idx.value();
        Eigen::Quaterniond q(pose.quat().toRotationMatrix());
        Eigen::Vector3d p(pose.p());
        qs[idx][0] = q.x();
        qs[idx][1] = q.y();
        qs[idx][2] = q.z();
        qs[idx][3] = q.w();

        ps[idx][0] = p.x();
        ps[idx][1] = p.y();
        ps[idx][2] = p.z();

        problem.AddParameterBlock(qs[idx], 4, quat_manifold);
        problem.AddParameterBlock(ps[idx], 3);

        if (timestamp == oldest_keyframe_timestamp_)
        {
            problem.SetParameterBlockConstant(qs[idx]);
            problem.SetParameterBlockConstant(ps[idx]);
        }
        else if (timestamp == reference_keyframe_timestamp_)
        {
            problem.SetParameterBlockConstant(ps[idx]);
        }
    }

    // Add residuals for each feature
    uint32_t all_feature_nums = all_features_.size();
    double feature_3d[kMaxFeaturesForSfm][3] = {0.f};
    for (const auto &[feature_id, feature] : all_features_)
    {
        uint32_t feature_idx = indexInMap<uint32_t, Feature>(all_features_, feature_id).value();
        feature_3d[feature_idx][0] = feature._pwf(0);
        feature_3d[feature_idx][1] = feature._pwf(1);
        feature_3d[feature_idx][2] = feature._pwf(2);
        problem.AddParameterBlock(feature_3d[feature_idx], 3);

        for (const auto &[timestamp, single_obs] : feature._visual_obs_buffer)
        {
            uint32_t pose_idx = indexInMap<double, Pose>(keyframe_poses_, timestamp).value();
            ceres::CostFunction *cost_function = new FeatureReprojectionFactor(single_obs);
            problem.AddResidualBlock(cost_function, loss_function, feature_3d[feature_idx], qs[pose_idx], ps[pose_idx]);
        }
    }

    std::cout << "Pose before optimization: " << std::endl;
    for (auto &[timestamp, pose] : keyframe_poses_)
    {
        Eigen::Vector3d rpy = MathUtils::R2rpy(pose.quat().toRotationMatrix()) * RAD2DEG;
        std::cout << cv::format("timestamp: %f, rpy: [%f, %f, %f], p: [%f, %f, %f]", timestamp, rpy.x(), rpy.y(), rpy.z(), pose.p().x(),
                                pose.p().y(), pose.p().z())
                << std::endl;
    }

    // Solve full bundle adjustment
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 15;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // if (summary.termination_type == ceres::CONVERGENCE)
    // {
    //     LOG(INFO) << "SFM finished with " << summary.final_cost << " cost";
    // }
    // else
    // {
    //     LOG(INFO) << "SFM optimization failed";

    //     for (int i = 0; i < kRequiredKeyframesForSfm; i++)
    //     {
    //         Eigen::Quaterniond q_updated(qs[i][3], qs[i][0], qs[i][1], qs[i][2]);   // w, x, y, z
    //         Eigen::Vector3d p_updated(ps[i][0], ps[i][1], ps[i][2]);
    //         Eigen::Vector3d rpy = MathUtils::R2rpy(q_updated.toRotationMatrix()) * RAD2DEG;
    //         std::cout << cv::format("index: %d, rpy: [%f, %f, %f], p: [%f, %f, %f]", i, rpy.x(), rpy.y(), rpy.z(), p_updated.x(),
    //                                 p_updated.y(), p_updated.z())
    //                 << std::endl;
    //     }
    //     return false;
    // }

    // Update keyframe poses
    LOG(INFO) << "Pose after optimization: " << std::endl;
    for (auto& [timestamp, pose] : keyframe_poses_)
    {
        uint32_t pose_idx = indexInMap<double, Pose>(keyframe_poses_, timestamp).value();
        Eigen::Quaterniond q_updated(qs[pose_idx][3], qs[pose_idx][0], qs[pose_idx][1], qs[pose_idx][2]);  // w, x, y, z
        Eigen::Vector3d p_updated(ps[pose_idx][0], ps[pose_idx][1], ps[pose_idx][2]);
        Eigen::Vector3d rpy = MathUtils::R2rpy(q_updated.toRotationMatrix()) * RAD2DEG;
        LOG(INFO) << fmt::format("timestamp: {:f}, rpy: [{:f}, {:f}, {:f}], p: [{:f}, {:f}, {:f}]", timestamp, rpy.x(), rpy.y(), rpy.z(),
                                 p_updated.x(), p_updated.y(), p_updated.z())
                  << std::endl;
        keyframe_poses_[timestamp].set_pose(q_updated.toRotationMatrix(), p_updated);
    }

    // Update features
    for (auto &[feature_id, feature] : all_features_)
    {
        uint32_t feature_idx = indexInMap<uint32_t, Feature>(all_features_, feature_id).value();
        feature._pwf << feature_3d[feature_idx][0], feature_3d[feature_idx][1], feature_3d[feature_idx][2];
    }

    return true;
}