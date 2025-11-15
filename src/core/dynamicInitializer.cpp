#include "dynamicInitializer.h"

bool DynamicInitializer::isReadyToInitialize() const
{
    if (!sfm_solver->isReady())
    {
        LOG(INFO) << fmt::format("SFM solver is not ready({:d}/{:d}), need more movement",
                                 static_cast<int>(sfm_solver->getAllFeatureObservations().size()), Sfm::kRequiredKeyframesForSfm);
        return false;
    }

    const auto& sfm_all_feature_observatioins = sfm_solver->getAllFeatureObservations();
    assert(imu_preIntegration_map_.size() == sfm_all_feature_observatioins.size() - 1);
    for (const auto& x : imu_preIntegration_map_)
    {
        if (sfm_all_feature_observatioins.find(x.second.start_ts()) == sfm_all_feature_observatioins.end() ||
            sfm_all_feature_observatioins.find(x.second.end_ts()) == sfm_all_feature_observatioins.end())
        {
            LOG(WARNING) << fmt::format("Can not find {:f} (or {:f}) feature observations in SFM-solver", x.second.start_ts(), x.second.end_ts());
            return false;
        }
    }
    return true;
}

bool DynamicInitializer::feedVisualMeasurement(const std::pair<double, cv::Mat> image,
                                               const std::pair<double, std::vector<CameraObs>> feature_observes,
                                               ImuState imu_state)
{
    if (!sfm_solver->MaybeAddSfmKeyframes(feature_observes, image))
    {
        // LOG(INFO) << "Parallex is not large enough, need more movement";
        return false;
    }
    std::shared_ptr<ImuState> imu_state_ptr = std::make_shared<ImuState>();
    imu_state_ptr->set_ts(feature_observes.first);
    imu_state_map_.insert_or_assign(feature_observes.first, imu_state_ptr);

    return true;
}

void DynamicInitializer::feedImuPreIntegration(ImuPreintegrator imu_preintegration)
{
    imu_preIntegration_map_.insert_or_assign(imu_preintegration.start_ts(), imu_preintegration);
}

void DynamicInitializer::reset()
{
    is_orientation_initialized = false;
    is_bias_initialized = false;
    is_position_initialized = false;
    is_velocity_initialized = false;
    feature_obs_buffer_.clear();
    imu_data.clear();
    imu_state_map_.clear();
    imu_preIntegration_map_.clear();
    sfm_poses_.clear();
    sfm_solver->Reset();
}

// Align SFM poses with IMU pre-integration
bool DynamicInitializer::solveGyroscopeBias()
{
    Eigen::Matrix3d A_total = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b_total = Eigen::Vector3d::Zero();
    for (const auto& [timestamp, pre_integration] : imu_preIntegration_map_)
    {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        Pose sfm_pose_i = sfm_poses_.at(pre_integration.start_ts());
        Pose sfm_pose_j = sfm_poses_.at(pre_integration.end_ts());
        Eigen::Matrix3d Rij_sfm = sfm_pose_i.R().transpose() * sfm_pose_j.R();
        Eigen::Matrix3d Rij_imu = pre_integration.dR();

        A = pre_integration.dR_dbg();

        Sophus:SO3d delta_R(Rij_imu.transpose() * Rij_sfm);
        b = delta_R.log();

        A_total += A.transpose() * A;
        b_total += A.transpose() * b;
    }

    Eigen::Vector3d delta_bg = A_total.ldlt().solve(b_total);
    if (!delta_bg.allFinite())
    {
        LOG(INFO) << "Gyroscope bias estimation failed, has NaN or Inf";
        return false;
    }

    for (auto& [timestamp, imu_state] : imu_state_map_)
    {
        imu_state->bg()->update(delta_bg);
        if (imu_preIntegration_map_.find(timestamp) != imu_preIntegration_map_.end())
        {
            imu_preIntegration_map_[timestamp].Propagate(imu_state_map_[timestamp]->ba()->vec(), imu_state_map_[timestamp]->bg()->vec());
        }
    }

    Eigen::Vector3d bg = imu_state_map_.begin()->second->bg()->vec();
    LOG(INFO) << fmt::format("Gyroscope bias estimation successful, estimated bg: [{:f}, {:f}, {:f}]", bg.x(), bg.y(), bg.z());
    return true;
}

template <typename KeyType, typename ValueType>
std::optional<uint32_t> indexInMap(const std::map<KeyType, ValueType>& map, const KeyType& key)
{
    auto it = map.find(key);
    if (it != map.end())
    {
        return static_cast<uint32_t>(std::distance(map.begin(), it));
    }
    return std::nullopt;  // Not found
}

bool DynamicInitializer::LinearAlignment(Eigen::VectorXd& x)
{
    constexpr uint32_t kGravityDim = 3;
    constexpr uint32_t kScaleDim = 1;
    constexpr double kGravityNormTolerance = 0.5;  // Tolerance for gravity norm
    constexpr double kGravityNorm = 9.81;  // Expected gravity norm in m/s^2

    const uint32_t H_rows = 6 * imu_preIntegration_map_.size();
    const uint32_t H_cols = 3 * imu_state_map_.size() + kGravityDim + kScaleDim;

    // z = Hx * x
    // x : [v1, v2, ..., vn, gravity, scale]
    Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(H_rows, H_cols);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(H_rows);

    for (auto &[timestamp, pre_integration] : imu_preIntegration_map_)
    {
        double delta_t = pre_integration.end_ts() - pre_integration.start_ts();

        Eigen::Vector3d Alpha = pre_integration.dp();
        Eigen::Vector3d Beta = pre_integration.dv();

        // Get the sfm pose at the start and end timestamps of the pre-integration
        Pose sfm_pose = sfm_poses_.at(pre_integration.start_ts());
        Pose sfm_pose_next = sfm_poses_.at(pre_integration.end_ts());
        Eigen::Matrix3d R_bk0toC0 = sfm_pose.R();
        Eigen::Matrix3d R_bk1toC0 = sfm_pose_next.R();
        Eigen::Vector3d p_Ck0inC0 = sfm_pose.p();
        Eigen::Vector3d p_Ck1inC0 = sfm_pose_next.p();

        const uint32_t pre_integration_id = indexInMap<double, ImuPreintegrator>(imu_preIntegration_map_, timestamp).value();
        const uint32_t var_id = indexInMap<double, std::shared_ptr<ImuState>>(imu_state_map_, pre_integration.start_ts()).value();

        /* Fill Hx */
        // 1. For Velocity
        Hx.block<3, 3>(6 * pre_integration_id,      3 * var_id) = -Eigen::Matrix3d::Identity() * delta_t;
        Hx.block<3, 3>(6 * pre_integration_id + 3,  3 * var_id) = -Eigen::Matrix3d::Identity();
        Hx.block<3, 3>(6 * pre_integration_id + 3,  3 * var_id + 3) = R_bk0toC0.transpose() * R_bk1toC0;

        // 2. For Gravity
        Hx.block<3, 3>(6 * pre_integration_id,      H_cols - 4) = 0.5 * R_bk0toC0.transpose() * delta_t * delta_t;
        Hx.block<3, 3>(6 * pre_integration_id + 3,  H_cols - 4) = R_bk0toC0.transpose() * delta_t;

        // 3. For Scale
        Hx.block<3, 1>(6 * pre_integration_id,      H_cols - 1) = R_bk0toC0.transpose() * (p_Ck1inC0 - p_Ck0inC0);

        /* Fill z */
        // 1. For Alpha
        z.segment<3>(6 * pre_integration_id) = Alpha - p_CinI_ + R_bk0toC0.transpose() * R_bk1toC0 * p_CinI_;

        // 2. For Beta
        z.segment<3>(6 * pre_integration_id + 3) = Beta;
    }

    Eigen::MatrixXd ATA = Hx.transpose() * Hx;
    Eigen::MatrixXd ATb = Hx.transpose() * z;
    x = ATA.ldlt().solve(ATb);

    const double s = x.tail<1>()(0);                     // Sfm scale
    Eigen::Vector3d gravity = x.segment<3>(H_cols - 4);  // Gravity in C0 frame
    LOG(INFO) << fmt::format("Gravity: [{:f}, {:f}, {:f}], Gravity norm: {:f}, Scale: {:f}", gravity.x(), gravity.y(), gravity.z(), gravity.norm(), s);
    if (fabs(gravity.norm() - kGravityNorm) > kGravityNormTolerance || s < 0)
    {
        LOG(ERROR) << "Linear alignment failed, gravity norm: " << gravity.norm() << ", scale: " << s;
        return false;
    }

    return true;
}

void DynamicInitializer::assignImuState(const Eigen::VectorXd velocity_gravity_scale)
{
    const double scale = velocity_gravity_scale.tail<1>()(0);                      // scale
    Eigen::Vector3d gravity = velocity_gravity_scale.segment<3>(velocity_gravity_scale.size() - 4);   // Gravity in C0 frame
    Eigen::Vector3d gravity_in_I = R_CtoI_ * gravity;           // Gravity in b0 frame
    Eigen::Matrix3d R_b0toG = Eigen::Quaterniond::FromTwoVectors(gravity_in_I.normalized(), Eigen::Vector3d(0, 0, -1.0)).toRotationMatrix();
    Eigen::Vector3d rpy = utils::math::R2rpy(R_b0toG);
    rpy(2) = 0;  // Set yaw to zero
    R_b0toG = utils::math::rpy2R(rpy);

    // Assign all states in sliding window
    for (auto it = imu_state_map_.begin(); it != imu_state_map_.end(); ++it)
    {
        std::shared_ptr<ImuState> imu_state_i = it->second;
        Eigen::Matrix3d R_bitoC0 = sfm_poses_.at(imu_state_i->ts()).R();
        Eigen::Vector3d p_CkinC0_bar = sfm_poses_.at(imu_state_i->ts()).p();
        uint32_t idx = indexInMap<double, std::shared_ptr<ImuState>>(imu_state_map_, imu_state_i->ts()).value();
        Eigen::Vector3d v_bi = velocity_gravity_scale.segment<3>(3 * idx);  // Velocity in body frame

        // Calculate the R_IinG
        Eigen::Matrix3d R_bitoG = R_b0toG * R_CtoI_ * R_bitoC0;

        // Calculate the position in G frame
        Eigen::Vector3d p_biinC0 = scale * p_CkinC0_bar + R_bitoC0 * (-p_CinI_);
        Eigen::Vector3d p_biinb0 = R_CtoI_ * p_biinC0 + p_CinI_;
        Eigen::Vector3d p_biinG = R_b0toG * p_biinb0;

        // Calculate the velocity in G frame
        Eigen::Vector3d v_biinG = R_bitoG * v_bi;

        // Assign Imu states
        imu_state_i->set_pose(R_bitoG, p_biinG);
        imu_state_i->set_velocity(v_biinG);

        Eigen::Vector3d rpy = utils::math::R2rpy(R_bitoG) * 180.0 / M_PI;  // Convert to degrees
        LOG(INFO) << fmt::format(
            "\033[32mDynamic initialized imu state at {:f}: RPY: [{:f}, {:f}, {:f}], Position: [{:f}, {:f}, {:f}], Velocity: [{:f}, {:f}, {:f}]\033[0m",
            imu_state_i->ts(), rpy.x(), rpy.y(), rpy.z(), p_biinG.x(), p_biinG.y(), p_biinG.z(), v_biinG.x(), v_biinG.y(), v_biinG.z());
    }

    // initialize position and velocity
    std::shared_ptr<ImuState> last_imu_state = imu_state_map_.rbegin()->second;

    state_->reset();
    state_->set_ts_sec(last_imu_state->ts());
    Eigen::MatrixXd value = Eigen::MatrixXd::Zero(16, 1);
    value << last_imu_state->q()->q().coeffs(), last_imu_state->p()->vec(), last_imu_state->v()->vec(), last_imu_state->bg()->vec(), last_imu_state->ba()->vec();
    state_->_imu_state->set_value(value);

    if (param_.use_fej)
    {
        state_->_imu_state->pose()->set_pose_fej(last_imu_state->q()->q().toRotationMatrix(), last_imu_state->p()->vec());
    }

    // initialize dynamic initialization imu covariance
    const uint32_t kQId = state_->getImuState()->q()->id();
    const uint32_t kPId = state_->getImuState()->p()->id();
    const uint32_t kVId = state_->getImuState()->v()->id();
    const uint32_t kBgId = state_->getImuState()->bg()->id();
    const uint32_t kBaId = state_->getImuState()->ba()->id();
    const uint32_t kRicLeftId = state_->enableEstimateRic() ? state_->mutable_Qic(LEFT_CAM)->id() : -1;
    const uint32_t kRicRightId = state_->enableEstimateRic() && state_->CameraNum() == 2 ? state_->mutable_Qic(RIGHT_CAM)->id() : -1;
    const uint32_t kTdVisualId = state_->enableEstimateTdVisual() ? state_->td_visual().id() : -1;

    // Eigen::MatrixXd init_covariance = state_->getImuState().covariance();
    Eigen::MatrixXd init_covariance = Eigen::MatrixXd::Identity(state_->dim(), state_->dim());
    init_covariance.block(kQId, kQId, 3, 3) = std::pow(kInitSigmaRotation, 2) * Eigen::Matrix3d::Identity();     // q
    init_covariance.block(kPId, kPId, 3, 3) = std::pow(kInitSigmaPosition, 2) * Eigen::Matrix3d::Identity();     // p
    init_covariance.block(kVId, kVId, 3, 3) = std::pow(kInitSigmaVelocity, 2) * Eigen::Matrix3d::Identity();     // v
    init_covariance.block(kBgId, kBgId, 3, 3) = std::pow(kInitSigmaGyroBias, 2) * Eigen::Matrix3d::Identity();   // bg
    init_covariance.block(kBaId, kBaId, 3, 3) = std::pow(kInitSigmaAccelBias, 2) * Eigen::Matrix3d::Identity();  // ba
    if (kRicLeftId != -1)
    {
        init_covariance.block(kRicLeftId, kRicLeftId, 3, 3) = std::pow(kInitSigmaRic, 2) * Eigen::Matrix3d::Identity();  // qic left
    }
    if (kRicRightId != -1)
    {
        init_covariance.block(kRicRightId, kRicRightId, 3, 3) = std::pow(kInitSigmaRic, 2) * Eigen::Matrix3d::Identity();  // qic right
    }
    if (kTdVisualId != -1)
    {
        init_covariance(kTdVisualId, kTdVisualId) = std::pow(kInitSigmaTdVisual, 2);  // td_visual
    }

    state_->SetCovariance(init_covariance);

    Eigen::MatrixXd sqrt_init_covariance = init_covariance.llt().matrixL().transpose();
    state_->SetSqrtPt(sqrt_init_covariance);
}

void DynamicInitializer::assignFeatureBase(const double scale)
{
    constexpr double kMaxFeatureDistanceForInitialization = 15.0;

    std::map<uint32_t, Feature> all_features = sfm_solver->getAllFeatures();
    std::shared_ptr<Pose> first_pose = imu_state_map_.begin()->second->pose();
    Eigen::Matrix3d R_I0toG = first_pose->R();

    // Scale up the feature positions and transform them to the IMU frame
    for (auto it = all_features.begin(); it != all_features.end();)
    {
        Feature& feature = it->second;

        feature._valid = true;
        feature._is_triangulated = true;
        Eigen::Vector3d pwf = R_I0toG * (scale * R_CtoI_ * feature._pwf + p_CinI_);

        if (pwf.norm() > kMaxFeatureDistanceForInitialization)
        {
            it = all_features.erase(it);  // Remove feature if it is too far away
        }

        feature._pwf = pwf;
        it = std::next(it);
    }

    std::map<double, std::vector<CameraObs>> all_sfm_feature_observations = sfm_solver->getAllFeatureObservations();
    std::vector<CameraObs> latest_observations = all_sfm_feature_observations.rbegin()->second;
    std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> init_features;

    for (const auto& obs : latest_observations)
    {
        if (all_features.find(obs.feat_id) != all_features.end())
        {
            init_features.try_emplace(obs.feat_id, std::make_pair(obs, all_features.at(obs.feat_id)._pwf));
        }
    }

    visual_manager_->ResetFeatureBase();
    visual_manager_->InitFeatureBase(init_features);

    // // Debug: Show feature reprojections on images
    // std::map<double, cv::Mat> keyframe_images = sfm_solver->getKeyframeImages();
    // std::map<double, std::vector<CameraObs>> all_feature_observes = sfm_solver->getAllFeatureObservations();
    // for (const auto& [timestamp, feature_observes] : all_feature_observes)
    // {
    //     cv::Mat image = keyframe_images.at(timestamp);
    //     cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    //     std::shared_ptr<Pose> imu_pose = imu_state_map_.at(timestamp)->pose();
    //     for (const auto& obs : feature_observes)
    //     {
    //         if (all_features.find(obs.feat_id) != all_features.end())
    //         {
    //             const Feature& feature = all_features.at(obs.feat_id);
    //             if (feature._valid && feature._is_triangulated)
    //             {
    //                 // Draw the feature position on the image
    //                 cv::circle(image, cv::Point2f(obs.u, obs.v), 3, cv::Scalar(0, 255, 0), -1);

    //                 // Project point from IMU frame to camera frame
    //                 Eigen::Matrix3d R_ItoG = imu_pose->R();
    //                 Eigen::Vector3d pif = feature._pwf - imu_pose->p();
    //                 Eigen::Vector3d pcf = R_CtoI_.transpose() * (R_ItoG.transpose() * pif - p_CinI_);
    //                 const double depth = pcf.z();
    //                 if (depth > 0)
    //                 {
    //                     // Project the point to the image
    //                     std::ostringstream depth_ss;
    //                     depth_ss << std::fixed << std::setprecision(2) << depth;
    //                     std::string depth_text = depth_ss.str();
    //                     cv::putText(image, depth_text, cv::Point2f(obs.u, obs.v), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    //                     Eigen::Vector2d uv = camera_model_->project_left(pcf);
    //                     cv::circle(image, cv::Point2f(uv.x(), uv.y()), 4, cv::Scalar(0, 0, 255), 1);
    //                 }
    //             }
    //         }
    //     }
    //     cv::imshow("Feature Reprojections", image);
    //     cv::waitKey(0);  // Display the image for a short time
    // }
}

bool DynamicInitializer::visualInertialAlignment()
{
    if (!solveGyroscopeBias())
    {
        LOG(INFO) << "Rotation and gyro bias estimation failed, reset initializer";
        return false;
    }

    Eigen::VectorXd vgs;
    if (!LinearAlignment(vgs))
    {
        LOG(INFO) << "Linear alignment failed, reset initializer";
        return false;
    }

    assignImuState(vgs);

    assignFeatureBase(vgs.tail<1>()(0));

    is_orientation_initialized = true;
    is_position_initialized = true;
    is_velocity_initialized = true;
    is_bias_initialized = true;

    return true;
}

bool DynamicInitializer::TryInitialize()
{
    if (!isReadyToInitialize())
    {
        LOG(INFO) << "Dynamic initializer is not ready, need more movement";
        return false;
    }

    if (!InitializeSystem())
    {
        LOG(INFO) << "Dynamic initializer failed to initialize the system, reset initializer...";
        reset();
        return false;
    }

    return true;
}

bool DynamicInitializer::InitializeSystem()
{
    if (!sfm_solver->Optimization())
    {
        LOG(INFO) << "SFM optimization failed, reset initializer";
        return false;
    }

    sfm_poses_ = sfm_solver->getSfmPoses();
    // Convert sfm poses from R_CkinC0 to R_IkinC0, p_CkinC0 maintains the same
    for (auto& [timestamp, pose] : sfm_poses_)
    {
        Eigen::Matrix3d R_ItoC0 = pose.R() * R_CtoI_.transpose();
        pose.set_pose(R_ItoC0, pose.p());
    }

    if (!visualInertialAlignment())
    {
        LOG(INFO) << "Visual initial alignment failed, reset initializer";
        return false;
    }

    LOG(INFO) << "Dynamic initializer has successfully initialized the system";

    return true;
}