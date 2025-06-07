#include "dynamicInitializer.h"

bool DynamicInitializer::isReadyToInitialize() const
{
    if (!sfm_solver->isReady())
    {
        LOG(INFO) << "SFM solver is not ready, need more movement";
        return false;
    }

    const auto &sfm_all_feature_observatioins = sfm_solver->getAllFeatureObservations();
    assert(imu_preIntegration_map_.size() == sfm_all_feature_observatioins.size() - 1);
    for (const auto &x : imu_preIntegration_map_)
    {
        if (sfm_all_feature_observatioins.find(x.second.start_ts()) == sfm_all_feature_observatioins.end() ||
            sfm_all_feature_observatioins.find(x.second.end_ts()) == sfm_all_feature_observatioins.end())
        {
            LOG(WARNING) << cv::format("Can not find %f (or %f) feature observations in SFM-solver", x.second.start_ts(), x.second.end_ts());
            // Reset();
            return false;
        }
    }
    return true;
}

void DynamicInitializer::feedVisualMeasurement(std::pair<double, std::vector<CameraObs>> feature_observes)
{
    sfm_solver->MaybeAddSfmKeyframes(feature_observes);
}

void DynamicInitializer::feedImuPreIntegration(ImuPreintegrator imu_preintegration)
{
    imu_preIntegration_map_.insert_or_assign(imu_preintegration.start_ts(), imu_preintegration);
}

void DynamicInitializer::Reset()
{
    Initializer::reset();
    is_ready_to_initialize_ = false;
    imu_state_map_->clear();
    imu_preIntegration_map_.clear();
    sfm_solver->Reset();
}

// Align SFM poses with IMU pre-integration
bool DynamicInitializer::solveGyroscopeBias()
{
    Eigen::Matrix3d A_total = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b_total = Eigen::Vector3d::Zero();
    for (const auto &[timestamp, pre_integration] : imu_preIntegration_map_)
    {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        Pose sfm_pose_i = sfm_poses_.at(pre_integration.start_ts());
        Pose sfm_pose_j = sfm_poses_.at(pre_integration.end_ts());
        Eigen::Matrix3d Rij_sfm = sfm_pose_i.R().transpose() * sfm_pose_j.R();
        Eigen::Matrix3d Rij_imu = pre_integration.dR();

        A = pre_integration.dR_dbg();
        b = Sophus::log(Rij_imu.transpose() * Rij_sfm).matrix();
        A_total += A.transpose() * A;
        b_total += A.transpose() * b;
    }

    Eigen::Vector3d delta_bg = A_total.ldlt().solve(b_total);
    if (delta_bg.hasNaN() || delta_bg.hasInf())
    {
        LOG(INFO) << "Gyroscope bias estimation failed, has NaN or Inf";
        return false;
    }

    for (auto &[timestamp, imu_state] : imu_state_map_)
    {
        imu_state->bg()->update(delta_bg);
    }

    for (const auto &[timestamp, pre_integration] : imu_preIntegration_map_)
    {
        pre_integration.Propagate(Eigen::Vector3d::Zero(), imu_state_map_->bg()->vec());
    }

    LOG(INFO) << cv::format("Gyroscope bias estimation successful, delta_bg: [%f, %f, %f]", delta_bg.x(), delta_bg.y(), delta_bg.z());
    return true;
}

template <typename KeyType, typename ValueType>
std::optional<uint32_t> indexInMap(const std::map<KeyType, ValueType> &map, const KeyType &key) const
{
    auto it = map.find(key);
    if (it != map.end())
    {
        return static_cast<uint32_t>(std::distance(map.begin(), it));
    }
    return std::nullopt; // Not found
}

bool DynamicInitializer::LinearAlignment(Eigen::VectorXd& x)
{
    constexpr uint32_t kGravityDim = 3;
    constexpr uint32_t kScaleDim = 1;

    const uint32_t H_rows = 6 * imu_preIntegration_map_.size();
    const uint32_t H_cols = 3 * imu_state_map_.size() + kGravityDim + kScaleDim;

    // Z = H * X
    // x : [v1, v2, ..., vn, gravity, scale]
    Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(H_rows, H_cols);
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(H_rows, 1);

    for (auto it = imu_preIntegration_map_.begin(); it != imu_preIntegration_map_.end(); ++it)
    {
        const auto &[timestamp, pre_integration] = *it;
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

        const uint32_t pre_integration_id = indexInMap<imu_preIntegration_map_, double>(imu_preIntegration_map_, timestamp).value();
        const uint32_t var_id = indexInMap<imu_state_map_, double>(imu_state_map_, pre_integration.start_ts()).value();

        /* Fill Hx */
        // 1. For Velocity
        Hx.block<3, 3>(3 * pre_integration_id,      3 * var_id) = -Eigen::Matrix3d::Identity() * delta_t;
        Hx.block<3, 3>(3 * pre_integration_id + 3,  3 * var_id) = -Eigen::Matrix3d::Identity();
        Hx.block<3, 3>(3 * pre_integration_id + 3,  3 * var_id + 3) = R_bk0toC0.transpose() * R_bk1toC0;

        // 2. For Gravity
        Hx.block<3, 3>(3 * pre_integration_id,      3 * var_id + 6) = 0.5 * R_bk0toC0.transpose() * delta_t * delta_t;
        Hx.block<3, 3>(3 * pre_integration_id + 3,  3 * var_id + 6) = R_bk0toC0.transpose() * delta_t;

        // 3. For Scale
        Hx.block<3, 1>(3 * pre_integration_id,      3 * var_id + 9) = R_bk0toC0.transpose() * (p_Ck1inC0 - p_Ck0inC0);

        /* Fill z */
        // 1. For Alpha
        z.segment<3>(3 * pre_integration_id) = Alpha - p_CinI + R_bk0toC0.transpose() * R_bk1toC0 * p_CinI;

        // 2. For Beta
        z.segment<3>(3 * pre_integration_id + 3) = Beta;
    }

    Eigen::MatrixXd ATA = Hx.transpose() * Hx;
    Eigen::MatrixXd ATb = Hx.transpose() * z;
    x = ATA.ldlt().solve(ATb);

    const double s = x.tail<1>()(0);    // Sfm scale
    Eigen::Vector3d gravity = x.segment<3>(H_cols - 4); // Gravity in C0 frame
    LOG(INFO) << cv::format("Gravity: [%f, %f, %f], Scale: %f", gravity.x(), gravity.y(), gravity.z(), s);
    if (fabs(gravity.norm() - 9.81) > 0.5 || s < 0)
    {
        LOG(ERROR) << "Linear alignment failed, gravity norm: " << gravity.norm() << ", scale: " << s;
        return false;
    }

    return true;
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

    const double scale = vgs.tail<1>()(0);    // scale
    Eigen::Vector3d gravity = vgs.segment<3>(H_cols - 4); // Gravity in C0 frame
    Eigen::Vector3d gravity_in_I = R_CtoI_ * gravity; // Gravity in b0 frame
    Eigen::Matrix3d R_b0toG = Eigen::FromTwoVectors(gravity_in_I.normalized(), Eigen::Vector3d(0, 0, -1.0));
    Eigen::Vector3d rpy = MathUtils::R2rpy(R_b0toG);
    rpy(2) = 0; // Set yaw to zero
    R_b0toG = MathUtils::rpy2R(rpy);

    // Assign all states in sliding window
    for (uint32_t i = 0; i < imu_state_map_->size(); ++i)
    {
        std::shared_ptr<ImuState> &imu_state_i = imu_state_map_->at(i);
        Eigen::Matrix3d R_bitoC0 = sfm_poses_.at(imu_state_i->ts()).R();
        Eigen::Vector3d p_CkinC0_bar = sfm_poses_.at(imu_state_i->ts()).p();
        Eigen::Vector3d v_bi = vgs.segment<3>(3 * i); // Velocity in body frame

        // Calculate the R_IinG
        Eigen::Matrix3d R_bitoG = R_b0toG * R_CtoI * R_bitoC0;

        // Calculate the position in G frame
        Eigen::Vector3d p_biinC0 = scale * p_CkinC0_bar + R_bitoC0 * (-p_CinI_);
        Eigen::Vector3d p_biinb0 = R_CtoI * p_biinC0 + p_CinI_;
        Eigen::Vector3d p_biinG = R_b0toG * p_biinb0;

        // Calculate the velocity in G frame
        Eigen::Vector3d v_biinG = R_bitoG * v_bi;

        // Assign State
        imu_state_i->set_pose(R_bitoG, p_biinG);
        imu_state_i->set_velocity(v_biinG);
    }

    return true;
}

bool DynamicInitializer::InitializeSystem()
{
    if (!sfm_solver->Optimization())
    {
        LOG(INFO) << "SFM optimization failed, reset initializer";
        Reset();
        return false;
    }

    sfm_poses_ = sfm_solver->getAllSfmPoses();
    // Convert sfm poses from R_CkinC0 to R_IkinC0, p_CkinC0 maintains the same
    for (auto &[timestamp, pose] : sfm_poses_)
    {
        Eigen::Matrix3d R_ItoC0 = pose.R() * R_CtoI_.transpose();
        pose.set_pose(R_ItoC0, pose.p());
    }

    if (!visualInertialAlignment())
    {
        LOG(INFO) << "Visual initial alignment failed, reset initializer";
        Reset();
        return false;
    }

    LOG(INFO) << "Dynamic initializer has successfully initialized the system";

    return true;
}