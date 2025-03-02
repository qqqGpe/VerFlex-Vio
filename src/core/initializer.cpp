#include "initializer.h"
#include <glog/logging.h>

namespace
{
constexpr double kTimeDurationForInit = 1.0f;
constexpr double kVisualParallexForInit = 5.f;
constexpr int kMaxVisualStereoSize = 5;
constexpr int kMinStereoFeaturesForInit = 25;
}  // namespace

void Initializer::FeedImuMeasurement(const ImuData& data)
{
    if (!imu_data.empty() && data.ts_sec <= imu_data.back().ts_sec)
    {
        return;
    }
    imu_data.push_back(data);
    while (imu_data.size() > kInitializeImuQueSize)
    {
        imu_data.pop_front();
    }
}

// Gram-Schmidt正交化
Eigen::Matrix3d Initializer::Gram_Schmidt(const Eigen::Vector3d& gravity_inI)
{
    Eigen::Vector3d z_axis = -gravity_inI / gravity_inI.norm();
    Eigen::Vector3d x_axis, y_axis;
    Eigen::Vector3d e_1(1.0, 0.0, 0.0);
    Eigen::Vector3d e_2(0.0, 1.0, 0.0);
    double inner1 = e_1.dot(z_axis) / z_axis.norm();
    double inner2 = e_2.dot(z_axis) / z_axis.norm();
    if (fabs(inner1) < fabs(inner2))
    {
        x_axis = z_axis.cross(e_1);
        x_axis = x_axis / x_axis.norm();
        y_axis = z_axis.cross(x_axis);
        y_axis = y_axis / y_axis.norm();
    }
    else
    {
        x_axis = z_axis.cross(e_2);
        x_axis = x_axis / x_axis.norm();
        y_axis = z_axis.cross(x_axis);
        y_axis = y_axis / y_axis.norm();
    }
    Eigen::Matrix3d R_GtoI;
    R_GtoI.block<3, 1>(0, 0) = x_axis;
    R_GtoI.block<3, 1>(0, 1) = y_axis;
    R_GtoI.block<3, 1>(0, 2) = z_axis;
    return R_GtoI;
}

double Initializer::PixelDistance(CameraObs obs_a, CameraObs obs_b) const
{
    double dx = obs_a.u_norm - obs_b.u_norm;
    double dy = obs_a.v_norm - obs_b.v_norm;
    return sqrt(dx * dx + dy * dy);
}

double Initializer::calcVisualObsParallex(std::unordered_map<uint32_t, CameraObs> visual_obs_a,
                                          std::unordered_map<uint32_t, CameraObs> visual_obs_b) const
{
    double average_parallex = 0.0;
    int count = 0;
    for (const auto& [feature_id, obs_a] : visual_obs_a)
    {
        if (visual_obs_b.find(feature_id) != visual_obs_b.end())
        {
            CameraObs obs_b = visual_obs_b.at(feature_id);
            double parallex = PixelDistance(obs_a, obs_b);
            average_parallex += parallex;
            count++;
        }
    }
    if (count > 0)
    {
        average_parallex = average_parallex / count;
    }
    return average_parallex;
}

bool Initializer::is_initialized()
{
    return is_orientation_initialized && is_bias_initialized && is_position_initialized && is_velocity_initialized;
}

bool Initializer::StereoVisualInitialize(const std::pair<double, std::vector<CameraObs>> feature_observes)
{
    if (is_initialized())
    {
        return false;
    }
    else if (is_orientation_initialized == false || is_bias_initialized == false)
    {
        LOG(INFO) << "Stereo visual initialization failed reason: orientation or bias not initialized!";
        return false;
    }

    LOG(INFO) << "trying to initialize with stereo visual measurements";

    double ts_sec = feature_observes.first;
    std::unordered_map<uint32_t, CameraObs> current_feature_umap;
    for (auto& obs : feature_observes.second)
    {
        current_feature_umap.insert({obs.feat_id, obs});
    }

    // Store current feature maps
    if (current_feature_umap.size() > kMinStereoFeaturesForInit)
    {
        feature_obs_buffer_.push_back(current_feature_umap);
        if (feature_obs_buffer_.size() > kMaxVisualStereoSize)
        {
            feature_obs_buffer_.pop_front();
        }
    }

    // Store stereo visual observations for first entry
    if (feature_obs_buffer_.size() < 2)
    {
        LOG(INFO) << "Stereo visual initialization failed reason: first entry, exit";
        return false;
    }

    // Calculate visual parallex between current and previous observations
    std::unordered_map<uint32_t, CameraObs> candidate_obs_umap;
    double max_average_parallex = -1.f;
    for (const auto& prev_obs_umap : feature_obs_buffer_)
    {
        double average_parallex = calcVisualObsParallex(prev_obs_umap, current_feature_umap);
        // std::cout << cv::format("average_parallex: %f, feature_obs_buffer_.size(): %d\n", average_parallex,
        // static_cast<int>(feature_obs_buffer_.size()));
        if (average_parallex > max_average_parallex)
        {
            max_average_parallex = average_parallex;
            candidate_obs_umap = prev_obs_umap;
        }
    }

    if (candidate_obs_umap.empty())
    {
        LOG(INFO) << "Stereo visual initialization failed reason: not enough parallex!";
        return false;
    }

    // Prepare stereo observation pairs
    std::vector<std::pair<CameraObs, CameraObs>> stereo_obs_pairs;
    for (const auto& [feature_id, prev_obs] : candidate_obs_umap)
    {
        if (current_feature_umap.find(feature_id) != current_feature_umap.end())
        {
            CameraObs cur_obs = current_feature_umap.at(feature_id);
            stereo_obs_pairs.push_back({prev_obs, cur_obs});
        }
    }
    if (stereo_obs_pairs.size() < kMinStereoFeaturesForInit)
    {
        LOG(INFO) << "Stereo visual initialization failed reason: not enough stereo observations";
        return false;
    }

    // triangulate stereo observations
    std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>>
        stereo_obs_triangulated;  // {feature_id, {cur_obs, pwf}}
    for (auto& [obs_prev, obs_cur] : stereo_obs_pairs)
    {
        Eigen::Vector3d pcf;
        if (!visual_manager_->StereoLeastSqureTriangulation(camera_model_, obs_prev, pcf))
        {
            continue;
        }
        stereo_obs_triangulated.try_emplace(obs_prev.feat_id, std::make_pair(obs_cur, pcf));
    }

    // calculate relative pose using Perspective-n-Point (PnP) algorithm
    Eigen::Matrix3d R_12;  // R_prev_to_curr
    Eigen::Vector3d p_12;  // t_prev_in_curr
    if (!visual_manager_->PnpRansac(camera_model_, stereo_obs_triangulated, R_12, p_12))
    {
        return false;
    }

    std::shared_ptr<IMU_state>& imu_state = state_->_imu_state;
    assert(imu_state->ts() == ts_sec);
    Eigen::Matrix3d R_ItoG = imu_state->q()->Rot();
    Eigen::Matrix3d R_CtoI = state_->_Tic->quat().toRotationMatrix();
    Eigen::Vector3d p_CpinG = -R_ItoG * R_CtoI * p_12;  // t_prev_in_G
    auto& [obs_prev, obs_cur] = stereo_obs_pairs[0];
    double delta_ts = abs(obs_cur.ts_sec - obs_prev.ts_sec);
    Eigen::Vector3d v_CinG = p_CpinG / delta_ts;  // initial velocity of camera in global coordinate
    LOG(INFO) << cv::format("Stereo visual initialization success! Initial velocity: [%f, %f, %f]", v_CinG.x(),
                            v_CinG.y(), v_CinG.z());

    // initialize position and velocity
    state_->set_ts_sec(ts_sec);
    imu_state->p()->set_value(Eigen::Vector3d::Zero());
    imu_state->v()->set_value(v_CinG);

    // initialize stereo initialization imu covariance
    Eigen::MatrixXd stereo_init_covariance = imu_state->covariance();
    stereo_init_covariance.block(imu_state->p()->id(), imu_state->p()->id(), 3, 3) = std::pow(0.1, 2) * Eigen::Matrix3d::Identity();  // p
    stereo_init_covariance.block(imu_state->v()->id(), imu_state->v()->id(), 3, 3) = std::pow(0.1, 2) * Eigen::Matrix3d::Identity();  // v
    imu_state->set_covariance(stereo_init_covariance);
    state_->_covariance.block(imu_state->id(), imu_state->id(), imu_state->size(), imu_state->size()) = stereo_init_covariance;

    // Init feature base
    std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_obs_global;
    for (auto& [feature_id, stereo_obs] : stereo_obs_triangulated)
    {
        Eigen::Vector3d feature_pwf = R_ItoG * (R_CtoI * stereo_obs.second + state_->_Tic->p());
        stereo_obs_global.try_emplace(feature_id, std::make_pair(stereo_obs.first, feature_pwf));
    }
    visual_manager_->ResetFeatureBase();
    visual_manager_->InitFeatureBase(stereo_obs_global);

    is_position_initialized = true;
    is_velocity_initialized = true;

    return true;
}

bool Initializer::static_initialize()
{
    LOG(INFO) << "trying to initialize with static states";
    if (is_orientation_initialized && is_bias_initialized)
    {
        LOG(WARNING) << "orientation has already been initialized!";
        return false;
    }

    if (imu_data.size() < kInitializeImuQueSize)
    {
        LOG(INFO) << "Static initialization failed reason: not enough imu data for initialization";
        return false;
    }

    std::vector<ImuData> imu_data_for_init;
    for (const auto& data : imu_data)
    {
        double start_ts = imu_data.begin()->ts_sec;
        if (data.ts_sec - start_ts <= kTimeDurationForInit)
        {
            imu_data_for_init.push_back(data);
        }
    }

    Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();

    for (const ImuData& data : imu_data_for_init)
    {
        acc_mean += data.am;
        gyro_mean += data.wm;
    }
    acc_mean = acc_mean / imu_data_for_init.size();    // average acceleration
    gyro_mean = gyro_mean / imu_data_for_init.size();  // average angular velocity

    double acc_var = 0.0;
    for (const ImuData& data : imu_data_for_init)
    {
        acc_var += (data.am - acc_mean).dot(data.am - acc_mean);
    }
    acc_var = acc_var / imu_data_for_init.size();

    if (acc_var > static_acc_var_thres && abs(acc_mean.norm() - gravity_mag) < 0.5)
    {
        LOG(INFO) << "Static initialization failed reason: platform is moving";
        return false;
    }

    Eigen::Matrix3d R_GtoI = Gram_Schmidt(acc_mean);
    Eigen::Vector3d gravity_inG(0, 0, -gravity_mag);
    Eigen::Vector3d init_bg = gyro_mean;
    Eigen::Vector3d init_ba = acc_mean - R_GtoI * gravity_inG;

    std::shared_ptr<IMU_state>& imu_state = state_->_imu_state;

    // initialize static imu state
    Eigen::VectorXd init_imu_state = Eigen::VectorXd::Zero(16);
    Eigen::Quaterniond q_GtoI(R_GtoI);
    init_imu_state.block<4, 1>(0, 0) = q_GtoI.inverse().coeffs();  // q_ItoG
    init_imu_state.block<3, 1>(10, 0) = init_bg;
    init_imu_state.block<3, 1>(13, 0) = init_ba;
    imu_state->set_value(init_imu_state);

    // initialize static imu covariance
    Eigen::MatrixXd init_imu_covariance =
        std::pow(0.02, 2) * Eigen::MatrixXd::Identity(imu_state->size(), imu_state->size());
    // init_imu_covariance.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity(); // p
    // init_imu_covariance.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity(); // v (static)
    imu_state->set_covariance(init_imu_covariance);
    state_->_covariance.block(imu_state->id(), imu_state->id(), imu_state->size(), imu_state->size()) =
        init_imu_covariance;
    imu_state->set_ts(imu_data_for_init.back().ts_sec);

    is_orientation_initialized = true;
    is_bias_initialized = true;

    std::cout << "ba: " << imu_state->ba()->vec().transpose() << std::endl;
    std::cout << "bg: " << imu_state->bg()->vec().transpose() << std::endl;

    LOG(INFO) << "orientation inialization success!";
    LOG(INFO) << "IMU bias initialization success!";

    return true;
}