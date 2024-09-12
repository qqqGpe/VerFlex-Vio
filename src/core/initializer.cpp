#include "initializer.h"
#include <glog/logging.h>

namespace {
    constexpr double kTimeDurationForInit = 1.0f;
}

void Initializer::feed_imu_measurement(const ImuData & data) {
    imu_data.push_back(data);
    std::sort(imu_data.begin(), imu_data.end());
    while (imu_data.size() > kInitializeImuQueSize)
    {
        imu_data.pop_front();
    }
}

// Gram-Schmidt正交化
Eigen::Matrix3d Initializer::Gram_Schmidt(const Eigen::Vector3d &gravity_inI) {
    Eigen::Vector3d e1(0, 1.0, 0);
    Eigen::Vector3d z_axis = -gravity_inI.normalized();
    Eigen::Vector3d x_axis = z_axis.cross(e1);
    x_axis = x_axis.eval().normalized();
    Eigen::Vector3d y_axis = z_axis.cross(x_axis);
    y_axis = y_axis.eval().normalized();
    Eigen::Matrix3d R_GtoI;
    R_GtoI.block<3, 1>(0, 0) = x_axis;
    R_GtoI.block<3, 1>(0, 1) = y_axis;
    R_GtoI.block<3, 1>(0, 2) = z_axis;
    return R_GtoI;
}

bool Initializer::static_initialize(std::shared_ptr<State> &state)
{
    LOG(INFO) << "trying to initialize with static states";
    if(is_initialized) {
        LOG(WARNING) << "system has already been initialized!";
        return false;
    }

    if(imu_data.size() < kInitializeImuQueSize) {
        LOG(INFO) << "Static initialization failed reason: not enough imu data for initialization";
        return false;
    }

    std::vector<ImuData> imu_data_for_init;
    for(const auto &data : imu_data) {
        double start_ts = imu_data.begin()->ts_sec;
        if(data.ts_sec - start_ts <= kTimeDurationForInit) {
            imu_data_for_init.push_back(data);
        }
    }

    Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();

    for(const ImuData &data : imu_data_for_init) {
        acc_mean += data.am;
        gyro_mean += data.wm;
    }
    acc_mean = acc_mean / imu_data_for_init.size();     // average acceleration
    gyro_mean = gyro_mean / imu_data_for_init.size();   // average angular velocity

    double acc_var = 0.0;
    for(const ImuData &data : imu_data_for_init) {
        acc_var += (data.am - acc_mean).dot(data.am - acc_mean);
    }
    acc_var = acc_var / imu_data_for_init.size();       // 加计的方差，若方差小于阈值则认为系统处于静止状态。

    if(acc_var > static_acc_var_thres) {
        LOG(INFO) << "Static initialization failed reason: platform is moving";
        return false;
    }

    Eigen::Matrix3d R_GtoI = Gram_Schmidt(acc_mean);

    Eigen::Vector3d gravity_inG(0, 0, -gravity_mag);
    Eigen::Vector3d init_bg = gyro_mean;
    Eigen::Vector3d init_ba = acc_mean - R_GtoI * gravity_inG;

    std::shared_ptr<IMU_state> &imu_state = state->_imu_state;

    // initialize static imu state
    Eigen::VectorXd init_imu_state = Eigen::VectorXd::Zero(16);
    Eigen::Quaterniond q_GtoI(R_GtoI);
    init_imu_state.block<4, 1>(0, 0) = q_GtoI.inverse().coeffs();   // q_ItoG
    init_imu_state.block<3, 1>(10, 0) = init_bg;
    init_imu_state.block<3, 1>(13, 0) = init_ba;
    imu_state->set_value(init_imu_state);

    // initialize static imu covariance
    Eigen::MatrixXd init_imu_covariance = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(imu_state->size(), imu_state->size());
    init_imu_covariance.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity(); // p
    init_imu_covariance.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity(); // v (static)
    imu_state->set_covariance(init_imu_covariance);
    state->_covariance.block(imu_state->id(), imu_state->id(), imu_state->size(), imu_state->size()) = init_imu_covariance;
    imu_state->set_ts(imu_data_for_init.back().ts_sec);

    LOG(INFO) << "static inialization success!";
    std::cout << "ba: " << imu_state->ba()->vec().transpose() << std::endl;
    std::cout << "bg: " << imu_state->bg()->vec().transpose() << std::endl;

    is_initialized = true;
    return true;
}