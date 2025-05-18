#include "imuPreIntegration.h"

inline static ImuPreintegrator ImuPreintegrator::midPointIntegrate(std::shared_ptr<State> state,
                                         const std::vector<ImuData> imu_input,
                                         const double ts_start,
                                         const double ts_end)
{
    if (ts_start >= ts_end || imu_input.empty())
    {
        LOG(WARNING) << "Invalid imu data or time range for mid-point integration";
        return;
    }

    const uint32_t p_id = state->_imu_state->p()->id();
    const uint32_t v_id = state->_imu_state->v()->id();
    const uint32_t theta_id = state->_imu_state->q()->id();
    // const uint32_t ba_id = state->_imu_state->ba()->id();
    // const uint32_t bg_id = state->_imu_state->bg()->id();

    constexpr uint32_t kNoiseAccId = 0;
    constexpr uint32_t kNoiseGyroId = 3;
    // constexpr uint32_t kNoiseGyroBiasId = 6;
    // constexpr uint32_t kNoiseAccBiasId = 9;

    Eigen::MatrixXd Cov_m = Eigen::MatrixXd::Identity(6, 6);
    Cov_m.block(kNoiseAccId, kNoiseAccId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_na, 2);
    Cov_m.block(kNoiseGyroId, kNoiseGyroId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_nw, 2);
    // Cov_m.block(kNoiseGyroBiasId, kNoiseGyroBiasId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_bg, 2);
    // Cov_m.block(kNoiseAccBiasId, kNoiseAccBiasId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_ba, 2);

    Eigen::Matrix3d dp = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Cov = Eigen::Matrix3d::Zero();

    Eigen::Vector3d ba = state->getImuState()->ba()->vec();
    Eigen::Vector3d bg = state->getImuState()->bg()->vec();

    for (int i = 0; i < imu_input.size() - 1; i++)
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(9, 9);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(9, 6);

        double dt = imu_input.at(i + 1).ts_sec - imu_input.at(i).ts_sec;
        double dt2 = dt * dt;
        if (dt > 0)
        {
            Eigen::Vector3d am_mid = 0.5 * (imu_input.at(i).am + imu_input.at(i + 1).am) - ba;
            Eigen::Vector3d wm_mid = 0.5 * (imu_input.at(i).wm + imu_input.at(i + 1).wm) - bg;

            dp = dp + dv * dt + 0.5 * dR * am_mid * dt2;
            dv = dv + dR * am_mid * dt;

            // For P
            A.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
            A.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
            A.block<3, 3>(p_id, theta_id) = -0.5 * dR * MathUtils::skew(am_mid * dt2);

            // For V
            A.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
            A.block<3, 3>(v_id, theta_id)  = -dR * MathUtils::skew(am_mid * dt);

            B.block<3, 3>(p_id, kNoiseAccId) = 0.5 * dR * dt2;
            B.block<3, 3>(v_id, kNoiseAccId) = dR * dt;

            dp_dba = dp_dba + dv_dba * dt - 0.5 * dR * dt2;
            dp_dbg = dp_dbg + dv_dbg * dt - 0.5 * dR * dt2 * MathUtils::skew(am_mid) * dR_dbg;
            dv_dba = dv_dba - dR * dt;
            dv_dbg = dv_dbg - dR * dt * MathUtils::skew(am_mid) * dR_dbg;

            // For R
            Eigen::Matrix3d delta_R = SO3d::exp(wm_mid * dt).matrix();
            dR = dR * delta_R;

            A.block<3, 3>(theta_id, theta_id) = delta_R.transpose();
            B.block<3, 3>(theta_id, kNoiseGyroId) = MathUtils::Jr(wm_mid) * dt;
            dR_dbg = delta_R.transpose() * dR_dbg - MathUtils::Jr(wm_mid) * dt;

            Cov = A * Cov * A.transpose() + B * Cov_m * B.transpose();
        }
    }

    ImuPreintegrator preIntegrator;
    preIntegrator.start_ts_ = ts_start;
    preIntegrator.end_ts_ = ts_end;
    preIntegrator.imu_data_ = imu_input;
    preIntegrator.imu_state_ = *state->getImuState();
    preIntegrator.set_dp(dp);
    preIntegrator.set_dv(dv);
    preIntegrator.set_dR(dR);
    preIntegrator.set_dp_dba(dp_dba);
    preIntegrator.set_dp_dbg(dp_dbg);
    preIntegrator.set_dv_dba(dv_dba);
    preIntegrator.set_dv_dbg(dv_dbg);
    preIntegrator.set_dR_dbg(dR_dbg);
    preIntegrator.set_Cov(Cov);

    return preIntegrator;
}

void ImuPreintegrator::update(const Eigen::Vector3d delta_ba, const Eigen::Vector3d delta_bg)
{
    dp_ = dp_ + dp_dba_ * delta_ba + dp_dbg_ * delta_bg;
    dv_ = dv_ + dv_dba_ * delta_ba + dv_dbg_ * delta_bg;
    dR_ = dR_ * SO3d::exp(dR_dbg_ * delta_bg).matrix();
}