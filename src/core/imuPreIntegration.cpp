#include "imuPreIntegration.h"

void ImuPreintegrator::Propagate(const Eigen::Vector3d ba, const Eigen::Vector3d bg)
{
    if (start_ts_ >= end_ts_ || imu_data_.empty())
    {
        LOG(ERROR) << cv::format(
            "Invalid data for pre-integration propagate: start_ts_ = %f, end_ts_ = %f, imu_data_.size() = %d",
            start_ts_, end_ts_, static_cast<int>(imu_data_.size()));
        return;
    }

    const uint32_t p_id = static_cast<uint32_t>(ImuState::VariableId::kP);
    const uint32_t v_id = static_cast<uint32_t>(ImuState::VariableId::kV);
    const uint32_t theta_id = static_cast<uint32_t>(ImuState::VariableId::kQ);

    constexpr uint32_t kNoiseAccId = 0;
    constexpr uint32_t kNoiseGyroId = 3;

    Eigen::MatrixXd Cov_m = Eigen::MatrixXd::Identity(6, 6);
    Cov_m.block(kNoiseAccId, kNoiseAccId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_na, 2);
    Cov_m.block(kNoiseGyroId, kNoiseGyroId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_nw, 2);

    Eigen::Vector3d dp = Eigen::Vector3d::Zero();
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
    Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();
    Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(9, 9);

    for (int i = 0; i < imu_data_.size() - 1; i++)
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(9, 9);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(9, 6);

        double dt = imu_data_.at(i + 1).ts_sec - imu_data_.at(i).ts_sec;
        double dt2 = dt * dt;
        if (dt > 0)
        {
            Eigen::Vector3d am_mid = 0.5 * (imu_data_.at(i).am + imu_data_.at(i + 1).am) - ba;
            Eigen::Vector3d wm_mid = 0.5 * (imu_data_.at(i).wm + imu_data_.at(i + 1).wm) - bg;

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

    set_dp(dp);
    set_dv(dv);
    set_dR(dR);
    set_dp_dba(dp_dba);
    set_dp_dbg(dp_dbg);
    set_dv_dba(dv_dba);
    set_dv_dbg(dv_dbg);
    set_dR_dbg(dR_dbg);
    set_Cov(Cov);
}

void ImuPreintegrator::update(const Eigen::Vector3d delta_ba, const Eigen::Vector3d delta_bg)
{
    dp_ = dp_ + dp_dba_ * delta_ba + dp_dbg_ * delta_bg;
    dv_ = dv_ + dv_dba_ * delta_ba + dv_dbg_ * delta_bg;
    dR_ = dR_ * SO3d::exp(dR_dbg_ * delta_bg).matrix();
}