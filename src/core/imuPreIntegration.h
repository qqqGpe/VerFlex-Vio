#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

#include "state.h"
#include "Imu_state.h"
#include "sensor_data.h"
#include "ImuManager.h"

using namespace Sophus;

class ImuPreintegrator
{
   public:
    ImuPreintegrator();
    virtual ~ImuPreintegrator() = default;

    void feedImuData(const ImuData& imu_data)
    {
        if (imu_data_.empty() || imu_data.ts_sec > imu_data_.back().ts_sec)
        {
            imu_data_.push_back(imu_data);
        }
        else
        {
            LOG(WARNING) << "imu data is not in order, drop this data";
        }
    }

    static ImuPreintegrator midPointIntegrate(std::shared_ptr<State> state, const std::vector<ImuData> imu_input, const double ts_start, const double ts_end);

    void update(const Eigen::Vector3d delta_ba, const Eigen::Vector3d delta_bg);

    Eigen::Matrix3d dR() const { return dR_; }

    Eigen::Vector3d dp() const { return dp_; }

    Eigen::Vector3d dv() const { return dv_; }

    Eigen::Matrix3d Cov() const { return Cov_; }

    void set_dR (const Eigen::Matrix3d& dR) { dR_ = dR; }

    void set_dp (const Eigen::Vector3d& dp) { dp_ = dp; }

    void set_dv (const Eigen::Vector3d& dv) { dv_ = dv; }

    void set_Cov (const Eigen::MatrixXd& Cov) { Cov_ = Cov; }

    void set_dR_dbg (const Eigen::Matrix3d& dR_dbg) { dR_dbg_ = dR_dbg; }

    void set_dp_dba (const Eigen::Matrix3d& dp_dba) { dp_dba_ = dp_dba; }

    void set_dp_dbg (const Eigen::Matrix3d& dp_dbg) { dp_dbg_ = dp_dbg; }

    void set_dv_dba (const Eigen::Matrix3d& dv_dba) { dv_dba_ = dv_dba; }

    void set_dv_dbg (const Eigen::Matrix3d& dv_dbg) { dv_dbg_ = dv_dbg; }

   private:
    double start_ts_ = 0.f;
    double end_ts_ = 0.f;
    std::vector<ImuData> imu_data_;
    ImuState imu_state_;
    Eigen::Matrix3d dR_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d dp_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
    Eigen::MatrixXd Cov_ = Eigen::MatrixXd::Zero(15, 15);

    Eigen::Matrix3d dR_dbg_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dba_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dbg_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_dba_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_dbg_ = Eigen::Matrix3d::Zero();
};
