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
    ImuPreintegrator() = default;
    virtual ~ImuPreintegrator() = default;

    void feedImuMeasuremnts(const std::vector<ImuData>& imu_data)
    {
        imu_data_ = imu_data;
        start_ts_ = imu_data_.front().ts_sec;
        end_ts_ = imu_data_.back().ts_sec;
    }

    void Propagate(const Eigen::Vector3d ba, const Eigen::Vector3d bg);

    void update(const Eigen::Vector3d delta_ba, const Eigen::Vector3d delta_bg);

    Eigen::Matrix3d dR() const { return dR_; }

    Eigen::Vector3d dp() const { return dp_; }

    Eigen::Vector3d dv() const { return dv_; }

    Eigen::Matrix3d dR_dbg() const { return dR_dbg_; }

    Eigen::Matrix3d dp_dba() const { return dp_dba_; }

    Eigen::Matrix3d dp_dbg() const { return dp_dbg_; }

    Eigen::Matrix3d dv_dba() const { return dv_dba_; }

    Eigen::Matrix3d dv_dbg() const { return dv_dbg_; }

    Eigen::Matrix3d Cov() const { return Cov_; }

    double start_ts() const { return start_ts_; }

    double end_ts() const { return end_ts_; }

private:

    void set_dR (const Eigen::Matrix3d& dR) { dR_ = dR; }

    void set_dp (const Eigen::Vector3d& dp) { dp_ = dp; }

    void set_dv (const Eigen::Vector3d& dv) { dv_ = dv; }

    void set_Cov (const Eigen::MatrixXd& Cov) { Cov_ = Cov; }

    void set_dR_dbg (const Eigen::Matrix3d& dR_dbg) { dR_dbg_ = dR_dbg; }

    void set_dp_dba (const Eigen::Matrix3d& dp_dba) { dp_dba_ = dp_dba; }

    void set_dp_dbg (const Eigen::Matrix3d& dp_dbg) { dp_dbg_ = dp_dbg; }

    void set_dv_dba (const Eigen::Matrix3d& dv_dba) { dv_dba_ = dv_dba; }

    void set_dv_dbg (const Eigen::Matrix3d& dv_dbg) { dv_dbg_ = dv_dbg; }

    double start_ts_ = 0.f;
    double end_ts_ = 0.f;

    std::shared_ptr<ImuState> imu_state_;
    std::vector<ImuData> imu_data_;

    Eigen::Matrix3d dR_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d dp_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d dv_ = Eigen::Vector3d::Zero();
    Eigen::MatrixXd Cov_ = Eigen::MatrixXd::Zero(9, 9);

    Eigen::Matrix3d dR_dbg_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dba_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dbg_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_dba_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_dbg_ = Eigen::Matrix3d::Zero();
};
