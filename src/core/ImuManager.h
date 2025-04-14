#ifndef __IMU_MANAGER__
#define __IMU_MANAGER__

#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include "solver.h"

class SolverBase;

class ImuManager {
public:
    ImuManager(const Param& param, std::shared_ptr<State> state, std::shared_ptr<MsckfSolverBase> solver)
    {
        _imu_acc_var_static_thres = param.imu_acc_var_static_thres;
        _imu_gyro_static_thres = param.imu_gyro_static_thres;
        _gravity_magn = Eigen::Vector3d(0, 0, -param.gravity_magn);
        _data = std::make_shared<std::deque<ImuData>>();
        solver_ = solver;
    }

    ~ImuManager() { }

    bool FeedImuMeasurement(const ImuData& imu_measurement);

    void ZuptUpdate(std::shared_ptr<State> state);

    void ConstructZuptConstraint(std::shared_ptr<State> state, ImuData imu_data, Eigen::MatrixXd& Hx,
        std::vector<std::shared_ptr<Type>>& _Hx_order, std::unordered_map<std::shared_ptr<Type>, size_t>& _map_hx, Eigen::VectorXd& res);

    bool IsStaticStatus();

    ImuData InterpolateImuData(const ImuData& imu_1, const ImuData& imu_2, double timestamp);

    ImuData GetImuData(double timestamp);

    std::vector<ImuData> AccessIntervalImuMeasurements(const double ts_start, const double ts_end);

    void ClearExpiredMeasurements(const double ts);

    void SetImuNoise(const double sigma_na, const double sigma_nw, const double sigma_ba, const double sigma_bg);

    double _imu_latest_timestamp = 0;

    inline static double _sigma_na;
    inline static double _sigma_nw;
    inline static double _sigma_ba;
    inline static double _sigma_bg;

private:
    std::shared_ptr<std::deque<ImuData>> _data;
    Eigen::Vector3d _gravity_magn;
    double _imu_acc_var_static_thres = 0.5;
    double _imu_gyro_static_thres = 0.5;
    std::shared_ptr<MsckfSolverBase> solver_;
};
#endif