#ifndef __IMU_MANAGER__
#define __IMU_MANAGER__

#include "sensor_data.h"
#include "state.h"
#include "parameter.h"
#include "eskf_solver.h"

class ImuManager {
public:

    ImuManager(const Param &param, std::shared_ptr<State> state)
    {
        _sigma_na = param.sigma_ba;
        _sigma_nw = param.sigma_nw;
        _sigma_ba = param.sigma_ba;
        _sigma_bw = param.sigma_bw;
        _gravity_magn = Eigen::Vector3d(0, 0, -param.gravity_magn);
        _state = state;
        _data = std::make_shared<std::deque<ImuData>>();
        _data->clear();
    }

    ~ImuManager(){}

    bool feed_imu_measurement(const ImuData &imu_measurement);

    void zupt_update(std::shared_ptr<State> state);

    void construct_zupt_constraint(std::shared_ptr<State> state, ImuData imu_data, Eigen::MatrixXd &Hx,
        std::vector<std::shared_ptr<Type>>& _Hx_order, std::unordered_map<std::shared_ptr<Type>, size_t>& _map_hx, Eigen::VectorXd& res);

    bool static_status();

    std::shared_ptr<std::deque<ImuData>> access_observations() const {return _data;}

    ImuData interpolate_data(const ImuData &imu_1, const ImuData &imu_2, double timestamp);

    std::vector<ImuData> access_interval_imu_measurment(const double ts_start, const double ts_end);

    void delete_old_measurements(const double ts);

    double _sigma_na;
    double _sigma_nw;
    double _sigma_ba;
    double _sigma_bw;

private:
    std::shared_ptr<State> _state;
    std::shared_ptr<std::deque<ImuData>> _data;
    Eigen::Vector3d _gravity_magn;
    double _last_static_ts = -1;
    bool _last_static_position_valid = false;
    Eigen::Vector3d _last_static_position = Eigen::Vector3d::Zero();

};

#endif