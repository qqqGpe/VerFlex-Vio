#ifndef __IMU_MANAGER__
#define __IMU_MANAGER__

#include "sensor_data.h"
#include "states/state.h"
#include "parameter.h"

class ImuManager {
public:

    ImuManager(const Param &param){
        _sigma_na = param.sigma_ba;
        _sigma_nw = param.sigma_nw;
        _sigma_ba = param.sigma_ba;
        _sigma_bw = param.sigma_bw;

        _data = std::make_shared<std::deque<ImuData>>();
        _data->clear();
    }
    ~ImuManager(){}

    bool feed_imu_measurement(const ImuData &imu_measurement);

    std::shared_ptr<std::deque<ImuData>> access_observations() const {return _data;}

    ImuData interpolate_data(const ImuData &imu_1, const ImuData &imu_2, double timestamp);

    std::vector<ImuData> access_interval_imu_measurment(const double ts_start, const double ts_end);

    void delete_old_measurements(const double ts);

    double _sigma_na;
    double _sigma_nw;
    double _sigma_ba;
    double _sigma_bw;

private:
    std::shared_ptr<std::deque<ImuData>> _data;

};

#endif