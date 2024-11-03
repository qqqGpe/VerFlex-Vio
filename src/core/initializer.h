#ifndef __VIO_INITIALIZER__
#define __VIO_INITIALIZER__
#include "Imu_state.h"
#include "state.h"
#include "sensor_data.h"

namespace {
    constexpr uint32_t kInitializeImuQueSize = 200;
}

class Initializer {
public:

    Initializer() = default;
    ~Initializer(){};

    void feed_imu_measurement(const ImuData & data);
    bool static_initialize(std::shared_ptr<State> &state);
    Eigen::Matrix3d Gram_Schmidt(const Eigen::Vector3d &gravity_body);

    bool is_initialized = false;

protected:

    double init_win_time = 1.0;     // 用于初始化的IMU窗口长度
    double gravity_mag = 9.81;
    double static_acc_var_thres = 0.1;  // 静止检测加计阈值
    std::deque<ImuData> imu_data;

};


#endif