#ifndef __VIO_INITIALIZER__
#define __VIO_INITIALIZER__
#include "Imu_state.h"
#include "cameraModel.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include "utils.h"
#include "visualManager.h"

namespace
{
constexpr uint32_t kInitializeImuQueSize = 200;
}

class Initializer
{
   public:
    Initializer() = default;
    Initializer(const Param& paramters,
                const std::shared_ptr<VisualManager> visual_manager,
                const std::shared_ptr<CameraModel>& camera_model,
                std::shared_ptr<State>& state)
    {
        visual_manager_ = visual_manager;
        camera_model_ = camera_model;
        state_ = state;
        gravity_mag = paramters.gravity_magn;
    }
    ~Initializer() {};

    bool StereoVisualInitialize(const std::pair<double, std::vector<CameraObs>> feature_observes);

    void FeedImuMeasurement(const ImuData& data);

    bool static_initialize();

    Eigen::Matrix3d Gram_Schmidt(const Eigen::Vector3d& gravity_body);

    bool IsInitialized();

    bool is_orientation_initialized = false;
    bool is_position_initialized = false;
    bool is_velocity_initialized = false;
    bool is_bias_initialized = false;

   private:
    double calcVisualObsParallex(std::unordered_map<uint32_t, CameraObs> visual_obs_a,
                                 std::unordered_map<uint32_t, CameraObs> visual_obs_b) const;

    double PixelDistance(CameraObs obs_a, CameraObs obs_b) const;

    std::shared_ptr<VisualManager> visual_manager_;
    std::shared_ptr<CameraModel> camera_model_;
    std::shared_ptr<State> state_;
    std::deque<std::unordered_map<uint32_t, CameraObs>> feature_obs_buffer_;
    double init_win_time = 1.0;  // 用于初始化的IMU窗口长度
    double gravity_mag = 9.81;
    double static_acc_var_thres = 0.1;  // 静止检测加计阈值
    std::deque<ImuData> imu_data;
};

#endif