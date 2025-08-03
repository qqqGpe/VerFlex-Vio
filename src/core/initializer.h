#ifndef __VIO_INITIALIZER__
#define __VIO_INITIALIZER__
#include "Imu_state.h"
#include "camModel.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include "utils.h"
#include "visualManager.h"

namespace
{
constexpr uint32_t kInitializeImuQueSize = 200;
}

enum class InitializerType
{
    kStatic = 0,
    kDynamic = 1
};

class Initializer
{
   public:
    static constexpr double kInitSigmaRotation = 1e-2;   // rad
    static constexpr double kInitSigmaPosition = 1e-1;   // m
    static constexpr double kInitSigmaVelocity = 1e-1;   // m/s
    static constexpr double kInitSigmaGyroBias = 1e-3;   // rad
    static constexpr double kInitSigmaAccelBias = 1e-2;  // m/s
    static constexpr double kInitSigmaRic = 1e-3;        // rad
    static constexpr double kInitSigmaTdVisual = 1e-3;   // sec

    Initializer() = default;
    Initializer(const Param& paramters, const std::shared_ptr<VisualManager> visual_manager, std::shared_ptr<State>& state)
    {
        state_ = state;
        visual_manager_ = visual_manager;
        gravity_mag = paramters.gravity_magn;
        init_type = static_cast<InitializerType>(paramters.initial_type);
    }
    virtual ~Initializer() {};

    virtual void reset();

    bool StereoVisualInitialize(const std::pair<double, std::vector<CameraObs>> feature_observes);

    void FeedImuMeasurement(const ImuData& data);

    bool InitializeOrientation();

    Eigen::Matrix3d Gram_Schmidt(const Eigen::Vector3d& gravity_body);

    bool IsInitialized();

    InitializerType init_type;
    bool is_bias_initialized = false;
    bool is_orientation_initialized = false;
    bool is_position_initialized = false;
    bool is_velocity_initialized = false;

   protected:
    std::shared_ptr<VisualManager> visual_manager_;
    std::shared_ptr<State> state_;
    std::deque<std::unordered_map<uint32_t, CameraObs>> feature_obs_buffer_;
    double init_win_time = 1.0;  // 用于初始化的IMU窗口长度
    double gravity_mag = 9.81;
    double static_acc_var_thres = 0.1;  // 静止检测加计阈值
    std::deque<ImuData> imu_data;
};

#endif