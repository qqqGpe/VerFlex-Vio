#pragma once

#include "imuPreIntegration.h"
#include "initializer.h"
#include "Imu_state.h"
#include "cameraModel.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include "utils.h"
#include "visualManager.h"
#include "sfm.h"

class DynamicInitializer : public Initializer
{
public:
    DynamicInitializer() = default;
    DynamicInitializer(const Param &paramters,
                       const std::shared_ptr<VisualManager> visual_manager,
                       const std::shared_ptr<CameraModel> &camera_model,
                       std::shared_ptr<State> &state)
        : Initializer(paramters, visual_manager, camera_model, state)
    {
        sfm_solver = std::make_unique<Sfm>(paramters, camera_model);
    }

    virtual ~DynamicInitializer() {};

    void Reset();

    bool isReadyToInitialize() const;

    void feedVisualMeasurement(std::pair<double, std::vector<CameraObs>> feature_observes);

    void feedImuMeasurements(std::vector<ImuData> imu_data);

    void feedImuPreIntegration(ImuPreintegrator imu_preintegration);

    bool InitializeSystem();

private:
    bool visualInertialAlignment();

    bool solveRotationAndGyroBias();

    bool LinearAlignment();

    std::map<double, Pose> sfm_poses_;

    std::unique_ptr<Sfm> sfm_solver;

    bool is_ready_to_initialize_ = false;

    std::map<double, ImuState> imu_state_map_;

    std::map<double, ImuPreintegrator> imu_preIntegration_map_;
};