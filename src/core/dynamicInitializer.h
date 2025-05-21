#pragma once

#include "imuPreIntegrator.h"
#include "initializer.h"
#include "Imu_state.h"
#include "cameraModel.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include "utils.h"
#include "visualManager.h"

class DynamicInitializer : public Initializer
{
   public:
    DynamicInitializer() = default;
    DynamicInitializer(const Param& paramters,
                       const std::shared_ptr<VisualManager> visual_manager,
                       const std::shared_ptr<CameraModel>& camera_model,
                       std::shared_ptr<State>& state)
        : Initializer(paramters, visual_manager, camera_model, state)
    {
    }
    virtual ~DynamicInitializer() {};

    bool isReadyToInitialize() const { return is_ready_to_initialize_; }

    void feedVisualMeasurement(std::pair<double, std::vector<CameraObs>> feature_observes);

    void feedImuMeasurements(std::vector<ImuData> imu_data);

    bool InitializeSystem();

   private:
    bool is_ready_to_initialize_ = false;
    std::map<double, ImuState> imu_state_map_;
    std::map<double, ImuPreintegrator>  imu_preIntegration_map_;
};