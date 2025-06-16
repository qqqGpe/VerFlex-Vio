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

        R_CtoI_ = paramters.Ric[0];

        p_CinI_ = paramters.tic[0];
    }

    static constexpr double kInitSigmaRotation = 1e-2;   // rad
    static constexpr double kInitSigmaPosition = 1e-1;   // m
    static constexpr double kInitSigmaVelocity = 1e-1;   // m/s
    static constexpr double kInitSigmaGyroBias = 1e-3;   // rad
    static constexpr double kInitSigmaAccelBias = 1e-2;  // m/s

    virtual ~DynamicInitializer() {};

    virtual void reset() override;

    bool isReadyToInitialize() const;

    bool feedVisualMeasurement(const std::pair<double, cv::Mat> image,
                               std::pair<double, std::vector<CameraObs>> feature_observes,
                               ImuState imu_state);

    void feedImuPreIntegration(ImuPreintegrator imu_preintegration);

    bool InitializeSystem();

    std::optional<double> getLastestFeatureMeasurementTimestamp() const
    {
        const std::map<double, std::vector<CameraObs>>& all_sfm_feature_observations = sfm_solver->getAllFeatureObservations();
        if (all_sfm_feature_observations.empty())
        {
            return std::nullopt;
        }
        return all_sfm_feature_observations.rbegin()->first;
    }

private:
    bool visualInertialAlignment();

    bool solveGyroscopeBias();

    bool LinearAlignment(Eigen::VectorXd& x);

    void assignImuState(const Eigen::VectorXd velocity_gravity_scale);

    void assignFeatureBase(const double scale);

    Eigen::Matrix3d R_CtoI_;
    Eigen::Vector3d p_CinI_;
    std::map<double, Pose> sfm_poses_;
    std::unique_ptr<Sfm> sfm_solver;
    std::map<double, std::shared_ptr<ImuState>> imu_state_map_;
    std::map<double, ImuPreintegrator> imu_preIntegration_map_;
};