#pragma once

#include "imuPreIntegration.h"
#include "initializer.h"
#include "ImuState.h"
#include "camModel.h"
#include "parameter.h"
#include "sensorType.h"
#include "vioState.h"
#include "utils.h"
#include "visualManager.h"
#include "sfm.h"

class DynamicInitializer : public Initializer
{
public:
    DynamicInitializer() = default;
    DynamicInitializer(const Param &paramters,
                       const std::shared_ptr<VisualManager> visual_manager,
                       std::shared_ptr<State> &state)
        : Initializer(paramters, visual_manager, state)
    {
        sfm_solver = std::make_unique<Sfm>(paramters);
        R_CtoI_ = paramters.Ric[0];
        p_CinI_ = paramters.tic[0];
    }

    virtual ~DynamicInitializer() {};

    virtual void reset() override;

    bool isReadyToInitialize() const;

    bool feedVisualMeasurement(const std::pair<double, cv::Mat> image,
                               std::pair<double, std::vector<CameraObs>> feature_observes,
                               ImuState imu_state);

    void feedImuPreIntegration(ImuPreintegrator imu_preintegration);

    bool TryInitialize();

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

    bool InitializeSystem();

    Eigen::Matrix3d R_CtoI_;
    Eigen::Vector3d p_CinI_;
    std::map<double, Pose> sfm_poses_;
    std::unique_ptr<Sfm> sfm_solver;
    std::map<double, std::shared_ptr<ImuState>> imu_state_map_;
    std::map<double, ImuPreintegrator> imu_preIntegration_map_;
};