#include "dynamicInitializer.h"

bool DynamicInitializer::isReadyToInitialize() const
{
    const bool is_sfm_ready = sfm_solver->isReady();
    const auto &sfm_all_feature_observatioins = sfm_solver->getAllFeatureObservations();
    assert(imu_preIntegration_map_.size() == sfm_all_feature_observatioins.size() - 1);
    for (const auto &x : imu_preIntegration_map_)
    {
        if (sfm_all_feature_observatioins.find(x.second.start_ts()) == sfm_all_feature_observatioins.end() ||
            sfm_all_feature_observatioins.find(x.second.end_ts()) == sfm_all_feature_observatioins.end())
        {
            LOG(WARNING) << cv::format("Can not find %f (or %f) feature observations in SFM-solver", x.second.start_ts(), x.second.end_ts());
            // Reset();
            return false;
        }
    }
    return true;
}

void DynamicInitializer::feedVisualMeasurement(std::pair<double, std::vector<CameraObs>> feature_observes)
{
    sfm_solver->MaybeAddSfmKeyframes(feature_observes);
}

void DynamicInitializer::feedImuPreIntegration(ImuPreintegrator imu_preintegration)
{
    imu_preIntegration_map_.insert_or_assign(imu_preintegration.start_ts(), imu_preintegration);
}

void DynamicInitializer::Reset()
{
    Initializer::reset();
    is_ready_to_initialize_ = false;
    imu_state_map_.clear();
    imu_preIntegration_map_.clear();
    sfm_solver->Reset();
}

bool DynamicInitializer::solveRotationAndGyroBias()
{

}

bool DynamicInitializer::LinearAlignment()
{

}

bool DynamicInitializer::visualInertialAlignment()
{
    if (!solveRotationAndGyroBias())
    {
        LOG(INFO) << "Rotation and gyro bias estimation failed, reset initializer";
        return false;
    }

    if (!LinearAlignment())
    {
        LOG(INFO) << "Linear alignment failed, reset initializer";
        return false;
    }

    return true;
}

bool DynamicInitializer::InitializeSystem()
{
    if (!sfm_solver->Optimization())
    {
        LOG(INFO) << "SFM optimization failed, reset initializer";
        Reset();
        return false;
    }

    if (!visualInertialAlignment())
    {
        LOG(INFO) << "Visual initial alignment failed, reset initializer";
        Reset();
        return false;
    }

    LOG(INFO) << "Dynamic initializer has successfully initialized the system";

    return true;
}