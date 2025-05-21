#pragma once

#include "visualManager.h"
#include "cameraModel.h"
#include "parameter.h"
#include "sensor_data.h"
#include "Pose.h"
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

class Sfm
{
   public:
    Sfm() = default;
    Sfm(const Param& parameters, const std::shared_ptr<CameraModel> camera_model) { camera_model_ = camera_model; }
    virtual ~Sfm() = default;

    void ResetSfm()
    {
        all_feature_observes_.clear();
        keyframe_poses_.clear();
        latest_keyframe_observe_.first = 0;
        latest_keyframe_observe_.second.clear();
    }

    bool MaybeAddSfmKeyframes(const std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool Optimization();

    bool isReady() const { return all_feature_observes_.size() >= kMinRequiredKeyframesForSfm; }

    std::map<double, Pose> getSfmPoses() const { return keyframe_poses_; }

    static constexpr uint32_t kMinRequiredFeaturesPerFrame = 20;
    static constexpr double kMaxTimeIntervalBetweenKeyframes = 1.0f;  // seconds
    static constexpr double kMinPixelParallexBetweenKeyframes = 5.0f;  // pixels
    static constexpr uint32_t kMinRequiredKeyframesForSfm = 20;

   private:
    std::shared_ptr<CameraModel> camera_model_;
    std::map<double, std::vector<CameraObs>> all_feature_observes_;
    std::map<double, Pose> keyframe_poses_;
    std::pair<double, std::vector<CameraObs>> latest_keyframe_observe_;
};