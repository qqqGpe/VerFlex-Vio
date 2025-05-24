#pragma once

#include "visualManager.h"
#include "cameraModel.h"
#include "parameter.h"
#include "sensor_data.h"
#include "Pose.h"
#include "utils.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/local_parameterization.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

struct FeatureReprojectionFactor
{
    FeatureReprojectionFactor(const CameraObs& obs, const Eigen::Vector3d& pwf, const std::shared_ptr<CameraModel>& camera_model)
        : obs_(obs), pwf_(pwf), camera_model_(camera_model) {}

    template <typename T>
    bool operator()(const T* const q, const T* const p, T* residuals) const;

    static ceres::CostFunction* Create(const CameraObs& obs, const Eigen::Vector3d& pwf, const std::shared_ptr<CameraModel>& camera_model);

   private:
    CameraObs obs_;
    Eigen::Vector3d pwf_;
    std::shared_ptr<CameraModel> camera_model_;
};

class Sfm
{
   public:
    Sfm() = default;
    Sfm(const Param& parameters, const std::shared_ptr<CameraModel> camera_model) { camera_model_ = camera_model; }
    virtual ~Sfm() = default;

    void ResetSfm()
    {
        all_features_.clear();
        all_feature_observes_.clear();
        keyframe_poses_.clear();
        latest_keyframe_observe_.first = 0.f;
        latest_keyframe_observe_.second.clear();
        oldest_keyframe_timestamp_ = 0.f;
        reference_keyframe_timestamp_ = 0.f;
    }

    void triangulateFramePoints(const std::vector<CameraObs>& obs_A, const std::vector<CameraObs>& obs_B, const Pose& pose_a, const Pose& pose_b);

    Eigen::Vector3d triangulatePoint(const Pose pose0, const Pose pose1, const Vector2d& point0, const Vector2d& point1);

    bool initSfmSolver();

    bool MaybeAddSfmKeyframes(const std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool solveFrameByPnp(const std::vector<CameraObs> current_obsv, Pose &current_pose);

    double findReferenceKeyframeTimestamp();

    bool Optimization();

    bool isReady() const { return all_feature_observes_.size() == kRequiredKeyframesForSfm; }

    bool calcRelativePose(const std::vector<CameraObs>& obs_a,
                          const std::vector<CameraObs>& obs_b,
                          Eigen::Matrix3d& R_relative,
                          Eigen::Vector3d& p_relative);

    std::map<double, Pose> getSfmPoses() const { return keyframe_poses_; }

    static constexpr uint32_t kMinRequiredObservTimesPerFeature = 3;
    static constexpr uint32_t kMinRequiredFeaturesPerFrame = 20;
    static constexpr double kMaxTimeIntervalBetweenKeyframes = 2.0f;  // seconds
    static constexpr double kMinPixelParallexBetweenKeyframes = 5.0f;  // pixels
    static constexpr uint32_t kMinRequiredFeaturesForSfm = 50;
    static constexpr uint32_t kRequiredKeyframesForSfm = 10;

   private:
    std::shared_ptr<CameraModel> camera_model_;
    std::map<double, std::vector<CameraObs>> all_feature_observes_;
    std::map<uint32_t, Feature> all_features_;
    std::map<double, Pose> keyframe_poses_;
    std::pair<double, std::vector<CameraObs>> latest_keyframe_observe_;
    double oldest_keyframe_timestamp_ = 0.0;
    double reference_keyframe_timestamp_ = 0.0;
};