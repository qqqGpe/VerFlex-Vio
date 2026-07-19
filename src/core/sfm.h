#pragma once

#include "visualManager.h"
#include "camera_model.h"
#include "parameter.h"
#include "sensorType.h"
#include "mathematical_tools.h"
#include "Pose.h"
#include "utils.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

class Sfm
{
public:
    Sfm() = default;
    Sfm(const Parameter &parameters) : params_(parameters) {}
    virtual ~Sfm() = default;

    void Reset()
    {
        all_features_.clear();
        all_feature_observes_.clear();
        keyframe_poses_.clear();
        latest_keyframe_observe_.first = 0.f;
        latest_keyframe_observe_.second.clear();
        oldest_keyframe_timestamp_ = 0.f;
        reference_keyframe_timestamp_ = 0.f;
        keyframe_images_.clear();
    }

    static constexpr uint32_t kMinRequiredObservTimesPerFeature = 3;
    static constexpr uint32_t kMinRequiredFeaturesPerFrame = 30;
    static constexpr double kMaxTimeIntervalBetweenKeyframes = 2.0f;  // seconds
    static constexpr double kMinPixelParallexBetweenKeyframes = 7.f; // pixels
    static constexpr uint32_t kMinRequiredFeaturesForSfm = 30;
    static constexpr uint32_t kMaxFeaturesForSfm = 300;
    static constexpr uint32_t kRequiredKeyframesForSfm = 10;

    template <typename KeyType, typename ValueType>
    std::optional<uint32_t> indexInMap(const std::map<KeyType, ValueType> &map, const KeyType &key) const;

    void triangulateFramePoints(const std::vector<CameraObs> &obs_A, const std::vector<CameraObs> &obs_B, const Pose &pose_a, const Pose &pose_b);

    bool initSfmSolver();

    bool MaybeAddSfmKeyframes(const std::pair<double, std::vector<CameraObs>>& current_feature_observe,
                              std::optional<std::pair<double, cv::Mat>> image);

    bool solveFrameByPnp(const std::vector<CameraObs> current_obsv, Pose &current_pose);

    double findReferenceKeyframeTimestamp();

    bool Optimization();

    bool isReady() const;

    bool calcRelativePose(const std::vector<CameraObs> &obs_a,
                          const std::vector<CameraObs> &obs_b,
                          Eigen::Matrix3d &R_BtoA,
                          Eigen::Vector3d &p_BinA);

    std::map<double, Pose> getSfmPoses() const { return keyframe_poses_; }

    void ShowPoseWrtFirstFrame(const Pose current_pose);

    const std::map<double, std::vector<CameraObs>>& getAllFeatureObservations() const { return all_feature_observes_; }

    const std::map<uint32_t, Feature>& getAllFeatures() const { return all_features_; }

    const std::map<double, cv::Mat>& getKeyframeImages() const { return keyframe_images_; }

    void showKeyframeImages() const;

    void feedKeyframeImage(const double timestamp, const cv::Mat& image) { keyframe_images_[timestamp] = image; }

   private:
    Parameter params_;
    std::map<double, cv::Mat> keyframe_images_;
    std::map<double, std::vector<CameraObs>> all_feature_observes_;
    std::map<uint32_t, Feature> all_features_;
    std::map<double, Pose> keyframe_poses_;
    std::pair<double, std::vector<CameraObs>> latest_keyframe_observe_;
    double oldest_keyframe_timestamp_ = 0.0;
    double reference_keyframe_timestamp_ = 0.0;
    Eigen::Matrix3d latest_Rwc_;
};

struct FeatureReprojectionFactor : ceres::SizedCostFunction<2, 3, 4, 3>
{
public:
    FeatureReprojectionFactor(const CameraObs &obs) : obs_(obs) {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d p_finG(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond q_CtoG(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d p_CinG(parameters[2][0], parameters[2][1], parameters[2][2]);

        Eigen::Matrix3d R_CtoG = q_CtoG.toRotationMatrix();
        Eigen::Vector3d p_finC = R_CtoG.transpose() * (p_finG - p_CinG);

        residuals[0] = (p_finC(0) / p_finC(2)) - obs_.uv_norm.at(LEFT_CAM).x();
        residuals[1] = (p_finC(1) / p_finC(2)) - obs_.uv_norm.at(LEFT_CAM).y();

        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> dz_dpcf;
        dz_dpcf <<
            1.0 / p_finC(2), 0, -p_finC(0) / (p_finC(2) * p_finC(2)),
            0, 1.0 / p_finC(2), -p_finC(1) / (p_finC(2) * p_finC(2));

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_dz_dpwf(jacobians[0]);
                Eigen::Matrix<double, 2, 3> dz_dpwf = dz_dpcf * R_CtoG.transpose();
                J_dz_dpwf = dz_dpwf;
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_dz_dqwc(jacobians[1]);
                Eigen::Matrix<double, 2, 3> dz_dqwc = dz_dpcf * R_CtoG.transpose() * utils::math::skew(p_finG - p_CinG);
                J_dz_dqwc.leftCols<3>() = 2 * dz_dqwc;
                J_dz_dqwc.rightCols<1>().setZero();
            }

            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_dz_dpwc(jacobians[2]);
                Eigen::Matrix<double, 2, 3> dz_dpwc = dz_dpcf * (-R_CtoG.transpose());
                J_dz_dpwc = dz_dpwc;
            }
        }
        return true;
    }

private:
    CameraObs obs_;
};