/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-11-07 01:40:59
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
 */
#ifndef __VIO_SENSOR_DATA__
#define __VIO_SENSOR_DATA__
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace
{
constexpr uint32_t kLeft = 0;
constexpr uint32_t kRight = 1;
}  // namespace

enum class KeyFrameStatus
{
    kNone = 0,
    kLargeParallex = 1,
    kFeatureLostTooMuch = 2,
    kTooFewFeatureTracked = 3
};

struct ImuData
{
    double ts_sec = 0.f;
    Eigen::Vector3d am = Eigen::Vector3d::Zero();
    Eigen::Vector3d wm = Eigen::Vector3d::Zero();
    bool operator<(const ImuData& other) { return ts_sec < other.ts_sec; }
};

class GroundTruth
{
   public:
    GroundTruth() = default;
    GroundTruth(double ts_sec, Eigen::Vector3d p, Eigen::Vector3d v) : ts_sec(ts_sec), p_(p), v_(v) {}

    double ts_sec = 0.f;
    Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
};

struct CameraData
{
    double ts_sec = 0.f;
    int cam_id = 0;
    cv::Mat image;
    cv::Mat mask;
    bool operator<(const CameraData& other) { return ts_sec < other.ts_sec; }
};

struct CameraObs
{
    CameraObs() = default;
    CameraObs(double ts_sec, float u, float v, float ur, float vr) : ts_sec(ts_sec)
    {
        uv[kLeft] = Eigen::Vector2d(u, v);
        uv[kRight] = Eigen::Vector2d(ur, vr);
    }

    CameraObs(float u, float v, float u_norm, float v_norm)
    {
        uv[kLeft] = Eigen::Vector2d(u, v);
        uv_norm[kRight] = Eigen::Vector2d(u_norm, v_norm);
    }

    void setInvalid()
    {
        ts_sec = 0;
        feat_id = -1;
        valid = false;
        obs_times_n = 0;
        std::map<int, Eigen::Vector2d>().swap(uv);
        std::map<int, Eigen::Vector2d>().swap(uv_norm);
    }

    double ts_sec = 0;
    int32_t feat_id = -1;
    bool valid = false;
    uint32_t obs_times_n = 0;
    std::map<int, Eigen::Vector2d> uv;
    std::map<int, Eigen::Vector2d> uv_norm;
};

enum class FeatureType
{
    kUnknown = 0,
    kMsckfPoint = 1,
    kSlamPoint = 2
};

struct Feature
{
    uint32_t _id = -1;
    FeatureType _type = FeatureType::kUnknown;
    bool _valid = false;
    bool _is_triangulated = false;
    double _parallex = 0.f;
    Eigen::Vector3d _pwf = Eigen::Vector3d::Zero();
    double _theta_parallex = 0.f;
    std::map<double, CameraObs> _visual_obs_buffer;  // <ts_sec, obs>

    // Reset feature to initial state
    void reset() { *this = Feature{}; }
};

#endif