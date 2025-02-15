#ifndef __VIO_SENSOR_DATA__
#define __VIO_SENSOR_DATA__
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

enum CameraId {
    LEFT_CAM = 0,
    RIGHT_CAM = 1,
    MAX_CAM_NUM = 2
};

enum class KeyFrameType {
    not_keyframe = 0,
    large_parallex_flag,
    feat_lost_too_much
};

struct ImuData {
    double ts_sec = 0.f;
    Eigen::Vector3d am = Eigen::Vector3d::Zero();
    Eigen::Vector3d wm = Eigen::Vector3d::Zero();;
    bool operator<(const ImuData& other) { return ts_sec < other.ts_sec; }
};

class GroundTruth {
public:
    GroundTruth() = default;
    GroundTruth(double ts_sec, Eigen::Vector3d p, Eigen::Vector3d v)
        : ts_sec(ts_sec)
        , p_(p)
        , v_(v)
    {}

    double ts_sec = 0.f;
    Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
};

struct CameraData {
    double ts_sec = 0.f;
    int cam_id = 0;
    cv::Mat image;
    cv::Mat mask;
    bool operator<(const CameraData& other) { return ts_sec < other.ts_sec; }
};

struct CameraObs {
    CameraObs() = default;
    CameraObs(double ts_sec, float u, float v, float ur, float vr)
        : ts_sec(ts_sec)
        , u(u)
        , v(v)
        , ur(ur)
        , vr(vr)
    {}

    CameraObs(float u, float v, float u_norm, float v_norm)
        : u(u)
        , v(v)
        , u_norm(u_norm)
        , v_norm(v_norm)
    {}

    void set_invalid()
    {
        ts_sec = 0;
        feat_id = -1;
        valid = false;
        obs_times_n = 0;

        u = 0;
        v = 0;
        u_norm = 0;
        v_norm = 0;

        ur = 0;
        vr = 0;
        ur_norm = 0;
        vr_norm = 0;
        image_left = cv::Mat();
        image_right = cv::Mat();
    }

    double ts_sec = 0;
    int32_t feat_id = -1;
    bool valid = false;
    uint32_t obs_times_n = 0;
    float u = 0.f;
    float v = 0.f;
    float u_norm = 0.f;
    float v_norm = 0.f;
    float ur = 0.f;
    float vr = 0.f;
    float ur_norm = 0.f;
    float vr_norm = 0.f;
    cv::Mat image_left;
    cv::Mat image_right;
};

class CamObsHash {
    public:
    std::size_t operator()(const CameraObs &camObs) const {
        return std::hash<uint32_t>()(camObs.feat_id);
    }
};

struct Feature {
    Feature() = default;

    void reset()
    {
        _id = -1;
        _pwf.setZero();
        _valid = false;
        _is_triangulated = false;
        parallex = 0.f;
        _visual_obs_buffer.clear();
    }

    int _id = -1;
    bool _valid = false;
    Eigen::Vector3d _pwf = Eigen::Vector3d::Zero();
    bool _is_triangulated = false;
    double parallex = 0.f;
    std::map<double, CameraObs> _visual_obs_buffer; // <ts_sec, obs>
};

#endif