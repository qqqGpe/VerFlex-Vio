#ifndef __VIO_SENSOR_DATA__
#define __VIO_SENSOR_DATA__
#include <Eigen/Core>
#include <opencv2/opencv.hpp>


enum class keyframe_flag_e {
    not_keyframe = 0,
    large_parallex_flag,
    feat_lost_too_much
};

struct ImuData {
    double ts_sec;
    Eigen::Vector3d am;
    Eigen::Vector3d wm;
    bool operator<(const ImuData& other) { return ts_sec < other.ts_sec; }
};

// struct FeatureData {
//     double ts_sec;
//     int cam_id;
//     int keyframe;
//     std::vector<Eigen::Vector2d> cam_obs;
//     std::vector<bool> valid;
//     std::vector<uint32_t> feat_id;
//     std::vector<uint32_t> obs_times;
//     bool operator<(const FeatureData& other) { return ts_sec < other.ts_sec; }
// };

struct CameraData {
    double ts_sec;
    int cam_id;
    cv::Mat image;
    cv::Mat mask;
    bool operator<(const CameraData& other) { return ts_sec < other.ts_sec; }
};

struct cam_obs_t {
    cam_obs_t() { }
    cam_obs_t(double u, double v, double u_norm, double v_norm)
        : u(u)
        , v(v)
        , u_norm(u_norm)
        , v_norm(v_norm)
    {
        ts_sec = 0;
        valid = false;
        feat_id = -1;
        obs_times_n = 0;
    }

    void set_invalid()
    {
        ts_sec = 0;
        feat_id = -1;
        valid = false;
        obs_times_n = 0;
    }

    double ts_sec;
    uint32_t feat_id;
    bool valid;
    uint32_t obs_times_n;
    double u, v;
    double u_norm, v_norm;
};

struct Feature {
    Feature() = default;

    void reset()
    {
        _id = -1;
        _valid = false;
        _is_triangulated = false;
        _visual_obs_buffer.clear();
    }

    int _id = -1;
    bool _valid = false;
    Eigen::Vector3d _pwf;
    bool _is_triangulated = false;
    std::map<double, cam_obs_t> _visual_obs_buffer; // <ts_sec, obs>
};

#endif