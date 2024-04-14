#include <Eigen/Eigen>
#include <cstdint>
#include <opencv2/opencv.hpp>

struct ImuData {
    double ts_sec;
    Eigen::Vector3d am;
    Eigen::Vector3d wm;

    bool operator<(const ImuData& other) { return ts_sec < other.ts_sec; }
};

struct FeatureData {
    double ts_sec;
    int cam_id;
    std::vector<Eigen::Vector2d> cam_obs;
    std::vector<bool> valid;
    std::vector<uint32_t> feat_id;
    std::vector<uint32_t> obs_times;
    bool operator<(const FeatureData& other) { return ts_sec < other.ts_sec; }
};

struct CameraData {
    double ts_sec;
    int cam_id;
    cv::Mat image;
    cv::Mat mask;
    bool operator<(const CameraData& other) { return ts_sec < other.ts_sec; }
};