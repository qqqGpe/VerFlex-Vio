#ifndef VIO_VIEWER_PANGOLIN_VIEWER_H
#define VIO_VIEWER_PANGOLIN_VIEWER_H

#include <pangolin/pangolin.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "parameter.h"  // Parameter
#include "vioManager.h" // VioOutput

/**
 * @brief ORB-SLAM2-style Pangolin 3D viewer for VerFlex-VIO.
 *
 * Runs its own render thread. The VIO pipeline feeds it through Update(), which
 * is the only hand-off point (called from the VioOutput callback on the
 * processing thread). All shared state is protected by a mutex.
 *
 * Draws: current body pose (green frustum), trajectory (red line), keyframes
 * (blue frustums, one per real keyframe event surfaced via VioOutput::is_keyframe),
 * and accumulated MSCKF feature points (black cloud).
 */
class PangolinViewer
{
  public:
    explicit PangolinViewer(const Parameter& params);
    ~PangolinViewer();

    PangolinViewer(const PangolinViewer&) = delete;
    PangolinViewer& operator=(const PangolinViewer&) = delete;

    /// Spawn the render thread (idempotent).
    void Start();

    /// Request the render thread to finish and join it (idempotent).
    void Stop();

    /// Block until the render thread exits on its own (e.g. user closes the window).
    void Wait();

    /// Thread-safe feed from the VioOutput callback.
    void Update(const VioOutput& output);

  private:
    void Run();
    void DrawFrustum(const pangolin::OpenGlMatrix& Twc, float r, float g, float b, float size, float linewidth);
    void DrawTrajectory(const std::deque<Eigen::Vector3d>& traj);
    void DrawKeyFrames(const std::vector<Eigen::Quaterniond>& kf_q, const std::vector<Eigen::Vector3d>& kf_p);
    void DrawMapPoints(const std::vector<Eigen::Vector3d>& points, float r, float g, float b);

    /// Build a column-major OpenGL matrix from the body->world pose (q_WI, p_WI).
    static pangolin::OpenGlMatrix ToGl(const Eigen::Quaterniond& q_WI, const Eigen::Vector3d& p_WI);

    std::thread thread_;
    std::atomic<bool> started_{false};
    std::atomic<bool> finish_requested_{false};

    // Shared state (written by Update on the processing thread, read by Run on the render thread).
    std::mutex mutex_;
    bool has_pose_ = false;
    Eigen::Quaterniond cur_q_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d cur_p_ = Eigen::Vector3d::Zero();
    std::deque<Eigen::Vector3d> trajectory_;
    std::vector<Eigen::Quaterniond> keyframe_q_;
    std::vector<Eigen::Vector3d> keyframe_p_;
    std::unordered_map<uint32_t, Eigen::Vector3d> map_points_; // feature_id -> latest world position (dedup'd)
    std::unordered_set<uint32_t> active_ids_;                  // feature_ids in the current MSCKF window (drawn blue)
    cv::Mat latest_image_; // latest image_with_features (BGR) for the inset
    int img_w_ = 0;
    int img_h_ = 0;
    bool record_viewer_ = false;
    std::string record_dir_ = "viewer_frames";

    // Viewer settings (ORB-SLAM2 EuRoC-like defaults).
    const float viewpoint_x_ = 0.0f;
    const float viewpoint_y_ = -1.0f;
    const float viewpoint_z_ = -0.8f;
    const float viewpoint_f_ = 500.0f;
    const float chase_dist_ = 1.0f;   // chase cam: distance behind the camera (m)
    const float chase_height_ = 0.8f; // chase cam: height above the camera (m)
    const float keyframe_size_ = 0.05f;
    const float keyframe_linewidth_ = 1.0f;
    const float camera_size_ = 0.08f;
    const float camera_linewidth_ = 2.0f;
    const float point_size_ = 2.0f;
    const float trajectory_linewidth_ = 2.0f;
    const int max_trajectory_points_ = 20000;
    const int render_dt_ms_ = 30; // ~33 fps render cap
};

#endif // VIO_VIEWER_PANGOLIN_VIEWER_H
