#include "pangolin_viewer.h"

#include <pangolin/pangolin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.h" // LOG_ERROR / LOG_INFO

namespace
{
constexpr int kWindowWidth = 1332;
constexpr int kWindowHeight = 999;
constexpr const char* kMapWindowName = "VerFlex-VIO: Map Viewer";
constexpr int kMaxRecordFrames = 600; // safety cap for full-run time-lapse recording
} // namespace

PangolinViewer::PangolinViewer(const Parameter& params)
{
    img_w_ = params.img_width;
    img_h_ = params.img_height;
    record_viewer_ = params.record_viewer;
}

PangolinViewer::~PangolinViewer()
{
    Stop();
}

void PangolinViewer::Start()
{
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true))
    {
        return;
    }
    thread_ = std::thread(&PangolinViewer::Run, this);
}

void PangolinViewer::Stop()
{
    finish_requested_ = true;
    if (thread_.joinable())
    {
        thread_.join();
    }
}

void PangolinViewer::Wait()
{
    if (thread_.joinable())
    {
        thread_.join();
    }
}

void PangolinViewer::Update(const VioOutput& output)
{
    std::lock_guard<std::mutex> lock(mutex_);
    cur_q_ = output.orientation;
    cur_p_ = output.position;
    has_pose_ = true;

    trajectory_.push_back(output.position);
    while (static_cast<int>(trajectory_.size()) > max_trajectory_points_)
    {
        trajectory_.pop_front();
    }

    // Active keyframes = current clone-window poses (sliding). Overwrite each tick so only the
    // currently-active keyframes are shown; old ones drop out as the window marginalizes.
    keyframe_q_ = output.active_keyframe_q;
    keyframe_p_ = output.active_keyframe_p;

    // Feature points: keep only the latest position per feature_id (overwrite old, dedup new).
    // active_ids_ = features in the current MSCKF window (this tick's feat_msckf_) -> drawn blue.
    active_ids_.clear();
    for (size_t i = 0; i < output.feature_points.size(); ++i)
    {
        const uint32_t id = output.feature_ids[i];
        map_points_[id] = output.feature_points[i];
        active_ids_.insert(id);
    }

    if (!output.image_with_features.empty())
    {
        latest_image_ = output.image_with_features;
    }
}

pangolin::OpenGlMatrix PangolinViewer::ToGl(const Eigen::Quaterniond& q_WI, const Eigen::Vector3d& p_WI)
{
    pangolin::OpenGlMatrix M;
    M.SetIdentity();
    const Eigen::Matrix3d R = q_WI.toRotationMatrix(); // R_WI: body -> world
    M.m[0] = R(0, 0); M.m[1] = R(1, 0); M.m[2] = R(2, 0);  M.m[3] = 0.0;
    M.m[4] = R(0, 1); M.m[5] = R(1, 1); M.m[6] = R(2, 1);  M.m[7] = 0.0;
    M.m[8] = R(0, 2); M.m[9] = R(1, 2); M.m[10] = R(2, 2); M.m[11] = 0.0;
    M.m[12] = p_WI(0); M.m[13] = p_WI(1); M.m[14] = p_WI(2); M.m[15] = 1.0;
    return M;
}

void PangolinViewer::DrawFrustum(const pangolin::OpenGlMatrix& Twc, float r, float g, float b, float size, float linewidth)
{
    const float w = size;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glPushMatrix();
    glMultMatrixd(Twc.m);
    glLineWidth(linewidth);
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0); glVertex3f(w, h, z);
    glVertex3f(0, 0, 0); glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w, h, z);
    glVertex3f(w, h, z); glVertex3f(w, -h, z);
    glVertex3f(-w, h, z); glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z); glVertex3f(w, h, z);
    glVertex3f(-w, -h, z); glVertex3f(w, -h, z);
    glEnd();
    glPopMatrix();
}

void PangolinViewer::DrawTrajectory(const std::deque<Eigen::Vector3d>& traj)
{
    if (traj.size() < 2)
    {
        return;
    }
    glLineWidth(trajectory_linewidth_);
    glColor3f(1.0f, 0.0f, 0.0f); // red
    glBegin(GL_LINE_STRIP);
    for (const auto& p : traj)
    {
        glVertex3d(p(0), p(1), p(2));
    }
    glEnd();
}

void PangolinViewer::DrawKeyFrames(const std::vector<Eigen::Quaterniond>& kf_q, const std::vector<Eigen::Vector3d>& kf_p)
{
    for (size_t i = 0; i < kf_p.size(); ++i)
    {
        DrawFrustum(ToGl(kf_q[i], kf_p[i]), 0.0f, 0.0f, 1.0f, keyframe_size_, keyframe_linewidth_); // blue
    }
}

void PangolinViewer::DrawMapPoints(const std::vector<Eigen::Vector3d>& points, float r, float g, float b)
{
    if (points.empty())
    {
        return;
    }
    glPointSize(point_size_);
    glColor3f(r, g, b);
    glBegin(GL_POINTS);
    for (const auto& p : points)
    {
        glVertex3d(p(0), p(1), p(2));
    }
    glEnd();
}

void PangolinViewer::Run()
{
    try
    {
        pangolin::CreateWindowAndBind(kMapWindowName, kWindowWidth, kWindowHeight);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowTrajectory("menu.Show Trajectory", true, true);
        pangolin::Var<bool> menuShowImage("menu.Show Image", true, true);
        pangolin::Var<bool> menuReset("menu.Reset", false, false);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(kWindowWidth, kWindowHeight, viewpoint_f_, viewpoint_f_, kWindowWidth / 2, kWindowHeight / 2, 0.1, 1000),
            pangolin::ModelViewLookAt(viewpoint_x_, viewpoint_y_, viewpoint_z_, 0, 0, 0, 0.0, 0.0, -1.0));

        pangolin::View& d_cam = pangolin::CreateDisplay()
                                    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0,
                                               -static_cast<float>(kWindowWidth) / static_cast<float>(kWindowHeight))
                                    .SetHandler(new pangolin::Handler3D(s_cam));

        // Latest-frame image inset (bottom-right), aspect-correct.
        const int inset_w = std::max(1, img_w_ / 2);
        const int inset_h = std::max(1, img_h_ / 2);
        pangolin::View& d_img = pangolin::CreateDisplay()
                                    .SetBounds(pangolin::Attach::Pix(8), pangolin::Attach::Pix(8 + inset_h),
                                               pangolin::Attach::ReversePix(inset_w + 8), pangolin::Attach::ReversePix(8),
                                               static_cast<double>(img_w_) / static_cast<double>(std::max(1, img_h_)));
        std::unique_ptr<pangolin::GlTexture> img_tex;

        LOG_INFO("Pangolin viewer: 3D window ready ('VerFlex-VIO: Map Viewer')");

        if (record_viewer_)
        {
            std::filesystem::create_directories(record_dir_);
            LOG_INFO("Pangolin viewer: recording up to {} frames to '{}'", kMaxRecordFrames, record_dir_);
        }
        int record_idx = 0;
        std::chrono::steady_clock::time_point last_record_time;

        while (!pangolin::ShouldQuit() && !finish_requested_)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Snapshot shared state under the lock; render outside so the VIO callback is never blocked by GL.
            pangolin::OpenGlMatrix Twc;
            Twc.SetIdentity();
            Eigen::Quaterniond cur_q_snap = Eigen::Quaterniond::Identity();
            Eigen::Vector3d cur_p_snap = Eigen::Vector3d::Zero();
            bool has_pose_snap = false;
            std::deque<Eigen::Vector3d> traj_snap;
            std::vector<Eigen::Quaterniond> kf_q_snap;
            std::vector<Eigen::Vector3d> kf_p_snap;
            std::vector<Eigen::Vector3d> mp_active_snap;
            std::vector<Eigen::Vector3d> mp_inactive_snap;
            cv::Mat img_snap;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (has_pose_)
                {
                    Twc = ToGl(cur_q_, cur_p_);
                    cur_q_snap = cur_q_;
                    cur_p_snap = cur_p_;
                    has_pose_snap = true;
                }
                traj_snap = trajectory_;
                kf_q_snap = keyframe_q_;
                kf_p_snap = keyframe_p_;
                mp_active_snap.reserve(active_ids_.size());
                mp_inactive_snap.reserve(map_points_.size());
                for (const auto& kv : map_points_)
                {
                    if (active_ids_.count(kv.first))
                    {
                        mp_active_snap.push_back(kv.second);
                    }
                    else
                    {
                        mp_inactive_snap.push_back(kv.second);
                    }
                }
                img_snap = latest_image_;
            }

            if (menuFollowCamera && has_pose_snap)
            {
                // Chase cam behind the camera, level (world-Z up). Forward = body +Z in world
                // (the camera optical axis ~ body +Z); the view sits behind it on the horizontal plane.
                const Eigen::Vector3d forward = cur_q_snap.toRotationMatrix() * Eigen::Vector3d::UnitZ();
                Eigen::Vector3d forward_h(forward.x(), forward.y(), 0.0);
                if (forward_h.norm() < 1e-6)
                {
                    forward_h = Eigen::Vector3d::UnitX();
                }
                else
                {
                    forward_h.normalize();
                }
                const Eigen::Vector3d cam_pos = cur_p_snap - chase_dist_ * forward_h + Eigen::Vector3d(0.0, 0.0, -chase_height_);
                const Eigen::Vector3d look_at = cur_p_snap + forward_h;
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    cam_pos(0), cam_pos(1), cam_pos(2), look_at(0), look_at(1), look_at(2), 0.0, 0.0, -1.0));
            }

            d_cam.Activate(s_cam);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // white background

            if (menuShowTrajectory)
            {
                DrawTrajectory(traj_snap);
            }
            DrawFrustum(Twc, 0.0f, 1.0f, 0.0f, camera_size_, camera_linewidth_); // current body pose (green)
            if (menuShowKeyFrames)
            {
                DrawKeyFrames(kf_q_snap, kf_p_snap);
            }
            if (menuShowPoints)
            {
                DrawMapPoints(mp_inactive_snap, 0.0f, 0.0f, 0.0f); // black: not in window
                DrawMapPoints(mp_active_snap, 0.0f, 0.0f, 1.0f);   // blue: in window
            }

            if (menuShowImage && !img_snap.empty())
            {
                if (!img_tex || img_tex->width != img_snap.cols || img_tex->height != img_snap.rows)
                {
                    img_tex = std::make_unique<pangolin::GlTexture>(
                        img_snap.cols, img_snap.rows, GL_RGB8, true, 0, GL_RGB, GL_UNSIGNED_BYTE);
                }
                cv::Mat rgb;
                cv::cvtColor(img_snap, rgb, cv::COLOR_BGR2RGB);
                img_tex->Upload(rgb.data, GL_RGB, GL_UNSIGNED_BYTE);
                d_img.Activate();
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                img_tex->RenderToViewportFlipY();
            }

            if (record_viewer_ && has_pose_snap && record_idx < kMaxRecordFrames)
            {
                const auto now = std::chrono::steady_clock::now();
                if (record_idx == 0 || now - last_record_time >= std::chrono::milliseconds(500))
                {
                    cv::Mat shot(kWindowHeight, kWindowWidth, CV_8UC3);
                    glPixelStorei(GL_PACK_ALIGNMENT, 1);
                    glReadPixels(0, 0, kWindowWidth, kWindowHeight, GL_BGR, GL_UNSIGNED_BYTE, shot.data);
                    cv::flip(shot, shot, 0);
                    char fn[256];
                    std::snprintf(fn, sizeof(fn), "%s/frame_%05d.png", record_dir_.c_str(), record_idx);
                    cv::imwrite(fn, shot);
                    ++record_idx;
                    last_record_time = now;
                }
            }

            pangolin::FinishFrame();

            std::this_thread::sleep_for(std::chrono::milliseconds(render_dt_ms_));

            if (menuReset)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                trajectory_.clear();
                keyframe_q_.clear();
                keyframe_p_.clear();
                map_points_.clear();
                has_pose_ = false;
                latest_image_.release();
                menuReset = false;
            }
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Pangolin viewer terminated: {}", e.what());
    }
    catch (...)
    {
        LOG_ERROR("Pangolin viewer terminated with unknown exception");
    }
}
