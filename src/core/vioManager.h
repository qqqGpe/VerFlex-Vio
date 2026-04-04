#ifndef __VIO_MANAGER__
#define __VIO_MANAGER__
#include <Eigen/Core>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>

#include "solver.h"
#include "eskf_solver.h"
#include "sqrt_eskf_solver.h"
#include "ImuManager.h"
#include "camModel.h"
#include "vioFrontend.h"
#include "initializer.h"
#include "dynamicInitializer.h"
#include "logger.h"
#include "mathematical_tools.h"
#include "parameter.h"
#include "sensorType.h"
#include "vioState.h"
#include "visualManager.h"

enum class FrameOptions
{
    kStatusOk = 0,
    kStatusError = 1,
    kWaitForImu = 2,
    kSkipFrame = 3
};

struct VioOutput
{
    double timestamp = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> feature_points;
    cv::Mat image_with_features;
};

using VioOutputCallback = std::function<void(const VioOutput&)>;

class VioManager
{
   public:
    VioManager() = default;
    VioManager(const Parameter& params);

    ~VioManager() {}

    void SetOutputCallback(VioOutputCallback callback) { output_callback_ = std::move(callback); }

    void FeedImuData(const ImuData& data);

    void FeedImageData(double ts_sec, const std::vector<cv::Mat>& images);

    void FeedGroundTruth(const GroundTruth& gt);

    bool TryFrontendTrack(const std::pair<double, std::vector<cv::Mat>>& images, std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool TryDynamicInitialization(const std::pair<double, std::vector<cv::Mat>>& image,
                                  const std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool TryStaticInitialization(const std::pair<double, std::vector<cv::Mat>>& image,
                                 const std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool TryVisualUpdate(const std::pair<double, std::vector<CameraObs>>& feature_observes);

    bool TryZuptUpdate(const double ts_sec);

    bool CheckVioState(const double ts_sec) const;

    void ClearExpiredMeasurements();

    void PublishVioMessages(const double ts_sec);

    void SetInitialTimeStamp(double initial_timestamp) { _initial_timestamp = initial_timestamp; }

    FrameOptions CheckMeasurements() const;

    void ProcessMeasurementOnce();

    void ResetSystem();

    GroundTruth InterpolateGroundTruth(const double ts) const;

    void FrontendLoop();

    void BackendLoop();

    void StartFrontendThread();

    void StartBackendThread();

    double _initial_timestamp = 0.f;
    double lazy_time_ = 0.2; // In seconds
    std::shared_ptr<State> state;
    std::shared_ptr<Initializer> initializer;
    std::shared_ptr<ImuManager> _imu_manager;
    std::shared_ptr<VisualManager> _visual_manager;
    std::shared_ptr<MsckfSolverBase> solver;
    std::map<double, std::pair<cv::Mat, cv::Mat>> image_bak;
    std::map<double, GroundTruth> ground_truth_;

   private:
    void SaveResultsToFile();

    Parameter params_;
    bool use_zupt_ = false;
    VioOutputCallback output_callback_;
    uint8_t visual_updated_this_tick_ = false;
    uint8_t zupt_updated_this_tick_ = false;
    double last_update_timestamp_ = -1.0;
    std::pair<double, std::vector<CameraObs>> last_feature_observes_;

    std::thread frontend_thread_;
    std::thread backend_thread_;
    std::queue<std::pair<double, std::vector<CameraObs>>> feature_observes_queue_; // (ts_sec, feature_observes)
    std::queue<std::pair<double, std::vector<cv::Mat>>> image_queue_; // (ts_sec, images)
    mutable std::mutex feature_observes_queue_mutex_; // Mutex for feature_observes_queue_
};

#endif