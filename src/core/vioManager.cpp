#include <glog/logging.h>
#include <sophus/so3.hpp>

#include "vioManager.h"
#include "camModel.h"
#include "imuPreIntegration.h"
#include "mathematical_tools.h"
#include "utils.h"
#include "visualizer.h"

using namespace Sophus;

namespace
{
constexpr double kRad2Deg = 180.0 / M_PI;
constexpr uint32_t kNoInputDataCntThres = 100;
constexpr double kMaxAllowedSysUpdateInterval = 1.0f;  // 2s
constexpr uint32_t kMinVisualFeaturesForUpdate = 10;
}  // namespace

/**
 * @brief Constructor for VioManager
 * @param nh Shared pointer to ROS NodeHandle
 * @param params Configuration parameters
 */
VioManager::VioManager(std::shared_ptr<ros::NodeHandle>& nh, const Param& params)
{
    nh_ = nh;
    params_ = params;

    // Solver configuration
    if (params.solver_type == static_cast<int>(SolverType::ESKF))
    {
        solver = std::make_shared<eskfSolver>(params.use_fej, params.enable_schmidt_eskf);
    }
    else if (params.solver_type == static_cast<int>(SolverType::SQRT_ESKF))
    {
        solver = std::make_shared<SqrtEskfSolver>(params.use_fej);
    }

    // Initialize state
    state = std::make_shared<State>(params);

    // Initialize IMU manager
    _imu_manager = std::make_shared<ImuManager>(params, state, solver);

    // Initialize visual manager
    _visual_manager = std::make_shared<VisualManager>(nh, params, state, solver);

    // Initializer configuration
    if (params.initial_type == static_cast<int>(InitializerType::kStatic) && params.camera_num == 2)
    {
        initializer = std::make_shared<Initializer>(params, _visual_manager, state);
    }
    else if (params.initial_type == static_cast<int>(InitializerType::kDynamic))
    {
        initializer = std::make_shared<DynamicInitializer>(params, _visual_manager, state);
    }
    else
    {
        LOG(FATAL) << "Unsupported initialization type or camera type!";
        exit(1);
    }

    lazy_time_ = params.lazy_time;
    use_zupt_ = params.use_zupt;
}

/**
 * @brief Reset the VIO system to its initial state
 */
void VioManager::ResetSystem()
{
    state->reset();
    _visual_manager->reset();
    initializer->reset();
    last_update_timestamp_ = -1.0;
    LOG(INFO) << "VIO system reset";
}

/**
 * @brief Interpolate ground truth data for a given timestamp
 * @param ts Timestamp for interpolation
 * @return Interpolated GroundTruth object
 */
GroundTruth VioManager::InterpolateGroundTruth(const double ts) const
{
    auto interpolate = [](std::pair<double, GroundTruth> pv1, std::pair<double, GroundTruth> pv2, double ts)
    {
        double lambda = (ts - pv1.first) / (pv2.first - pv1.first);
        GroundTruth pv_interp;
        pv_interp.p_ = pv1.second.p_ + lambda * (pv2.second.p_ - pv1.second.p_);
        pv_interp.v_ = pv1.second.v_ + lambda * (pv2.second.v_ - pv1.second.v_);
        return pv_interp;
    };

    // if ground truth buffer is empty, return empty ground truth
    if (ground_truth_.empty())
    {
        return GroundTruth();
    }

    auto it = ground_truth_.lower_bound(ts);
    if (it == ground_truth_.end())
    {
        // If timestamp is greater than the largest key, return the largest value
        return std::prev(it)->second;
    }
    else if (it == ground_truth_.begin())
    {
        // If timestamp is smaller than the smallest key, return the smallest value
        return it->second;
    }
    else
    {
        // Otherwise, interpolate between the closest values
        return interpolate(*std::prev(it), *it, ts);
    }
}

/**
 * @brief Attempt to track features using the visual frontend
 * @param images Pair of timestamp and vector of images
 * @param feature_observes Output pair of timestamp and vector of camera observations
 */
bool VioManager::TryFrontendTrack(const std::pair<double, std::vector<cv::Mat>> &images,
                                  std::pair<double, std::vector<CameraObs>> &feature_observes)
{
    const double td_visual = state->enableEstimateTdVisual() ? state->td_visual().data() : 0.0;
    std::vector<ImuData> imu_data =
        _imu_manager->AccessIntervalImuMeasurements(state->ts_sec(), images.first + td_visual);
    bool do_prediction_flag = params_.frontend_prediction && initializer->IsInitialized();

    Eigen::Matrix3d Rwc = Eigen::Matrix3d::Identity();
    if (!imu_data.empty())
    {
        ImuPreintegrator pre_integration;
        pre_integration.feedImuMeasuremnts(imu_data);
        pre_integration.Propagate(state->_imu_state->ba()->vec(), state->_imu_state->bg()->vec());
        Eigen::Matrix3d Rwi = state->_imu_state->q()->Rot() * pre_integration.dR();
        Rwc = Rwi * CamModel::getInstance().Ric(LEFT_CAM);
    }
    else
    {
        do_prediction_flag = false;
    }

    if (params_.camera_num == CamType::MONO)
    {
        return _visual_manager->vio_frontend->TrackMonocular(images, Rwc, do_prediction_flag, feature_observes);
    }
    else if (params_.camera_num == CamType::STEREO)
    {
        return _visual_manager->vio_frontend->TrackStereo(images, Rwc, do_prediction_flag, feature_observes);
    }
    else
    {
        LOG(ERROR) << "Unsupported camera type: " << params_.camera_num;
        return false;
    }
    return false;
}

/**
 * @brief Attempt dynamic initialization of the VIO system
 * @param image Pair of timestamp and vector of images
 * @param feature_observes Pair of timestamp and vector of camera observations
 * @return True if initialization is successful, false otherwise
 */
bool VioManager::TryDynamicInitialization(const std::pair<double, std::vector<cv::Mat>>& image,
                                          const std::pair<double, std::vector<CameraObs>>& feature_observes)
{
    if (!initializer->IsInitialized())
    {
        std::shared_ptr<DynamicInitializer> dynamic_initializer =
            std::dynamic_pointer_cast<DynamicInitializer>(initializer);

        std::optional<double> lastest_obv_ts = dynamic_initializer->getLastestFeatureMeasurementTimestamp();
        std::pair<double, cv::Mat> mono_image = std::make_pair(feature_observes.first, image.second[LEFT_CAM]);
        if (!dynamic_initializer->feedVisualMeasurement(mono_image, feature_observes, *state->_imu_state))
        {
            if (feature_observes.second.size() < Sfm::kMinRequiredFeaturesPerFrame)
            {
                _visual_manager->SetKeyframeState(
                    KeyFrameStatus::kFeatureLostTooMuch); // Switch keyframe if too few features
                LOG(INFO) << fmt::format("Switch visual frontend keyframe for dynamic initialization");
            }
            return false;
        }

        std::vector<ImuData> imu_data;
        if (lastest_obv_ts.has_value())
        {
            ImuPreintegrator pre_integration;
            imu_data = _imu_manager->AccessIntervalImuMeasurements(lastest_obv_ts.value(), feature_observes.first);
            if (imu_data.empty())
            {
                LOG(WARNING) << fmt::format("No IMU data available for dynamic initialization at {:f}s",
                                            feature_observes.first);
                initializer->reset();
                return false;
            }

            pre_integration.feedImuMeasuremnts(imu_data);
            pre_integration.Propagate(state->_imu_state->ba()->vec(), state->_imu_state->bg()->vec());
            dynamic_initializer->feedImuPreIntegration(pre_integration);
        }

        if (dynamic_initializer->TryInitialize())
        {
            last_update_timestamp_ = state->ts_sec();
            solver->StochasticClone(state, &imu_data);
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        LOG(INFO) << "Vio system is already initialized.";
        return true;  // Already initialized
    }
}

/**
 * @brief Attempt static initialization of the VIO system
 * @param image Pair of timestamp and vector of images
 * @param feature_observes Pair of timestamp and vector of camera observations
 * @return True if initialization is successful, false otherwise
 */
bool VioManager::TryStaticInitialization(const std::pair<double, std::vector<cv::Mat>>& image,
                                         const std::pair<double, std::vector<CameraObs>>& feature_observes)
{
    if (!initializer->IsInitialized())
    {
        if (!initializer->is_orientation_initialized || !initializer->is_bias_initialized)
        {
            if (!initializer->InitializeOrientation())
            {
                LOG(ERROR) << "Failed to initialize orientation and bias";
                return false;
            }
        }

        if (feature_observes.second.size() < Sfm::kMinRequiredFeaturesPerFrame)
        {
            _visual_manager->SetKeyframeState(
                KeyFrameStatus::kFeatureLostTooMuch); // Switch keyframe if too few features
            LOG(INFO) << fmt::format("Switch visual frontend keyframe for dynamic initialization");
        }

        std::vector<ImuData> imu_data =
            _imu_manager->AccessIntervalImuMeasurements(state->_imu_state->ts(), feature_observes.first);
        solver->PropagateStateAndCovariance(imu_data, feature_observes.first, state);
        if (initializer->StereoVisualInitialize(feature_observes))
        {
            last_update_timestamp_ = state->ts_sec();
            solver->StochasticClone(state, &imu_data);
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        LOG(INFO) << "Vio system is already initialized.";
        return true;  // Already initialized
    }
}

/**
 * @brief Perform ZUPT update of the VIO system if conditions are met
 * @param ts_sec Current timestamp in seconds
 */
bool VioManager::TryZuptUpdate(const double ts_sec)
{
    if (use_zupt_)
    {
        if (_imu_manager->IsStaticStatus())
        {
            std::vector<ImuData> imu_data =
                _imu_manager->AccessIntervalImuMeasurements(state->_imu_state->ts(), ts_sec);
            if (imu_data.empty())
            {
                LOG(WARNING) << fmt::format("No IMU data available for ZUPT update at {:f}s", ts_sec);
                return false;
            }

            solver->PropagateStateAndCovariance(imu_data, ts_sec, state);
            _imu_manager->ZuptUpdate(state);
            return true;
        }
    }

    return false;
}

/**
 * @brief Perform visual update of the VIO system using feature observations
 * @param feature_observes Pair of timestamp and vector of camera observations
 */
bool VioManager::TryVisualUpdate(const std::pair<double, std::vector<CameraObs>>& feature_observes)
{
    double time0 = state->ts_sec();
    double time_comp = feature_observes.first;
    if (state->enableEstimateTdVisual())
    {
        time_comp += state->td_visual().data();
    }

    std::vector<ImuData> imu_data = _imu_manager->AccessIntervalImuMeasurements(time0, time_comp);
    if (imu_data.empty())
    {
        LOG(WARNING) << fmt::format("No IMU data available for visual update at {:f}s, time0: {:f}s, time_comp: {:f}s",
                                    feature_observes.first, time0, time_comp);
        return false;
    }

    solver->PropagateStateAndCovariance(imu_data, time_comp, state);

    solver->StochasticClone(state, &imu_data);

    _visual_manager->UpdateFeatureStatistic(state->ts_sec(), feature_observes);

    if (_visual_manager->VisualUpdate())
    {
        LOG(INFO) << fmt::format(GREEN "VIO updated, current state ts: {:f}, pos: [{:.3f}, {:.3f}, {:.3f}], vel: "
                                 "[{:.3f}, {:.3f}, {:.3f}]" RESET,
                                 state->ts_sec(), state->_imu_state->p()->vec().x(), state->_imu_state->p()->vec().y(),
                                 state->_imu_state->p()->vec().z(), state->_imu_state->v()->vec().x(),
                                 state->_imu_state->v()->vec().y(), state->_imu_state->v()->vec().z());

        // if (params_.estimate_td_visual)
        // {
        //     LOG(INFO) << fmt::format("td_visual: {:f}s", state->td_visual().data());
        // }

        return true;
    }

    return false;
}

/**
 * @brief Check the VIO system state for validity
 * @param ts_sec Current timestamp in seconds
 */
bool VioManager::CheckVioState(const double ts_sec) const
{
    if (std::abs(ts_sec - last_update_timestamp_) > kMaxAllowedSysUpdateInterval)
    {
        LOG(WARNING) << fmt::format("VIO update intervals: {:f}, is larger than {:f}, reset vio system",
                                   std::abs(ts_sec - last_update_timestamp_), kMaxAllowedSysUpdateInterval);
        return false;
    }

    return true;
}

/**
 * @brief Publish VIO state and features to ROS topics
 * @param ts_sec Current timestamp in seconds
 */
void VioManager::PublishVioMessages(const double ts_sec)
{
    Visualizer::getInstance().PublishVioState(state->_imu_state);

    if (visual_updated_this_tick_)
    {
        Visualizer::getInstance().PublishFeatures(state->ts_sec(), _visual_manager->feat_msckf_);
    }

    std::pair<double, cv::Mat> image_with_features = _visual_manager->vio_frontend->getImageWithFeatures();
    if (abs(image_with_features.first - ts_sec) < 0.01 && !image_with_features.second.empty())
    {
        Visualizer::getInstance().PublishImageWithFeatures(image_with_features.first, image_with_features.second);
    }
}

/**
 * @brief Check the availability and validity of measurements for processing
 * @return FrameOptions indicating the status of measurements
 */
FrameOptions VioManager::CheckMeasurements() const
{
    const double td_visual = state->enableEstimateTdVisual() ? state->td_visual().data() : 0.f;

    // Lock the input image buffer for checking
    std::lock_guard<std::mutex> lock(_visual_manager->input_image_buffer_mutex_);

    if (_visual_manager->_input_image_buffer.empty())
    {
        LOG(WARNING) << "No input image data available, skipping measurement processing.";
        return FrameOptions::kStatusError;
    }

    if (_imu_manager->_imu_latest_timestamp - lazy_time_ <
        _visual_manager->_input_image_buffer.front().first + td_visual)
    {
        // LOG(WARNING) << fmt::format(
        //     "IMU latest timestamp: {:f}s, lazy_time: {:f}, (imu_time - lazy_time) is slower than the input image
        //     timestamp: {:f}s, waiting for IMU data", _imu_manager->_imu_latest_timestamp, lazy_time_,
        //     _visual_manager->_input_image_buffer.front().first + td_visual);
        return FrameOptions::kWaitForImu;
    }

    if (_visual_manager->_input_image_buffer.front().first + td_visual < state->ts_sec())
    {
        LOG(WARNING) << fmt::format("Input image timestamp: {:f} is older than the current state timestamp: {:f}, "
                                    "skipping measurement processing",
                                    _visual_manager->_input_image_buffer.front().first + td_visual, state->ts_sec());
        return FrameOptions::kSkipFrame;
    }

    return FrameOptions::kStatusOk;
}

/**
 * @brief Clear expired IMU and visual measurements from their respective buffers
 * @note Measurements older than a certain threshold relative to the oldest cloned pose are removed
 */
void VioManager::ClearExpiredMeasurements()
{
    constexpr double kClearToTimestampThreshold = 0.5; // In second
    const double clear_to_timestamp = state->_clone_pose.begin()->first - kClearToTimestampThreshold;

    // Clear expired IMU measurements
    _imu_manager->ClearExpiredMeasurements(clear_to_timestamp);

    // Clear expired visual measurements
    _visual_manager->ClearExpiredMeasurements(clear_to_timestamp);
}

/**
 * @brief Process a single measurement from the input buffer
 * @note This function handles the entire VIO processing pipeline for a single measurement, including feature tracking,
 * initialization, state updates, and publishing results
 */
void VioManager::ProcessMeasurementOnce()
{
    while (!_visual_manager->_input_image_buffer.empty())
    {
        double ts_sec = 0;
        visual_updated_this_tick_ = false;
        zupt_updated_this_tick_ = false;

        FrameOptions frame_status = CheckMeasurements();
        switch (frame_status)
        {
            case FrameOptions::kStatusError:
            case FrameOptions::kSkipFrame:
                _visual_manager->_input_image_buffer.pop();
                continue;
            case FrameOptions::kWaitForImu:
                return;
            case FrameOptions::kStatusOk:
                break;
            default:
                LOG(ERROR) << "Unknown status: " << static_cast<int>(frame_status);
                return;
        }

        // Vio frontend process
        std::pair<double, std::vector<CameraObs>> feature_observes;
        std::pair<double, std::vector<cv::Mat>> new_image = _visual_manager->_input_image_buffer.front();
        _visual_manager->_input_image_buffer.pop();
        ts_sec = new_image.first;

        // Tracking
        if (!TryFrontendTrack(new_image, feature_observes))
        {
            LOG(WARNING) << fmt::format("Failed to track features in image at ts: {:f}", new_image.first);
            continue;
        }

        // Dynamic monocular visual initialization
        if (!initializer->IsInitialized())
        {
            if (initializer->init_type == InitializerType::kDynamic)
            {
                if (TryDynamicInitialization(new_image, feature_observes))
                {
                    LOG(INFO) << fmt::format("Dynamic initialized successfully at {:.3f}s", ts_sec);
                }
            }
            else if (initializer->init_type == InitializerType::kStatic)
            {
                if (TryStaticInitialization(new_image, feature_observes))
                {
                    LOG(INFO) << fmt::format("Static stereo visual initialized successfully at {:.3f}s", ts_sec);
                }
            }
            continue;
        }

        // Zupt update process
        if (TryZuptUpdate(ts_sec))
        {
            zupt_updated_this_tick_ = true;
            last_update_timestamp_ = state->ts_sec();
            ClearExpiredMeasurements();
        }

        // Visual update process
        if (TryVisualUpdate(feature_observes))
        {
            visual_updated_this_tick_ = true;
            last_update_timestamp_ = state->ts_sec();
            ClearExpiredMeasurements();
        }

        // Reset vio system if system goes not well
        if (!CheckVioState(ts_sec))
        {
            ResetSystem();
            continue;
        }

        // Publish VIO state and features
        PublishVioMessages(ts_sec);

        /* Save vio results to log */
        SaveResultsToFile();
    }
}

/**
 * @brief Start the frontend processing thread
 */
void VioManager::StartFrontendThread()
{
    frontend_thread_ = std::thread(&VioManager::FrontendLoop, this);
}

/**
 * @brief Start the backend processing thread
 */
void VioManager::StartBackendThread()
{
    backend_thread_ = std::thread(&VioManager::BackendLoop, this);
}

/**
 * @brief Frontend processing loop running in a separate thread
 * @note Continuously checks for new images, performs feature tracking, and stores the results in a thread-safe queue
 */
void VioManager::FrontendLoop()
{
    std::pair<double, std::vector<cv::Mat>> new_image;
    std::pair<double, std::vector<CameraObs>> feature_observes;

    for(;;)
    {
        FrameOptions frame_status = CheckMeasurements();
        {
            std::lock_guard<std::mutex> lock(_visual_manager->input_image_buffer_mutex_);
            if (frame_status == FrameOptions::kStatusOk)
            {
                new_image = _visual_manager->_input_image_buffer.front();
                _visual_manager->_input_image_buffer.pop();
            }
            else if (frame_status == FrameOptions::kSkipFrame)
            {
                if (!_visual_manager->_input_image_buffer.empty())
                {
                    _visual_manager->_input_image_buffer.pop();
                }
                continue;
            }
            else
            {
                continue;
            }
        }

        if (!TryFrontendTrack(new_image, feature_observes))
        {
            LOG(WARNING) << fmt::format("Failed to track features in image at ts: {:f}", new_image.first);
            continue;
        }

        // Lock the feature observes queue
        {
            std::lock_guard<std::mutex> lock(feature_observes_queue_mutex_);
            feature_observes_queue_.emplace(feature_observes);
            image_queue_.emplace(new_image);
        }
    }
}

/**
 * @brief Backend processing loop running in a separate thread
 * @note Continuously checks for new feature observations and performs VIO updates
 */
void VioManager::BackendLoop()
{
    for(;;)
    {
        std::pair<double, std::vector<CameraObs>> feature_observes;
        std::pair<double, std::vector<cv::Mat>> new_image;
        {
            std::lock_guard<std::mutex> lock(feature_observes_queue_mutex_);
            if (!feature_observes_queue_.empty())
            {
                feature_observes = feature_observes_queue_.front();
                new_image = image_queue_.front();
                feature_observes_queue_.pop();
                image_queue_.pop();
            }
            else
            {
                continue;
            }
        }

        double ts_sec = feature_observes.first;
        visual_updated_this_tick_ = false;
        zupt_updated_this_tick_ = false;

        // Dynamic monocular visual initialization
        if (!initializer->IsInitialized())
        {
            if (initializer->init_type == InitializerType::kDynamic)
            {
                if (TryDynamicInitialization(new_image, feature_observes))
                {
                    LOG(INFO) << fmt::format("Dynamic initialized successfully at {:.3f}s", ts_sec);
                }
            }
            else if (initializer->init_type == InitializerType::kStatic)
            {
                if (TryStaticInitialization(new_image, feature_observes))
                {
                    LOG(INFO) << fmt::format("Static stereo visual initialized successfully at {:.3f}s", ts_sec);
                }
            }
            continue;
        }

        // Zupt update process
        if (TryZuptUpdate(ts_sec))
        {
            zupt_updated_this_tick_ = true;
            last_update_timestamp_ = state->ts_sec();
            ClearExpiredMeasurements();
        }

        // Visual update process
        if (TryVisualUpdate(feature_observes))
        {
            visual_updated_this_tick_ = true;
            last_update_timestamp_ = state->ts_sec();
            ClearExpiredMeasurements();
        }

        // Reset vio system if system goes not well
        if (!CheckVioState(ts_sec))
        {
            ResetSystem();
            continue;
        }

        // Publish VIO state and features
        PublishVioMessages(ts_sec);

        /* Save vio results to log */
        SaveResultsToFile();
    }
}

/**
 * @brief Save VIO results to log files
 * @note Saves both full log and TUM format log based on configuration
 */
void VioManager::SaveResultsToFile()
{
    if (params_.save_full_log)
    {
        static std::string full_filename = [this]()
        {
            char time_str[100];
            std::time_t now = std::time(nullptr);
            std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
            return params_.log_path + "/" + params_.bag_name + "_" + std::string(time_str) + "_full.csv";
        }();

        utils::LogValueFull log_value;
        log_value.timestamp = state->_imu_state->ts();
        log_value.px = state->_imu_state->p()->vec().x();
        log_value.py = state->_imu_state->p()->vec().y();
        log_value.pz = state->_imu_state->p()->vec().z();
        log_value.vx = state->_imu_state->v()->vec().x();
        log_value.vy = state->_imu_state->v()->vec().y();
        log_value.vz = state->_imu_state->v()->vec().z();

        Eigen::Vector3d euler_angle = utils::math::R2rpy(state->_imu_state->q()->Rot()) * RAD2DEG;
        log_value.roll = euler_angle.x();
        log_value.pitch = euler_angle.y();
        log_value.yaw = euler_angle.z();

        log_value.bias_gyro_x = state->_imu_state->bg()->vec().x();
        log_value.bias_gyro_y = state->_imu_state->bg()->vec().y();
        log_value.bias_gyro_z = state->_imu_state->bg()->vec().z();

        log_value.bias_acc_x = state->_imu_state->ba()->vec().x();
        log_value.bias_acc_y = state->_imu_state->ba()->vec().y();
        log_value.bias_acc_z = state->_imu_state->ba()->vec().z();

        log_value.sigma_px = std::sqrt(state->Covariance()(state->_imu_state->p()->id(), state->_imu_state->p()->id()));
        log_value.sigma_py =
            std::sqrt(state->Covariance()(state->_imu_state->p()->id() + 1, state->_imu_state->p()->id() + 1));
        log_value.sigma_pz =
            std::sqrt(state->Covariance()(state->_imu_state->p()->id() + 2, state->_imu_state->p()->id() + 2));

        log_value.sigma_vx = std::sqrt(state->Covariance()(state->_imu_state->v()->id(), state->_imu_state->v()->id()));
        log_value.sigma_vy =
            std::sqrt(state->Covariance()(state->_imu_state->v()->id() + 1, state->_imu_state->v()->id() + 1));
        log_value.sigma_vz =
            std::sqrt(state->Covariance()(state->_imu_state->v()->id() + 2, state->_imu_state->v()->id() + 2));

        log_value.sigma_bias_acc_x =
            std::sqrt(state->Covariance()(state->_imu_state->ba()->id(), state->_imu_state->ba()->id()));
        log_value.sigma_bias_acc_y =
            std::sqrt(state->Covariance()(state->_imu_state->ba()->id() + 1, state->_imu_state->ba()->id() + 1));
        log_value.sigma_bias_acc_z =
            std::sqrt(state->Covariance()(state->_imu_state->ba()->id() + 2, state->_imu_state->ba()->id() + 2));

        log_value.sigma_bias_gyro_x =
            std::sqrt(state->Covariance()(state->_imu_state->bg()->id(), state->_imu_state->bg()->id()));
        log_value.sigma_bias_gyro_y =
            std::sqrt(state->Covariance()(state->_imu_state->bg()->id() + 1, state->_imu_state->bg()->id() + 1));
        log_value.sigma_bias_gyro_z =
            std::sqrt(state->Covariance()(state->_imu_state->bg()->id() + 2, state->_imu_state->bg()->id() + 2));

        log_value.visual_updated = visual_updated_this_tick_;
        log_value.zupt_updated = zupt_updated_this_tick_;
        log_value.keyframe = int(_visual_manager->GetKeyframeState());

        utils::Logger *full_logger = utils::Logger::GetInstance(full_filename);
        if (full_logger)
        {
            full_logger->SaveValues(log_value);
        }
        else
        {
            LOG(WARNING) << "Failed to get Full logger for: " << full_filename;
        }
    }

    /* Assign extracted log values in TUM format */
    if (params_.save_tum_log)
    {
        static std::string tum_filename = [this]() {
            char time_str[100];
            std::time_t now = std::time(nullptr);
            std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
            return params_.log_path + "/" + params_.bag_name + "_" + std::string(time_str) + "_tum.csv";
        }();

        utils::LogValueTUM log_value_tum;
        log_value_tum.timestamp = state->_imu_state->ts();
        log_value_tum.px = state->_imu_state->p()->vec().x();
        log_value_tum.py = state->_imu_state->p()->vec().y();
        log_value_tum.pz = state->_imu_state->p()->vec().z();
        log_value_tum.qw = state->_imu_state->q()->q().w();
        log_value_tum.qx = state->_imu_state->q()->q().x();
        log_value_tum.qy = state->_imu_state->q()->q().y();
        log_value_tum.qz = state->_imu_state->q()->q().z();

        utils::Logger* tum_logger = utils::Logger::GetInstance(tum_filename);
        if (tum_logger)
        {
            tum_logger->SaveValues(log_value_tum);
        }
        else
        {
            LOG(WARNING) << "Failed to get TUM logger for: " << tum_filename;
        }
    }
}

void VioManager::GroundTruthCallback(const geometry_msgs::PointStamped::ConstPtr& msg)
{
    double ts_sec = msg->header.stamp.toSec() - _initial_timestamp;
    GroundTruth gt_pv;
    gt_pv.p_ << msg->point.x, msg->point.y, msg->point.z;
    if (!ground_truth_.empty())
    {
        auto it = ground_truth_.rbegin();
        gt_pv.v_ = (gt_pv.p_ - it->second.p_) / (ts_sec - it->first);
    }
    ground_truth_.try_emplace(ts_sec, gt_pv);
}

/**
 * @brief IMU data callback function
 * @param msg Pointer to the incoming IMU message
 */
void VioManager::ImuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    ImuData data;
    data.ts_sec = msg->header.stamp.toSec() - _initial_timestamp;
    data.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    data.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    initializer->FeedImuMeasurement(data);
    _imu_manager->FeedImuMeasurement(data);
}

/**
 * @brief Stereo camera data callback function
 * @param msg0 Pointer to the left camera image message
 * @param msg1 Pointer to the right camera image message
 */
void VioManager::CallbackStereo(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1)
{
    double ts_sec = msg0->header.stamp.toSec() - _initial_timestamp;
    std::vector<cv::Mat> images;

    if (msg0 != nullptr && msg1 != nullptr)
    {
        cv::Mat image_l, image_l_rectify;
        cv::Mat image_r, image_r_rectify;
        utils::transfer_image(msg0, image_l);
        utils::transfer_image(msg1, image_r);
        CamModel::getInstance().RectifyImage(0, image_l, &image_l_rectify);
        CamModel::getInstance().RectifyImage(1, image_r, &image_r_rectify);
        images.push_back(image_l_rectify);
        images.push_back(image_r_rectify);
        _visual_manager->FeedImages(std::make_pair(ts_sec, images));
    }
}

/**
 * @brief Monocular camera data callback function
 * @param msg0 Pointer to the monocular camera image message
 */
void VioManager::CallbackMonocular(const sensor_msgs::ImageConstPtr& msg0)
{
    double ts_sec = msg0->header.stamp.toSec() - _initial_timestamp;
    std::vector<cv::Mat> images;

    if (msg0 != nullptr)
    {
        cv::Mat image_l, image_l_rectify;
        utils::transfer_image(msg0, image_l);
        CamModel::getInstance().RectifyImage(0, image_l, &image_l_rectify);
        images.push_back(image_l_rectify);
        _visual_manager->FeedImages(std::make_pair(ts_sec, images));
    }
}

/**
 * @brief Camera data callback function for both mono and stereo setups
 * @param msg0 Pointer to the first camera image message (left or mono)
 * @param msg1 Pointer to the second camera image message (right), can be nullptr for mono
 */
void VioManager::CameraCallback(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1)
{
    double ts_sec = msg0->header.stamp.toSec() - _initial_timestamp;
    std::vector<cv::Mat> images;

    if (params_.camera_num == CamType::MONO && msg0 != nullptr)
    {
        cv::Mat image_l, image_l_rectify;
        utils::transfer_image(msg0, image_l);
        // CamModel::getInstance().RectifyImage(0, image_l, &image_l_rectify);
        if (params_.use_histequal)
        {
            cv::equalizeHist(image_l_rectify, image_l_rectify);
        }

        images.push_back(image_l_rectify);
        _visual_manager->FeedImages(std::make_pair(ts_sec, images));
    }

    else if (params_.camera_num == CamType::STEREO && msg0 != nullptr && msg1 != nullptr)
    {
        cv::Mat image_l, image_l_rectify;
        cv::Mat image_r, image_r_rectify;
        utils::transfer_image(msg0, image_l);
        utils::transfer_image(msg1, image_r);
        // CamModel::getInstance().RectifyImage(0, image_l, &image_l_rectify);
        // CamModel::getInstance().RectifyImage(1, image_r, &image_r_rectify);
        if (params_.use_histequal)
        {
            // cv::equalizeHist(image_l_rectify, image_l_rectify);
            // cv::equalizeHist(image_r_rectify, image_r_rectify);
            cv::equalizeHist(image_l, image_l);
            cv::equalizeHist(image_r, image_r);
        }
        images.push_back(image_l);
        images.push_back(image_r);
        _visual_manager->FeedImages(std::make_pair(ts_sec, images));
    }
}
