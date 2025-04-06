#include "vioManager.h"
#include <glog/logging.h>
#include <sophus/so3.hpp>
#include "format.h"
#include "mathematical_tools.h"
#include "utils.h"

using namespace Sophus;

namespace
{
constexpr uint32_t kNoInputDataCntThres = 100;
}  // namespace

void frontend_task_entry(std::shared_ptr<VisualManager> visual_manager)
{
    static uint32_t no_input_cnt = 0;

    while (true)
    {
        usleep(100);  // sleep for 0.01s -> 100Hz
        if (visual_manager->_input_image_buffer.empty())
        {
            no_input_cnt++;
            if (no_input_cnt > kNoInputDataCntThres)
            {
                LOG(ERROR) << "No Input data for vio frontend, going to exit...";
                exit(0);
            }
            continue;
        }
        else
        {
            no_input_cnt = 0;
        }

        std::pair<double, std::pair<cv::Mat, cv::Mat>> data = visual_manager->_input_image_buffer.front();
        visual_manager->_input_image_buffer.pop();
        std::pair<double, std::vector<CameraObs>> feature_observes;
        auto feature_base = visual_manager->GetFeatureBase();
        bool status = visual_manager->vio_frontend->TrackMonocular(data, feature_observes);
        if (status == true)
        {
            if (visual_manager->_keyframe != KeyFrameStatus::kNone)
            {
                while (!visual_manager->feature_obs_buffer.empty())
                {
                    visual_manager->feature_obs_buffer.pop();
                }
                visual_manager->_keyframe = KeyFrameStatus::kNone;
            }
            visual_manager->feature_obs_buffer.push(feature_observes);
        }
    }
}

void backend_task_entry(VioManager* vio)
{
    while (true)
    {
        usleep(100);  // sleep for 0.01s -> 100Hz
        if (!vio->initializer->IsInitialized())
        {
            bool status = vio->initializer->static_initialize();
            if (status == false)
            {
                continue;
            }
        }
        if (vio->_visual_manager->feature_obs_buffer.empty())
        {
            continue;
        }
        // std::pair<double, std::vector<CameraObs>> feature_observes = vio->_visual_manager->feature_obs_buffer.front();
        // vio->_visual_manager->feature_obs_buffer.pop();
        // vio->_visual_manager->UpdateFeature(feature_observes);
        // vio->PropagateStateAndCovariance(vio->state, feature_observes.first);
        // vio->_visual_manager->update();
    }
}

void VioManager::start_visual_system()
{
    std::thread frontend_thread(frontend_task_entry, _visual_manager);
    std::thread backend_thread(backend_task_entry, this);
    frontend_thread.detach();
    backend_thread.detach();
}

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

void VioManager::ProcessMeasurementOnce()
{
    if (!initializer->is_orientation_initialized || !initializer->is_bias_initialized)
    {
        if (!initializer->static_initialize())
        {
            LOG(ERROR) << "failed to initialize orientation and bias";
            return;
        }
    }

    while (!_visual_manager->_input_image_buffer.empty())
    {
        utils::LogValueFull log_value;
        utils::LogValueTUM log_value_tum;
        bool visual_updated = false;
        bool zupt_updated = false;

        vio_rT = boost::posix_time::microsec_clock::local_time();

        if (_visual_manager->_input_image_buffer.front().first >= _imu_manager->_imu_latest_timestamp - 0.1)
        {
            return;
        }

        if (_visual_manager->_input_image_buffer.front().first < state->ts_sec())
        {
            LOG(WARNING) << cv::format("Input image ts: %f, is older than current state ts: %f, skip current image",
                                       _visual_manager->_input_image_buffer.front().first, state->ts_sec());
            while (!_visual_manager->_input_image_buffer.empty() && _visual_manager->_input_image_buffer.front().first < state->ts_sec())
            {
                _visual_manager->_input_image_buffer.pop();
            }
            continue;
        }

        // track stereo features
        std::pair<double, std::pair<cv::Mat, cv::Mat>> new_image = _visual_manager->_input_image_buffer.front();
        _visual_manager->_input_image_buffer.pop();
        std::pair<double, std::vector<CameraObs>> feature_observes;
        if (!_visual_manager->vio_frontend->TrackStereo(new_image, feature_observes))
        {
            continue;
        }

        // stereo visual initialization
        if (!initializer->IsInitialized())
        {
            eskfSolver::PropagateStateAndCovariance(state, _imu_manager, feature_observes.first);
            if (initializer->StereoVisualInitialize(feature_observes))
            {
                log_value.init_vnorm = state->_imu_state->v()->vec().norm();
                GroundTruth gt_pv = InterpolateGroundTruth(feature_observes.first);
                log_value.groundtruth_vnorm = gt_pv.v_.norm();
                state->StochasticClone(state->_imu_state->pose());
            }
            continue;
        }

        if (_imu_manager->static_status())
        {
            eskfSolver::PropagateStateAndCovariance(state, _imu_manager, _imu_manager->_imu_latest_timestamp - 0.1);
            _imu_manager->ZuptUpdate(state);
            zupt_updated = true;
            LOG(INFO) << "ZUPT updated";
        }
        else
        {
            eskfSolver::PropagateStateAndCovariance(state, _imu_manager, feature_observes.first);
            state->StochasticClone(state->_imu_state->pose());
            _visual_manager->UpdateFeature(feature_observes);  // visual update
            if (_visual_manager->VisualUpdate())
            {
                visual_updated = true;
                LOG(INFO) << "Visual updated";
            }
        }

        /* Assign full log values */
        log_value.timestamp = state->_imu_state->ts();
        log_value.px = state->_imu_state->p()->vec().x();
        log_value.py = state->_imu_state->p()->vec().y();
        log_value.pz = state->_imu_state->p()->vec().z();
        log_value.vx = state->_imu_state->v()->vec().x();
        log_value.vy = state->_imu_state->v()->vec().y();
        log_value.vz = state->_imu_state->v()->vec().z();

        Eigen::Vector3d euler_angle = MathUtils::R2rpy(state->_imu_state->q()->Rot()) * RAD2DEG;
        log_value.roll = euler_angle.x();
        log_value.pitch = euler_angle.y();
        log_value.yaw = euler_angle.z();

        log_value.bias_gyro_x = state->_imu_state->bg()->vec().x();
        log_value.bias_gyro_y = state->_imu_state->bg()->vec().y();
        log_value.bias_gyro_z = state->_imu_state->bg()->vec().z();

        log_value.bias_acc_x = state->_imu_state->ba()->vec().x();
        log_value.bias_acc_y = state->_imu_state->ba()->vec().y();
        log_value.bias_acc_z = state->_imu_state->ba()->vec().z();

        log_value.sigma_px = std::sqrt(state->_covariance(state->_imu_state->p()->id(), state->_imu_state->p()->id()));
        log_value.sigma_py = std::sqrt(state->_covariance(state->_imu_state->p()->id() + 1, state->_imu_state->p()->id() + 1));
        log_value.sigma_pz = std::sqrt(state->_covariance(state->_imu_state->p()->id() + 2, state->_imu_state->p()->id() + 2));

        log_value.sigma_vx = std::sqrt(state->_covariance(state->_imu_state->v()->id(), state->_imu_state->v()->id()));
        log_value.sigma_vy = std::sqrt(state->_covariance(state->_imu_state->v()->id() + 1, state->_imu_state->v()->id() + 1));
        log_value.sigma_vz = std::sqrt(state->_covariance(state->_imu_state->v()->id() + 2, state->_imu_state->v()->id() + 2));

        log_value.sigma_bias_acc_x = std::sqrt(state->_covariance(state->_imu_state->ba()->id(), state->_imu_state->ba()->id()));
        log_value.sigma_bias_acc_y = std::sqrt(state->_covariance(state->_imu_state->ba()->id() + 1, state->_imu_state->ba()->id() + 1));
        log_value.sigma_bias_acc_z = std::sqrt(state->_covariance(state->_imu_state->ba()->id() + 2, state->_imu_state->ba()->id() + 2));

        log_value.sigma_bias_gyro_x = std::sqrt(state->_covariance(state->_imu_state->bg()->id(), state->_imu_state->bg()->id()));
        log_value.sigma_bias_gyro_y = std::sqrt(state->_covariance(state->_imu_state->bg()->id() + 1, state->_imu_state->bg()->id() + 1));
        log_value.sigma_bias_gyro_z = std::sqrt(state->_covariance(state->_imu_state->bg()->id() + 2, state->_imu_state->bg()->id() + 2));

        log_value.visual_updated = visual_updated;
        log_value.ZuptUpdated = zupt_updated;
        log_value.keyframe = int(_visual_manager->GetKeyframeState());

        vio_logger->SaveValues(log_value);

        /* Assign extracted log values in TUM format */
        log_value_tum.timestamp = state->_imu_state->ts();
        log_value_tum.px = state->_imu_state->p()->vec().x();
        log_value_tum.py = state->_imu_state->p()->vec().y();
        log_value_tum.pz = state->_imu_state->p()->vec().z();
        log_value_tum.qw = state->_imu_state->q()->q().w();
        log_value_tum.qx = state->_imu_state->q()->q().x();
        log_value_tum.qy = state->_imu_state->q()->q().y();
        log_value_tum.qz = state->_imu_state->q()->q().z();
        vio_logger_tum->SaveValues(log_value_tum);
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

void VioManager::ImuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    ImuData data;
    data.ts_sec = msg->header.stamp.toSec() - _initial_timestamp;
    data.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    data.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    initializer->FeedImuMeasurement(data);
    _imu_manager->FeedImuMeasurement(data);
}

void VioManager::CameraCallback(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1)
{
    double ts_sec = msg0->header.stamp.toSec() - _initial_timestamp;
    cv::Mat image_l, image_l_rectify;
    cv::Mat image_r, image_r_rectify;
    Utils::transfer_image(msg0, image_l);
    Utils::transfer_image(msg1, image_r);
    _camera_model_0->RectifyStereoImages(image_l, image_r, image_l_rectify, image_r_rectify);
    _visual_manager->feed_image({ts_sec, {image_l_rectify, image_r_rectify}});
}
