#include "vioManager.h"
#include <glog/logging.h>
#include <sophus/so3.hpp>
#include "format.h"
#include "mathematical_tools.h"
#include "utils.h"

namespace
{
constexpr int kImuOutputHz = 200;
constexpr double kMaxImuToleranceDelayTime = 1.0 / kImuOutputHz * 5;
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
        auto feature_base = visual_manager->get_feature_base();
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
        if (!vio->initializer->is_initialized())
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
        // vio->propagate_state_and_covariance(vio->state, feature_observes.first);
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

void VioManager::process_measurememt_once()
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
        utils::LogValue log_value;
        bool visual_updated = false;
        bool ZuptUpdated = false;

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
        std::pair<double, std::pair<cv::Mat, cv::Mat>> image_data = _visual_manager->_input_image_buffer.front();
        _visual_manager->_input_image_buffer.pop();
        std::pair<double, std::vector<CameraObs>> feature_observes;
        if (!_visual_manager->vio_frontend->TrackStereo(image_data, feature_observes))
        {
            continue;
        }

        // stereo visual initialization
        if (!initializer->is_initialized())
        {
            propagate_state_and_covariance(state, feature_observes.first);
            if (initializer->StereoVisualInitialize(feature_observes))
            {
                log_value.init_vnorm = state->_imu_state->v()->vec().norm();
                GroundTruth gt_pv = InterpolateGroundTruth(feature_observes.first);
                log_value.groundtruth_vnorm = gt_pv.v_.norm();
                state->stochastic_clone(state->_imu_state->pose());
            }
            continue;
        }

        if (_imu_manager->static_status() && 0)
        {
            propagate_state_and_covariance(state, _imu_manager->_imu_latest_timestamp - 0.1);
            _imu_manager->ZuptUpdate(state);
            ZuptUpdated = true;
            std::cout << "zupt updated" << std::endl;
        }
        else
        {
            propagate_state_and_covariance(state, feature_observes.first);
            state->stochastic_clone(state->_imu_state->pose());
            _visual_manager->UpdateFeature(feature_observes);  // visual update
            if (_visual_manager->VisualUpdate())
            {
                visual_updated = true;
                // std::cout << "visual udpated" << std::endl;
            }

            //     image_bak.insert(image_data);
            //     for (auto it = image_bak.begin(); it != image_bak.end();)
            //     {
            //         if (it->first < state->_clone_pose.begin()->first)
            //         {
            //             it = image_bak.erase(it);
            //             it ++;
            //             continue;
            //         }
            //         break;
            //     }

            //     std::vector<cv::Mat> keyframe_images;
            //     std::vector<double> keyframe_ts;
            //     for (auto it = state->_clone_pose.begin(); it != state->_clone_pose.end(); it++)
            //     {
            //         double ts_sec = it->first;
            //         if (image_bak.find(it->first) != image_bak.end())
            //         {
            //             keyframe_images.push_back(image_bak[it->first].first);
            //             keyframe_ts.push_back(ts_sec);
            //         }
            //     }
            //     Utils::ShowGridImages(keyframe_images);

            //     // double frontend_tracking_duration = (vio_rT1 - vio_rT).total_microseconds() * 1e-6;
            //     // double update_feature_duration = (vio_rT2 - vio_rT1).total_microseconds() * 1e-6;
            //     // double propagate_duration = (vio_rT3 - vio_rT2).total_microseconds() * 1e-6;
            //     // double visual_update_duration = (vio_rT4 - vio_rT3).total_microseconds() * 1e-6;
            //     // LOG(INFO) << cv::format("frontend tracking duration: %f", frontend_tracking_duration);
            //     // LOG(INFO) << cv::format("update obs feature duration: %f", update_feature_duration);
            //     // LOG(INFO) << cv::format("propagate state duration: %f", propagate_duration);
            //     // LOG(INFO) << cv::format("visual update duration: %f", visual_update_duration);
        }

        // Assign log values
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

        log_value.visual_updated = visual_updated;
        log_value.ZuptUpdated = ZuptUpdated;
        log_value.keyframe = int(_visual_manager->get_keyframe());

        vio_logger->save_to_file(log_value);
    }
}

void VioManager::groundtruth_callback(const geometry_msgs::PointStamped::ConstPtr& msg)
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

void VioManager::imu_callback(const sensor_msgs::Imu::ConstPtr& msg)
{
    ImuData data;
    data.ts_sec = msg->header.stamp.toSec() - _initial_timestamp;
    data.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    data.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    initializer->FeedImuMeasurement(data);
    _imu_manager->FeedImuMeasurement(data);
}

void VioManager::camera_callback(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1)
{
    double ts_sec = msg0->header.stamp.toSec() - _initial_timestamp;
    cv::Mat image_l, image_l_rectify;
    cv::Mat image_r, image_r_rectify;
    Utils::transfer_image(msg0, image_l);
    Utils::transfer_image(msg1, image_r);
    _camera_model_0->RectifyStereoImages(image_l, image_r, image_l_rectify, image_r_rectify);
    _visual_manager->feed_image({ts_sec, {image_l_rectify, image_r_rectify}});
}

bool VioManager::propagate_state_and_covariance(std::shared_ptr<State> state, double ts)
{
    using namespace Sophus;

    if (ts <= state->_imu_state->ts())
    {
        LOG(WARNING) << cv::format("curent state timestamp: %f, must be later than imu_state ts: %f", ts, state->_imu_state->ts());
        return false;
    }
    std::vector<ImuData> imu_data = _imu_manager->AccessIntervalImuMeasurements(state->_imu_state->ts(), ts);
    if (imu_data.empty() || imu_data.back().ts_sec < ts)
    {
        LOG(WARNING) << cv::format("wait for imu data, current state timestamp: %f but latest imu ts: %f", state->ts_sec(),
                                   _imu_manager->_imu_latest_timestamp - 0.1);
        return false;
    }

    // Eigen::Vector3d new_p_IinG = state->_imu_state->pose()->p();
    // Eigen::Matrix3d new_R_ItoG = state->_imu_state->pose()->quat().toRotationMatrix();
    // Eigen::Vector3d new_v_IinG = state->_imu_state->v()->vec();
    // Eigen::Vector3d ba = state->_imu_state->ba()->vec();
    // Eigen::Vector3d bg = state->_imu_state->bg()->vec();

    Eigen::Vector3d P = state->_imu_state->pose()->p();
    Eigen::Matrix3d R = state->_imu_state->pose()->quat().toRotationMatrix();
    Eigen::Vector3d V = state->_imu_state->v()->vec();
    Eigen::Vector3d ba = state->_imu_state->ba()->vec();
    Eigen::Vector3d bg = state->_imu_state->bg()->vec();

    Eigen::Vector3d P_next = P;
    Eigen::Matrix3d R_next = R;
    Eigen::Vector3d V_next = V;
    Eigen::Vector3d ba_next = ba;
    Eigen::Vector3d bg_next = bg;

    int dim = state->_imu_state->size();
    int q_id = state->_imu_state->q()->id();
    int p_id = state->_imu_state->p()->id();
    int v_id = state->_imu_state->v()->id();
    int bg_id = state->_imu_state->bg()->id();
    int ba_id = state->_imu_state->ba()->id();

    Eigen::MatrixXd Phi_sum = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::MatrixXd Qd_new = state->_imu_state->covariance().block<15, 15>(state->_imu_state->id(), state->_imu_state->id());

    uint32_t na_id = 0;
    uint32_t ng_id = 3;
    uint32_t nbg_id = 6;
    uint32_t nba_id = 9;
    double sigma_a2 = std::pow(_imu_manager->_sigma_na, 2);
    double sigma_w2 = std::pow(_imu_manager->_sigma_nw, 2);
    double sigma_bg2 = std::pow(_imu_manager->_sigma_bg, 2);
    double sigma_ba2 = std::pow(_imu_manager->_sigma_ba, 2);
    Eigen::MatrixXd Cov_m = Eigen::MatrixXd::Identity(12, 12);
    Cov_m.block(na_id, na_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_a2;
    Cov_m.block(ng_id, ng_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_w2;
    Cov_m.block(nbg_id, nbg_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_bg2;
    Cov_m.block(nba_id, nba_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_ba2;

    // std::cout << "Cov before predict: \n" << Qd_new << std::endl;

    for (int i = 0; i < imu_data.size() - 1; i++)
    {
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(15, 15);
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(15, 12);
        Eigen::MatrixXd Q = Qd_new;

        double dt = imu_data.at(i + 1).ts_sec - imu_data.at(i).ts_sec;
        if (dt > 0 && dt < kMaxImuToleranceDelayTime)
        {
            P = P_next;
            V = V_next;
            R = R_next;
            ba = ba_next;
            bg = bg_next;

            Eigen::Vector3d am_mid = 0.5 * (imu_data.at(i).am + imu_data.at(i + 1).am) - ba;
            Eigen::Vector3d wm_mid = 0.5 * (imu_data.at(i).wm + imu_data.at(i + 1).wm) - bg;

            // /*for R*/
            // F.block<3, 3>(q_id, q_id) = SO3d::exp(-wm_mid * dt).matrix();
            // F.block<3, 3>(q_id, bg_id) = -Eigen::Matrix3d::Identity() * dt;

            // /*for p*/
            // F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
            // F.block<3, 3>(p_id, q_id) = -0.5 * R * MathUtils::skew(am_mid * dt * dt);
            // F.block<3, 3>(p_id, ba_id) = -0.5 * R * dt * dt;

            // /*for v*/
            // F.block<3, 3>(v_id, q_id) = -R * MathUtils::skew(am_mid) * dt;
            // F.block<3, 3>(v_id, ba_id) = -R * dt;

            // /*for bg*/
            // F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();

            // /*for ba*/
            // F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();

            // for R
            F.block<3, 3>(q_id, q_id) = SO3d::exp(-wm_mid * dt).matrix();
            F.block<3, 3>(q_id, bg_id) = -Eigen::Matrix3d::Identity() * dt;

            // for p
            F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
            F.block<3, 3>(p_id, q_id) = -0.5 * R * MathUtils::skew(am_mid * dt * dt);
            F.block<3, 3>(p_id, ba_id) = -0.5 * R * dt * dt;

            // for veloity
            F.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(v_id, q_id) = -R * MathUtils::skew(am_mid * dt);
            F.block<3, 3>(v_id, ba_id) = -R * dt;

            // for bg
            F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();

            // for ba
            F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();

            // for sigma noise
            G.block<3, 3>(q_id, ng_id) = -Eigen::Matrix3d::Identity() * dt;
            G.block<3, 3>(p_id, na_id) = -0.5 * R * dt * dt;
            G.block<3, 3>(v_id, na_id) = -R * dt;
            G.block<3, 3>(bg_id, nbg_id) = Eigen::Matrix3d::Identity();
            G.block<3, 3>(ba_id, nba_id) = Eigen::Matrix3d::Identity();

            // Q.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_na, 2) * dt;
            // Q.block<3, 3>(q_id, q_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_nw, 2) * dt;
            // Q.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_bg, 2);
            // Q.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_ba, 2);

            // state propagation
            P_next = P + V * dt - 0.5 * state->_imu_state->gravity_inG * dt * dt + 0.5 * (R * am_mid * dt * dt);
            V_next = V - state->_imu_state->gravity_inG * dt + R * am_mid * dt;
            R_next = R * SO3d::exp(wm_mid * dt).matrix();
            ba_next = ba;
            bg_next = bg;

            // Phi_sum = F * Phi_sum;
            Qd_new = G * Cov_m * G.transpose() + F * Q * F.transpose();
            // Qd_new = Q + F * Qd_new.eval() * F.transpose();
            Qd_new = 0.5 * (Qd_new.eval() + Qd_new.eval().transpose());

            // std::cout << "F: \n" << F << std::endl;
            // std::cout << "G: \n" << G << std::endl;
            // std::cout << "Q: \n" << Q << std::endl;
        }
        else
        {
            LOG(WARNING) << utils::Format("Imu delayed for {0}s", dt);
            exit(0);
        }
    }

    state->_imu_state->q()->set_value(Eigen::Quaterniond(R_next).normalized().coeffs());
    state->_imu_state->p()->set_value(P_next);
    state->_imu_state->v()->set_value(V_next);

    state->_imu_state->set_ts(ts);
    state->_imu_state->set_covariance(Qd_new);
    state->_covariance.block(state->_imu_state->id(), state->_imu_state->id(), state->_imu_state->size(), state->_imu_state->size()) = Qd_new;

    // std::cout << "Cov after predict: \n" << Qd_new << std::endl;

    return true;
}
