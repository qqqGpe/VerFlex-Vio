#include "vioManager.h"
#include "utils/format.h"
#include "mathematical_tools.h"
#include <glog/logging.h>

namespace {
    constexpr int kImuOutputHz = 200;
    constexpr double kMaxImuToleranceDelayTime = 1.0 / kImuOutputHz * 10;
}

// void VioManager::feature_callback(const sensor_msgs::PointCloud::ConstPtr &msg)
// {
// }

void VioManager::imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    ImuData data;
    data.ts_sec = msg->header.stamp.toSec();
    data.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    data.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    if(!initializer->is_initialized) {
        initializer->feed_imu_measurement(data);
    } else {
        _imu_manager->feed_imu_measurement(data);
    }

    std::thread thread(
        [&]{
            if(!initializer->is_initialized) {
                bool status = initializer->static_initialize(state->_imu_state);
            } else {
                LOG(INFO) << "vio has already initialized!";
            }
        }
    );
    // thread.join();
    thread.detach();

}

void VioManager::propagate_state_and_covariance(std::shared_ptr<State> state, double ts)
{
    if(ts <= state->_imu_state->ts()) {
        LOG(WARNING) << utils::Format("curent timestamp: {0}, must be later than imu_state ts: {1}", ts, state->_imu_state->ts());
        // LOG(WARNING) << "current timestamp: " << ts << ", must be later than imu_state ts: " << state->_imu_state->ts();
        return;
    }
    std::vector<ImuData> imu_data = _imu_manager->access_interval_imu_measurment(state->_imu_state->ts(), ts);

    Eigen::Vector3d new_p_IinG = state->_imu_state->pose()->p();
    Eigen::Matrix3d new_R_ItoG = state->_imu_state->pose()->quat().toRotationMatrix();
    Eigen::Vector3d new_v_IinG = state->_imu_state->v()->vec();
    Eigen::Vector3d ba = state->_imu_state->ba()->vec();
    Eigen::Vector3d bg = state->_imu_state->bg()->vec();

    int dim = state->_imu_state->size();
    int q_id = state->_imu_state->q()->id();
    int p_id = state->_imu_state->p()->id();
    int v_id = state->_imu_state->v()->id();
    int bg_id = state->_imu_state->bg()->id();
    int ba_id = state->_imu_state->ba()->id();

    Eigen::MatrixXd Phi_sum = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::MatrixXd Qd_old = state->_imu_state->covariance();
    Eigen::MatrixXd Qd_new = Qd_old.block<15, 15>(state->_imu_state->id(), state->_imu_state->id());

    for (int i = 0; i < imu_data.size() - 1; i++) {

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(dim, dim);
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(dim, dim);

        double dt = imu_data.at(i + 1).ts_sec - imu_data.at(i).ts_sec;
        if(dt > 0 && dt < kMaxImuToleranceDelayTime) {
            Eigen::Vector3d am_mid = .5 * (imu_data.at(i).am + imu_data.at(i + 1).am) - ba;
            Eigen::Vector3d wm_mid = .5 * (imu_data.at(i).wm + imu_data.at(i + 1).wm) - bg;

            new_p_IinG = new_p_IinG + new_v_IinG * dt - 0.5 * state->_imu_state->gravity_inG * dt * dt + 0.5 * (new_R_ItoG * am_mid * dt * dt);
            new_v_IinG = new_v_IinG - state->_imu_state->gravity_inG * dt + new_R_ItoG * am_mid * dt;
            new_R_ItoG = new_R_ItoG * mathematical::Rodrigues(wm_mid.normalized(), wm_mid.norm() * dt);

            // for R
            F.block<3, 3>(q_id, q_id) = Eigen::Matrix3d::Identity() - mathematical::skew(wm_mid) * dt;
            F.block<3, 3>(q_id, bg_id) = -Eigen::Matrix3d::Identity() * dt;
            // for p
            F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
            // compute v
            F.block<3, 3>(v_id, q_id) = -new_R_ItoG * mathematical::skew(am_mid) * dt;
            F.block<3, 3>(v_id, ba_id) = -new_R_ItoG * dt;
            F.block<3, 3>(v_id, bg_id) = Eigen::Matrix3d::Identity() * dt;
            // for bg
            F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();
            // for ba
            F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();

            Q.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_na, 2) * dt;
            Q.block<3, 3>(q_id, q_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_nw, 2) * dt;
            Q.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_bw, 2);
            Q.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity() * std::pow(_imu_manager->_sigma_ba, 2);

            state->_imu_state->q()->set_value(Eigen::Quaterniond(new_R_ItoG).coeffs());
            state->_imu_state->p()->set_value(new_p_IinG);
            state->_imu_state->v()->set_value(new_v_IinG);

            Phi_sum = F * Phi_sum;
            Qd_new = Q + F * Qd_new * F.transpose();
            Qd_new = 0.5 * (Qd_new + Qd_new.transpose());
        } else {
            LOG(WARNING) << utils::Format("Imu delayed for {0}s", dt);
        }
    }

    state->_imu_state->set_ts(ts);
    state->_imu_state->set_covariance(Qd_new);

    state->stochastic_clone(state->_imu_state->pose());

    return;
}
