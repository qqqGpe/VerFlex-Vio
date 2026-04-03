#ifndef __SQRT_ESKF_SOLVER__
#define __SQRT_ESKF_SOLVER__
#include "solver.h"
#include "utils.h"
#include "ImuManager.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fmt/format.h>
#include <opencv2/core/core.hpp>
#include <sophus/so3.hpp>

using namespace Sophus;
class SqrtEskfSolver : public MsckfSolverBase
{
   public:
    SqrtEskfSolver() = default;
    SqrtEskfSolver(const bool use_fej) : use_fej_(use_fej) {}
    virtual ~SqrtEskfSolver() = default;

    /**
     * @brief Clone the given pose into the state with stochastic cloning
     * @param state The state to be cloned
     * @param imu_data The imu measurements between the last state and the current state
     * @return void
     */
    virtual void StochasticClone(std::shared_ptr<State> state, std::vector<ImuData>* imu_data) override
    {
        Eigen::MatrixXd SqrtPt_old = state->Sqrt_Pt();
        std::shared_ptr<Pose> pose_to_clone = state->_imu_state->pose();
        const int clone_pose_size = pose_to_clone->size();
        const int old_rows = SqrtPt_old.rows();
        const int old_cols = SqrtPt_old.cols();
        const int new_cols = old_cols + clone_pose_size;

        Eigen::MatrixXd SqrtPt_new = Eigen::MatrixXd::Zero(old_rows, new_cols);
        SqrtPt_new.topLeftCorner(old_rows, old_cols) = SqrtPt_old;
        SqrtPt_new.rightCols(clone_pose_size) = SqrtPt_new.leftCols(clone_pose_size);

        std::shared_ptr<Pose> clone_pose = std::dynamic_pointer_cast<Pose>(pose_to_clone->clone());
        clone_pose->set_local_id(old_cols);
        state->_clone_pose.insert(std::make_pair(clone_pose->ts(), clone_pose));
        state->_variables.push_back(clone_pose);
        state->_dim += clone_pose->size();

        if (state->enableEstimateTdVisual())
        {
            Eigen::Vector3d last_w = imu_data->back().wm;
            Eigen::MatrixXd J_td = Eigen::MatrixXd::Zero(clone_pose_size, 1);
            J_td << last_w, state->_imu_state->v()->vec();
            SqrtPt_new.rightCols(clone_pose_size) += SqrtPt_new.block(0, state->td_visual().id(), SqrtPt_new.rows(), state->td_visual().size()) * J_td.transpose();
        }

        state->SetSqrtPt(SqrtPt_new);

        Eigen::MatrixXd Cov_new = SqrtPt_new.transpose() * SqrtPt_new;
        state->SetCovariance(0.5 * (Cov_new + Cov_new.transpose()));
    }

    /**
     * @brief Marginalize the given state variable from the state
     * @param marge_type The type of state to be marginalized (ClonePose, SlamFeature, etc.)
     * @param state The state containing the variable to be marginalized
     * @param state_to_marginalize The state variable to be marginalized
     * @note SLAM Feature marginalization involves removing both the state variable and its mapping in the state.
     * @return void
     */
    virtual void MarginalizeState(MarginalizeType marge_type, std::shared_ptr<State> state, std::shared_ptr<Type> state_to_marginalize) override
    {
        if (state_to_marginalize == nullptr)
        {
            LOG(ERROR) << "marginalization failed, state_to_marginalize is nullptr";
            return;
        }

        const uint32_t id_to_marginalize = state_to_marginalize->id();
        const uint32_t size_to_marginalize = state_to_marginalize->size();

        // Erase the state to marginalize from the state vector
        auto iter_marginalize = std::find(state->_variables.begin(), state->_variables.end(), state_to_marginalize);
        if (iter_marginalize == state->_variables.end())
        {
            LOG(ERROR) << "marginalization failed, no such variable in states";
            return;
        }
        state->_variables.erase(iter_marginalize);
        state->_dim = state->_dim - size_to_marginalize;
        for (auto it = state->_variables.begin(); it != state->_variables.end(); it++)
        {
            if ((*it)->id() > id_to_marginalize)
            {
                (*it)->set_local_id((*it)->id() - size_to_marginalize);
            }
        }

        // Marginalize clone pose
        if (marge_type == MarginalizeType::ClonePose)
        {
            if (state->_clone_pose.find(state_to_marginalize->ts()) != state->_clone_pose.end())
            {
                state->_clone_pose.erase(state_to_marginalize->ts());
            }
        }

        // Marginalize slam feature (works identically to ESKF, removing the state from the feature map)
        if (marge_type == MarginalizeType::SlamFeature)
        {
            for (auto &[id, feature] : state->slam_features())
            {
                if (feature._state_ptr == state_to_marginalize)
                {
                    state->mutable_slam_features().erase(id);
                    break;
                }
            }
        }

        // Update marginalized square-root covariance matrix. P = S^T * S, where S is SqrtPt.
        Eigen::MatrixXd SqrtPt_old = state->Sqrt_Pt();
        const uint32_t old_rows = SqrtPt_old.rows();
        const uint32_t old_cols = SqrtPt_old.cols();
        const uint32_t new_cols = old_cols - size_to_marginalize;
        Eigen::MatrixXd SqrtPt_small = Eigen::MatrixXd::Zero(old_rows, new_cols);
        if (id_to_marginalize + size_to_marginalize < old_cols)
        {
            SqrtPt_small.topLeftCorner(old_rows, id_to_marginalize) = SqrtPt_old.topLeftCorner(old_rows, id_to_marginalize);

            SqrtPt_small.topRightCorner(old_rows, old_cols - id_to_marginalize - size_to_marginalize) =
                SqrtPt_old.topRightCorner(old_rows, old_cols - id_to_marginalize - size_to_marginalize);
        }
        else
        {
            SqrtPt_small.topLeftCorner(old_rows, id_to_marginalize) = SqrtPt_old.topLeftCorner(old_rows, id_to_marginalize);
        }

        state->SetSqrtPt(SqrtPt_small);
    }

    /**
     * @brief Update the state with given measurements
     * @param state The state to be updated
     * @param Hx The measurement Jacobian matrix
     * @param res The measurement residual
     * @param Hx_order The order of the state variables in the Hx matrix
     * @param map_hx The mapping from state variable to its column index in the Hx matrix
     * @param R The measurement noise covariance
     * @return void
     */
    virtual void update(std::shared_ptr<State>& state,
                        const Eigen::Ref<Eigen::MatrixXd>& Hx,
                        const Eigen::Ref<Eigen::MatrixXd>& res,
                        const std::vector<std::shared_ptr<Type>>& Hx_order,
                        const std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx,
                        const Eigen::MatrixXd& R) override
    {
        assert(R.rows() == res.rows());
        assert(Hx.rows() == res.rows());
        Eigen::MatrixXd SqrtPt_predict = state->Sqrt_Pt();
        Eigen::MatrixXd M_all = Eigen::MatrixXd::Zero(R.rows() + SqrtPt_predict.rows(), R.cols() + SqrtPt_predict.cols());

        int32_t current_it = 0;
        std::vector<int32_t> H_id;
        for (const auto& meas_var : Hx_order)
        {
            H_id.push_back(current_it);
            current_it += meas_var->size();
        }

        Eigen::MatrixXd Hx_all = Eigen::MatrixXd::Zero(res.rows(), SqrtPt_predict.cols());
        for (int i = 0; i < Hx_order.size(); i++)
        {
            std::shared_ptr<Type> var = Hx_order[i];
            Hx_all.block(0, var->id(), res.rows(), var->size()) = Hx.block(0, H_id[i], res.rows(), var->size());
        }

        // Construct M matrix
        M_all.topLeftCorner(R.rows(), R.cols()) = R.llt().matrixL().transpose();
        M_all.bottomLeftCorner(SqrtPt_predict.rows(), Hx_all.rows()) = SqrtPt_predict * Hx_all.transpose();
        M_all.bottomRightCorner(SqrtPt_predict.rows(), SqrtPt_predict.cols()) = SqrtPt_predict;

        // QR decomposition
        Eigen::MatrixXd rhks = utils::math::GivensRotation(M_all, M_all.cols());
        Eigen::MatrixXd rh = rhks.topLeftCorner(R.rows(), R.cols());
        Eigen::MatrixXd K_hat = rhks.topRightCorner(R.rows(), SqrtPt_predict.cols()).transpose();
        Eigen::MatrixXd SqrtPt_update = rhks.bottomRightCorner(SqrtPt_predict.rows(), SqrtPt_predict.cols());

        // Calculate K matrix
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(rh);
        Eigen::MatrixXd I_mat = Eigen::MatrixXd::Identity(rh.rows(), rh.cols());
        Eigen::MatrixXd K = K_hat * qr.solve(I_mat);

        // Check covariance matrix definition
        Eigen::MatrixXd Cov_full = SqrtPt_update.transpose() * SqrtPt_update;
        Eigen::VectorXd diags = Cov_full.diagonal();
        for (int i = 0; i < diags.rows(); i++)
        {
            if (diags(i) < 0.0)
            {
                LOG(ERROR) << fmt::format(RED "Diagonal is negative when update, diags" RESET);
                LOG(ERROR) << "diags: " << diags.transpose();
                std::exit(EXIT_FAILURE);
            }
        }

        // Update state
        Eigen::VectorXd dx = K * res;
        for (size_t i = 0; i < state->_variables.size(); i++)
        {
            state->_variables[i]->update(dx.block(state->_variables[i]->id(), 0, state->_variables[i]->size(), 1));
        }

        // Update Covariance
        state->SetSqrtPt(SqrtPt_update);
        state->SetCovariance(0.5 * (Cov_full + Cov_full.transpose()));
    }

    /**
     * @brief Propagate the state with given imu measurements
     * @param state The state to be propagated
     * @param dt The time difference between the last state and the current state
     * @param meas_k0 The imu measurement at time k0
     * @param meas_k1 The imu measurement at time k1
     * @param new_q The propagated orientation
     * @param new_p The propagated position
     * @param new_v The propagated velocity
     * @return void
     */
    void PropagateState(const std::shared_ptr<State> state,
                        const double dt,
                        const ImuData meas_k0,
                        const ImuData meas_k1,
                        Eigen::Quaterniond& new_q,
                        Eigen::Vector3d& new_p,
                        Eigen::Vector3d& new_v)
    {
        Eigen::Quaterniond q_ItoG = state->_imu_state->pose()->quat();
        Eigen::Vector3d p_IinG = state->_imu_state->pose()->p();
        Eigen::Vector3d v_IinG = state->_imu_state->v()->vec();

        Eigen::Vector3d wm_mid = 0.5 * (meas_k0.wm + meas_k1.wm) - state->_imu_state->bg()->vec();
        Eigen::Vector3d am_mid = 0.5 * (meas_k0.am + meas_k1.am) - state->_imu_state->ba()->vec();
        Eigen::Matrix3d dR = Eigen::AngleAxisd(wm_mid.norm() * dt, wm_mid.normalized()).toRotationMatrix();

        new_q = (q_ItoG * Eigen::Quaterniond(dR)).normalized();

        new_p = p_IinG + v_IinG * dt - 0.5 * state->_imu_state->gravity_inG * dt * dt + 0.5 * (q_ItoG.toRotationMatrix() * am_mid * dt * dt);

        new_v = v_IinG + (q_ItoG.toRotationMatrix() * am_mid - state->_imu_state->gravity_inG) * dt;
    }


    /**
     * @brief Compute the state transition matrix F and the process noise matrix G
     * @param state The state to be propagated
     * @param dt The time difference between the last state and the current state
     * @param meas_k0 The imu measurement at time k0
     * @param meas_k1 The imu measurement at time k1
     * @param new_q The propagated orientation
     * @param F The state transition matrix to be computed
     * @param G The process noise matrix to be computed
     * @return void
     */
    void Compute_F_and_G(const std::shared_ptr<State> state,
                         const double dt,
                         const ImuData meas_k0,
                         const ImuData meas_k1,
                         const Eigen::Quaterniond new_q,
                         Eigen::MatrixXd& F,
                         Eigen::MatrixXd& G)
    {
        const uint32_t imu_state_dim = state->_imu_state->size();
        const uint32_t imu_id = state->_imu_state->id();
        const uint32_t th_id = state->_imu_state->q()->id();
        const uint32_t p_id = state->_imu_state->p()->id();
        const uint32_t v_id = state->_imu_state->v()->id();
        const uint32_t bg_id = state->_imu_state->bg()->id();
        const uint32_t ba_id = state->_imu_state->ba()->id();

        F = Eigen::MatrixXd::Identity(15, 15);
        G = Eigen::MatrixXd::Zero(15, 12);

        Eigen::Vector3d p_IinG = state->_imu_state->pose()->p();
        Eigen::Matrix3d R_ItoG = state->_imu_state->pose()->quat().toRotationMatrix();
        Eigen::Vector3d v_IinG = state->_imu_state->v()->vec();
        Eigen::Vector3d ba = state->_imu_state->ba()->vec();
        Eigen::Vector3d bg = state->_imu_state->bg()->vec();
        Eigen::Vector3d wm_mid = 0.5 * (meas_k0.wm + meas_k1.wm) - state->_imu_state->bg()->vec();
        Eigen::Vector3d am_mid = 0.5 * (meas_k0.am + meas_k1.am) - state->_imu_state->ba()->vec();

        if (use_fej_)
        {
            p_IinG = state->_imu_state->pose()->p_fej();
            R_ItoG = state->_imu_state->pose()->quat_fej().toRotationMatrix();
        }

        Eigen::Matrix3d dR = new_q.toRotationMatrix().transpose() * R_ItoG;

        // For R
        F.block<3, 3>(th_id, th_id) = dR;
        F.block<3, 3>(th_id, bg_id) = -Eigen::Matrix3d::Identity() * dt;

        // For p
        F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(p_id, th_id) = -0.5 * R_ItoG * utils::math::skew(am_mid * dt * dt);
        F.block<3, 3>(p_id, ba_id) = -0.5 * R_ItoG * dt * dt;

        // For v
        F.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(v_id, th_id) = -R_ItoG * utils::math::skew(am_mid * dt);
        F.block<3, 3>(v_id, ba_id) = -R_ItoG * dt;

        // For bg
        F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();

        // For ba
        F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();

        // For measurement noise
        G.block<3, 3>(th_id, kNoiseGyroId) = -Eigen::Matrix3d::Identity() * dt;
        G.block<3, 3>(p_id, kNoiseAccId) = -0.5 * R_ItoG * dt * dt;
        G.block<3, 3>(v_id, kNoiseAccId) = -R_ItoG * dt;
        G.block<3, 3>(bg_id, kNoiseGyroBiasId) = Eigen::Matrix3d::Identity();
        G.block<3, 3>(ba_id, kNoiseAccBiasId) = Eigen::Matrix3d::Identity();
    }

    /**
     * @brief Propagate the state and covariance with given imu measurements
     * @param imu_data The imu measurements between the last state and the current state
     * @param visual_ts The timestamp of the current state
     * @param state The state to be propagated
     * @return true if the propagation is successful, false otherwise
     */
    virtual bool PropagateStateAndCovariance(const std::vector<ImuData> imu_data, const double visual_ts, std::shared_ptr<State> state) override
    {
        if (visual_ts <= state->_imu_state->ts())
        {
            LOG(WARNING) << fmt::format("Propagation failed, curent state timestamp: {}, must be later than imu_state timestamp: {}", visual_ts,
                                       state->_imu_state->ts());
            return false;
        }

        if (imu_data.empty() || imu_data.back().ts_sec < state->ts_sec())
        {
            LOG(WARNING) << fmt::format("Propagation failed, waiting for imu data, current state timestamp: {} but latest imu timestamp: {}",
                                       state->ts_sec(), imu_data.back().ts_sec);
            return false;
        }

        const uint32_t imu_state_dim = state->_imu_state->size();

        Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(12, 12);
        Q.block(kNoiseAccId, kNoiseAccId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_na, 2);
        Q.block(kNoiseGyroId, kNoiseGyroId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_nw, 2);
        Q.block(kNoiseGyroBiasId, kNoiseGyroBiasId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_bg, 2);
        Q.block(kNoiseAccBiasId, kNoiseAccBiasId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_ba, 2);

        Eigen::MatrixXd Phi_sum = Eigen::MatrixXd::Identity(imu_state_dim, imu_state_dim);
        Eigen::MatrixXd Q_sum = Eigen::MatrixXd::Zero(imu_state_dim, imu_state_dim);

        for (int i = 0; i < imu_data.size() - 1; i++)
        {
            Eigen::MatrixXd F = Eigen::MatrixXd::Identity(15, 15);
            Eigen::MatrixXd G = Eigen::MatrixXd::Zero(15, 12);
            double dt = imu_data.at(i + 1).ts_sec - imu_data.at(i).ts_sec;

            if (dt > 0 && dt < kMaxImuToleranceDelayTime)
            {
                Eigen::Quaterniond new_q;
                Eigen::Vector3d new_p;
                Eigen::Vector3d new_v;

                // 1. Propagate state
                PropagateState(state, dt, imu_data.at(i), imu_data.at(i + 1), new_q, new_p, new_v);

                // 2. Compute F and G
                Compute_F_and_G(state, dt, imu_data.at(i), imu_data.at(i + 1), new_q, F, G);

                // 3. Propagate covariance
                Phi_sum = F * Phi_sum;
                Q_sum = F * Q_sum * F.transpose() + G * Q * G.transpose();

                // 4. Update state
                state->set_ts_sec(imu_data.at(i + 1).ts_sec);  // Set current state timestamp to visual timestamp
                state->_imu_state->q()->set_value(new_q.coeffs());
                state->_imu_state->p()->set_value(new_p);
                state->_imu_state->v()->set_value(new_v);

                if (use_fej_)
                {
                    state->_imu_state->pose()->set_pose_fej(new_q.toRotationMatrix(), new_p);
                }
            }
            else
            {
                LOG(WARNING) << fmt::format("Imu delayed for {}s", dt);
                exit(0);
            }
        }

        // Update sqrt root covariance
        Eigen::MatrixXd SqrtPt_imu = state->_imu_state->Sqrt_Pt();
        Eigen::MatrixXd SqrtPt_tmp = Eigen::MatrixXd::Zero(state->Sqrt_Pt().rows() + Q_sum.rows(), state->Sqrt_Pt().cols());
        SqrtPt_tmp.block(0, 0, state->Sqrt_Pt().rows(), state->Sqrt_Pt().cols()) = state->Sqrt_Pt();
        SqrtPt_tmp.block(0, 0, imu_state_dim, imu_state_dim) = SqrtPt_imu * Phi_sum.transpose();
        SqrtPt_tmp.block(state->Sqrt_Pt().rows(), 0, Q_sum.rows(), Q_sum.cols()) = Q_sum.llt().matrixL().transpose();
        Eigen::MatrixXd SqrtPt_triangulated = utils::math::GivensRotation(SqrtPt_tmp, SqrtPt_tmp.cols());
        Eigen::MatrixXd SqrtPt_propagated = SqrtPt_triangulated.block(0, 0, SqrtPt_tmp.cols(), SqrtPt_tmp.cols());

        assert(SqrtPt_tmp.cols() == state->Sqrt_Pt().cols());
        state->SetSqrtPt(SqrtPt_propagated);

        Eigen::MatrixXd Cov_full = SqrtPt_propagated.transpose() * SqrtPt_propagated;
        state->SetCovariance(0.5 * (Cov_full + Cov_full.transpose()));

        return true;
    }

   private:
    bool use_fej_ = true;
    constexpr static  uint32_t kNoiseAccId = 0;
    constexpr static  uint32_t kNoiseGyroId = 3;
    constexpr static uint32_t kNoiseGyroBiasId = 6;
    constexpr static uint32_t kNoiseAccBiasId = 9;
};

#endif