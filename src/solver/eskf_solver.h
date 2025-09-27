#ifndef __ESKF_SOLVER__
#define __ESKF_SOLVER__

#include "ImuManager.h"
#include "solver.h"
#include "utils.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <sophus/so3.hpp>

using namespace Sophus;

class eskfSolver : public MsckfSolverBase
{
   public:
    eskfSolver() = default;
    eskfSolver(const bool use_fej) : use_fej_(use_fej) {}
    virtual ~eskfSolver() {}

    /**
     * @brief Stochastic clone the given pose into the state
     * @param state The state to be cloned
     * @param imu_data The imu measurements between the last state and the current state
     * @return void
     */
    virtual void StochasticClone(std::shared_ptr<State> state, std::vector<ImuData>* imu_data) override
    {
        Eigen::MatrixXd Cov_old = state->Covariance();
        std::shared_ptr<Pose> pose_to_clone = state->_imu_state->pose();
        const int clone_pose_size = pose_to_clone->size();
        const int old_rows = Cov_old.rows();
        const int old_cols = Cov_old.cols();
        const int new_rows = old_rows + pose_to_clone->size();
        const int new_cols = old_cols + pose_to_clone->size();

        Eigen::MatrixXd Cov_new = Eigen::MatrixXd::Zero(new_rows, new_cols);
        Cov_new.topLeftCorner(old_rows, old_cols) = Cov_old;
        Cov_new.bottomRightCorner(clone_pose_size, clone_pose_size) = Cov_old.block(pose_to_clone->id(), pose_to_clone->id(), clone_pose_size, clone_pose_size);
        Cov_new.topRightCorner(old_rows, clone_pose_size) = Cov_old.block(0, pose_to_clone->id(), old_rows, clone_pose_size);
        Cov_new.bottomLeftCorner(clone_pose_size, old_cols) = Cov_old.block(pose_to_clone->id(), 0, clone_pose_size, old_cols);

        std::shared_ptr<Type> clone_pose = pose_to_clone->clone();
        clone_pose->set_local_id(old_cols);
        state->_clone_pose.insert(std::make_pair(clone_pose->ts(), std::dynamic_pointer_cast<Pose>(clone_pose)));
        state->_variables.push_back(clone_pose);
        state->_dim += clone_pose->size();

        // Consider the time delay of visual measurement when agument the covariance
        if (state->enable_estimate_td_visual_)
        {
            Eigen::Vector3d last_w = imu_data->back().wm;
            Eigen::MatrixXd J_td = Eigen::MatrixXd::Zero(clone_pose_size, 1);
            J_td << last_w, state->_imu_state->v()->vec();
            Cov_new.rightCols(clone_pose_size) += Cov_new.block(0, state->td_visual().id(), new_rows, state->td_visual().size()) * J_td.transpose();
            Cov_new.bottomRows(clone_pose_size) += J_td * Cov_new.block(state->td_visual().id(), 0, state->td_visual().size(), new_cols);
        }

        state->SetCovariance(Cov_new);
    }

    /**
     * @brief Marginalize the given state variable from the state
     * @param state The state containing the variable to be marginalized
     * @param state_to_marginalize The state variable to be marginalized
     * @return void
     */
    virtual void MarginalizeState(std::shared_ptr<State> state, std::shared_ptr<Type> state_to_marginalize) override
    {
        if (state_to_marginalize == nullptr)
        {
            LOG(ERROR) << "State marginalization failed, state_to_marginalize is nullptr";
            return;
        }

        const uint32_t id_to_marginalize = state_to_marginalize->id();
        const uint32_t size_to_marginalize = state_to_marginalize->size();

        // Erase the state to marginalize from the state vector
        auto iter_marginalize = std::find(state->_variables.begin(), state->_variables.end(), state_to_marginalize);
        if (iter_marginalize == state->_variables.end())
        {
            LOG(ERROR) << "State marginalization failed, no such variable in states";
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

        if (state->_clone_pose.find(state_to_marginalize->ts()) != state->_clone_pose.end())
        {
            state->_clone_pose.erase(state_to_marginalize->ts());
        }

        // Update marginalized covariance matrix
        Eigen::MatrixXd Cov_old = state->Covariance();
        const uint32_t old_dim = Cov_old.rows();
        const uint32_t new_dim = old_dim - size_to_marginalize;
        Eigen::MatrixXd Cov_small = Eigen::MatrixXd::Zero(new_dim, new_dim);
        if (id_to_marginalize + size_to_marginalize < old_dim)
        {
            Cov_small.topLeftCorner(id_to_marginalize, id_to_marginalize) = Cov_old.topLeftCorner(id_to_marginalize, id_to_marginalize);

            Cov_small.bottomLeftCorner(old_dim - id_to_marginalize - size_to_marginalize, id_to_marginalize) =
                Cov_old.bottomLeftCorner(old_dim - id_to_marginalize - size_to_marginalize, id_to_marginalize);

            Cov_small.topRightCorner(id_to_marginalize, old_dim - id_to_marginalize - size_to_marginalize) =
                Cov_old.topRightCorner(id_to_marginalize, old_dim - id_to_marginalize - size_to_marginalize);

            Cov_small.bottomRightCorner(old_dim - id_to_marginalize - size_to_marginalize, old_dim - id_to_marginalize - size_to_marginalize) =
                Cov_old.bottomRightCorner(old_dim - id_to_marginalize - size_to_marginalize, old_dim - id_to_marginalize - size_to_marginalize);
        }
        else
        {
            Cov_small.topLeftCorner(id_to_marginalize, id_to_marginalize) = Cov_old.topLeftCorner(id_to_marginalize, id_to_marginalize);
        }
        state->SetCovariance(Cov_small);
    }

    /**
     * @brief Compute the marginal covariance of the given state variables
     * @param state The state containing the full covariance matrix
     * @param Hx_order The order of the state variables to compute the marginal covariance
     * @return The marginal covariance matrix
     */
    static Eigen::MatrixXd MarginalCovariance(const std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> Hx_order)
    {
        int32_t matrix_size = 0;
        for (const auto& x : Hx_order)
        {
            matrix_size += x->size();
        }

        Eigen::MatrixXd Cov_full = state->Covariance();
        Eigen::MatrixXd Cov_marginal = Eigen::MatrixXd::Zero(matrix_size, matrix_size);

        int32_t current_row = 0;
        for (int32_t i = 0; i < Hx_order.size(); i++)
        {
            std::shared_ptr<Type> var_i = Hx_order[i];
            int32_t current_col = 0;
            for (int32_t j = 0; j < Hx_order.size(); j++)
            {
                std::shared_ptr<Type> var_j = Hx_order[j];
                Cov_marginal.block(current_row, current_col, var_i->size(), var_j->size()) =
                    Cov_full.block(var_i->id(), var_j->id(), var_i->size(), var_j->size());
                current_col += var_j->size();
            }
            current_row += var_i->size();
        }
        return Cov_marginal;
    }

    /**
     * @brief Update the state with given measurements
     * @param state The state to be updated
     * @param Hx The measurement Jacobian matrix
     * @param res The measurement residual
     * @param Hx_order The order of the state variables in the Hx matrix
     * @param map_hx The mapping from state variable to its column index in the Hx matrix
     * @param R The measurement noise covariance
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
        Eigen::MatrixXd Cov_old = state->Covariance();
        Eigen::MatrixXd M_all = Eigen::MatrixXd::Zero(Cov_old.rows(), res.rows());

        int32_t current_it = 0;
        std::vector<int32_t> H_id;
        for (const auto& meas_var : Hx_order)
        {
            H_id.push_back(current_it);
            current_it += meas_var->size();
        }

        Eigen::MatrixXd Hx_all = Eigen::MatrixXd::Zero(res.rows(), Cov_old.rows());
        for (int i = 0; i < Hx_order.size(); i++)
        {
            std::shared_ptr<Type> var = Hx_order[i];
            Hx_all.block(0, var->id(), res.rows(), var->size()) = Hx.block(0, H_id[i], res.rows(), var->size());
        }
        M_all.noalias() = Cov_old * Hx_all.transpose();

        Eigen::MatrixXd Cov_involved = MarginalCovariance(state, Hx_order);

        // Residual covariance S = H * Cov * H' + R
        Eigen::MatrixXd S = Hx * Cov_involved * Hx.transpose() + R;
        Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
        S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
        Eigen::MatrixXd K = M_all * Sinv.selfadjointView<Eigen::Upper>();
        // Eigen::MatrixXd K = M_all * S.inverse();

        // Update Covariance
        Eigen::MatrixXd Cov_update = Cov_old - K * M_all.transpose();
        state->SetCovariance(0.5 * (Cov_update + Cov_update.transpose()));

        // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
        Eigen::VectorXd diags = state->Covariance().diagonal();
        for (int i = 0; i < diags.rows(); i++)
        {
            if (diags(i) < 0.0)
            {
                LOG(ERROR) << "\033[31m" << fmt::format("Diagonal is negative when update, diags") << "\033[0m";
                LOG(ERROR) << "diags: " << diags.transpose();
                std::exit(EXIT_FAILURE);
            }
        }

        Eigen::VectorXd dx = K * res;
        for (size_t i = 0; i < state->_variables.size(); i++)
        {
            state->_variables[i]->update(dx.block(state->_variables[i]->id(), 0, state->_variables[i]->size(), 1));
        }
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
            LOG(WARNING) << fmt::format("Propagation failed, curent state timestamp: %f, must be later than imu_state timestamp: %f", visual_ts,
                                        state->_imu_state->ts());
            return false;
        }

        if (imu_data.empty() || imu_data.back().ts_sec < visual_ts)
        {
            LOG(WARNING) << fmt::format("Propagation failed, waiting for imu data, current state timestamp: %f but latest imu timestamp: %f",
                                        state->ts_sec(), imu_data.back().ts_sec);
            return false;
        }

        Eigen::MatrixXd Phi_sum = Eigen::MatrixXd::Identity(state->_imu_state->size(), state->_imu_state->size());
        Eigen::MatrixXd Q_sum = state->_imu_state->covariance().block<15, 15>(state->_imu_state->id(), state->_imu_state->id());

        Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(12, 12);
        Q.block(kNoiseAccId, kNoiseAccId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_na, 2);
        Q.block(kNoiseGyroId, kNoiseGyroId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_nw, 2);
        Q.block(kNoiseGyroBiasId, kNoiseGyroBiasId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_bg, 2);
        Q.block(kNoiseAccBiasId, kNoiseAccBiasId, 3, 3) = Eigen::Matrix3d::Identity() * std::pow(ImuManager::_sigma_ba, 2);

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
                Q_sum = F * Q_sum.eval() * F.transpose() + G * Q * G.transpose();
                Q_sum = 0.5 * (Q_sum.eval() + Q_sum.eval().transpose());

                // 4. Update state
                state->set_ts_sec(imu_data.at(i + 1).ts_sec);
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
                LOG(WARNING) << "\033[31m" << fmt::format("Imu delayed for {}s", dt) << "\033[0m";
                exit(0);
            }
        }

        state->_imu_state->set_covariance(Q_sum);
        state->_covariance.block(state->_imu_state->id(), state->_imu_state->id(), state->_imu_state->size(), state->_imu_state->size()) = Q_sum;

        const Eigen::MatrixXd Cov = state->Covariance();
        const uint32_t imu_dim = state->_imu_state->size();
        if (Cov.rows() != imu_dim)
        {
            Eigen::MatrixXd Cov_ic = state->Covariance().block(0, imu_dim, imu_dim, Cov.rows() - imu_dim);
            state->_covariance.block(0, imu_dim, imu_dim, Cov.rows() - imu_dim) = Phi_sum * Cov_ic;
            state->_covariance.block(imu_dim, 0, Cov.rows() - imu_dim, imu_dim) = Cov_ic.transpose() * Phi_sum.transpose();
        }

        return true;
    }

   private:
    bool use_fej_ = true;
    static constexpr uint32_t kNoiseAccId = 0;
    static constexpr uint32_t kNoiseGyroId = 3;
    static constexpr uint32_t kNoiseGyroBiasId = 6;
    static constexpr uint32_t kNoiseAccBiasId = 9;
};

#endif