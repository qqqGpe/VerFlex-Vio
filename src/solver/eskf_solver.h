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
    ~eskfSolver() {}

    virtual void StochasticClone(std::shared_ptr<State> state) override
    {
        Eigen::MatrixXd Cov_old = state->Covariance();
        std::shared_ptr<Pose> pose_to_clone = state->_imu_state->pose();
        const int clone_size = pose_to_clone->size();
        const int old_rows = Cov_old.rows();
        const int old_cols = Cov_old.cols();
        const int new_rows = old_rows + pose_to_clone->size();
        const int new_cols = old_cols + pose_to_clone->size();

        Eigen::MatrixXd Cov_new = Eigen::MatrixXd::Zero(new_rows, new_cols);
        Cov_new.topLeftCorner(old_rows, old_cols) = Cov_old;

        int old_loc = pose_to_clone->id();
        Cov_new.block(old_rows, old_cols, clone_size, clone_size) = Cov_old.block(old_loc, old_loc, clone_size, clone_size);
        Cov_new.block(0, old_cols, old_rows, clone_size) = Cov_old.block(0, old_loc, old_rows, clone_size);
        Cov_new.block(old_rows, 0, clone_size, old_cols) = Cov_old.block(old_loc, 0, clone_size, old_cols);

        std::shared_ptr<Type> clone = pose_to_clone->clone();
        clone->set_local_id(old_cols);
        state->_clone_pose.insert(std::make_pair(clone->ts(), std::dynamic_pointer_cast<Pose>(clone)));
        state->_variables.push_back(clone);
        state->_dim += clone->size();
        state->SetCovariance(Cov_new);
    }

    virtual void MarginalizeState(std::shared_ptr<State> state, std::shared_ptr<Type> state_to_marginalize) override
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
        Eigen::VectorXd diags = state->_covariance.diagonal();
        for (int i = 0; i < diags.rows(); i++)
        {
            if (diags(i) < 0.0)
            {
                LOG(ERROR) << cv::format("diagonal is negative when update");
                LOG(ERROR) << "diags: " << diags.transpose();
                std::cout << "diag size: " << diags.size() << std::endl;
                std::cout << "variable size: " << state->_variables.size() << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        Eigen::VectorXd dx = K * res;
        for (size_t i = 0; i < state->_variables.size(); i++)
        {
            state->_variables[i]->update(dx.block(state->_variables[i]->id(), 0, state->_variables[i]->size(), 1));
        }
    }

    virtual bool PropagateStateAndCovariance(const std::vector<ImuData> imu_data, const double ts, std::shared_ptr<State> state) override
    {
        if (ts <= state->_imu_state->ts())
        {
            LOG(WARNING) << cv::format("curent state timestamp: %f, must be later than imu_state ts: %f", ts, state->_imu_state->ts());
            return false;
        }

        if (imu_data.empty() || imu_data.back().ts_sec < ts)
        {
            LOG(WARNING) << cv::format("wait for imu data, current state timestamp: %f but latest imu ts: %f", state->ts_sec(),
                                       imu_data.back().ts_sec);
            return false;
        }

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

        uint32_t dim = state->_imu_state->size();
        uint32_t th_id = state->_imu_state->q()->id();
        uint32_t p_id = state->_imu_state->p()->id();
        uint32_t v_id = state->_imu_state->v()->id();
        uint32_t bg_id = state->_imu_state->bg()->id();
        uint32_t ba_id = state->_imu_state->ba()->id();

        uint32_t na_id = 0;
        uint32_t ng_id = 3;
        uint32_t nbg_id = 6;
        uint32_t nba_id = 9;

        double sigma_a2 = std::pow(ImuManager::_sigma_na, 2);
        double sigma_w2 = std::pow(ImuManager::_sigma_nw, 2);
        double sigma_bg2 = std::pow(ImuManager::_sigma_bg, 2);
        double sigma_ba2 = std::pow(ImuManager::_sigma_ba, 2);
        Eigen::MatrixXd Cov_m = Eigen::MatrixXd::Identity(12, 12);
        Cov_m.block(na_id, na_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_a2;
        Cov_m.block(ng_id, ng_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_w2;
        Cov_m.block(nbg_id, nbg_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_bg2;
        Cov_m.block(nba_id, nba_id, 3, 3) = Eigen::Matrix3d::Identity() * sigma_ba2;

        Eigen::MatrixXd Phi_sum = Eigen::MatrixXd::Identity(dim, dim);
        Eigen::MatrixXd Cov_imu_old = state->_imu_state->covariance().block<15, 15>(state->_imu_state->id(), state->_imu_state->id());
        Eigen::MatrixXd Cov_imu_new = Cov_imu_old;

        for (int i = 0; i < imu_data.size() - 1; i++)
        {
            Eigen::MatrixXd F = Eigen::MatrixXd::Identity(15, 15);
            Eigen::MatrixXd G = Eigen::MatrixXd::Zero(15, 12);
            Eigen::MatrixXd Q = Cov_imu_new;

            double dt = imu_data.at(i + 1).ts_sec - imu_data.at(i).ts_sec;
            if (dt > 0 && dt < kMaxImuToleranceDelayTime)
            {
                Eigen::Vector3d am_mid = 0.5 * (imu_data.at(i).am + imu_data.at(i + 1).am) - ba;
                Eigen::Vector3d wm_mid = 0.5 * (imu_data.at(i).wm + imu_data.at(i + 1).wm) - bg;

                P = P_next;
                V = V_next;
                R = R_next;
                ba = ba_next;
                bg = bg_next;

                // for R
                F.block<3, 3>(th_id, th_id) = SO3d::exp(-wm_mid * dt).matrix();
                F.block<3, 3>(th_id, bg_id) = -Eigen::Matrix3d::Identity() * dt;

                // for p
                F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
                F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
                F.block<3, 3>(p_id, th_id) = -0.5 * R * MathUtils::skew(am_mid * dt * dt);
                F.block<3, 3>(p_id, ba_id) = -0.5 * R * dt * dt;

                // for v
                F.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
                F.block<3, 3>(v_id, th_id) = -R * MathUtils::skew(am_mid * dt);
                F.block<3, 3>(v_id, ba_id) = -R * dt;

                // for bg
                F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();

                // for ba
                F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();

                // for sigma noise
                G.block<3, 3>(th_id, ng_id) = -Eigen::Matrix3d::Identity() * dt;
                G.block<3, 3>(p_id, na_id) = -0.5 * R * dt * dt;
                G.block<3, 3>(v_id, na_id) = -R * dt;
                G.block<3, 3>(bg_id, nbg_id) = Eigen::Matrix3d::Identity();
                G.block<3, 3>(ba_id, nba_id) = Eigen::Matrix3d::Identity();

                // state propagation
                P_next = P + V * dt - 0.5 * state->_imu_state->gravity_inG * dt * dt + 0.5 * (R * am_mid * dt * dt);
                V_next = V - state->_imu_state->gravity_inG * dt + R * am_mid * dt;
                R_next = R * SO3d::exp(wm_mid * dt).matrix();
                ba_next = ba;
                bg_next = bg;

                Phi_sum = F * Phi_sum;
                Cov_imu_new = G * Cov_m * G.transpose() + F * Q * F.transpose();
                Cov_imu_new = 0.5 * (Cov_imu_new.eval() + Cov_imu_new.eval().transpose());
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
        state->_imu_state->set_covariance(Cov_imu_new);
        state->_covariance.block(state->_imu_state->id(), state->_imu_state->id(), state->_imu_state->size(), state->_imu_state->size()) =
            Cov_imu_new;

        uint32_t clone_state_size = state->_clone_pose.size();
        if (clone_state_size > 0)
        {
            Eigen::MatrixXd Cov_ic = state->_covariance.block(0, 15, 15, 6 * clone_state_size);
            state->_covariance.block(0, 15, 15, 6 * clone_state_size) = Phi_sum * Cov_ic;
            state->_covariance.block(15, 0, 6 * clone_state_size, 15) = Cov_ic.transpose() * Phi_sum.transpose();
        }

        return true;
    }
};

#endif