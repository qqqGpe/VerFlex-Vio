#ifndef __SQRT_ESKF_SOLVER__
#define __SQRT_ESKF_SOLVER__
#include "solver.h"
#include "utils.h"
#include "ImuManager.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <sophus/so3.hpp>

using namespace Sophus;

class SqrtEskfSolver : public MsckfSolverBase
{
   public:
    SqrtEskfSolver() = default;
    virtual ~SqrtEskfSolver() {}

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

        std::shared_ptr<Type> clone_pose = pose_to_clone->clone();
        clone_pose->set_local_id(old_cols);
        state->_clone_pose.insert(std::make_pair(clone_pose->ts(), std::dynamic_pointer_cast<Pose>(clone_pose)));
        state->_variables.push_back(clone_pose);
        state->_dim += clone_pose->size();

        if (state->enable_estimate_td_visual_)
        {
            Eigen::Vector3d last_w = imu_data->back().wm;
            Eigen::MatrixXd J_td = Eigen::MatrixXd::Zero(clone_pose_size, 1);
            J_td << last_w, state->_imu_state->v()->vec();
            SqrtPt_new.rightCols(clone_pose_size) += SqrtPt_new.block(0, state->td_visual().id(), SqrtPt_new.rows(), state->td_visual().size()) * J_td.transpose();
        }

        state->SetSqrtPt(SqrtPt_new);
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
        Eigen::MatrixXd rhks = MathUtils::GivensRotation(M_all, M_all.cols());
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
                LOG(ERROR) << cv::format("diagonal is negative when update");
                LOG(ERROR) << "diags: " << diags.transpose();
                // std::cout << "diag size: " << diags.size() << std::endl;
                // std::cout << "variable size: " << state->_variables.size() << std::endl;
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

    virtual bool PropagateStateAndCovariance(const std::vector<ImuData> imu_data, const double visual_ts, std::shared_ptr<State> state) override
    {
        if (visual_ts <= state->_imu_state->ts())
        {
            LOG(WARNING) << cv::format("Propagation failed, curent state timestamp: %f, must be later than imu_state timestamp: %f", visual_ts,
                                       state->_imu_state->ts());
            return false;
        }

        if (imu_data.empty() || imu_data.back().ts_sec < state->ts_sec())
        {
            LOG(WARNING) << cv::format("Propagation failed, waiting for imu data, current state timestamp: %f but latest imu timestamp: %f",
                                       state->ts_sec(), imu_data.back().ts_sec);
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

        uint32_t imu_state_dim = state->_imu_state->size();
        uint32_t imu_id = state->_imu_state->id();
        uint32_t th_id = state->_imu_state->q()->id();
        uint32_t p_id = state->_imu_state->p()->id();
        uint32_t v_id = state->_imu_state->v()->id();
        uint32_t bg_id = state->_imu_state->bg()->id();
        uint32_t ba_id = state->_imu_state->ba()->id();

        constexpr uint32_t kNoiseAccId = 0;
        constexpr uint32_t kNoiseGyroId = 3;
        constexpr uint32_t kNoiseGyroBiasId = 6;
        constexpr uint32_t kNoiseAccBiasId = 9;

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
                Eigen::Vector3d am_mid = 0.5 * (imu_data.at(i).am + imu_data.at(i + 1).am) - ba;
                Eigen::Vector3d wm_mid = 0.5 * (imu_data.at(i).wm + imu_data.at(i + 1).wm) - bg;

                P = P_next;
                V = V_next;
                R = R_next;
                ba = ba_next;
                bg = bg_next;

                // For R
                F.block<3, 3>(th_id, th_id) = SO3d::exp(-wm_mid * dt).matrix();
                F.block<3, 3>(th_id, bg_id) = -Eigen::Matrix3d::Identity() * dt;

                // For p
                F.block<3, 3>(p_id, p_id) = Eigen::Matrix3d::Identity();
                F.block<3, 3>(p_id, v_id) = Eigen::Matrix3d::Identity() * dt;
                F.block<3, 3>(p_id, th_id) = -0.5 * R * MathUtils::skew(am_mid * dt * dt);
                F.block<3, 3>(p_id, ba_id) = -0.5 * R * dt * dt;

                // For v
                F.block<3, 3>(v_id, v_id) = Eigen::Matrix3d::Identity();
                F.block<3, 3>(v_id, th_id) = -R * MathUtils::skew(am_mid * dt);
                F.block<3, 3>(v_id, ba_id) = -R * dt;

                // For bg
                F.block<3, 3>(bg_id, bg_id) = Eigen::Matrix3d::Identity();

                // For ba
                F.block<3, 3>(ba_id, ba_id) = Eigen::Matrix3d::Identity();

                // For measurement noise
                G.block<3, 3>(th_id, kNoiseGyroId) = -Eigen::Matrix3d::Identity() * dt;
                G.block<3, 3>(p_id, kNoiseAccId) = -0.5 * R * dt * dt;
                G.block<3, 3>(v_id, kNoiseAccId) = -R * dt;
                G.block<3, 3>(bg_id, kNoiseGyroBiasId) = Eigen::Matrix3d::Identity();
                G.block<3, 3>(ba_id, kNoiseAccBiasId) = Eigen::Matrix3d::Identity();

                // State propagation
                P_next = P + V * dt - 0.5 * state->_imu_state->gravity_inG * dt * dt + 0.5 * (R * am_mid * dt * dt);
                V_next = V - state->_imu_state->gravity_inG * dt + R * am_mid * dt;
                R_next = R * SO3d::exp(wm_mid * dt).matrix();
                ba_next = ba;
                bg_next = bg;

                // Phi and Q_sum propagation
                Phi_sum = F * Phi_sum;
                Q_sum = F * Q_sum * F.transpose() + G * Q * G.transpose();
            }
            else
            {
                LOG(WARNING) << fmt::format("Imu delayed for {}s", dt);
                exit(0);
            }
        }

        // Update state
        state->_imu_state->set_ts(visual_ts);   // Set current state timestamp to visual timestamp
        state->_imu_state->q()->set_value(Eigen::Quaterniond(R_next).normalized().coeffs());
        state->_imu_state->p()->set_value(P_next);
        state->_imu_state->v()->set_value(V_next);

        // Update sqrt root covariance
        Eigen::MatrixXd SqrtPt_imu = state->_imu_state->Sqrt_Pt();
        Eigen::MatrixXd SqrtPt_tmp = Eigen::MatrixXd::Zero(state->Sqrt_Pt().rows() + Q_sum.rows(), state->Sqrt_Pt().cols());
        SqrtPt_tmp.block(0, 0, state->Sqrt_Pt().rows(), state->Sqrt_Pt().cols()) = state->Sqrt_Pt();
        SqrtPt_tmp.block(0, 0, imu_state_dim, imu_state_dim) = SqrtPt_imu * Phi_sum.transpose();
        SqrtPt_tmp.block(state->Sqrt_Pt().rows(), 0, Q_sum.rows(), Q_sum.cols()) = Q_sum.llt().matrixL().transpose();
        Eigen::MatrixXd SqrtPt_triangulated = MathUtils::GivensRotation(SqrtPt_tmp, SqrtPt_tmp.cols());
        Eigen::MatrixXd SqrtPt_propagated = SqrtPt_triangulated.block(0, 0, SqrtPt_tmp.cols(), SqrtPt_tmp.cols());

        assert(SqrtPt_tmp.cols() == state->Sqrt_Pt().cols());
        state->SetSqrtPt(SqrtPt_propagated);

        Eigen::MatrixXd Cov_full = SqrtPt_propagated.transpose() * SqrtPt_propagated;
        state->SetCovariance(0.5 * (Cov_full + Cov_full.transpose()));

        return true;
    }
};

#endif