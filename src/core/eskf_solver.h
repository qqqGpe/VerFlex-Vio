#ifndef __ESKF_SOLVER__
#define __ESKF_SOLVER__

#include "state.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

class eskfSolver {
public:
    eskfSolver() = default;
    ~eskfSolver() { }

    Eigen::MatrixXd construct_involved_covariance(std::shared_ptr<State>& state, const std::vector<std::shared_ptr<Type>>& Hx_order)
    {
        int32_t size = 0;
        for (const auto& x : Hx_order) {
            size += x->size();
        }
        Eigen::MatrixXd covariance_small = Eigen::MatrixXd::Zero(size, size);

        int32_t current_row = 0;
        for (int32_t i = 0; i < Hx_order.size(); i++) {
            std::shared_ptr<Type> var_i = Hx_order[i];
            int32_t current_col = 0;
            for (int32_t j = 0; j < Hx_order.size(); j++) {
                std::shared_ptr<Type> var_j = Hx_order[j];
                covariance_small.block(current_row, current_col, var_i->size(), var_j->size()) = state->_covariance.block(var_i->id(), var_j->id(), var_i->size(), var_j->size());
                current_col += var_j->size();
            }
            current_row += var_i->size();
        }
        return covariance_small;
    }

    void update(std::shared_ptr<State>& state, const Eigen::Ref<Eigen::MatrixXd>& Hx, const Eigen::Ref<Eigen::MatrixXd>& res,
        const std::vector<std::shared_ptr<Type>>& Hx_order, const std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx,
        const Eigen::MatrixXd& R)
    {
        assert(R.rows() == res.rows());
        assert(Hx.rows() == res.rows());
        Eigen::MatrixXd M_all = Eigen::MatrixXd::Zero(state->_covariance.rows(), res.rows());

        int32_t current_it = 0;
        std::vector<int32_t> H_id;
        for (const auto& meas_var : Hx_order) {
            H_id.push_back(current_it);
            current_it += meas_var->size();
        }

        for (const auto& var : state->_variables) {
            Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
            for (int32_t i = 0; i < Hx_order.size(); i++) {
                std::shared_ptr<Type> meas_var = Hx_order[i];
                M_i.noalias() += state->_covariance.block(var->id(), meas_var->id(), var->size(), meas_var->size()) * Hx.block(0, H_id[i], res.rows(), meas_var->size()).transpose();
            }
            M_all.block(var->id(), 0, var->size(), res.rows()) = M_i;
        }

        Eigen::MatrixXd cov_involved = construct_involved_covariance(state, Hx_order);

        // Residual covariance S = H*Cov*H' + R
        Eigen::MatrixXd S(R.rows(), R.rows());
        S.triangularView<Eigen::Upper>() = Hx * cov_involved * Hx.transpose();
        S.triangularView<Eigen::Upper>() += R;
        // Eigen::MatrixXd S = H * P_small * H.transpose() + R;

        Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
        S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
        Eigen::MatrixXd K = M_all * Sinv.selfadjointView<Eigen::Upper>();
        // Eigen::MatrixXd K = M_a * S.inverse();

        // Update Covariance
        state->_covariance.triangularView<Eigen::Upper>() -= K * M_all.transpose();
        state->_covariance = state->_covariance.selfadjointView<Eigen::Upper>();
        // Cov -= K * M_a.transpose();
        // Cov = 0.5*(Cov+Cov.transpose());

        // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
        Eigen::VectorXd diags = state->_covariance.diagonal();
        for (int i = 0; i < diags.rows(); i++) {
            if (diags(i) < 0.0) {
                LOG(ERROR) << cv::format("diagonal is negative when update");
                std::exit(EXIT_FAILURE);
            }
        }

        Eigen::VectorXd dx = K * res;
        for (size_t i = 0; i < state->_variables.size(); i++) {
            state->_variables[i]->update(dx.block(state->_variables[i]->id(), 0, state->_variables[i]->size(), 1));
        }
    }
};

#endif