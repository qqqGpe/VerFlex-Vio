#ifndef __ESKF_SOLVER__
#define __ESKF_SOLVER__

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "state.h"
#include "utils.h"

class eskfSolver
{
   public:
    eskfSolver() = default;
    ~eskfSolver() {}

    static Eigen::MatrixXd construct_involved_covariance(std::shared_ptr<State>& state, const std::vector<std::shared_ptr<Type>>& Hx_order)
    {
        int32_t size = 0;
        for (const auto& x : Hx_order)
        {
            size += x->size();
        }
        Eigen::MatrixXd covariance_small = Eigen::MatrixXd::Zero(size, size);

        int32_t current_row = 0;
        for (int32_t i = 0; i < Hx_order.size(); i++)
        {
            std::shared_ptr<Type> var_i = Hx_order[i];
            int32_t current_col = 0;
            for (int32_t j = 0; j < Hx_order.size(); j++)
            {
                std::shared_ptr<Type> var_j = Hx_order[j];
                covariance_small.block(current_row, current_col, var_i->size(), var_j->size()) =
                    state->_covariance.block(var_i->id(), var_j->id(), var_i->size(), var_j->size());
                current_col += var_j->size();
            }
            current_row += var_i->size();
        }
        return covariance_small;
    }

    static void update(std::shared_ptr<State>& state,
                       const Eigen::Ref<Eigen::MatrixXd>& Hx,
                       const Eigen::Ref<Eigen::MatrixXd>& res,
                       const std::vector<std::shared_ptr<Type>>& Hx_order,
                       const std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx,
                       const Eigen::MatrixXd& R)
    {
        assert(R.rows() == res.rows());
        assert(Hx.rows() == res.rows());
        Eigen::MatrixXd M_all = Eigen::MatrixXd::Zero(state->_covariance.rows(), res.rows());

        int32_t current_it = 0;
        std::vector<int32_t> H_id;
        for (const auto& meas_var : Hx_order)
        {
            H_id.push_back(current_it);
            current_it += meas_var->size();
        }
        // // std::cout << "state->variable size: " << state->_variables.size() << std::endl;
        // // std::cout << "state->_covariance: " << state->_covariance << std::endl;
        // for (const auto& var : state->_variables)
        // {
        //     Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        //     for (int32_t i = 0; i < Hx_order.size(); i++)
        //     {
        //         std::shared_ptr<Type> meas_var = Hx_order[i];
        //         M_i += state->_covariance.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
        //                          Hx.block(0, H_id[i], Hx.rows(), meas_var->size()).transpose();
        //     }
        //     M_all.block(var->id(), 0, var->size(), res.rows()) = M_i;
        // }

        Eigen::MatrixXd Hx_all = Eigen::MatrixXd::Zero(res.rows(), state->_covariance.rows());
        for (int i = 0; i < Hx_order.size(); i++)
        {
            std::shared_ptr<Type> var = Hx_order[i];
            Hx_all.block(0, var->id(), res.rows(), var->size()) = Hx.block(0, H_id[i], res.rows(), var->size());
        }
        M_all.noalias() = state->_covariance * Hx_all.transpose();

        // Utils::show_eigen_matrix(Hx_all, "Hx_all");
        // Utils::show_eigen_matrix(Hx, "Hx");
        // Utils::show_eigen_matrix(M_all, "M_all");

        // std::cout << "Hx:\n" << std::setprecision(3) << Hx << std::endl;
        // std::cout << "M_all:\n" << std::setprecision(3) << M_all << std::endl;

        // std::cout << "M_all: " << M_all << std::endl;

        Eigen::MatrixXd cov_involved = construct_involved_covariance(state, Hx_order);

        // Residual covariance S = H * Cov * H' + R
        // Eigen::MatrixXd S(R.rows(), R.rows());
        // S.triangularView<Eigen::Upper>() = Hx * cov_involved * Hx.transpose();
        // S.triangularView<Eigen::Upper>() += R;
        Eigen::MatrixXd S = Hx * cov_involved * Hx.transpose() + R;

        // std::cout << "S:\n" << S << std::endl;

        Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
        S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
        Eigen::MatrixXd K = M_all * Sinv.selfadjointView<Eigen::Upper>();
        // Eigen::MatrixXd K = M_all * S.inverse();

        // Update Covariance
        // std::cout << "Cov before update: \n" << std::setprecision(2) << state->_covariance << std::endl;
        state->_covariance.triangularView<Eigen::Upper>() -= K * M_all.transpose();
        state->_covariance = state->_covariance.selfadjointView<Eigen::Upper>();
        // state->_covariance -= K * M_all.transpose();
        // std::cout << "K: \n" << std::setprecision(2) << K << std::endl;
        // std::cout << "K * M_all.transpose(): \n" << std::setprecision(2) <<  K * M_all.transpose() << std::endl;
        // std::cout << "Cov after update: \n" << std::setprecision(2) << state->_covariance << std::endl;
        state->_covariance = 0.5 * (state->_covariance.eval() + state->_covariance.eval().transpose());
        state->_imu_state->set_covariance(
            state->_covariance.block(state->_imu_state->id(), state->_imu_state->id(), state->_imu_state->size(), state->_imu_state->size()));

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
            // std::cout << state->_variables[i]->state_name << ": \n" << dx.block(state->_variables[i]->id(), 0, state->_variables[i]->size(),
            // 1).transpose() << std::endl;
        }
    }
};

#endif