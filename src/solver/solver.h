/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-04-16 22:27:58
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
 */
#ifndef __SOLVER__
#define __SOLVER__
#include "vioState.h"
#include "Type.h"
#include "sensorType.h"
#include "mathematical_tools.h"

#include <glog/logging.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

enum class SolverType
{
    ESKF = 0,
    SQRT_ESKF = 1
};

class MsckfSolverBase
{
   public:
    MsckfSolverBase() = default;
    virtual ~MsckfSolverBase() = default;

    virtual void update(std::shared_ptr<State>& state,
                        const Eigen::Ref<Eigen::MatrixXd>& Hx,
                        const Eigen::Ref<Eigen::MatrixXd>& res,
                        const std::vector<std::shared_ptr<Type>>& Hx_order,
                        const std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx,
                        const Eigen::MatrixXd& R) = 0;

    virtual bool PropagateStateAndCovariance(const std::vector<ImuData> imu_data, const double visual_ts, std::shared_ptr<State> state) = 0;

    virtual void StochasticClone(std::shared_ptr<State> state, std::vector<ImuData>* imu_data) = 0;

    virtual void MarginalizeState(std::shared_ptr<State> state, std::shared_ptr<Type> state_to_marginalize) = 0;

   protected:
    constexpr static uint32_t kImuOutputHz = 200;
    constexpr static double kMaxImuToleranceDelayTime = 1.0 / kImuOutputHz * 5;
};

#endif