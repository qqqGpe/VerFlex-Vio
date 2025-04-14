#ifndef __SOLVER__
#define __SOLVER__
#include "state.h"
#include "Type.h"
#include "sensor_data.h"
#include "mathematical_tools.h"

#include <memory>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace
{
constexpr uint32_t kImuOutputHz = 200;
constexpr double kMaxImuToleranceDelayTime = 1.0 / kImuOutputHz * 5;
}  // namespace

enum class SolverType
{
    ESKF = 0,
    SQRT_ESKF = 1
};

class MsckfSolverBase
{
   public:
    MsckfSolverBase() = default;
    ~MsckfSolverBase() = default;

    virtual void update(std::shared_ptr<State>& state,
                        const Eigen::Ref<Eigen::MatrixXd>& Hx,
                        const Eigen::Ref<Eigen::MatrixXd>& res,
                        const std::vector<std::shared_ptr<Type>>& Hx_order,
                        const std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx,
                        const Eigen::MatrixXd& R) = 0;

    virtual bool PropagateStateAndCovariance(const std::vector<ImuData> imu_data, const double ts, std::shared_ptr<State> state) = 0;

    virtual void StochasticClone(std::shared_ptr<State> state) = 0;

    virtual void MarginalizeState(std::shared_ptr<State> state, std::shared_ptr<Type> state_to_marginalize) = 0;
};

#endif