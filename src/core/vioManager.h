#ifndef __VIO_MANAGER__
#define __VIO_MANAGER__
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <Eigen/Core>
#include <thread>

#include "solver.h"
#include "eskf_solver.h"
#include "sqrt_eskf_solver.h"
#include "ImuManager.h"
#include "cameraModel.h"
#include "frontend.h"
#include "initializer.h"
#include "dynamicInitializer.h"
#include "logger.h"
#include "mathematical_tools.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include "visualManager.h"


class VioManager
{
   public:
    VioManager() = default;
    VioManager(const Param& params)
    {
        params_ = params;

        if (params.solver_type == static_cast<int>(SolverType::ESKF))
        {
            solver = std::make_shared<eskfSolver>();
        }
        else if (params.solver_type == static_cast<int>(SolverType::SQRT_ESKF))
        {
            solver = std::make_shared<SqrtEskfSolver>();
        }
        else
        {
            LOG(ERROR) << "Solver type not supported!";
            exit(0);
        }

        state = std::make_shared<State>();

        _camera_model_0 = std::make_shared<CameraModel>(params);

        _imu_manager = std::make_shared<ImuManager>(params, state, solver);
        _imu_manager->SetImuNoise(params.sigma_na, params.sigma_nw, params.sigma_ba, params.sigma_bg);

        _visual_manager = std::make_shared<VisualManager>(params, state, _camera_model_0, solver);

        initializer = std::make_shared<Initializer>(params, _visual_manager, _camera_model_0, state);

        dynamic_initializer = std::make_unique<DynamicInitializer>(params, _visual_manager, _camera_model_0, state);

        vio_logger = std::make_shared<utils::LoggerFull>(params.log_path);

        vio_logger_tum = std::make_shared<utils::LoggerTUM>(params.log_path);

        // set camera extrinsic coeff
        Eigen::Quaterniond qic(params.Ric[0]);
        Eigen::Vector3d tic = params.tic[0];
        state->set_extrinsic(qic.normalized(), tic);
    }
    ~VioManager() {}

    void SetInitialTimeStamp(double initial_timestamp) { _initial_timestamp = initial_timestamp; }

    void ProcessMeasurementOnce();

    void start_visual_system();

    void GroundTruthCallback(const geometry_msgs::PointStamped::ConstPtr& msg);

    void ImuCallback(const sensor_msgs::Imu::ConstPtr& msg);

    void CameraCallback(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1);

    void ResetSystem();

    GroundTruth InterpolateGroundTruth(const double ts) const;

    double _initial_timestamp = 0.f;
    std::shared_ptr<State> state;
    std::shared_ptr<Initializer> initializer;
    std::unique_ptr<DynamicInitializer> dynamic_initializer;
    std::shared_ptr<ImuManager> _imu_manager;
    std::shared_ptr<VisualManager> _visual_manager;
    std::shared_ptr<CameraModel> _camera_model_0;
    std::shared_ptr<utils::LoggerFull> vio_logger;
    std::shared_ptr<utils::LoggerTUM> vio_logger_tum;
    std::shared_ptr<MsckfSolverBase> solver;
    std::map<double, std::pair<cv::Mat, cv::Mat>> image_bak;
    std::map<double, GroundTruth> ground_truth_;

    boost::posix_time::ptime vio_rT, vio_rT1, vio_rT2, vio_rT3, vio_rT4;
    boost::posix_time::ptime pro_rT, pro_rT1, pro_rT2, pro_rT3, pro_rT4;

   private:
    Param params_;
    double last_update_timestamp_ = -1.0;
    std::pair<double, std::vector<CameraObs>> last_feature_observes_;
};

void frontend_task_entry(std::shared_ptr<VisualManager> visual_manager);

void backend_task_entry(VioManager* vio);

#endif