#include <Eigen/Core>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PointStamped.h>
#include <thread>

#include "ImuManager.h"
#include "cameraModel.h"
#include "frontend.h"
#include "initializer.h"
#include "mathematical_tools.h"
#include "parameter.h"
#include "sensor_data.h"
#include "state.h"
#include "visualManager.h"
#include "logger.h"

class VioManager;

void frontend_task_entry(std::shared_ptr<VisualManager> visual_manager);

void backend_task_entry(VioManager* vio);

class VioManager {
public:
    VioManager() = default;
    VioManager(const Param& params)
    {
        state = std::make_shared<State>();
        _camera_model_0 = std::make_shared<CameraModel>(params);
        _imu_manager = std::make_shared<ImuManager>(params, state);
        _visual_manager = std::make_shared<VisualManager>(params, state, _camera_model_0);
        initializer = std::make_shared<Initializer>(params, _visual_manager, _camera_model_0, state);
        vio_logger = std::make_shared<utils::Logger>(params.log_path);

        // set camera extrinsic coeff
        Eigen::Quaterniond qic(params.Ric[0]);
        Eigen::Vector3d tic = params.tic[0];
        state->set_extrinsic(qic.normalized(), tic);
        // std::cout << "Ric: \n" << state->_Tic->quat().toRotationMatrix() << std::endl;
        // std::cout << "tic: \n" << state->_Tic->p().transpose() << std::endl;
    }
    ~VioManager() { }

    void set_initial_timestamp(double initial_timestamp) { _initial_timestamp = initial_timestamp; }

    void process_measurememt_once();

    void start_visual_system();

    void groundtruth_callback(const geometry_msgs::PointStamped::ConstPtr& msg);

    void imu_callback(const sensor_msgs::Imu::ConstPtr& msg);

    void camera_callback(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1);

    bool propagate_state_and_covariance(std::shared_ptr<State> state, double ts);

    GroundTruth InterpolateGroundTruth(const double ts) const;

    double _initial_timestamp = 0.f;
    std::shared_ptr<State> state;
    std::shared_ptr<Initializer> initializer;
    std::shared_ptr<ImuManager> _imu_manager;
    std::shared_ptr<VisualManager> _visual_manager;
    std::shared_ptr<CameraModel> _camera_model_0;
    std::shared_ptr<utils::Logger> vio_logger;
    std::map<double, std::pair<cv::Mat, cv::Mat>> image_bak;

    std::map<double, GroundTruth> ground_truth_;

    boost::posix_time::ptime vio_rT, vio_rT1, vio_rT2, vio_rT3, vio_rT4;
    boost::posix_time::ptime pro_rT, pro_rT1, pro_rT2, pro_rT3, pro_rT4;
};