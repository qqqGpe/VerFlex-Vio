#include <Eigen/Core>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
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

class VioManager {
public:
    VioManager() = default;
    VioManager(const Param& params)
    {
        state = std::make_shared<State>();
        initializer = std::make_shared<Initializer>();
        _camera_model_0 = std::make_shared<CameraModel>(CameraType::PINHOLE, params);
        _imu_manager = std::make_shared<ImuManager>(params);
        _visual_manager = std::make_shared<VisualManager>(params, state, _camera_model_0);

        // set camera extrinsic coeff
        Eigen::Quaterniond qic(params.Ric[0]);
        Eigen::Vector3d tic = params.tic[0];
        state->set_extrinsic(qic, tic);
    }
    ~VioManager() { }

    void set_initial_timestamp(double initial_timestamp) { _initial_timestamp = initial_timestamp; }

    void process_measurememt_once();

    void start_visual_system();

    void imu_callback(const sensor_msgs::Imu::ConstPtr& msg);

    void camera_callback(const sensor_msgs::ImageConstPtr& msg);

    bool propagate_state_and_covariance(std::shared_ptr<State> state, double ts);

    double _initial_timestamp = 0.f;
    std::shared_ptr<State> state;
    std::shared_ptr<Initializer> initializer;
    std::shared_ptr<ImuManager> _imu_manager;
    std::shared_ptr<VisualManager> _visual_manager;
    std::shared_ptr<CameraModel> _camera_model_0;

    boost::posix_time::ptime vio_rT, vio_rT1, vio_rT2, vio_rT3, vio_rT4;
    boost::posix_time::ptime pro_rT, pro_rT1, pro_rT2, pro_rT3, pro_rT4;
};

void frontend_task_entry(std::shared_ptr<VisualManager> visual_manager);

void backend_task_entry(VioManager* vio);