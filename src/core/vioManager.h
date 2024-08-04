#include <thread>
#include <Eigen/Core>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>

#include "frontend.h"
#include "state.h"
#include "initializer.h"
#include "sensor_data.h"
#include "parameter.h"
#include "ImuManager.h"
#include "visualManager.h"
#include "mathematical_tools.h"
#include "camera_model.h"

class VioManager {
public:
    VioManager() = default;
    VioManager(const Param &params) {
        state = std::make_shared<State>();
        initializer = std::make_shared<Initializer>();
        _camera_model_0 = std::make_shared<CameraModel>(CameraType::PINHOLE, params.intrinsic_cam_0, params.distortion_cam_0);

        _imu_manager = std::make_shared<ImuManager>(params);
        _visual_manager = std::make_shared<VisualManager>(params, state, _camera_model_0);
    }
    ~VioManager(){}

    void start_visual_system();

    void imu_callback(const sensor_msgs::Imu::ConstPtr &msg);

    void camera_callback(const sensor_msgs::ImageConstPtr &msg);

    void propagate_state_and_covariance(std::shared_ptr<State> state, double ts);

    std::atomic<bool> backend_thread_running = false;
    std::shared_ptr<State> state;
    std::shared_ptr<Initializer> initializer;
    std::shared_ptr<ImuManager> _imu_manager;
    std::shared_ptr<VisualManager> _visual_manager;
    std::shared_ptr<CameraModel> _camera_model_0;

};

void frontend_task_entry(std::shared_ptr<VisualManager> visual_manager);

void backend_task_entry(VioManager *vio);