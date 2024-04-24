#include <thread>
#include <Eigen/Eigen>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <glog/logging.h>

#include "states/state.h"
#include "initializer.h"
#include "sensor_data.h"
#include "parameter.h"

class VioManager {
public:
    VioManager() = default;
    VioManager(const Param &params) {
        state = std::make_shared<State>();
        initializer = std::make_shared<Initializer>();
    }
    ~VioManager(){}

    void imu_callback(const sensor_msgs::Imu::ConstPtr &msg);
    void feature_callback(const sensor_msgs::PointCloud::ConstPtr &msg);

    std::shared_ptr<State> state;
    std::shared_ptr<Initializer> initializer;

};