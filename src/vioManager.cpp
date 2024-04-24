#include "vioManager.h"

// void VioManager::feature_callback(const sensor_msgs::PointCloud::ConstPtr &msg)
// {
// }

void VioManager::imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    ImuData data;
    data.ts_sec = msg->header.stamp.toSec();
    data.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    data.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

    if(!initializer->is_initialized) {
        initializer->feed_imu_measurement(data);
    }

    std::thread thread(
        [&]{
            if(!initializer->is_initialized) {
                bool status = initializer->static_initialize(state->_imu_state);
            } else {
                LOG(INFO) << "vio has already initialized!";
            }
        }
    );

    thread.join();
    // thread.detach();

}
