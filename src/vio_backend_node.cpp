#include "ros/init.h"
#include "ros/node_handle.h"
#include <cstdlib>
#include <memory>
#include <ros/ros.h>
#include <glog/logging.h>
#include <rosbag/bag.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "parameter.h"
#include "vioManager.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vio_backend");
    std::shared_ptr<ros::NodeHandle> nh = std::make_shared<ros::NodeHandle>("~");
    Param params(nh);
    params.load_params();

    rosbag::Bag bag;
    bag.open(params.path_bag, rosbag::bagmode::Read);
    rosbag::View view_full;
    rosbag::View view;

    VioManager vio_manager(params);

    view_full.addQuery(bag);
    ros::Time time_init = view_full.getBeginTime();
    time_init += ros::Duration(params.bag_start);
    ros::Time time_finish = (params.bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(params.bag_durr);
    view.addQuery(bag, time_init, time_finish);

    for (const rosbag::MessageInstance& msg : view) {

        if (msg.getTopic() == params.imu_topic) {
            vio_manager.imu_callback(msg.instantiate<sensor_msgs::Imu>());
        }
    }

    return EXIT_SUCCESS;
}