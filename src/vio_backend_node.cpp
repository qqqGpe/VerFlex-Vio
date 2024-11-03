#include <cstdlib>
#include <glog/logging.h>
#include <memory>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <signal.h>

#include "core/parameter.h"
#include "core/vioManager.h"

void mySigintHandler(int sig)
{
  ros::shutdown();
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(*argv);
    ros::init(argc, argv, "vio_backend");
    std::shared_ptr<ros::NodeHandle> nh = std::make_shared<ros::NodeHandle>("~");

    signal(SIGINT, mySigintHandler);

    // load vio_backend parameters
    Param params(nh);
    params.load_params();

    // set log level
    fLI::FLAGS_stderrthreshold = params.log_level;  // 0: info, 1: warning, 2: error, 3: fatal

    // prepare rosbag
    rosbag::Bag bag;
    rosbag::View view, view_full;
    bag.open(params.path_bag, rosbag::bagmode::Read);
    view_full.addQuery(bag);
    ros::Time time_init = view_full.getBeginTime();
    time_init += ros::Duration(params.bag_start);
    ros::Time time_finish = (params.bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(params.bag_durr);
    view.addQuery(bag, time_init, time_finish);

    // initialize vio_backend
    VioManager vio_manager(params);
    vio_manager.set_initial_timestamp(time_init.toSec());

    // start vio updater
    if (params.use_multi_thread)
    {
        vio_manager.start_visual_system();
    }

    // load data from rosbag
    for (const rosbag::MessageInstance& msg : view) {
        if (msg.getTopic() == params.imu_topic)
        {
            vio_manager.imu_callback(msg.instantiate<sensor_msgs::Imu>());
        }
        else if (msg.getTopic() == params.camera_topic[0])
        {
            vio_manager.camera_callback(msg.instantiate<sensor_msgs::Image>());
            if (!params.use_multi_thread) {
                vio_manager.process_measurememt_once();
            }
        }
    }

    // waiting for program to exit
    ros::spin();
    google::ShutdownGoogleLogging();
    return 0;
}