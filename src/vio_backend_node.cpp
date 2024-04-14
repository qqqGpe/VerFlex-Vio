#include "ros/init.h"
#include "ros/node_handle.h"
#include <cstdlib>
#include <memory>
#include <ros/ros.h>
#include <glog/logging.h>
#include <rosbag/bag.h>

#include "parameter.h"
#include "vioManager.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vio_backend");
    std::shared_ptr<ros::NodeHandle> nh = std::make_shared<ros::NodeHandle>("~");
    Param params_(nh);
    params_.load_params();
    
    return EXIT_SUCCESS;
}