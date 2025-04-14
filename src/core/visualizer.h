#ifndef __VISUALIZER__
#define __VISUALIZER__

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include "vioManager.h"
#include "eskf_solver.h"

class Visualizer
{
public:
    Visualizer(std::shared_ptr<ros::NodeHandle> &nh, std::shared_ptr<VioManager> &vio_manager);

    ~Visualizer() = default;

    void PublishVioState();

protected:
    std::shared_ptr<ros::NodeHandle> nh_;

    std::shared_ptr<VioManager> app_;

    ros::Publisher pub_pose_;

    ros::Publisher pub_path_;

    ros::Publisher pub_points_msckf_;

    uint32_t pose_seq_ = 0;

    std::vector<geometry_msgs::PoseStamped> imu_path_;

    uint32_t pub_path_div_ = 4;

    nav_msgs::Path path_output;
};

#endif