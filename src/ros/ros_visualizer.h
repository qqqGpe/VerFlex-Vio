#ifndef __ROS_VISUALIZER__
#define __ROS_VISUALIZER__

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include "vioManager.h"

class RosVisualizer
{
public:
    ~RosVisualizer() = default;

    void Init(std::shared_ptr<ros::NodeHandle> &nh, const Parameter &params);

    void PublishVioOutput(const VioOutput &output);

  private:
    void publishStaticCameraTransform(Eigen::Quaterniond qic, Eigen::Vector3d tic)
    {
        geometry_msgs::TransformStamped static_transform;
        static_transform.header.stamp = ros::Time(0);
        static_transform.header.frame_id = "imu";
        static_transform.child_frame_id = "camera";
        static_transform.transform.translation.x = tic.x();
        static_transform.transform.translation.y = tic.y();
        static_transform.transform.translation.z = tic.z();
        static_transform.transform.rotation.w = qic.w();
        static_transform.transform.rotation.x = qic.x();
        static_transform.transform.rotation.y = qic.y();
        static_transform.transform.rotation.z = qic.z();
        static_tf_broadcaster_.sendTransform(static_transform);
    }

    uint32_t frame_id_ = 0;
    std::shared_ptr<ros::NodeHandle> nh_;
    ros::Publisher pose_publisher_;
    ros::Publisher path_publisher_;
    ros::Publisher feature_publisher_;
    ros::Publisher image_with_features_publisher_;
    nav_msgs::Path path_output;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;
    std::unordered_map<uint32_t, geometry_msgs::Point32> all_features_;
};

#endif
