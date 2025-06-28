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
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include "vioManager.h"

class Visualizer
{
public:
    ~Visualizer() = default;
    Visualizer(const Visualizer&) = delete;
    Visualizer& operator=(const Visualizer&) = delete;
    Visualizer(Visualizer&&) = delete;
    Visualizer& operator=(Visualizer&&) = delete;

    static Visualizer& getInstance()
    {
        static Visualizer* instance = new Visualizer();
        return *instance;
    }

    void Init(std::shared_ptr<ros::NodeHandle> &nh, const Param &params)
    {
        nh_ = nh;
        path_publisher_ = nh_->advertise<nav_msgs::Path>("/vio/path", 10, true);
        pose_publisher_ = nh_->advertise<geometry_msgs::PoseStamped>("/vio/pose", 10);
        feature_publisher_ = nh_->advertise<sensor_msgs::PointCloud>("/vio/msckf_points", 1000, true);

        Eigen::Quaterniond qic(params.Ric[LEFT_CAM]);
        Eigen::Vector3d tic(params.tic[LEFT_CAM]);
        publishStaticCameraTransform(qic, tic);
    }

    void PublishVioState(const std::shared_ptr<ImuState>& imu_state);

    void PublishFeatures(const double timestamp, const std::vector<Feature*>& features);

private:
    Visualizer() = default;

    void publishStaticCameraTransform(Eigen::Quaterniond qic, Eigen::Vector3d tic) {
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
    uint32_t pub_path_div_ = 4;
    std::shared_ptr<ros::NodeHandle> nh_;
    ros::Publisher pose_publisher_;
    ros::Publisher path_publisher_;
    ros::Publisher feature_publisher_;
    nav_msgs::Path path_output;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;
    std::unordered_map<uint32_t, geometry_msgs::Point32> all_features_;
};

#endif