#include "ros_visualizer.h"

void RosVisualizer::Init(std::shared_ptr<ros::NodeHandle> &nh, const Parameter &params)
{
    nh_ = nh;
    path_publisher_ = nh_->advertise<nav_msgs::Path>("/vio/path", 10, true);
    pose_publisher_ = nh_->advertise<geometry_msgs::PoseStamped>("/vio/pose", 10);
    feature_publisher_ = nh_->advertise<sensor_msgs::PointCloud>("/vio/msckf_points", 1000, true);
    image_with_features_publisher_ = nh_->advertise<sensor_msgs::Image>("/vio/image_with_features", 10, true);

    Eigen::Quaterniond qic(params.Ric[0]);
    Eigen::Vector3d tic(params.tic[0]);
    publishStaticCameraTransform(qic, tic);
}

void RosVisualizer::PublishVioOutput(const VioOutput &output)
{
    // Publish pose
    geometry_msgs::PoseStamped pose_imu;
    pose_imu.header.stamp = ros::Time(output.timestamp);
    pose_imu.header.seq = frame_id_;
    pose_imu.header.frame_id = "global";
    pose_imu.pose.position.x = output.position.x();
    pose_imu.pose.position.y = output.position.y();
    pose_imu.pose.position.z = output.position.z();
    pose_imu.pose.orientation.w = output.orientation.w();
    pose_imu.pose.orientation.x = output.orientation.x();
    pose_imu.pose.orientation.y = output.orientation.y();
    pose_imu.pose.orientation.z = output.orientation.z();
    pose_publisher_.publish(pose_imu);

    // Publish path
    path_output.header = pose_imu.header;
    path_output.header.seq = frame_id_;
    path_output.header.frame_id = "global";
    path_output.poses.push_back(pose_imu);
    path_publisher_.publish(path_output);

    // Publish tf
    geometry_msgs::TransformStamped transform;
    transform.header.stamp = ros::Time(output.timestamp);
    transform.header.frame_id = "global";
    transform.child_frame_id = "imu";
    transform.transform.translation.x = output.position.x();
    transform.transform.translation.y = output.position.y();
    transform.transform.translation.z = output.position.z();
    transform.transform.rotation = pose_imu.pose.orientation;
    tf_broadcaster_.sendTransform(transform);

    // Publish features
    if (!output.feature_points.empty())
    {
        sensor_msgs::PointCloud feature_cloud;
        feature_cloud.header.frame_id = "global";
        feature_cloud.header.stamp = ros::Time(output.timestamp);
        for (const auto& pt : output.feature_points)
        {
            geometry_msgs::Point32 point;
            point.x = pt.x();
            point.y = pt.y();
            point.z = pt.z();
            feature_cloud.points.push_back(point);
        }
        feature_publisher_.publish(feature_cloud);
    }

    // Publish image with features
    if (!output.image_with_features.empty())
    {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output.image_with_features).toImageMsg();
        msg->header.stamp = ros::Time(output.timestamp);
        msg->header.frame_id = "global";
        image_with_features_publisher_.publish(msg);
    }

    frame_id_++;
}
