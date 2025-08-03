#include "visualizer.h"

void Visualizer::PublishFeatures(const double timestamp, const std::vector<Feature*>& features)
{
    sensor_msgs::PointCloud current_feature_cloud;
    sensor_msgs::PointCloud all_feature_cloud;

    current_feature_cloud.header.frame_id = "global";
    current_feature_cloud.header.stamp = ros::Time(timestamp);
    for (const auto& feature : features)
    {
        if (feature->_valid && feature->_is_triangulated)
        {
            geometry_msgs::Point32 point;
            point.x = feature->_pwf.x();
            point.y = feature->_pwf.y();
            point.z = feature->_pwf.z();
            current_feature_cloud.points.push_back(point);

            // Add features to all features map
            all_features_.insert_or_assign(feature->_id, point);
        }
    }
    feature_publisher_.publish(current_feature_cloud);

    // Publish all features as a PointCloud message
    all_feature_cloud.header.frame_id = "global";
    all_feature_cloud.header.stamp = ros::Time(timestamp);
    for (const auto& [id, point] : all_features_)
    {
        all_feature_cloud.points.push_back(point);
    }
    feature_publisher_.publish(all_feature_cloud);
}

void Visualizer::PublishVioState(const std::shared_ptr<ImuState>& imu_state)
{
    // Publish imu pose
    geometry_msgs::PoseStamped pose_imu;
    pose_imu.header.stamp = ros::Time(imu_state->ts());
    pose_imu.header.seq = frame_id_;
    pose_imu.header.frame_id = "global";
    pose_imu.pose.position.x = imu_state->p()->vec().x();
    pose_imu.pose.position.y = imu_state->p()->vec().y();
    pose_imu.pose.position.z = imu_state->p()->vec().z();
    pose_imu.pose.orientation.w = imu_state->q()->q().w();
    pose_imu.pose.orientation.x = imu_state->q()->q().x();
    pose_imu.pose.orientation.y = imu_state->q()->q().y();
    pose_imu.pose.orientation.z = imu_state->q()->q().z();
    pose_publisher_.publish(pose_imu);

    // Publish imu path
    path_output.header = pose_imu.header;
    path_output.header.seq = frame_id_;
    path_output.header.frame_id = "global";
    path_output.poses.push_back(pose_imu);
    path_publisher_.publish(path_output);

    // Publish tf transformation
    geometry_msgs::TransformStamped transform;
    transform.header.stamp = ros::Time(imu_state->ts());
    transform.header.frame_id = "global";
    transform.child_frame_id = "imu";
    transform.transform.translation.x = pose_imu.pose.position.x;
    transform.transform.translation.y = pose_imu.pose.position.y;
    transform.transform.translation.z = pose_imu.pose.position.z;
    transform.transform.rotation = pose_imu.pose.orientation;
    tf_broadcaster_.sendTransform(transform);
    frame_id_++;
}

void Visualizer::PublishImageWithFeatures(const double timestamp, const cv::Mat& image_with_features)
{
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_with_features).toImageMsg();
    msg->header.stamp = ros::Time(timestamp);
    msg->header.frame_id = "global";
    image_with_features_publisher_.publish(msg);
}