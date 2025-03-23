#include "visualizer.h"

Visualizer::Visualizer(std::shared_ptr<ros::NodeHandle> &nh, std::shared_ptr<VioManager> &vio_manager)
{
    nh_ = nh;

    app_ = vio_manager;

    pub_pose_ = nh_->advertise<geometry_msgs::PoseWithCovarianceStamped>("/vio/pose", 1000);

    pub_path_ = nh_->advertise<nav_msgs::Path>("/vio/path", 1000);

    pub_points_msckf_ = nh_->advertise<sensor_msgs::PointCloud2>("/vio/points", 1000);
}

void Visualizer::PublishVioState()
{
    std::shared_ptr<IMU_state> imu_state = app_->state->_imu_state;

    geometry_msgs::PoseWithCovarianceStamped pose_imu;
    pose_imu.header.stamp = ros::Time(imu_state->ts());
    pose_imu.header.seq = pose_seq_;
    pose_imu.header.frame_id = "global";
    pose_imu.pose.pose.position.x = imu_state->p()->vec().x();
    pose_imu.pose.pose.position.y = imu_state->p()->vec().y();
    pose_imu.pose.pose.position.z = imu_state->p()->vec().z();
    pose_imu.pose.pose.orientation.w = imu_state->q()->q().w();
    pose_imu.pose.pose.orientation.x = imu_state->q()->q().x();
    pose_imu.pose.pose.orientation.y = imu_state->q()->q().y();
    pose_imu.pose.pose.orientation.z = imu_state->q()->q().z();

    std::vector<std::shared_ptr<Type>> state_pub;
    state_pub.push_back(app_->state->_imu_state->p());
    state_pub.push_back(app_->state->_imu_state->q());

    Eigen::MatrixXd covariance_small = app_->state->GetMarginalCovariance(state_pub);
    for (int r = 0; r < 6; r++)
    {
        for (int c = 0; c < 6; c++)
        {
            pose_imu.pose.covariance[6 * r + c] = covariance_small(r, c);
        }
    }
    pub_pose_.publish(pose_imu);

    // publish imu paths
    geometry_msgs::PoseStamped pose_imu_current;
    pose_imu_current.header = pose_imu.header;
    pose_imu_current.pose = pose_imu.pose.pose;
    imu_path_.push_back(pose_imu_current);

    if (pose_seq_ % pub_path_div_ == 0)
    {
        path_output.header = pose_imu.header;
        path_output.header.seq = pose_seq_;
        path_output.header.frame_id = "global";
        path_output.poses.push_back(pose_imu_current);
        pub_path_.publish(path_output);
    }

    pose_seq_++;
}