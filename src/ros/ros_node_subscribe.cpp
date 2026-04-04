#include <glog/logging.h>
#include "utils.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <signal.h>
#include <cstdlib>
#include <memory>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include "core/camModel.h"
#include "core/parameter.h"
#include "core/vioManager.h"
#include "ros_visualizer.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Sync;

// Shared pointer accessible by callbacks
static std::shared_ptr<VioManager> g_vio_manager;

void ImuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    ImuData data;
    data.ts_sec = msg->header.stamp.toSec();
    data.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    data.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
    g_vio_manager->FeedImuData(data);
}

void StereoCallback(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1)
{
    double ts_sec = msg0->header.stamp.toSec();
    std::vector<cv::Mat> images;
    try
    {
        images.push_back(cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8)->image.clone());
        images.push_back(cv_bridge::toCvShare(msg1, sensor_msgs::image_encodings::MONO8)->image.clone());
    }
    catch (cv_bridge::Exception& e)
    {
        LOG_ERROR("cv_bridge exception: {}", e.what());
        return;
    }
    g_vio_manager->FeedImageData(ts_sec, images);
}

void MonoCallback(const sensor_msgs::ImageConstPtr& msg0)
{
    double ts_sec = msg0->header.stamp.toSec();
    std::vector<cv::Mat> images;
    try
    {
        images.push_back(cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8)->image.clone());
    }
    catch (cv_bridge::Exception& e)
    {
        LOG_ERROR("cv_bridge exception: {}", e.what());
        return;
    }
    g_vio_manager->FeedImageData(ts_sec, images);
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(*argv);
    ros::init(argc, argv, "vio_backend");
    std::shared_ptr<ros::NodeHandle> nh = std::make_shared<ros::NodeHandle>("~");
    ros::Subscriber sub_imu;
    ros::Subscriber sub_camera;
    std::shared_ptr<Sync> sync_;
    message_filters::Subscriber<sensor_msgs::Image> left_sub;
    message_filters::Subscriber<sensor_msgs::Image> right_sub;

    // Load parameters from JSON config
    std::string config_path;
    nh->param<std::string>("config_path", config_path, "");

    Parameter params;
    if (!config_path.empty())
    {
        if (!params.load_from_yaml(config_path))
        {
            LOG_ERROR("Failed to load parameters from: {}", config_path);
            return -1;
        }
    }
    else
    {
        LOG_ERROR("config_path parameter is required!");
        return -1;
    }

    // Initialize camera model
    CamModel::getInstance().Init(params);

    // Initialize ROS visualizer
    RosVisualizer ros_visualizer;
    ros_visualizer.Init(nh, params);

    // Initialize vio_backend
    g_vio_manager = std::make_shared<VioManager>(params);

    // Set output callback
    g_vio_manager->SetOutputCallback([&ros_visualizer](const VioOutput& output) {
        ros_visualizer.PublishVioOutput(output);
    });

    // Start frontend and backend threads if multi-thread is enabled
    if (params.use_multi_thread)
    {
        g_vio_manager->StartFrontendThread();
        g_vio_manager->StartBackendThread();
    }

    // set log level
    fLI::FLAGS_stderrthreshold = params.log_level;

    // Register subscriber for imu topics
    sub_imu = nh->subscribe(params.imu_topic, 1000, ImuCallback);

    // Register subscriber for camera topics
    if (params.camera_num == CamType::MONO)
    {
        sub_camera = nh->subscribe(params.camera_topic[0], 1000, MonoCallback);
    }
    else if (params.camera_num == CamType::STEREO)
    {
        left_sub.subscribe(*nh, params.camera_topic[0], 10);
        right_sub.subscribe(*nh, params.camera_topic[1], 10);
        sync_ = std::make_shared<Sync>(SyncPolicy(10), left_sub, right_sub);
        sync_->registerCallback(boost::bind(&StereoCallback, _1, _2));
    }

    // Start loop
    while(ros::ok())
    {
        ros::spinOnce();
        if (!params.use_multi_thread && g_vio_manager->_visual_manager->_input_image_buffer.size() > 0)
        {
            g_vio_manager->ProcessMeasurementOnce();
        }
    }

    return 0;
}
