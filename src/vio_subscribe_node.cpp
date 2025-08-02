#include <glog/logging.h>
#include <ros/ros.h>
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
#include "core/visualizer.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Sync;

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

    // Load vio_backend parameters
    Param params(nh);
    if (!params.load_params())
    {
        LOG(ERROR) << "Failed to load parameters!";
        return -1;
    }

    // Initialize camera model
    CamModel::getInstance().Init(params);

    // Initialize ros visualizer
    Visualizer::getInstance().Init(nh, params);

    // Initialize vio_backend
    std::shared_ptr<VioManager> vio_manager = std::make_shared<VioManager>(params);

    // set log level
    fLI::FLAGS_stderrthreshold = params.log_level;  // 0: info, 1: warning, 2: error, 3: fatal

    // Register subscriber for imu topics
    sub_imu = nh->subscribe(params.imu_topic, 1000, &VioManager::ImuCallback, vio_manager.get());

    // Register subscriber for camera topics
    if (params.camera_num == CamType::MONO)
    {
        sub_camera = nh->subscribe(params.camera_topic[0], 1000, &VioManager::CallbackMonocular, vio_manager.get());
    }
    else if (params.camera_num == CamType::STEREO)
    {
        left_sub.subscribe(*nh, params.camera_topic[0], 10);
        right_sub.subscribe(*nh, params.camera_topic[1], 10);
        sync_ = std::make_shared<Sync>(SyncPolicy(10), left_sub, right_sub);
        sync_->registerCallback(boost::bind(&VioManager::CallbackStereo, vio_manager.get(), _1, _2));
    }

    // Start loop
    while(ros::ok())
    {
        ros::spinOnce();
        if (vio_manager->_visual_manager->_input_image_buffer.size() > 0)
        {
            vio_manager->ProcessMeasurementOnce();
        }
    }

    return 0;
}