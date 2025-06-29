#include <glog/logging.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <signal.h>
#include <cstdlib>
#include <memory>

#include "core/camModel.h"
#include "core/visualizer.h"
#include "core/parameter.h"
#include "core/vioManager.h"

namespace
{
constexpr double kStereoTimeDeviation = 0.02;
}

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

    // prepare rosbag
    rosbag::Bag bag;
    rosbag::View view, view_full;
    bag.open(params.path_bag, rosbag::bagmode::Read);
    view_full.addQuery(bag);
    ros::Time time_init = view_full.getBeginTime();
    time_init += ros::Duration(params.bag_start);
    ros::Time time_finish = (params.bag_duration < 0) ? view_full.getEndTime() : time_init + ros::Duration(params.bag_duration);
    view.addQuery(bag, time_init, time_finish);

    if (params.set_init_timestamp_to_zero == true)
    {
        vio_manager->SetInitialTimeStamp(time_init.toSec());
    }
    else
    {
        vio_manager->SetInitialTimeStamp(0);
    }

    // start vio updater
    // if (params.use_multi_thread)
    // {
    //     vio_manager.start_visual_system();
    // }

    // load data from rosbag
    std::string ground_truth_topic = "/leica/position";
    std::vector<rosbag::MessageInstance> msgs;
    for (const rosbag::MessageInstance& msg : view)
    {
        if (!ros::ok())
        {
            break;
        }

        if (msg.getTopic() == params.imu_topic)
        {
            msgs.push_back(msg);
        }

        if (msg.getTopic() == ground_truth_topic)
        {
            msgs.push_back(msg);
        }

        for (int i = 0; i < params.camera_num; i++)
        {
            if (msg.getTopic() == params.camera_topic[i])
            {
                msgs.push_back(msg);
            }
        }
    }

    ros::Rate loop_rate(params.running_rate);
    for (int m = 0; m < msgs.size(); m++)
    {
        if (!ros::ok())
        {
            break;
        }

        // get groundtruth msgs
        if (msgs.at(m).getTopic() == ground_truth_topic)
        {
            vio_manager->GroundTruthCallback(msgs.at(m).instantiate<geometry_msgs::PointStamped>());
        }

        // get imu msgs
        if (msgs.at(m).getTopic() == params.imu_topic)
        {
            vio_manager->ImuCallback(msgs.at(m).instantiate<sensor_msgs::Imu>());
        }

        // get stereo visual msgs
        for (int cam_id = 0; cam_id < params.camera_num; cam_id++)
        {
            if (msgs.at(m).getTopic() != params.camera_topic.at(cam_id))
            {
                continue;
            }

            std::map<int, int> camid_to_msg_index;
            double meas_time = msgs.at(m).getTime().toSec();
            for (int cam_idt = 0; cam_idt < params.camera_num; cam_idt++)
            {
                if (cam_idt == cam_id)
                {
                    camid_to_msg_index.insert_or_assign(cam_id, m);
                    continue;
                }

                int cam_idt_idx = -1;
                for (int mt = m; mt < (int)msgs.size(); mt++)
                {
                    if (msgs.at(mt).getTopic() != params.camera_topic.at(cam_idt))
                    {
                        continue;
                    }

                    if (std::abs(msgs.at(mt).getTime().toSec() - meas_time) < kStereoTimeDeviation)
                    {
                        cam_idt_idx = mt;
                    }
                    break;
                }

                if (cam_idt_idx != -1)
                {
                    camid_to_msg_index.insert_or_assign(cam_idt, cam_idt_idx);
                }
            }

            if (static_cast<int>(camid_to_msg_index.size()) != params.camera_num)
            {
                continue;
            }

            if (params.camera_num == CamType::MONO)
            {
                auto msg0 = msgs.at(camid_to_msg_index.at(0));
                vio_manager->CameraCallback(msg0.instantiate<sensor_msgs::Image>(), nullptr);
            }
            else if (params.camera_num == CamType::STEREO)
            {
                auto msg0 = msgs.at(camid_to_msg_index.at(0));
                auto msg1 = msgs.at(camid_to_msg_index.at(1));
                vio_manager->CameraCallback(msg0.instantiate<sensor_msgs::Image>(), msg1.instantiate<sensor_msgs::Image>());
            }
            else
            {
                LOG(ERROR) << "Camera type not supported!";
                return -1;
            }

            vio_manager->ProcessMeasurementOnce();
            loop_rate.sleep();
        }
    }

    // waiting for program to exit
    google::ShutdownGoogleLogging();
    ros::shutdown();
    return 0;
}