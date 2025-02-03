#include <cstdlib>
#include <glog/logging.h>
#include <memory>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <signal.h>

#include "core/parameter.h"
#include "core/vioManager.h"

namespace {
  constexpr double kStereoTimeDeviation = 0.02;
}

void mySigintHandler(int sig) { ros::shutdown(); }

int main(int argc, char **argv) {
  google::InitGoogleLogging(*argv);
  ros::init(argc, argv, "vio_backend");
  std::shared_ptr<ros::NodeHandle> nh = std::make_shared<ros::NodeHandle>("~");

  signal(SIGINT, mySigintHandler);

  // load vio_backend parameters
  Param params(nh);
  params.load_params();

  // set log level
  fLI::FLAGS_stderrthreshold = params.log_level; // 0: info, 1: warning, 2: error, 3: fatal

  // prepare rosbag
  rosbag::Bag bag;
  rosbag::View view, view_full;
  bag.open(params.path_bag, rosbag::bagmode::Read);
  view_full.addQuery(bag);
  ros::Time time_init = view_full.getBeginTime();
  time_init += ros::Duration(params.bag_start);
  ros::Time time_finish = (params.bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(params.bag_durr);
  view.addQuery(bag, time_init, time_finish);

  // initialize vio_backend
  VioManager vio_manager(params);
  vio_manager.set_initial_timestamp(time_init.toSec());

  // start vio updater
  // if (params.use_multi_thread)
  // {
  //     vio_manager.start_visual_system();
  // }

  // load data from rosbag
  std::vector<rosbag::MessageInstance> msgs;
  for (const rosbag::MessageInstance &msg : view) {
    if (!ros::ok()) {
      break;
    }
    if (msg.getTopic() == params.imu_topic) {
      msgs.push_back(msg);
    }

    for (int i = 0; i < params.camera_num; i++) {
      if (msg.getTopic() == params.camera_topic[i]) {
        msgs.push_back(msg);
      }
    }
  }

  std::string ground_truth_topic = "/leica/position";
  for (int m = 0; m < msgs.size(); m++) {
    if (!ros::ok()) {
      break;
    }

    // get groundtruth msgs
    if (msgs.at(m).getTopic() == ground_truth_topic) {
        vio_manager.groundtruth_callback(msgs.at(m).instantiate<geometry_msgs::PointStamped>());
    }

    // get imu msgs
    if (msgs.at(m).getTopic() == params.imu_topic) {
      vio_manager.imu_callback(msgs.at(m).instantiate<sensor_msgs::Imu>());
    }

    // get stereo visual msgs
    for (int cam_id = 0; cam_id < params.camera_num; cam_id++) {
      if (msgs.at(m).getTopic() != params.camera_topic.at(cam_id)) {
        continue;
      }

      std::map<int, int> camid_to_msg_index;
      double meas_time = msgs.at(m).getTime().toSec();
      for (int cam_idt = 0; cam_idt < params.camera_num; cam_idt++) {
        if (cam_idt == cam_id) {
          camid_to_msg_index.insert({cam_id, m});
          continue;
        }
        int cam_idt_idx = -1;
        for (int mt = m; mt < (int)msgs.size(); mt++) {
            if (msgs.at(mt).getTopic() != params.camera_topic.at(cam_idt)) {
                continue;
            }

            if (std::abs(msgs.at(mt).getTime().toSec() - meas_time) < kStereoTimeDeviation) {
                cam_idt_idx = mt;
            }
            break;
        }
        if (cam_idt_idx != -1) {
          camid_to_msg_index.insert({cam_idt, cam_idt_idx});
        }
      }

      if ((int)camid_to_msg_index.size() != params.camera_num) {
        continue;
      }

      auto msg0 = msgs.at(camid_to_msg_index.at(0));
      auto msg1 = msgs.at(camid_to_msg_index.at(1));
      vio_manager.camera_callback(msg0.instantiate<sensor_msgs::Image>(),
                                  msg1.instantiate<sensor_msgs::Image>());
      vio_manager.process_measurememt_once();
    }
  }

  // waiting for program to exit
  ros::spin();
  google::ShutdownGoogleLogging();
  return 0;
}