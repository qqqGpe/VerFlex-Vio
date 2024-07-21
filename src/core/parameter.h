#ifndef __VIO_PARAMETER__
#define __VIO_PARAMETER__
#include "ros/node_handle.h"
#include <memory>
#include <ros/ros.h>

class Param {
public:
    Param(std::shared_ptr<ros::NodeHandle> nh)
        : _nh(nh)
    {
    }

    Param() = default;

    void load_params()
    {
        _nh->param<int>("img_width", img_width, 752);
        _nh->param<int>("img_height", img_height, 480);

        _nh->param<double>("bag_start", bag_start, 0);
        _nh->param<double>("bag_durr", bag_durr, -1);

        _nh->param<int>("max_feat_n", max_feat_n, 500);
        _nh->param<int>("max_clone_pose", max_clone_pose, 6);

        _nh->param<std::string>("log_path", log_path, "");
        _nh->param<std::string>("path_bag", path_bag, "");
        _nh->param<std::string>("cam_topic_0", cam_topic_0, "");
        _nh->param<std::string>("cam_topic_1", cam_topic_1, "");
        _nh->param<std::string>("imu_topic", imu_topic, "");
        _nh->param<std::string>("feature_topic", feature_topic, "");

        _nh->getParam("intrinsic_cam_0", intrinsic_cam_0);
        _nh->getParam("distortion_cam_0", distortion_cam_0);
        _nh->getParam("intrinsic_cam_1", intrinsic_cam_1);
        _nh->getParam("distortion_cam_1", distortion_cam_1);

        _nh->param<double>("sigma_na", sigma_na, 1e-3);
        _nh->param<double>("sigma_nw", sigma_nw, 1e-4);
        _nh->param<double>("sigma_ba", sigma_ba, 1e-3);
        _nh->param<double>("sigma_bw", sigma_bw, 1e-5);
    }

    int img_width, img_height;
    double bag_start, bag_durr;
    int max_feat_n;
    int max_clone_pose;
    std::string log_path;
    std::string path_bag;
    std::string cam_topic_0;
    std::string cam_topic_1;
    std::string imu_topic;
    std::string feature_topic;

    std::vector<double> intrinsic_cam_0;
    std::vector<double> distortion_cam_0;
    std::vector<double> intrinsic_cam_1;
    std::vector<double> distortion_cam_1;

    double sigma_na;
    double sigma_nw;
    double sigma_ba;
    double sigma_bw;

private:
    std::shared_ptr<ros::NodeHandle> _nh;
};

#endif