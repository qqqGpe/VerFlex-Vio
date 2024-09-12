#ifndef __VIO_PARAMETER__
#define __VIO_PARAMETER__
#include "ros/node_handle.h"
#include <memory>
#include <ros/ros.h>
#include <Eigen/Core>

class Param {
public:
    Param(std::shared_ptr<ros::NodeHandle> nh)
        : _nh(nh)
    {
    }

    Param() = default;

    void load_params()
    {
        _nh->param<bool>("use_multi_thread", use_multi_thread, false);
        _nh->param<int>("log_level", log_level, 2);
        _nh->param<int>("camera_num", camera_num, 1);

        _nh->param<int>("img_width", img_width, 752);
        _nh->param<int>("img_height", img_height, 480);

        _nh->param<double>("bag_start", bag_start, 0);
        _nh->param<double>("bag_durr", bag_durr, -1);

        _nh->param<int>("max_feat_n", max_feat_n, 225);
        _nh->param<int>("grid_w", grid_w, 15);
        _nh->param<int>("grid_h", grid_h, 15);
        _nh->param<int>("max_clone_pose", max_clone_pose, 6);

        _nh->param<std::string>("log_path", log_path, "");
        _nh->param<std::string>("path_bag", path_bag, "");

        _nh->getParam("intrinsic_cam_0", intrinsic_cam_0);
        _nh->getParam("distortion_cam_0", distortion_cam_0);
        _nh->getParam("intrinsic_cam_1", intrinsic_cam_1);
        _nh->getParam("distortion_cam_1", distortion_cam_1);

        camera_topic.resize(camera_num, "");
        _nh->getParam("camera_topic", camera_topic);
        _nh->param<std::string>("imu_topic", imu_topic, "");

        _nh->param<double>("sigma_na", sigma_na, 1e-3);
        _nh->param<double>("sigma_nw", sigma_nw, 1e-4);
        _nh->param<double>("sigma_ba", sigma_ba, 1e-3);
        _nh->param<double>("sigma_bw", sigma_bw, 1e-5);

        _nh->param<double>("gravity_magn", gravity_magn, 9.81);

        Ric.resize(camera_num, Eigen::Matrix3d::Identity());
        tic.resize(camera_num, Eigen::Vector3d::Zero());

        std::vector<double> Tic;
        _nh->getParam("Tic", Tic);
        for (int cam_id = 0; cam_id < camera_num; cam_id++) {
            const int offset = cam_id * 12;
            Ric[cam_id] << Tic[offset],     Tic[offset + 1], Tic[offset + 2],
                           Tic[offset + 3], Tic[offset + 4], Tic[offset + 5],
                           Tic[offset + 6], Tic[offset + 7], Tic[offset + 8];
            tic[cam_id] << Tic[offset + 9], Tic[offset + 10], Tic[offset + 11];
        }
    }

    bool use_multi_thread = false;
    int log_level = 2;
    int camera_num;
    double bag_start, bag_durr;
    int max_feat_n;
    int grid_h, grid_w;
    int max_clone_pose;
    int img_width, img_height;
    double gravity_magn;

    std::string log_path;
    std::string path_bag;
    std::string imu_topic;
    std::vector<std::string> camera_topic;

    std::vector<double> intrinsic_cam_0;
    std::vector<double> distortion_cam_0;
    std::vector<double> intrinsic_cam_1;
    std::vector<double> distortion_cam_1;

    double sigma_na;
    double sigma_nw;
    double sigma_ba;
    double sigma_bw;

    std::vector<Eigen::Matrix3d> Ric;
    std::vector<Eigen::Vector3d> tic;

private:
    std::shared_ptr<ros::NodeHandle> _nh;
};

#endif