#ifndef __VIO_PARAMETER__
#define __VIO_PARAMETER__
#include <ros/ros.h>
#include <Eigen/Core>
#include <memory>
#include "ros/node_handle.h"

class Param
{
public:
    Param(std::shared_ptr<ros::NodeHandle> nh) : _nh(nh) {}

    Param() = default;

    bool load_params()
    {
        _nh->param<bool>("set_init_timestamp_to_zero", set_init_timestamp_to_zero, false);
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

        for (int i = 0; i < camera_num; i++)
        {
            std::vector<double> intrinsic, distort;
            _nh->getParam("intrinsic_cam_" + std::to_string(i), intrinsic);
            _nh->getParam("distortion_cam_" + std::to_string(i), distort);
            Eigen::Matrix3d K;
            K << intrinsic[0], 0, intrinsic[2], 0, intrinsic[1], intrinsic[3], 0, 0, 1;
            intrinsics.push_back(K);
            distortion.push_back(Eigen::Map<Eigen::VectorXd>(distort.data(), distort.size()));
        }

        camera_topic.resize(camera_num, "");
        _nh->getParam("camera_topic", camera_topic);
        _nh->param<std::string>("imu_topic", imu_topic, "");

        _nh->param<double>("sigma_na", sigma_na, 2.0000e-3);
        _nh->param<double>("sigma_nw", sigma_nw, 1.6968e-04);
        _nh->param<double>("sigma_ba", sigma_ba, 3.0000e-3);
        _nh->param<double>("sigma_bg", sigma_bg, 1.9393e-05);

        _nh->param<double>("gravity_magn", gravity_magn, 9.81);

        _nh->param<double>("imu_acc_var_static_thres", imu_acc_var_static_thres, 0.5);
        _nh->param<double>("imu_gyro_static_thres", imu_gyro_static_thres, 0.5);

        _nh->param<int>("solver_type", solver_type, 0);

        Ric.resize(camera_num, Eigen::Matrix3d::Identity());
        tic.resize(camera_num, Eigen::Vector3d::Zero());

        std::vector<double> Tic;
        _nh->getParam("Tic", Tic);
        for (int cam_id = 0; cam_id < camera_num; cam_id++)
        {
            const int offset = cam_id * 12;
            Ric[cam_id] << Tic[offset], Tic[offset + 1], Tic[offset + 2], Tic[offset + 3], Tic[offset + 4], Tic[offset + 5], Tic[offset + 6],
                Tic[offset + 7], Tic[offset + 8];
            tic[cam_id] << Tic[offset + 9], Tic[offset + 10], Tic[offset + 11];
        }

        return CheckParams();
    }
    bool set_init_timestamp_to_zero = false;
    bool use_multi_thread = false;
    int log_level = 2;
    int camera_num;
    double bag_start, bag_durr;
    int max_feat_n;
    int grid_h, grid_w;
    int max_clone_pose;
    int img_width, img_height;
    double gravity_magn;
    double imu_acc_var_static_thres;
    double imu_gyro_static_thres;

    std::string log_path;
    std::string path_bag;
    std::string imu_topic;
    std::vector<std::string> camera_topic;

    std::vector<Eigen::Matrix3d> intrinsics;
    std::vector<Eigen::VectorXd> distortion;

    double sigma_na;
    double sigma_nw;
    double sigma_ba;
    double sigma_bg;

    std::vector<Eigen::Matrix3d> Ric;
    std::vector<Eigen::Vector3d> tic;

    int solver_type = 0; // 0: ESKF, 1: SqrtESKF

private:
    bool CheckParams()
    {
        if (camera_num < 1 || camera_num > 2)
        {
            std::cerr << "Error: camera_num must be 1 or 2!" << std::endl;
            return false;
        }
        if (img_width <= 0 || img_height <= 0)
        {
            std::cerr << "Error: img_width and img_height must be positive!" << std::endl;
            return false;
        }
        if (max_feat_n <= 0)
        {
            std::cerr << "Error: max_feat_n must be positive!" << std::endl;
            return false;
        }
        if (grid_w <= 0 || grid_h <= 0)
        {
            std::cerr << "Error: grid_w and grid_h must be positive!" << std::endl;
            return false;
        }
        if (max_clone_pose < 0)
        {
            std::cerr << "Error: max_clone_pose must be non-negative!" << std::endl;
            return false;
        }
        if (imu_acc_var_static_thres <= 0 || imu_gyro_static_thres <= 0)
        {
            std::cerr << "Error: imu_acc_var_static_thres and imu_gyro_static_thres must be positive!" << std::endl;
            return false;
        }
        return true;
    }

    std::shared_ptr<ros::NodeHandle> _nh;
};

#endif