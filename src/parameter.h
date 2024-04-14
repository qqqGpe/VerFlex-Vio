#include "ros/node_handle.h"
#include <memory>
#include <ros/ros.h>

class Param {
public:
    Param(std::shared_ptr<ros::NodeHandle> nh)
        : _nh(nh)
    {
    }

    void load_params()
    {
        _nh->param<int>("int", img_width, 752);
        _nh->param<int>("int", img_height, 480);
        _nh->param<double>("bag_start", bag_start, 0);
        _nh->param<double>("bag_durr", bag_durr, -1);
        _nh->param<int>("max_feat_n", max_feat_n, 500);
        _nh->param<int>("max_clone_pose", max_clone_pose, 6);
        _nh->param<std::string>("log_path", log_path, "");
        _nh->param<std::string>("data_path", data_path, "");
    }

    int img_width, img_height;
    double bag_start, bag_durr;
    int max_feat_n;
    int max_clone_pose;
    std::string log_path;
    std::string data_path;

private:
    std::shared_ptr<ros::NodeHandle> _nh;
};