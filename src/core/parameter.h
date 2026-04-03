#ifndef __VIO_PARAMETER__
#define __VIO_PARAMETER__
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <memory>

class Param
{
public:
    Param() = default;

    bool load_from_json(const std::string& json_path)
    {
        std::ifstream file(json_path);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open config file: " << json_path << std::endl;
            return false;
        }

        nlohmann::json j;
        try
        {
            file >> j;
        }
        catch (const nlohmann::json::parse_error& e)
        {
            std::cerr << "Error: JSON parse error: " << e.what() << std::endl;
            return false;
        }

        // Boolean flags
        set_init_timestamp_to_zero = j.value("set_init_timestamp_to_zero", false);
        use_multi_thread = j.value("use_multi_thread", false);
        estimate_ric = j.value("estimate_ric", true);
        estimate_td_visual = j.value("estimate_td_visual", true);
        use_zupt = j.value("use_zupt", false);
        use_fej = j.value("use_fej", false);
        frontend_prediction = j.value("frontend_prediction", false);
        save_full_log = j.value("save_full_log", true);
        save_tum_log = j.value("save_tum_log", true);
        use_pnp_ransac = j.value("use_pnp_ransac", true);
        use_slam_feature = j.value("use_slam_feature", false);
        use_census_transform = j.value("use_census_transform", true);
        use_histequal = j.value("use_histogram_equal", true);
        use_rate_limit = j.value("use_rate_limit", true);
        enable_schmidt_eskf = j.value("enable_schmidt_eskf", false);
        visualize_clone_poses = j.value("visualize_clone_poses", false);

        // Integer parameters
        log_level = j.value("log_level", 2);
        camera_num = j.value("camera_num", 1);
        running_rate = j.value("running_rate", 20);
        img_width = j.value("img_width", 752);
        img_height = j.value("img_height", 480);
        max_feat_n = j.value("max_feat_n", 225);
        grid_w = j.value("grid_w", 15);
        grid_h = j.value("grid_h", 15);
        max_clone_pose = j.value("max_clone_pose", 6);
        max_slam_feature = j.value("max_slam_feature", 25);
        solver_type = j.value("solver_type", 0);
        initial_type = j.value("initial_type", 1);

        // Double parameters
        lazy_time = j.value("lazy_time", 0.2);
        bag_start = j.value("bag_start", 0.0);
        bag_duration = j.value("bag_duration", -1.0);
        sigma_na = j.value("sigma_na", 2.0000e-3);
        sigma_nw = j.value("sigma_nw", 1.6968e-04);
        sigma_ba = j.value("sigma_ba", 3.0000e-3);
        sigma_bg = j.value("sigma_bg", 1.9393e-05);
        sigma_visual_pix = j.value("sigma_visual_pix", 1.0);
        init_td_visual_sigma = j.value("init_td_visual_sigma", 1e-4);
        init_ric_sigma = j.value("init_ric_sigma", 1e-3);
        gravity_magn = j.value("gravity_magn", 9.81);
        imu_acc_var_static_thres = j.value("imu_acc_var_static_thres", 0.5);
        imu_gyro_static_thres = j.value("imu_gyro_static_thres", 0.5);

        // String parameters
        bag_name = j.value("bag_name", std::string(""));
        log_path = j.value("log_path", std::string(""));
        bag_path = j.value("bag_path", std::string(""));
        imu_topic = j.value("imu_topic", std::string(""));
        dataset_dir = j.value("dataset_dir", std::string(""));

        // Camera topics
        if (j.contains("camera_topic"))
        {
            camera_topic = j["camera_topic"].get<std::vector<std::string>>();
        }
        else
        {
            camera_topic.resize(camera_num, "");
        }

        // Camera intrinsics and distortion
        for (int i = 0; i < camera_num; i++)
        {
            std::string intrinsic_key = "intrinsic_cam_" + std::to_string(i);
            std::string distortion_key = "distortion_cam_" + std::to_string(i);

            if (!j.contains(intrinsic_key) || !j.contains(distortion_key))
            {
                std::cerr << "Error: Missing " << intrinsic_key << " or " << distortion_key << std::endl;
                return false;
            }

            auto intrinsic = j[intrinsic_key].get<std::vector<double>>();
            auto distort = j[distortion_key].get<std::vector<double>>();
            Eigen::Matrix3d K;
            K << intrinsic[0], 0, intrinsic[2], 0, intrinsic[1], intrinsic[3], 0, 0, 1;
            intrinsics.push_back(K);
            distortion.push_back(Eigen::Map<Eigen::VectorXd>(distort.data(), distort.size()));
        }

        // Extrinsic parameters (Tic)
        Ric.resize(camera_num, Eigen::Matrix3d::Identity());
        tic.resize(camera_num, Eigen::Vector3d::Zero());
        if (j.contains("Tic"))
        {
            auto Tic = j["Tic"].get<std::vector<double>>();
            for (int cam_id = 0; cam_id < camera_num; cam_id++)
            {
                const int offset = cam_id * 12;
                Ric[cam_id] << Tic[offset], Tic[offset + 1], Tic[offset + 2], Tic[offset + 3], Tic[offset + 4], Tic[offset + 5], Tic[offset + 6],
                    Tic[offset + 7], Tic[offset + 8];
                tic[cam_id] << Tic[offset + 9], Tic[offset + 10], Tic[offset + 11];
            }
        }

        return CheckParams();
    }

    bool set_init_timestamp_to_zero = false;
    bool use_multi_thread = false;
    bool estimate_ric = true;
    bool estimate_td_visual = true;
    bool use_zupt = false;
    bool frontend_prediction = false;
    bool save_full_log = true;
    bool save_tum_log = true;
    bool use_census_transform = true;
    bool use_pnp_ransac = true;
    bool use_fej = false;
    bool use_rate_limit = true;
    bool use_slam_feature = false;
    bool use_histequal = true;

    int running_rate; // Hz
    int log_level = 2;
    int camera_num;
    int max_feat_n;
    int grid_h, grid_w;
    int max_clone_pose;
    int max_slam_feature;
    int img_width, img_height;
    int solver_type = 0; // 0: ESKF, 1: SqrtESKF
    bool enable_schmidt_eskf = false;
    int initial_type = 0; // 0: static initialization, 1: dynamic initialization
    bool visualize_clone_poses = false; // Enable clone pose visualization

    double sigma_na;
    double sigma_nw;
    double sigma_ba;
    double sigma_bg;
    double sigma_visual_pix;
    double init_td_visual_sigma;
    double init_ric_sigma;

    double lazy_time = 0.2; // In seconds
    double gravity_magn;
    double imu_acc_var_static_thres;
    double imu_gyro_static_thres;
    double bag_start, bag_duration;

    std::string log_path;
    std::string bag_path;
    std::string imu_topic;
    std::string bag_name;
    std::string dataset_dir;
    std::vector<std::string> camera_topic;

    std::vector<Eigen::Matrix3d> intrinsics;
    std::vector<Eigen::VectorXd> distortion;

    std::vector<Eigen::Matrix3d> Ric;
    std::vector<Eigen::Vector3d> tic;

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
        if (solver_type < 0 || solver_type > 1)
        {
            std::cerr << "Error: solver_type must be 0 (ESKF) or 1 (SqrtESKF)!" << std::endl;
            return false;
        }
        if (initial_type < 0 || initial_type > 1)
        {
            std::cerr << "Error: initial_type must be 0 (static initialization) or 1 (dynamic initialization)!" << std::endl;
            return false;
        }
        return true;
    }
};

#endif