#ifndef __VIO_PARAMETER__
#define __VIO_PARAMETER__
#include <Eigen/Core>
#include <iostream>
#include <yaml-cpp/yaml.h>

class Parameter
{
  public:
    Parameter() = default;

    bool load_from_yaml(const std::string &yaml_path)
    {
        YAML::Node yaml_node;
        try
        {
            yaml_node = YAML::LoadFile(yaml_path);
        }
        catch (const YAML::Exception &e)
        {
            std::cerr << "Error: Cannot load config file: " << yaml_path << " (" << e.what() << ")" << std::endl;
            return false;
        }

        try
        {
            auto get = [&](const std::string &key) { return yaml_node[key]; };
            // Boolean flags
            set_init_timestamp_to_zero = get("set_init_timestamp_to_zero") ? get("set_init_timestamp_to_zero").as<bool>() : false;
            use_multi_thread = get("use_multi_thread") ? get("use_multi_thread").as<bool>() : false;
            estimate_ric = get("estimate_ric") ? get("estimate_ric").as<bool>() : true;
            estimate_td_visual = get("estimate_td_visual") ? get("estimate_td_visual").as<bool>() : true;
            use_zupt = get("use_zupt") ? get("use_zupt").as<bool>() : false;
            use_fej = get("use_fej") ? get("use_fej").as<bool>() : false;
            do_warp_klt = get("do_warp_klt") ? get("do_warp_klt").as<bool>() : false;
            do_prediction = get("do_prediction") ? get("do_prediction").as<bool>() : false;
            save_full_log = get("save_full_log") ? get("save_full_log").as<bool>() : true;
            save_tum_log = get("save_tum_log") ? get("save_tum_log").as<bool>() : true;
            use_pnp_ransac = get("use_pnp_ransac") ? get("use_pnp_ransac").as<bool>() : true;
            use_slam_feature = get("use_slam_feature") ? get("use_slam_feature").as<bool>() : false;
            use_census_transform = get("use_census_transform") ? get("use_census_transform").as<bool>() : true;
            use_histequal = get("use_histogram_equal") ? get("use_histogram_equal").as<bool>() : true;
            use_rate_limit = get("use_rate_limit") ? get("use_rate_limit").as<bool>() : true;
            enable_schmidt_eskf = get("enable_schmidt_eskf") ? get("enable_schmidt_eskf").as<bool>() : false;
            visualize_clone_poses = get("visualize_clone_poses") ? get("visualize_clone_poses").as<bool>() : false;
            enable_pangolin_viewer = get("enable_pangolin_viewer") ? get("enable_pangolin_viewer").as<bool>() : false;
            record_viewer = get("record_viewer") ? get("record_viewer").as<bool>() : false;

            // Integer parameters
            log_level = get("log_level") ? get("log_level").as<int>() : 2;
            camera_num = get("camera_num") ? get("camera_num").as<int>() : 1;
            running_rate = get("running_rate") ? get("running_rate").as<int>() : 20;
            img_width = get("img_width") ? get("img_width").as<int>() : 752;
            img_height = get("img_height") ? get("img_height").as<int>() : 480;
            max_feat_n = get("max_feat_n") ? get("max_feat_n").as<int>() : 225;
            grid_w = get("grid_w") ? get("grid_w").as<int>() : 15;
            grid_h = get("grid_h") ? get("grid_h").as<int>() : 15;
            max_clone_pose = get("max_clone_pose") ? get("max_clone_pose").as<int>() : 6;
            max_slam_feature = get("max_slam_feature") ? get("max_slam_feature").as<int>() : 25;
            solver_type = get("solver_type") ? get("solver_type").as<int>() : 0;
            initial_type = get("initial_type") ? get("initial_type").as<int>() : 1;

            // Double parameters
            lazy_time = get("lazy_time") ? get("lazy_time").as<double>() : 0.2;
            keyframe_parallex_thres = get("keyframe_parallex_thres") ? get("keyframe_parallex_thres").as<double>() : 10.0;
            bag_start = get("bag_start") ? get("bag_start").as<double>() : 0.0;
            bag_duration = get("bag_duration") ? get("bag_duration").as<double>() : -1.0;
            sigma_na = get("sigma_na") ? get("sigma_na").as<double>() : 2.0000e-3;
            sigma_nw = get("sigma_nw") ? get("sigma_nw").as<double>() : 1.6968e-04;
            sigma_ba = get("sigma_ba") ? get("sigma_ba").as<double>() : 3.0000e-3;
            sigma_bg = get("sigma_bg") ? get("sigma_bg").as<double>() : 1.9393e-05;
            sigma_visual_pix = get("sigma_visual_pix") ? get("sigma_visual_pix").as<double>() : 1.0;
            init_td_visual_sigma = get("init_td_visual_sigma") ? get("init_td_visual_sigma").as<double>() : 1e-4;
            init_ric_sigma = get("init_ric_sigma") ? get("init_ric_sigma").as<double>() : 1e-3;
            gravity_magn = get("gravity_magn") ? get("gravity_magn").as<double>() : 9.81;
            imu_acc_var_static_thres = get("imu_acc_var_static_thres") ? get("imu_acc_var_static_thres").as<double>() : 0.5;
            imu_gyro_static_thres = get("imu_gyro_static_thres") ? get("imu_gyro_static_thres").as<double>() : 0.5;

            // String parameters
            bag_name = get("bag_name") ? get("bag_name").as<std::string>() : std::string("");
            log_path = get("log_path") ? get("log_path").as<std::string>() : std::string("");
            bag_path = get("bag_path") ? get("bag_path").as<std::string>() : std::string("");
            imu_topic = get("imu_topic") ? get("imu_topic").as<std::string>() : std::string("");
            dataset_dir = get("dataset_dir") ? get("dataset_dir").as<std::string>() : std::string("");

            // Camera topics
            if (get("camera_topic"))
            {
                camera_topic = get("camera_topic").as<std::vector<std::string>>();
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

                if (!get(intrinsic_key) || !get(distortion_key))
                {
                    std::cerr << "Error: Missing " << intrinsic_key << " or " << distortion_key << std::endl;
                    return false;
                }

                auto intrinsic = get(intrinsic_key).as<std::vector<double>>();
                auto distort = get(distortion_key).as<std::vector<double>>();
                Eigen::Matrix3d K;
                K << intrinsic[0], 0, intrinsic[2], 0, intrinsic[1], intrinsic[3], 0, 0, 1;
                intrinsics.push_back(K);
                distortion.push_back(Eigen::Map<Eigen::VectorXd>(distort.data(), distort.size()));
            }

            // Extrinsic parameters (Tic): list of [camera_num] 4x3 blocks (rows 0-2: R, row 3: t)
            Ric.resize(camera_num, Eigen::Matrix3d::Identity());
            tic.resize(camera_num, Eigen::Vector3d::Zero());
            if (get("Tic"))
            {
                auto Tic_node = get("Tic");
                for (int cam_id = 0; cam_id < camera_num; cam_id++)
                {
                    auto cam = Tic_node[cam_id];
                    Ric[cam_id] << cam[0][0].as<double>(), cam[0][1].as<double>(), cam[0][2].as<double>(), cam[1][0].as<double>(),
                        cam[1][1].as<double>(), cam[1][2].as<double>(), cam[2][0].as<double>(), cam[2][1].as<double>(), cam[2][2].as<double>();
                    tic[cam_id] << cam[3][0].as<double>(), cam[3][1].as<double>(), cam[3][2].as<double>();
                }
            }

            return CheckParams();

        } // try
        catch (const YAML::Exception &e)
        {
            std::cerr << "Error: Failed to parse config: " << yaml_path << " (" << e.what() << ")" << std::endl;
            return false;
        }
    }

    bool set_init_timestamp_to_zero = false;
    bool use_multi_thread = false;
    bool estimate_ric = true;
    bool estimate_td_visual = true;
    bool use_zupt = false;
    bool do_prediction = false;
    bool do_warp_klt = false;
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
    int initial_type = 0;               // 0: static initialization, 1: dynamic initialization
    bool visualize_clone_poses = false; // Enable clone pose visualization
    bool enable_pangolin_viewer = false; // Enable Pangolin 3D viewer (requires USE_PANGOLIN build)
    bool record_viewer = false;          // Record Pangolin frames to PNG (for GIF making)

    double sigma_na;
    double sigma_nw;
    double sigma_ba;
    double sigma_bg;
    double sigma_visual_pix;
    double init_td_visual_sigma;
    double init_ric_sigma;

    double keyframe_parallex_thres = 10.0; // Pixel parallex threshold for keyframe decision
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