#ifndef __LOGGER__
#define __LOGGER__
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <glog/logging.h>

namespace utils {

struct LogValue
{
    void reset_value()
    {
        LogValue temp;
        std::swap(temp, *this);
    }

    std::vector<std::string> value_name
    {
        "timestamp",
        "px",
        "py",
        "pz",
        "vx",
        "vy",
        "vz",
        "roll",
        "pitch",
        "yaw",

        "bias_acc_x",
        "bias_acc_y",
        "bias_acc_z",
        "bias_gyro_x",
        "bias_gyro_y",
        "bias_gyro_z",

        "sigma_px",
        "sigma_py",
        "sigma_pz",
        "sigma_vx",
        "sigma_vy",
        "sigma_vz",

        "sigma_roll",
        "sigma_pitch",
        "sigma_yaw",

        "sigma_bias_acc_x",
        "sigma_bias_acc_y",
        "sigma_bias_acc_z",
        "sigma_bias_gyro_x",
        "sigma_bias_gyro_y",
        "sigma_bias_gyro_z",

        "visual_updated",
        "zupt_updated",

        "diff_px",
        "diff_py",
        "diff_pz"
    };

    double timestamp = 0;

    double px = 0;
    double py = 0;
    double pz = 0;
    double vx = 0;
    double vy = 0;
    double vz = 0;
    double roll = 0;
    double pitch = 0;
    double yaw = 0;
    double bias_acc_x = 0;
    double bias_acc_y = 0;
    double bias_acc_z = 0;
    double bias_gyro_x = 0;
    double bias_gyro_y = 0;
    double bias_gyro_z = 0;

    double sigma_px = 0;
    double sigma_py = 0;
    double sigma_pz = 0;
    double sigma_vx = 0;
    double sigma_vy = 0;
    double sigma_vz = 0;
    double sigma_roll = 0;
    double sigma_pitch = 0;
    double sigma_yaw = 0;
    double sigma_bias_acc_x = 0;
    double sigma_bias_acc_y = 0;
    double sigma_bias_acc_z = 0;
    double sigma_bias_gyro_x = 0;
    double sigma_bias_gyro_y = 0;
    double sigma_bias_gyro_z = 0;

    int visual_updated = 0;
    int zupt_updated = 0;

    // pvdiff
    double diff_px = 0;
    double diff_py = 0;
    double diff_pz = 0;
};

class Logger {
public:
    Logger(const std::string& path) {

        if(!make_log_file(path))
        {
            LOG(ERROR) << "Failed to make logging file";
            std::exit(1);
        }
    }

    bool make_log_file(std::string root_dir)
    {
        char time_str[100];
        std::time_t now = std::time(nullptr);
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
        _file_name = root_dir + "/" + std::string(time_str) + ".csv";

        std::ofstream file(_file_name);
        std::string csv_title = last_log_value.value_name[0];
        for (int i = 1; i < last_log_value.value_name.size(); i++)
        {
            csv_title = csv_title + ", " + last_log_value.value_name[i];
        }

        if (file.is_open()) {
            file << csv_title << std::endl;
            file.close();
            return true;
        } else {
            std::cerr << "unable to open file" << _file_name << std::endl;
            return false;
        }
        return false;
    }

    void save_to_file(const LogValue log_value) {
        std::ofstream file(_file_name, std::ios_base::app); // 以追加模式打开文件
        if (file.is_open()) {
            file << log_value.timestamp << ", ";
            file << log_value.px << ", ";
            file << log_value.py << ", ";
            file << log_value.pz << ", ";
            file << log_value.vx << ", ";
            file << log_value.vy << ", ";
            file << log_value.vz << ", ";
            file << log_value.roll << ", ";
            file << log_value.pitch << ", ";
            file << log_value.yaw << ", ";
            file << log_value.bias_acc_x << ", ";
            file << log_value.bias_acc_y << ", ";
            file << log_value.bias_acc_z << ", ";
            file << log_value.bias_gyro_x << ", ";
            file << log_value.bias_gyro_y << ", ";
            file << log_value.bias_gyro_z << ", ";
            file << log_value.sigma_px << ", ";
            file << log_value.sigma_py << ", ";
            file << log_value.sigma_pz << ", ";
            file << log_value.sigma_vx << ", ";
            file << log_value.sigma_vy << ", ";
            file << log_value.sigma_vz << ", ";
            file << log_value.sigma_roll << ", ";
            file << log_value.sigma_pitch << ", ";
            file << log_value.sigma_bias_acc_x << ", ";
            file << log_value.sigma_bias_acc_y << ", ";
            file << log_value.sigma_bias_acc_z << ", ";
            file << log_value.sigma_bias_gyro_x << ", ";
            file << log_value.sigma_bias_gyro_y << ", ";
            file << log_value.sigma_bias_gyro_z << ", ";
            file << log_value.visual_updated << ", ";
            file << log_value.zupt_updated << ", ";
            file << log_value.diff_px << ", ";
            file << log_value.diff_py << ", ";
            file << log_value.diff_pz << std::endl;

            last_log_value = log_value; // backup current log value
            // 写入数据到文件
            file.close();
        } else {
            std::cerr << "无法打开文件：" << _file_name << std::endl;
        }
    }

    LogValue last_log_value;

private:
    std::string _path;
    std::string _file_name;
};

}

#endif