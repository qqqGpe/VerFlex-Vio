#ifndef __LOGGER__
#define __LOGGER__
#include <glog/logging.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

namespace utils
{

class LogValueTUM
{
   public:
    void Reset()
    {
        *this = std::move(LogValueTUM());
    }

    std::vector<std::string> value_name{"#timestamp" "px" "py" "pz" "qx" "qy" "qz" "qw"};
    double timestamp = 0;
    double px = 0;
    double py = 0;
    double pz = 0;
    double qx = 0;
    double qy = 0;
    double qz = 0;
    double qw = 0;
};

class LogValueFull
{
   public:
    void Reset()
    {
        *this = std::move(LogValueFull());
    }

    std::vector<std::string> value_name = {"#timestamp",
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
                                           "ZuptUpdated",
                                           "keyframe",

                                           "diff_px",
                                           "diff_py",
                                           "diff_pz",
                                           "init_vnorm",
                                           "ground_truth_vnorm"};

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
    int ZuptUpdated = 0;
    int keyframe = 0;

    // posiiton-velocity difference
    double diff_px = 0;
    double diff_py = 0;
    double diff_pz = 0;

    double init_vnorm = 0;
    double groundtruth_vnorm = 0;
};

template <typename Derived>
class LoggerBase
{
   public:
    LoggerBase(const std::string& dirname, const std::string bag_name, const std::string log_type = "default", const bool use_title = true)
    {
        if (!InitLogFile(dirname, bag_name, log_type, use_title))
        {
            LOG(ERROR) << "Failed to make logging file";
            std::exit(1);
        }
    }

    virtual void SaveValues(const Derived log_value) const = 0;

   protected:

    void Reset()
    {
        *this = std::move(LoggerBase());
    }

    bool InitLogFile(const std::string dirname, const std::string bag_name, const std::string log_type, const bool use_title)
    {
      char time_str[100];
      std::time_t now = std::time(nullptr);
      std::strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
      filename_ = dirname + "/" + bag_name + "_" + std::string(time_str) + "_" + log_type + ".csv";

      Derived value_temp;
      std::string log_title = value_temp.value_name[0];
      for (int i = 1; i < value_temp.value_name.size(); i++) {
        log_title = log_title + " " + value_temp.value_name[i];
      }

        std::ofstream file(filename_);
        if (file.is_open())
        {
            if (use_title)
            {
                file << log_title << std::endl;
                file.close();
            }
        }
        else
        {
            std::cerr << "Failed to create file" << filename_ << std::endl;
            return false;
        }
        return true;
    }

    std::string filename_;
};

/* Save Vio state values as TUM format */
class LoggerTUM : public LoggerBase<LogValueTUM>
{
   public:
    LoggerTUM(const std::string& dirname, const std::string& bag_name) : LoggerBase<LogValueTUM>(dirname, bag_name, "tum", false) {}

    virtual void SaveValues(const LogValueTUM log_value) const override
    {
        std::ofstream file(filename_, std::ios::app);
        file.setf(std::ios::scientific);
        file.precision(10);
        if (file.is_open())
        {
            file << log_value.timestamp << " " << log_value.px << " " << log_value.py << " " << log_value.pz << " " << log_value.qx << " "
                 << log_value.qy << " " << log_value.qz << " " << log_value.qw << "\n";
        }
        else
        {
            std::cerr << "Can not save TUM data to: " << filename_ << std::endl;
        }
        file.close();
    }
};

/* Save full log values for analysis */
class LoggerFull : public LoggerBase<LogValueFull>
{
   public:
    LoggerFull(const std::string& dirname, const std::string& bag_name) : LoggerBase<LogValueFull>(dirname, bag_name, "full", true) {}

    virtual void SaveValues(const LogValueFull log_value) const override
    {
        std::ofstream file(filename_, std::ios::app);
        file.setf(std::ios::scientific);
        file.precision(10);
        if (file.is_open())
        {
            file << log_value.timestamp << " " << log_value.px << " " << log_value.py << " " << log_value.pz << " " << log_value.vx << " "
                 << log_value.vy << " " << log_value.vz << " " << log_value.roll << " " << log_value.pitch << " " << log_value.yaw << " "
                 << log_value.bias_acc_x << " " << log_value.bias_acc_y << " " << log_value.bias_acc_z << " " << log_value.bias_gyro_x << " "
                 << log_value.bias_gyro_y << " " << log_value.bias_gyro_z << " " << log_value.sigma_px << " " << log_value.sigma_py << " "
                 << log_value.sigma_pz << " " << log_value.sigma_vx << " " << log_value.sigma_vy << " " << log_value.sigma_vz << " "
                 << log_value.sigma_roll << " " << log_value.sigma_pitch << " " << log_value.sigma_yaw << " " << log_value.sigma_bias_acc_x << " "
                 << log_value.sigma_bias_acc_y << " " << log_value.sigma_bias_acc_z << " " << log_value.sigma_bias_gyro_x << " "
                 << log_value.sigma_bias_gyro_y << " " << log_value.sigma_bias_gyro_z << " " << log_value.visual_updated << " "
                 << log_value.ZuptUpdated << " " << log_value.keyframe << " " << log_value.diff_px << " " << log_value.diff_py << " "
                 << log_value.diff_pz << " " << log_value.init_vnorm << " " << log_value.groundtruth_vnorm << std::endl;
            // last_log_value = log_value;  // backup current log value
        }
        else
        {
            std::cerr << "Can not save vio state to: " << filename_ << std::endl;
        }

        file.close();
    }

    // LogValueFull last_log_value;
};

}  // namespace utils

#endif