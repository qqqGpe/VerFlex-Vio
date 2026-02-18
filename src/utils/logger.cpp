/**
 * @file logger.cpp
 * @brief Implementation of Logger singleton class
 */

#include "logger.h"

namespace utils
{

// Static member definitions
std::map<std::string, std::unique_ptr<Logger>> Logger::instances_;
std::mutex Logger::instances_mutex_;

Logger::Logger(const std::string& filename)
    : filename_(filename), header_written_(false), is_tum_format_(true)  // Default to TUM format
{
    file_.open(filename, std::ios::out);
    if (file_.is_open())
    {
        file_ << std::fixed;
        file_.precision(10);
    }
}

Logger::~Logger()
{
    Close();
}

Logger* Logger::GetInstance(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(instances_mutex_);

    auto it = instances_.find(filename);
    if (it != instances_.end())
    {
        return it->second.get();
    }

    // Create new instance
    Logger* logger = new Logger(filename);
    if (!logger->file_.is_open())
    {
        delete logger;
        return nullptr;
    }

    instances_[filename] = std::unique_ptr<Logger>(logger);
    return logger;
}


bool Logger::SaveValues(const LogValueTUM& value)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    if (!file_.is_open())
    {
        return false;
    }

    if (!header_written_)
    {
        is_tum_format_ = true;
        WriteHeader<LogValueTUM>();
        header_written_ = true;
    }

    file_ << value.timestamp << " "
          << value.px << " "
          << value.py << " "
          << value.pz << " "
          << value.qx << " "
          << value.qy << " "
          << value.qz << " "
          << value.qw << "\n";

    file_.flush();
    return true;
}

bool Logger::SaveValues(const LogValueFull& value)
{
    std::lock_guard<std::mutex> lock(file_mutex_);

    if (!file_.is_open())
    {
        return false;
    }

    if (!header_written_)
    {
        is_tum_format_ = false;
        WriteHeader<LogValueFull>();
        header_written_ = true;
    }

    file_ << value.timestamp << " "
          << value.px << " "
          << value.py << " "
          << value.pz << " "
          << value.vx << " "
          << value.vy << " "
          << value.vz << " "
          << value.roll << " "
          << value.pitch << " "
          << value.yaw << " "
          << value.bias_acc_x << " "
          << value.bias_acc_y << " "
          << value.bias_acc_z << " "
          << value.bias_gyro_x << " "
          << value.bias_gyro_y << " "
          << value.bias_gyro_z << " "
          << value.sigma_px << " "
          << value.sigma_py << " "
          << value.sigma_pz << " "
          << value.sigma_vx << " "
          << value.sigma_vy << " "
          << value.sigma_vz << " "
          << value.sigma_roll << " "
          << value.sigma_pitch << " "
          << value.sigma_yaw << " "
          << value.sigma_bias_acc_x << " "
          << value.sigma_bias_acc_y << " "
          << value.sigma_bias_acc_z << " "
          << value.sigma_bias_gyro_x << " "
          << value.sigma_bias_gyro_y << " "
          << value.sigma_bias_gyro_z << " "
          << value.visual_updated << " "
          << value.zupt_updated << " "
          << value.keyframe << " "
          << value.diff_px << " "
          << value.diff_py << " "
          << value.diff_pz << " "
          << value.init_vnorm << " "
          << value.groundtruth_vnorm << "\n";

    file_.flush();
    return true;
}

void Logger::Close()
{
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_.is_open())
    {
        file_.close();
    }
}

}  // namespace utils
