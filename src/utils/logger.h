#ifndef __LOGGER__
#define __LOGGER__
#include <glog/logging.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace utils
{
// Macro to declare a field and register its name
// Defined outside structs to enable reuse across different logging structures.
// The inner struct's constructor is templated to accept any class pointer type (T*),
// allowing it to work with LogValueTUM, LogValueFull, or any other class that has
// a 'value_name' member (e.g., std::vector<std::string>).
#define LOG_FIELD(type, name, default_val) \
    type name = default_val;               \
    struct _reg_##name {                   \
        template <typename T>              \
        _reg_##name(T* self) { self->value_name.push_back(#name); } \
    } _reg_inst_##name{this}

/**
 * @brief TUM format data structure for logging
 * Simple struct with position and quaternion data
 */
struct LogValueTUM
{
    std::vector<std::string> value_name = {"#timestamp"};
    double timestamp = 0;
    LOG_FIELD(double, px, 0);
    LOG_FIELD(double, py, 0);
    LOG_FIELD(double, pz, 0);
    LOG_FIELD(double, qx, 0);
    LOG_FIELD(double, qy, 0);
    LOG_FIELD(double, qz, 0);
    LOG_FIELD(double, qw, 1.0);
};

/**
 * @brief Full format data structure for logging
 * Contains complete state information including biases and uncertainties
 */
struct LogValueFull
{
    std::vector<std::string> value_name = {"#timestamp"};

    double timestamp = 0;
    // Position, velocity, orientation
    LOG_FIELD(double, px, 0);
    LOG_FIELD(double, py, 0);
    LOG_FIELD(double, pz, 0);
    LOG_FIELD(double, vx, 0);
    LOG_FIELD(double, vy, 0);
    LOG_FIELD(double, vz, 0);
    LOG_FIELD(double, roll, 0);
    LOG_FIELD(double, pitch, 0);
    LOG_FIELD(double, yaw, 0);

    // IMU biases
    LOG_FIELD(double, bias_acc_x, 0);
    LOG_FIELD(double, bias_acc_y, 0);
    LOG_FIELD(double, bias_acc_z, 0);
    LOG_FIELD(double, bias_gyro_x, 0);
    LOG_FIELD(double, bias_gyro_y, 0);
    LOG_FIELD(double, bias_gyro_z, 0);

    // Uncertainties (1-sigma)
    LOG_FIELD(double, sigma_px, 0);
    LOG_FIELD(double, sigma_py, 0);
    LOG_FIELD(double, sigma_pz, 0);
    LOG_FIELD(double, sigma_vx, 0);
    LOG_FIELD(double, sigma_vy, 0);
    LOG_FIELD(double, sigma_vz, 0);
    LOG_FIELD(double, sigma_roll, 0);
    LOG_FIELD(double, sigma_pitch, 0);
    LOG_FIELD(double, sigma_yaw, 0);
    LOG_FIELD(double, sigma_bias_acc_x, 0);
    LOG_FIELD(double, sigma_bias_acc_y, 0);
    LOG_FIELD(double, sigma_bias_acc_z, 0);
    LOG_FIELD(double, sigma_bias_gyro_x, 0);
    LOG_FIELD(double, sigma_bias_gyro_y, 0);
    LOG_FIELD(double, sigma_bias_gyro_z, 0);

    // Update flags
    LOG_FIELD(int, visual_updated, 0);
    LOG_FIELD(int, zupt_updated, 0);
    LOG_FIELD(int, keyframe, 0);

    // Position difference & velocity norms
    LOG_FIELD(double, diff_px, 0);
    LOG_FIELD(double, diff_py, 0);
    LOG_FIELD(double, diff_pz, 0);
    LOG_FIELD(double, init_vnorm, 0);
    LOG_FIELD(double, groundtruth_vnorm, 0);

#undef LOG_FIELD
};


/**
 * @brief Thread-safe multi-instance singleton Logger
 *
 * Features:
 * - Multi-instance singleton by filename (GetInstance returns same instance for same filename)
 * - Thread-safe writes using mutex
 * - Supports both TUM and Full format (auto-detected by SaveValues overload)
 * - Writes header on first write
 * - Keeps file stream open for performance
 */
class Logger
{
   public:
    /**
     * @brief Get singleton instance for a specific filename
     * @param filename Full path to log file
     * @return Pointer to Logger instance, nullptr on failure
     */
    static Logger* GetInstance(const std::string& filename);

    /**
     * @brief Save a TUM format value to the log file
     * @param value The value to log
     * @return true on success, false on failure
     */
    bool SaveValues(const LogValueTUM& value);

    /**
     * @brief Save a Full format value to the log file
     * @param value The value to log
     * @return true on success, false on failure
     */
    bool SaveValues(const LogValueFull& value);

    /**
     * @brief Close the log file and cleanup
     */
    void Close();

    /**
     * @brief Destructor - public for unique_ptr compatibility
     */
    ~Logger();

   private:
    Logger(const std::string& filename);

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    template <typename T> void WriteHeader()
    {
        T temp;
        for (size_t i = 0; i < temp.value_name.size(); ++i)
        {
            if (i > 0)
                file_ << " ";
            file_ << temp.value_name[i];
        }
        file_ << "\n";
    }

    std::string filename_;
    std::ofstream file_;
    bool header_written_;
    bool is_tum_format_;  // true = TUM, false = Full
    std::mutex file_mutex_;

    // Static members for multi-instance singleton
    static std::map<std::string, std::unique_ptr<Logger>> instances_;
    static std::mutex instances_mutex_;
};

}  // namespace utils

#endif