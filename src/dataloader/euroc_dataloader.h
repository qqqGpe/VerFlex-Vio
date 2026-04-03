#ifndef __EUROC_DATALOADER_H__
#define __EUROC_DATALOADER_H__

#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <glog/logging.h>

class EurocDataLoader : public DataLoader
{
public:
    bool open(const std::string& dataset_path, int camera_num = 2) override
    {
        dataset_path_ = dataset_path;
        camera_num_ = camera_num;
        timeline_.clear();
        index_ = 0;

        loadImu(dataset_path + "/mav0/imu0/data.csv");
        loadCamera(0, dataset_path + "/mav0/cam0/data.csv", dataset_path + "/mav0/cam0/data");
        if (camera_num_ >= 2)
        {
            loadCamera(1, dataset_path + "/mav0/cam1/data.csv", dataset_path + "/mav0/cam1/data");
        }
        loadGroundTruth(dataset_path + "/mav0/state_groundtruth_estimate0/data.csv");

        std::sort(timeline_.begin(), timeline_.end(),
                  [](const SensorDataPtr& a, const SensorDataPtr& b) { return a->timestamp < b->timestamp; });

        LOG(INFO) << "EurocDataLoader: loaded " << timeline_.size() << " entries from " << dataset_path;
        return !timeline_.empty();
    }

    bool hasNext() const override { return index_ < timeline_.size(); }

    SensorDataPtr next() override { return timeline_[index_++]; }

    double firstTimestamp() const override
    {
        return timeline_.empty() ? 0.0 : timeline_.front()->timestamp;
    }

private:
    void loadImu(const std::string& csv_path)
    {
        std::ifstream file(csv_path);
        if (!file.is_open())
        {
            LOG(ERROR) << "Cannot open IMU file: " << csv_path;
            return;
        }

        std::string line;
        std::getline(file, line); // skip header
        int count = 0;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#') continue;
            auto values = parseCsvLine(line);
            if (values.size() >= 7)
            {
                ImuData imu;
                imu.ts_sec = values[0] * 1e-9;
                imu.wm << values[1], values[2], values[3];
                imu.am << values[4], values[5], values[6];
                timeline_.push_back(std::make_shared<ImuSensorData>(imu.ts_sec, imu));
                count++;
            }
        }
        LOG(INFO) << "Loaded " << count << " IMU measurements";
    }

    void loadCamera(int cam_id, const std::string& csv_path, const std::string& image_dir)
    {
        std::ifstream file(csv_path);
        if (!file.is_open())
        {
            LOG(ERROR) << "Cannot open camera file: " << csv_path;
            return;
        }

        std::string line;
        std::getline(file, line); // skip header
        int count = 0;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string ts_str, filename;
            std::getline(ss, ts_str, ',');
            std::getline(ss, filename, ',');
            filename.erase(0, filename.find_first_not_of(" \t"));
            filename.erase(filename.find_last_not_of(" \t\r\n") + 1);

            double ts = std::stod(ts_str) * 1e-9;
            std::string filepath = image_dir + "/" + filename;
            timeline_.push_back(std::make_shared<ImageSensorData>(ts, cam_id, filepath));
            count++;
        }
        LOG(INFO) << "Loaded " << count << " cam" << cam_id << " entries";
    }

    void loadGroundTruth(const std::string& csv_path)
    {
        std::ifstream file(csv_path);
        if (!file.is_open())
        {
            LOG(WARNING) << "No ground truth file: " << csv_path;
            return;
        }

        std::string line;
        std::getline(file, line); // skip header
        int count = 0;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#') continue;
            auto values = parseCsvLine(line);
            if (values.size() >= 11)
            {
                GroundTruth gt;
                gt.ts_sec = values[0] * 1e-9;
                gt.p_ << values[1], values[2], values[3];
                gt.v_ << values[8], values[9], values[10];
                timeline_.push_back(std::make_shared<GroundTruthSensorData>(gt.ts_sec, gt));
                count++;
            }
        }
        LOG(INFO) << "Loaded " << count << " ground truth entries";
    }

    std::vector<double> parseCsvLine(const std::string& line)
    {
        std::vector<double> values;
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            values.push_back(std::stod(token));
        }
        return values;
    }

    std::string dataset_path_;
    int camera_num_ = 2;
    size_t index_ = 0;
    std::vector<SensorDataPtr> timeline_;
};

#endif
