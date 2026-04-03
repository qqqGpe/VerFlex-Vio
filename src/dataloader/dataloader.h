#ifndef __DATALOADER_H__
#define __DATALOADER_H__

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "sensorType.h"

struct SensorData
{
    enum Type
    {
        IMU,
        IMAGE,
        GROUND_TRUTH
    };

    Type type;
    double timestamp = 0.0;

    virtual ~SensorData() = default;

protected:
    SensorData(Type t, double ts) : type(t), timestamp(ts) {}
};

struct ImuSensorData : public SensorData
{
    ImuData imu;
    ImuSensorData(double ts, const ImuData& d) : SensorData(IMU, ts), imu(d) {}
};

struct ImageSensorData : public SensorData
{
    int cam_id = 0;
    cv::Mat image;
    std::string filepath;
    ImageSensorData(double ts, int cid, const cv::Mat& img) : SensorData(IMAGE, ts), cam_id(cid), image(img) {}
    ImageSensorData(double ts, int cid, const std::string& path) : SensorData(IMAGE, ts), cam_id(cid), filepath(path) {}
};

struct GroundTruthSensorData : public SensorData
{
    GroundTruth gt;
    GroundTruthSensorData(double ts, const GroundTruth& g) : SensorData(GROUND_TRUTH, ts), gt(g) {}
};

using SensorDataPtr = std::shared_ptr<SensorData>;

class DataLoader
{
public:
    virtual ~DataLoader() = default;

    virtual bool open(const std::string& dataset_path, int camera_num = 2) = 0;

    virtual bool hasNext() const = 0;

    virtual SensorDataPtr next() = 0;

    virtual double firstTimestamp() const = 0;
};

#endif
