#include <glog/logging.h>
#include <fmt/format.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <map>

#include "camModel.h"
#include "parameter.h"
#include "vioManager.h"
#include "euroc_dataloader.h"

namespace fs = std::filesystem;
namespace
{
constexpr double kStereoTimeTolerance = 0.02; // 20ms
}

void printUsage(const char* prog)
{
    std::cerr << "Usage: " << prog << " --config <config.json> [--dataset <euroc_dataset_dir>]" << std::endl;
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(*argv);

    std::string config_path;
    std::string dataset_dir;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
            config_path = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc)
            dataset_dir = argv[++i];
        else if (arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (config_path.empty())
    {
        printUsage(argv[0]);
        return -1;
    }

    // Load parameters
    Param params;
    if (!params.load_from_json(config_path))
    {
        LOG(ERROR) << "Failed to load config: " << config_path;
        return -1;
    }

    if (!dataset_dir.empty())
    {
        params.dataset_dir = dataset_dir;
        params.bag_name = fs::path(dataset_dir).filename().string();
    }

    if (params.dataset_dir.empty())
    {
        LOG(ERROR) << "dataset_dir must be specified in config or via --dataset";
        return -1;
    }

    if (!params.log_path.empty())
    {
        fs::create_directories(params.log_path);
    }

    fLI::FLAGS_stderrthreshold = params.log_level;

    // Initialize camera model
    CamModel::getInstance().Init(params);

    // Initialize VIO manager
    auto vio_manager = std::make_shared<VioManager>(params);

    // Load dataset
    auto loader = std::make_unique<EurocDataLoader>();
    if (!loader->open(params.dataset_dir, params.camera_num))
    {
        LOG(ERROR) << "Failed to open dataset: " << params.dataset_dir;
        return -1;
    }

    // Set initial timestamp
    if (params.set_init_timestamp_to_zero)
    {
        vio_manager->SetInitialTimeStamp(loader->firstTimestamp());
    }

    // Pending images buffer for stereo synchronization: cam_id -> (timestamp, image)
    std::map<int, std::shared_ptr<ImageSensorData>> pending_images;

    // Main loop
    while (loader->hasNext())
    {
        auto data = loader->next();

        switch (data->type)
        {
            case SensorData::IMU:
            {
                auto imu = std::static_pointer_cast<ImuSensorData>(data);
                vio_manager->FeedImuData(imu->imu);
                break;
            }
            case SensorData::GROUND_TRUTH:
            {
                auto gt = std::static_pointer_cast<GroundTruthSensorData>(data);
                vio_manager->FeedGroundTruth(gt->gt);
                break;
            }
            case SensorData::IMAGE:
            {
                auto img_data = std::static_pointer_cast<ImageSensorData>(data);

                // Lazy load image from filepath
                if (img_data->image.empty() && !img_data->filepath.empty())
                {
                    img_data->image = cv::imread(img_data->filepath, cv::IMREAD_GRAYSCALE);
                    if (img_data->image.empty())
                    {
                        LOG(WARNING) << "Cannot load image: " << img_data->filepath;
                        break;
                    }
                }

                if (params.camera_num == 1)
                {
                    // Mono: feed directly
                    vio_manager->FeedImageData(img_data->timestamp, {img_data->image});
                    vio_manager->ProcessMeasurementOnce();
                }
                else
                {
                    // Stereo: synchronize left/right
                    int cam_id = img_data->cam_id;
                    int other_id = (cam_id == 0) ? 1 : 0;

                    pending_images[cam_id] = img_data;

                    // Check if both cameras have pending images with matching timestamps
                    if (pending_images.count(other_id))
                    {
                        double dt = std::abs(pending_images[0]->timestamp - pending_images[1]->timestamp);
                        if (dt < kStereoTimeTolerance)
                        {
                            double ts = pending_images[0]->timestamp;
                            std::vector<cv::Mat> images = {pending_images[0]->image, pending_images[1]->image};
                            pending_images.clear();

                            vio_manager->FeedImageData(ts, images);
                            vio_manager->ProcessMeasurementOnce();
                        }
                    }
                }
                break;
            }
        }
    }

    LOG(INFO) << "VIO offline processing completed.";
    google::ShutdownGoogleLogging();
    return 0;
}
