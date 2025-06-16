#include "core/sfm.h"
#include "gtest/gtest.h"
#include "types/Pose.h"
#include "utils/mathematical_tools.h"

#define DEG2RAD M_PI / 180
#define RAD2DEG 180 / M_PI

namespace
{
    constexpr uint32_t kCameraPoses = 10;
    constexpr double kSectorAngle = 72; // degrees
    constexpr double kSquareLength = 10.0f;
    constexpr uint32_t kFeatureNumForEachSqaureSide = 10;
    constexpr uint32_t kAllFeatureNum = kFeatureNumForEachSqaureSide * kFeatureNumForEachSqaureSide;
    constexpr double focal_x = 300;
    constexpr double focal_y = 300;
    constexpr double canvas_size_u = 640;
    constexpr double canvas_size_v = 480;
}

double generateRandomSmallDistance() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-3, 3);
    return dis(gen);
}

// Create a grid of feature points in a square area
// centered at the given center point
// The grid is kFeatureNumForEachSqaureSide x kFeatureNumForEachSqaureSide
// and each square has a length of kSquareLength
std::vector<Feature> CreateFeaturePoints(const double center_z)
{
    static uint32_t feature_id = 0;
    std::vector<Feature> features;
    features.reserve(kAllFeatureNum);
    const double stride = kSquareLength / (kFeatureNumForEachSqaureSide - 1);
    for (uint32_t i = 0; i < kFeatureNumForEachSqaureSide; ++i)
    {
        for (uint32_t j = 0; j < kFeatureNumForEachSqaureSide; ++j)
        {
            Feature feature;
            double x = -kSquareLength / 2 + i * stride;
            double y = -kSquareLength / 2 + j * stride;
            feature._pwf << x, y, center_z + generateRandomSmallDistance();
            feature._id = feature_id++;
            features.push_back(feature);
        }
    }
    return features;
}

std::vector<Pose> CreateCameraPoses(const double sector_r)
{
    static double timestamp = 0.1;
    std::vector<Pose> camera_poses;
    camera_poses.reserve(kCameraPoses);
    const double angle_stride = kSectorAngle / (kCameraPoses - 1);
    for (uint32_t i = 0; i < kCameraPoses; ++i)
    {
        double angle = (0 - i * angle_stride) * DEG2RAD;
        Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Vector3d::UnitY()).toRotationMatrix();
        double x = sector_r * sin(-angle);
        double y = 0.f;
        double z = sector_r - sector_r * cos(angle);
        Eigen::Vector3d t(x, y, z);

        Pose pose;
        pose.set_pose(R, t);
        pose.set_ts(timestamp);
        timestamp += 0.1;
        camera_poses.push_back(pose);
    }
    return camera_poses;
}

CameraObs ProjectToCamera(const Pose &camera_pose, Feature &feature)
{
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = focal_x;
    K(1, 1) = focal_y;
    K(0, 2) = canvas_size_u / 2;
    K(1, 2) = canvas_size_v / 2;

    Eigen::Vector3d p_finG = feature._pwf;
    Eigen::Vector3d p_CinG = camera_pose.p();
    Eigen::Matrix3d R_CtoG = camera_pose.quat().toRotationMatrix();
    Eigen::Vector3d pcf = R_CtoG.transpose() * (p_finG - p_CinG);

    CameraObs obs;
    obs.feat_id = feature._id;
    double u_norm = pcf(0) / pcf(2);
    double v_norm = pcf(1) / pcf(2);
    Eigen::Vector3d uv_norm(u_norm, v_norm, 1.0);
    Eigen::Vector3d uv = K * uv_norm;
    obs.u_norm = u_norm;
    obs.v_norm = v_norm;
    obs.u = uv.x();
    obs.v = uv.y();
    obs.ts_sec = camera_pose.ts();
    feature._visual_obs_buffer.insert({obs.ts_sec, obs});

    return obs;
}

void CreateObservations(const Pose &camera_pose,
                        std::vector<Feature> &features,
                        std::vector<CameraObs> &observations)
{
    for (auto& feature : features)
    {
        CameraObs obs = ProjectToCamera(camera_pose, feature);
        observations.push_back(obs);
    }
}

void ShowObservations(const std::vector<CameraObs> &observations)
{
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    for (const auto& obs : observations)
    {
        cv::circle(image, cv::Point(obs.u, obs.v), 3, cv::Scalar(0, 255, 0), -1);
    }
    cv::imshow("Observations", image);
    cv::waitKey(0);
}

TEST(VioTest, Sfm)
{
    std::shared_ptr<CameraModel> camera_model_ptr = std::make_shared<CameraModel>();
    camera_model_ptr->SetCameraIntrinsicMatrix({focal_x, focal_y, canvas_size_u / 2, canvas_size_v / 2});

    const double center_distance = 10.0f;
    std::vector<Feature> features = CreateFeaturePoints(center_distance);
    std::vector<Pose> camera_poses = CreateCameraPoses(center_distance);
    std::map<double, std::vector<CameraObs>> camera_observations;
    Sfm sfm_solver(Param(), camera_model_ptr);

    for (const auto& camera_pose : camera_poses)
    {
        std::vector<CameraObs> observations;
        CreateObservations(camera_pose, features, observations);
        // ShowObservations(observations);
        camera_observations.insert({camera_pose.ts(), observations});
        sfm_solver.MaybeAddSfmKeyframes(std::make_pair(camera_pose.ts(), observations), std::nullopt);
        if (sfm_solver.isReady())
        {
            break;
        }
    }

    EXPECT_TRUE(sfm_solver.Optimization());
    std::map<double, Pose> estimate_poses = sfm_solver.getSfmPoses();
    for (uint32_t i = 0; i < Sfm::kRequiredKeyframesForSfm; i++)
    {
        Pose pose_gt = camera_poses[i];
        Pose pose_est = estimate_poses.at(pose_gt.ts());
        Eigen::Matrix3d R_relative = pose_gt.R().transpose() * pose_est.R();
        Eigen::Vector3d rpy_relative = MathUtils::R2rpy(R_relative);
        EXPECT_NEAR(rpy_relative.norm(), 0.0, 1e-2);
    }
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
