#include "Pose.h"
#include "mathematical_tools.h"
#include "sfm.h"
#include "gtest/gtest.h"

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
    Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
    Eigen::Vector3d p_finG = feature._pwf;
    Eigen::Vector3d p_CinG = camera_pose.p();
    Eigen::Matrix3d R_CtoG = camera_pose.quat().toRotationMatrix();
    Eigen::Vector3d pcf = R_CtoG.transpose() * (p_finG - p_CinG);
    // std::cout << "Camera Pose: " << camera_pose.R().transpose() << ", Position: " << camera_pose.p().transpose() << std::endl;

    CameraObs obs;
    obs.ts_sec = camera_pose.ts();
    obs.feat_id = feature._id;
    const double u_norm = pcf(0) / pcf(2);
    const double v_norm = pcf(1) / pcf(2);
    Eigen::Vector3d uv_norm(u_norm, v_norm, 1.0);
    Eigen::Vector3d uv = K * uv_norm;
    // std::cout << "Camera Intrinsics: " << K << std::endl;
    // std::cout << "Feature ID: " << feature._id << ", UV: " << uv.transpose() << std::endl;

    obs.uv[LEFT_CAM] << uv.x(), uv.y();
    obs.uv_norm[LEFT_CAM] << u_norm, v_norm;

    feature._visual_obs_buffer.emplace(obs.ts_sec, obs);

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
        cv::circle(image, cv::Point(obs.uv.at(LEFT_CAM).x(), obs.uv.at(LEFT_CAM).y()), 3, cv::Scalar(0, 255, 0), -1);
    }
    cv::imshow("Observations", image);
    cv::waitKey(0);
}

TEST(VioTest, Sfm)
{
    Parameter params;
    params.camera_num = 1;
    Eigen::Matrix3d K;
    K << focal_x, 0, canvas_size_u / 2, 0, focal_y, canvas_size_v / 2, 0, 0, 1;
    params.img_width = canvas_size_u;
    params.img_height = canvas_size_v;
    Eigen::Matrix3d Ric = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Tic = Eigen::Vector3d::Zero();
    Eigen::VectorXd D = Eigen::VectorXd::Zero(5); // Assuming no distortion for simplicity
    params.intrinsics.push_back(K);
    params.distortion.push_back(D);
    params.Ric.push_back(Ric);
    params.tic.push_back(Tic);

    const double center_distance = 10.0f;
    CamModel::getInstance().Init(params);
    std::vector<Feature> features = CreateFeaturePoints(center_distance);
    std::vector<Pose> camera_poses = CreateCameraPoses(center_distance);
    std::map<double, std::vector<CameraObs>> camera_observations;
    Sfm sfm_solver(params);

    for (const auto& camera_pose : camera_poses)
    {
        std::vector<CameraObs> observations;
        CreateObservations(camera_pose, features, observations);
        camera_observations.emplace(camera_pose.ts(), observations);
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
        Eigen::Vector3d rpy_relative = utils::math::R2rpy(R_relative);
        EXPECT_NEAR(rpy_relative.norm(), 0.0, 1e-2);
    }
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
