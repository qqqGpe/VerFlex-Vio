#include "core/visualManager.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ctime>
#include <iostream>
#include <vector>

#include "Pose.h"

#define DEG2RAD M_PI / 180
#define POINT_NUM_N 10
#define SCALE 20
#define IMAGE_WIDTH 752
#define IMAGE_HEIGHT 480

using namespace Eigen;
using namespace std;

Matrix3d K;
MatrixXd pts_g; // 3d points in world frame
double camera_ts[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
double intrinsic[] = { 458.654, 457.296, 367.215, 248.375 }; // fu, fv, cu, cv
double baseline = 0.5;
std::map<double, CameraPose> camera_pose_buffer;
std::vector<Feature*> feats;

bool OOB(Eigen::Vector3d uv)
{
    return uv.x() < 10 || uv.x() > IMAGE_WIDTH - 10 || uv.y() < 10 || uv.y() > IMAGE_HEIGHT - 10;
}

void generate_camera_pose()
{
    CameraPose pose_0;
    pose_0.Rwc = Eigen::Matrix3d::Identity();
    pose_0.pwc = Eigen::Vector3d::Zero();
    camera_pose_buffer.insert(make_pair(camera_ts[0], pose_0));

    CameraPose pose_1;
    pose_1.Rwc = AngleAxisd(-45 * DEG2RAD, Eigen::Vector3d::UnitY()).toRotationMatrix();
    pose_1.pwc = Eigen::Vector3d(baseline, 0, 0);
    camera_pose_buffer.insert(make_pair(camera_ts[1], pose_1));

    CameraPose pose_2;
    pose_2.Rwc = AngleAxisd(45 * DEG2RAD, Eigen::Vector3d::UnitY()).toRotationMatrix();
    pose_2.pwc = Eigen::Vector3d(-baseline, 0, 0);
    camera_pose_buffer.insert(make_pair(camera_ts[2], pose_2));

    CameraPose pose_3;
    pose_3.Rwc = AngleAxisd(45 * DEG2RAD, Eigen::Vector3d::UnitX()).toRotationMatrix();
    pose_3.pwc = Eigen::Vector3d(0, baseline, 0);
    camera_pose_buffer.insert(make_pair(camera_ts[3], pose_3));

    CameraPose pose_4;
    pose_4.Rwc = AngleAxisd(-45 * DEG2RAD, Eigen::Vector3d::UnitX()).toRotationMatrix();
    pose_4.pwc = Eigen::Vector3d(0, -baseline, 0);
    camera_pose_buffer.insert(make_pair(camera_ts[4], pose_4));
}

Eigen::Vector3d back_project(Eigen::Vector3d &uv)
{
    uv.head(2) = uv.head(2).eval() + 2 * Vector2d::Random();
    return K.inverse() * uv;
}

void project_to_camera()
{
    for (int i = 0; i < POINT_NUM_N; i++) {
        Eigen::Vector3d p3d = pts_g.block<3, 1>(0, i).transpose();
        Feature* feat = new Feature();
        feats.push_back(feat);
        feat->_pwf = p3d;
        for (int j = 0; j < camera_pose_buffer.size(); j++) {
            auto pose = camera_pose_buffer.at(camera_ts[j]);
            Eigen::Matrix3d R_wc = pose.Rwc;
            Eigen::Vector3d p_wc = pose.pwc;
            if (p_wc.z() < 0) {
                continue;
            }
            Eigen::Vector3d p_inC = R_wc.transpose() * (p3d - p_wc);
            Eigen::Vector3d p_norm = p_inC / p_inC.z();
            Eigen::Vector3d uv = K * p_norm;
            p_norm = back_project(uv);
            // if (OOB(uv)) {
            //     continue;
            // }
            cam_obs_t obs = { uv.x(), uv.y(), p_norm.x(), p_norm.y() };
            feat->_visual_obs_buffer.insert(make_pair(camera_ts[j], obs));
            feat->_valid = true;
        }
    }
}

int main()
{
    K << intrinsic[0], 0, intrinsic[2],
        0, intrinsic[1], intrinsic[3],
        0, 0, 1;
    srand((unsigned)time(NULL));
    pts_g = MatrixXd::Random(3, POINT_NUM_N).array().abs();
    pts_g.block<2, POINT_NUM_N>(0, 0) *= 1;
    pts_g.block<1, POINT_NUM_N>(2, 0) *= 5;
    std::cout << "generated pwf: \n"
              << std::endl;
    std::cout << pts_g << std::endl;

    VisualManager visual_manager;
    generate_camera_pose();
    project_to_camera();
    std::shared_ptr<State> state = make_shared<State>();

    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;
    int map_id = 0;
    // std::shared_ptr<Pose> imu_to_cam_extrinsic = make_shared<Pose>();
    Eigen::Quaterniond q_ic(1, 0, 0, 0);
    Eigen::Vector3d tic = Eigen::Vector3d::Zero();
    Eigen::VectorXd extrinsic = Eigen::VectorXd::Zero(7);
    extrinsic << q_ic.coeffs(), tic;   // 外参：单位旋转 + 无平移
    state->_imu_to_cam_extrinsic->set_value(extrinsic);
    map_hx.insert(make_pair(state->_imu_to_cam_extrinsic, map_id));
    map_id += state->_imu_to_cam_extrinsic->size();
    for (int i = 0; i < camera_pose_buffer.size(); i++)
    {
        std::shared_ptr<Pose> clone_pose = make_shared<Pose>();
        CameraPose pose = camera_pose_buffer.at(camera_ts[i]);
        Eigen::Quaterniond q_clone(pose.Rwc);
        Eigen::Vector3d p_clone(pose.pwc);
        Eigen::VectorXd pose_vec = Eigen::VectorXd::Zero(7);
        pose_vec << q_clone.coeffs(), p_clone;
        clone_pose->set_value(pose_vec);
        clone_pose->set_ts(camera_ts[i]);
        state->_clone_pose.insert(make_pair(camera_ts[i], clone_pose));
        map_hx.insert(make_pair(clone_pose, map_id));
        map_id += clone_pose->size();
    }
    int total_hx = map_id;

    visual_manager._state = state;
    for (int i = 0; i < feats.size(); i++)
    {
        visual_manager.get_single_feature_jacobian(feats[i], map_hx, total_hx);
    }

    return 0;
}
