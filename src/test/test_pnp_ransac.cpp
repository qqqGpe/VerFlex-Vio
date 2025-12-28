#include "visualManager.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ctime>
#include <iostream>
#include <vector>

#include "Pose.h"
#include "solver.h"

#define DEG2RAD M_PI / 180
#define POINT_NUM_N 20
#define SCALE 20
#define IMAGE_WIDTH 752
#define IMAGE_HEIGHT 480

using namespace Eigen;
using namespace std;

Matrix3d K;
MatrixXd pts_g; // 3d points in world frame
std::vector<double> camera_ts = { 1.0, 2.0, 3.0, 4.0, 5.0 };
std::vector<double> intrinsic = { 458.654, 457.296, 367.215, 248.375 }; // fu, fv, cu, cv
double baseline = 1.0;
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
            CameraObs obs(uv.x(), uv.y(), p_norm.x(), p_norm.y());
            feat->_visual_obs_buffer.insert(make_pair(camera_ts[j], obs));
            feat->_valid = true;
        }
    }
}

int main()
{

    Param params;
    params.camera_num = 1;
    Eigen::Matrix3d K;
    params.img_width = IMAGE_WIDTH;
    params.img_height = IMAGE_HEIGHT;
    params.max_feat_n = POINT_NUM_N;
    K << intrinsic[0], 0, intrinsic[2], 0, intrinsic[1], intrinsic[3], 0, 0, 1;
    Eigen::Matrix3d Ric = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Tic = Eigen::Vector3d::Zero();
    Eigen::VectorXd D = Eigen::VectorXd::Zero(5); // Assuming no distortion for simplicity
    params.intrinsics.push_back(K);
    params.distortion.push_back(D);
    params.Ric.push_back(Ric);
    params.tic.push_back(Tic);

    CamModel::getInstance().Init(params);
    srand((unsigned)time(NULL));
    pts_g = MatrixXd::Random(3, POINT_NUM_N).array().abs();
    pts_g.block<2, POINT_NUM_N>(0, 0) *= 1;
    pts_g.block<1, POINT_NUM_N>(2, 0) *= 5;
    std::cout << "generated pwf: \n" << std::endl;
    std::cout << pts_g.transpose() << std::endl;

    std::vector<double> distortion;     // empty distortion coeff for debugging
    std::shared_ptr<State> state = make_shared<State>(Param());
    std::shared_ptr<ros::NodeHandle> nh = nullptr;
    VisualManager visual_manager(nh, params, state, nullptr);

    generate_camera_pose();
    project_to_camera();
    visual_manager.FeatureTriangulation(camera_pose_buffer, feats);

    std::cout << "\n feature triangulated: \n" << std::endl;
    for (auto x : feats) {
        if(x->_valid && x->_is_triangulated) {
            std::cout << x->_pwf.transpose() << std::endl;
        }
    }

    state->set_ts_sec(camera_ts.back());
    visual_manager.set_state(state);
    visual_manager.PnpRansacToRejectOutliers(feats);

    Eigen::Matrix3d S;
    Eigen::Matrix3d mat;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    S.triangularView<Eigen::Upper>() = mat;
    std::cout << "S: \n" << S << std::endl;
    Eigen::Matrix3d x = S.selfadjointView<Eigen::Upper>();
    std::cout << "S_adj: \n" << x;

    return 0;
}
