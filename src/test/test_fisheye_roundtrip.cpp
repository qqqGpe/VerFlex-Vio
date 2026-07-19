// Round-trip consistency: back_project (cv::fisheye::undistortPoints) must be the
// exact inverse of project_distort (hand-written KB). For a pixel uv:
//   uv -> back_project -> uv_norm -> project_distort -> uv'  should satisfy uv'~=uv.
// A large error (esp. at edges) means cv::fisheye and our KB disagree -> the
// uv_norm fed to triangulation is biased, explaining the "paf_opt z negative" surge.
#include <cmath>
#include <cstdio>
#include <Eigen/Core>

#include "camera_model.h"
#include "sensorType.h"

int main()
{
    Parameter params;
    params.camera_num = 1;
    params.img_width = 1024;
    params.img_height = 1024;
    params.camera_model = "fisheye";
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 380.8;
    K(1, 1) = 380.8;
    K(0, 2) = 510.3;
    K(1, 2) = 514.3;
    params.intrinsics.push_back(K);
    Eigen::VectorXd D(4);
    D << 0.010171, -0.010816, 0.005943, -0.001663;  // TUM-VI-ish
    params.distortion.push_back(D);
    params.Ric.push_back(Eigen::Matrix3d::Identity());
    params.tic.push_back(Eigen::Vector3d::Zero());

    CamModel::Init(params);
    CamModel& cam = CamModel::getInstance();

    // Pixels from center to extreme edge of a 1024x1024 image.
    double pts[][2] = {{511, 511}, {400, 400}, {700, 600}, {200, 200}, {900, 100},
                       {50, 50},   {970, 970}, {10, 500},  {1010, 500}, {500, 10}};
    const int n = sizeof(pts) / sizeof(pts[0]);
    double max_err = 0.0;
    for (int i = 0; i < n; i++)
    {
        double u = pts[i][0], v = pts[i][1];
        CameraObs obs;
        obs.uv[0] = Eigen::Vector2d(u, v);
        cam.back_project_undistort(obs);
        Eigen::Vector2d uv_norm = obs.uv_norm[0];
        Eigen::Vector3d p3d(uv_norm.x(), uv_norm.y(), 1.0);
        Eigen::Vector2d uv_re = cam.project_distort(0, p3d);
        double err = (uv_re - Eigen::Vector2d(u, v)).norm();
        max_err = err > max_err ? err : max_err;
        printf("uv=(%4.0f,%4.0f)  norm=(%+.4f,%+.4f)  reproj=(%7.2f,%7.2f)  err=%6.3f px\n",
               u, v, uv_norm.x(), uv_norm.y(), uv_re.x(), uv_re.y(), err);
    }
    printf("max round-trip error: %.4f px\n", max_err);
    return (max_err < 0.5) ? 0 : 1;
}
