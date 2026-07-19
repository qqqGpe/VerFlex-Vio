// Numerical self-check: verifies Fisheye::compute_distort_jacobian matches the
// finite-difference derivative of Fisheye::project_distort. Since both use the
// same KB formula, the analytic Jacobian and the numerical gradient should
// agree to ~1e-5. Exits non-zero if the max error exceeds the threshold.
#include <cmath>
#include <cstdio>
#include <Eigen/Core>

#include "camera_model.h"

namespace
{
double abs_max(double a, double b, double c, double d)
{
    double m = a > b ? a : b;
    m = m > c ? m : c;
    return m > d ? m : d;
}
}  // namespace

int main()
{
    Parameter params;
    params.camera_num = 1;
    params.img_width = 752;
    params.img_height = 480;
    params.camera_model = "fisheye";
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = 380.8;
    K(1, 1) = 380.8;
    K(0, 2) = 510.3;
    K(1, 2) = 514.3;
    params.intrinsics.push_back(K);
    Eigen::VectorXd D(4);
    D << 0.010171, -0.010816, 0.005943, -0.001663;  // TUM-VI-ish KB params
    params.distortion.push_back(D);
    params.Ric.push_back(Eigen::Matrix3d::Identity());
    params.tic.push_back(Eigen::Vector3d::Zero());

    CamModel::Init(params);
    CamModel& cam = CamModel::getInstance();

    const double h = 1e-6;
    double max_err = 0.0;
    // A spread of normalized coordinates (avoid the exact r=0 singularity).
    const double pts[][2] = {{0.10, 0.20}, {-0.30, 0.10}, {0.50, -0.40}, {-0.20, -0.20},
                             {0.40, 0.40},  {0.01, 0.01},  {-0.45, 0.35}};
    const int n = sizeof(pts) / sizeof(pts[0]);

    for (int i = 0; i < n; i++)
    {
        double x = pts[i][0], y = pts[i][1];
        Eigen::Vector2d u0 = cam.project_distort(0, Eigen::Vector3d(x, y, 1.0));
        Eigen::Vector2d ux = cam.project_distort(0, Eigen::Vector3d(x + h, y, 1.0));
        Eigen::Vector2d uy = cam.project_distort(0, Eigen::Vector3d(x, y + h, 1.0));

        Eigen::MatrixXd H(2, 2);
        cam.compute_distort_jacobian(0, Eigen::Vector2d(x, y), H);

        double num00 = (ux(0) - u0(0)) / h, num01 = (uy(0) - u0(0)) / h;
        double num10 = (ux(1) - u0(1)) / h, num11 = (uy(1) - u0(1)) / h;
        double e = abs_max(std::abs(num00 - H(0, 0)), std::abs(num01 - H(0, 1)),
                           std::abs(num10 - H(1, 0)), std::abs(num11 - H(1, 1)));
        max_err = e > max_err ? e : max_err;
        printf("(%.2f,%.2f) num=[%9.4f %9.4f; %9.4f %9.4f]  ana=[%9.4f %9.4f; %9.4f %9.4f]  err=%.2e\n",
               x, y, num00, num01, num10, num11, H(0, 0), H(0, 1), H(1, 0), H(1, 1), e);
    }
    printf("max abs error: %.3e\n", max_err);
    return (max_err < 1e-3) ? 0 : 1;
}
