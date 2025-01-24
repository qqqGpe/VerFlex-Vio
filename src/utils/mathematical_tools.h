#ifndef __MATH_TOOLS__
#define __MATH_TOOLS__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Jacobi>

class mathematical {
public:
    inline static Eigen::Matrix<double, 3, 3> skew(const Eigen::Vector3d& w)
    {
        Eigen::Matrix<double, 3, 3> w_x;
        w_x << 0, -w(2), w(1),
            w(2), 0, -w(0),
            -w(1), w(0), 0;
        return w_x;
    }

    inline static Eigen::Matrix3d Rodrigues(Eigen::Vector3d vec, double theta)
    {
        Eigen::Matrix3d R;
        Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
        if (theta < 1e-6) {
            R = I3x3 + skew(vec * theta);
        } else {
            R = cos(theta) * I3x3 + (1 - cos(theta)) * vec * vec.transpose() + sin(theta) * skew(vec);
        }
        return R;
    }

    static void NullSpaceProjectInplace(Eigen::MatrixXd& Hfx, int cols)
    {
        // Apply the left nullspace of H_f to all variables
        // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
        // See page 252, Algorithm 5.2.4 for how these two loops work
        // They use "matlab" index notation, thus we need to subtract 1 from all index
        Eigen::JacobiRotation<double> tempHo_GR;
        for (int n = 0; n < cols; n++) {
            for (int m = Hfx.rows() - 1; m > n; m--) {
                // Givens matrix G
                tempHo_GR.makeGivens(Hfx(m - 1, n), Hfx(m, n));
                // Multiply G to the corresponding lines (m-1,m) in each matrix
                // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
                //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
                (Hfx.block(m - 1, 0, 2, Hfx.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            }
        }
    }

    inline static Eigen::Vector3d RotationMatrixToEulerAngles(const Eigen::Matrix3d& R)
    {
        // euler (Z-Y-X，i.e. RPY) make sure in range [-pi/2, pi/2]
        Eigen::Vector3d euler_angle;
        Eigen::Matrix3d rot = R;
        euler_angle(0) = std::atan2(rot(2, 1), rot(2, 2));
        euler_angle(1) = std::atan2(-rot(2, 0), std::sqrt(rot(2, 1) * rot(2, 1) + rot(2, 2) * rot(2, 2)));
        euler_angle(2) = std::atan2(rot(1, 0), rot(0, 0));

        return euler_angle;
    }

    static double CalcStereoDepth(const Eigen::Vector2d uv_left, const Eigen::Vector2d uv_right, const Eigen::Matrix3d K, const double baseline)
    {
        Eigen::Vector3d p_left = K.inverse() * uv_left.homogeneous();
        Eigen::Vector3d p_right = K.inverse() * uv_right.homogeneous();
        double depth = baseline / (p_left - p_right).norm();
        return depth;
    }
};

#endif