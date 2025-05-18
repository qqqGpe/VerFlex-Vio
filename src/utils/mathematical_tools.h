#ifndef __MATH_TOOLS__
#define __MATH_TOOLS__
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Jacobi>

class MathUtils
{
   public:
   template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skew(const Eigen::MatrixBase<Derived>& w)
    {
        assert(w.rows() == 3 && w.cols() == 1);
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t, 3, 3> w_x;
        w_x << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
        return w_x;
    }

    template<typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> Rodrigues(const Eigen::MatrixBase<Derived> vec, const double theta)
    {
        assert(vec.rows() == 3 && vec.cols() == 1);
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t, 3, 3> R;
        const Eigen::Matrix<Scalar_t, 3, 3> I3x3 = Eigen::Matrix<Scalar_t, 3, 3>::Identity();
        if (theta < 1e-6)
        {
            R = I3x3 + skew(vec * theta);
        }
        else
        {
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
        for (int n = 0; n < cols; n++)
        {
            for (int m = Hfx.rows() - 1; m > n; m--)
            {
                // Givens matrix G
                tempHo_GR.makeGivens(Hfx(m - 1, n), Hfx(m, n));
                // Multiply G to the corresponding lines (m-1,m) in each matrix
                // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
                //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
                (Hfx.block(m - 1, 0, 2, Hfx.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            }
        }
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 1> R2rpy(const Eigen::MatrixBase<Derived>& R)
    {
        // euler (Z-Y-X，i.e. RPY) make sure in range [-pi/2, pi/2]
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t, 3, 1> euler_angle;
        Eigen::Matrix<Scalar_t, 3, 3> rot = R;
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

    template <typename Derived>
    static Derived cot(const Derived& tanx)
    {
        if (abs(tanx) < 1e-12)
        {
            return 0;
        }
        else
        {
            return 1.0 / tanx;
        }
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> Jl(const Eigen::MatrixBase<Derived> &phi)
    {
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t, 3, 3> Jl;
        Scalar_t theta = phi.norm();
        Eigen::Matrix<Scalar_t, 3, 1> vec = phi / theta;
        if (theta < 1e-12)
        {
            Jl = Eigen::Matrix<Scalar_t, 3, 3>::Identity();
        }
        else
        {
            Jl = (sin(theta) / theta) * Eigen::Matrix<Scalar_t, 3, 3>::Identity() + (1 - sin(theta) / theta) * vec * vec.transpose() +
                 (1 - cos(theta)) / theta * skew(vec);
        }
        return Jl;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> Jl_inv(const Eigen::MatrixBase<Derived>& phi)
    {
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t, 3, 3> Jl_inv;
        Scalar_t theta = phi.norm();
        if (theta < 1e-12)
        {
            Jl_inv = Eigen::Matrix<Scalar_t, 3, 3>::Identity();
        }
        else
        {
            Scalar_t half_theta = theta / 2;
            Eigen::Matrix<Scalar_t, 3, 1> vec = phi / theta;
            Jl_inv = half_theta * cot(half_theta) * Eigen::Matrix<Scalar_t, 3, 3>::Identity() +
                     (1 - half_theta * cot(half_theta)) * vec * vec.transpose() - half_theta * skew(vec);
        }
        return Jl_inv;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> Jr(const Eigen::MatrixBase<Derived>& phi)
    {
        return Jl(-phi);
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, -1, -1> GivensRotation(const Eigen::MatrixBase<Derived>& mat, const int32_t stop_col = -1)
    {
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t, -1, -1> mat_ret = mat;
        uint32_t rows = mat_ret.rows();
        uint32_t cols = mat_ret.cols();

        uint32_t row_end = stop_col < 0 ? std::min(rows, cols) : stop_col;

        for (uint32_t row_up = 0; row_up < row_end; row_up++)
        {
            uint32_t col = row_up;
            for (uint32_t row_down = col + 1; row_down < rows; row_down++)
            {
                Scalar_t A_mm = mat_ret(row_up, col);
                Scalar_t A_nm = mat_ret(row_down, col);
                Scalar_t r = std::sqrt(A_mm * A_mm + A_nm * A_nm);
                if (r < 1e-12)
                {
                    continue;
                }
                else
                {
                    Scalar_t c = A_mm / r;
                    Scalar_t s = A_nm / r;
                    for (uint32_t i = col; i < cols; i++)
                    {
                        Scalar_t a = mat_ret(row_up, i);
                        Scalar_t b = mat_ret(row_down, i);
                        mat_ret(row_up, i) = c * a + s * b;
                        mat_ret(row_down, i) = -s * a + c * b;
                    }
                }
            }
        }
        return mat_ret;
    }
};

#endif