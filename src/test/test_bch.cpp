#include <sophus/so3.hpp>
#include "gtest/gtest.h"
#include "mathematical_tools.h"

TEST(VioTest, bch)
{
    // Use a larger angle to avoid numerical issues near zero
    Eigen::Vector3d angle_rad(0.5, 0.3, 0.2);
    Eigen::Matrix3d Jl = utils::math::Jl(angle_rad);
    Eigen::Matrix3d Jl_sophus = Sophus::SO3d::leftJacobian(angle_rad).matrix();
    Eigen::Matrix3d Jl_error = Jl - Jl_sophus;
    EXPECT_TRUE(Jl_error.norm() < 1e-6) << "Jl and Jl_sophus are not equal within the tolerance.";

    Eigen::Matrix3d Jl_inv = utils::math::Jl_inv(angle_rad);
    Eigen::Matrix3d Jl_inv_sophus = Sophus::SO3d::leftJacobianInverse(angle_rad).matrix();
    Eigen::Matrix3d Jl_inv_error = Jl_inv - Jl_inv_sophus;
    EXPECT_TRUE(Jl_inv_error.norm() < 1e-6) << "Jl_inv and Jl_inv_sophus are not equal within the tolerance.";

    Eigen::Matrix3d Jl_Res = Eigen::Matrix3d::Identity() -  Jl * Jl_inv;
    EXPECT_TRUE(Jl_Res.norm() < 1e-6);

    Eigen::Matrix3d Jr = utils::math::Jr(angle_rad);
    Eigen::Matrix3d Jr_inv = utils::math::Jr_inv(angle_rad);
    Eigen::Matrix3d Jr_Res = Eigen::Matrix3d::Identity() - Jr * Jr_inv;
    EXPECT_TRUE(Jr_Res.norm() < 1e-6);
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}