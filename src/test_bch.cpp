#include <sophus/so3.hpp>
#include "gtest/gtest.h"
#include "utils/mathematical_tools.h"

TEST(VioTest, bch)
{
    Eigen::Vector3d angle_rad(0.1, 0.2, 0.3);
    Eigen::Matrix3d Jl = MathUtils::Jl(angle_rad);
    Eigen::Matrix3d Jl_sophus = Sophus::SO3d::leftJacobian(angle_rad).matrix();
    Eigen::Matrix3d Jl_error = Jl - Jl_sophus;
    EXPECT_TRUE(Jl_error.norm() < 1e-6) << "Jl and Jl_sophus are not equal within the tolerance.";
    // std::cout << "Jl:\n" << Jl << std::endl;
    // std::cout << "Jl_sophus:\n" << Jl_sophus << std::endl;

    Eigen::Matrix3d Jl_inv = MathUtils::Jl_inv(angle_rad);
    Eigen::Matrix3d Jl_inv_sophus = Sophus::SO3d::leftJacobianInverse(angle_rad).matrix();
    Eigen::Matrix3d Jl_inv_error = Jl_inv - Jl_inv_sophus;
    EXPECT_TRUE(Jl_inv_error.norm() < 1e-6) << "Jl_inv and Jl_inv_sophus are not equal within the tolerance.";
    // std::cout << "Jl_inv:\n" << Jl_inv << std::endl;
    // std::cout << "Jl_inv_sophus:\n" << Jl_inv_sophus << std::endl;

    Eigen::Matrix3d Jl_Res = Eigen::Matrix3d::Identity() -  Jl * Jl_inv;
    EXPECT_TRUE(Jl_Res.norm() < 1e-6);

    Eigen::Matrix3d Jr = MathUtils::Jr(angle_rad);
    Eigen::Matrix3d Jr_inv = MathUtils::Jr_inv(angle_rad);
    std::cout << "Jr * Jr_inv: \n" << Jr * Jr_inv << std::endl;
    Eigen::Matrix3d Jr_Res = Eigen::Matrix3d::Identity() - Jr * Jr_inv;
    EXPECT_TRUE(Jr_Res.norm() < 1e-6);

}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}