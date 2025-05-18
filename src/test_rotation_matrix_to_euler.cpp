#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "utils/mathematical_tools.h"
#include "gtest/gtest.h"

#define RAD2DEG 180 / M_PI
#define DEG2RAD M_PI / 180

TEST(VioTest, RotationMatrixToEuler)
{
    double roll = -30 * DEG2RAD;
    double pitch = -75 * DEG2RAD;
    double yaw = 20 * DEG2RAD;

    Eigen::AngleAxisd Rx = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d R = (Rz * Ry * Rx).toRotationMatrix();
    Eigen::Vector3d rpy = MathUtils::R2rpy(R) * RAD2DEG;

    EXPECT_NEAR(rpy.x(), roll * RAD2DEG, 1e-6);
    EXPECT_NEAR(rpy.y(), pitch * RAD2DEG, 1e-6);
    EXPECT_NEAR(rpy.z(), yaw * RAD2DEG, 1e-6);
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
