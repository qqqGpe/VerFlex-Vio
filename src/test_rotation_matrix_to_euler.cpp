#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "utils/mathematical_tools.h"

#define RAD2DEG 180 / M_PI
#define DEG2RAD M_PI / 180

int main()
{
    double roll = -30 * DEG2RAD;
    double pitch = -75 * DEG2RAD;
    double yaw = 200 * DEG2RAD;

    Eigen::AngleAxisd Rx = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d R = (Rz * Ry * Rx).toRotationMatrix();
    // Eigen::Vector3d rpy = R.eulerAngles(2,1,0) * RAD2DEG;
    Eigen::Vector3d rpy = mathematical::RotationMatrixToEulerAngles(R) * RAD2DEG;

    std::cout << "roll: " << rpy.x() << std::endl;
    std::cout << "pitch: " << rpy.y() << std::endl;
    std::cout << "yaw: " << rpy.z() << std::endl;

    return 0;
}
