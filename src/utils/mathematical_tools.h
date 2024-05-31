#ifndef __MATH_TOOLS__
#define __MATH_TOOLS__

#include <Eigen/Core>

class mathematical {
public:

inline static Eigen::Matrix<double, 3, 3> skew(const Eigen::Vector3d &w) {
  Eigen::Matrix<double, 3, 3> w_x;
  w_x << 0, -w(2), w(1),
         w(2), 0, -w(0),
        -w(1), w(0), 0;
  return w_x;
}

inline static Eigen::Matrix3d Rodrigues(Eigen::Vector3d vec, double theta) {
    Eigen::Matrix3d R;
    Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
    if(theta < 1e-6) {
        R = I3x3 + skew(vec * theta);
    } else {
        R = cos(theta) * I3x3 + (1 - cos(theta)) * vec * vec.transpose() + sin(theta) * skew(vec);
    }
    return R;
}


};



#endif