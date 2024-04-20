#ifndef __VIO_BACKEND_STATE__
#define __VIO_BACKEND_STATE__
#include <Eigen/Eigen>
#include <memory>
#include "types/Pose.h"
#include "types/Imu_state.h"

class State {
public:

    State(){
        _imu_state = std::make_shared<IMU_state>();
        _imu_to_cam_extrinsic = std::make_shared<Pose>();
    }
    ~State(){}

    std::shared_ptr<IMU_state> _imu_state;
    std::vector<std::shared_ptr<Pose>> _clone_pose;
    std::shared_ptr<Pose> _imu_to_cam_extrinsic;

};

#endif