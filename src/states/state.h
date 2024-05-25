#ifndef __VIO_BACKEND_STATE__
#define __VIO_BACKEND_STATE__
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <map>
#include "types/Pose.h"
#include "types/Imu_state.h"

struct CameraPose {

    Eigen::Matrix3d Rwi;
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d pwi;
    Eigen::Vector3d pwc;

    Eigen::Matrix3d Rwi_fej;
    Eigen::Matrix3d Rwc_fej;
    Eigen::Vector3d pwi_fej;
    Eigen::Vector3d pwc_fej;
};

class State {
public:

    State(){
        _imu_state = std::make_shared<IMU_state>();
        _imu_to_cam_extrinsic = std::make_shared<Pose>();

        // initialize local id
        _imu_state->set_local_id(_dim);
        _dim += _imu_state->size();
        _variables.push_back(_imu_state);
        // _imu_to_cam_extrinsic->set_local_id(_dim);
        // _dim += _imu_to_cam_extrinsic->size();

        // initialize state covariance
        _covariance = Eigen::MatrixXd::Zero(_dim, _dim);   // init covariance size;
    }
    ~State(){}

    void stochastic_clone(std::shared_ptr<Type> variable_to_clone)
    {
        int old_rows = _covariance.rows();
        int old_cols = _covariance.cols();
        int new_rows = old_rows + variable_to_clone->size();
        int new_cols = old_cols + variable_to_clone->size();
        int variable_size = variable_to_clone->size();

        _covariance.conservativeResize(new_rows, new_cols);

        int old_loc = variable_to_clone->id();

        _covariance.block(old_rows, old_cols, variable_size, variable_size) = _covariance.block(old_loc, old_loc, variable_size, variable_size);
        _covariance.block(0, old_cols, old_rows, variable_size) = _covariance.block(0, old_loc, old_rows, variable_size);
        _covariance.block(old_rows, 0, variable_size, old_cols) = _covariance.block(old_loc, 0, variable_size, old_cols);

        std::shared_ptr<Type> clone = variable_to_clone->clone();
        clone->set_local_id(old_cols);
        _clone_pose.insert(std::make_pair(clone->ts(), std::dynamic_pointer_cast<Pose>(clone)));
        _variables.push_back(clone);
        std::sort(_variables.begin(), _variables.end(),
                [](const std::shared_ptr<Type> &a, const std::shared_ptr<Type> b)->bool
                {return a->id() < b->id();});
    }

    std::map<double, CameraPose> access_clone_pose_buffer() const
    {
        std::map<double, CameraPose> camera_clone_poses;
        for (auto it = _clone_pose.begin(); it != _clone_pose.end(); it++) {
            CameraPose camera_pose;
            camera_pose.Rwi = it->second->quat().toRotationMatrix();
            camera_pose.pwi = it->second->p();
            Eigen::Matrix3d Ric = _imu_to_cam_extrinsic->quat().toRotationMatrix();
            Eigen::Vector3d pic = _imu_to_cam_extrinsic->p();
            camera_pose.Rwc = camera_pose.Rwi * Ric;
            camera_pose.pwc = camera_pose.pwi + camera_pose.Rwc * pic;
            camera_clone_poses.insert(std::make_pair(it->first, camera_pose));
        }
        return camera_clone_poses;
    }

    int _dim = 0;
    std::shared_ptr<IMU_state> _imu_state;
    std::map<double, std::shared_ptr<Pose>> _clone_pose;
    std::shared_ptr<Pose> _imu_to_cam_extrinsic;
    std::vector<std::shared_ptr<Type>> _variables;
    Eigen::MatrixXd _covariance;

};

#endif