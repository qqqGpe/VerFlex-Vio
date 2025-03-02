#ifndef __VIO_BACKEND_STATE__
#define __VIO_BACKEND_STATE__
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <map>
#include "Pose.h"
#include "Imu_state.h"

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
        _Tic = std::make_shared<Pose>();

        // initialize local id
        _dim = 0;
        // _imu_state->set_local_id(_dim);
        // _variables.push_back(_imu_state);
        // _dim += _imu_state->size();
        _imu_state->set_local_id(_dim);

        _variables.push_back(_imu_state->q());
        _dim += _imu_state->q()->size();
        _variables.push_back(_imu_state->p());
        _dim += _imu_state->p()->size();
        _variables.push_back(_imu_state->v());
        _dim += _imu_state->v()->size();
        _variables.push_back(_imu_state->bg());
        _dim += _imu_state->bg()->size();
        _variables.push_back(_imu_state->ba());
        _dim += _imu_state->ba()->size();

        if (_do_calibration_update) {
            _Tic->set_local_id(_dim);
            _variables.push_back(_Tic);
            _dim += _Tic->size();
        }

        // initialize state covariance
        _covariance = Eigen::MatrixXd::Identity(_dim, _dim);   // initialize covariance size;
    }
    ~State(){}

    void set_ts_sec(double ts_sec) { _imu_state->set_ts(ts_sec); }  // for debug

    void set_extrinsic(Eigen::Quaterniond qic, Eigen::Vector3d tic)
    {
        Eigen::VectorXd extrin_vector = Eigen::VectorXd::Zero(7);
        extrin_vector << qic.coeffs(), tic;
        _Tic->set_value(extrin_vector);
    }

    double ts_sec() { return _imu_state->ts(); }

    void stochastic_clone(std::shared_ptr<Type> variable_to_clone)
    {
        int old_rows = _covariance.rows();
        int old_cols = _covariance.cols();
        int new_rows = old_rows + variable_to_clone->size();
        int new_cols = old_cols + variable_to_clone->size();
        int variable_size = variable_to_clone->size();

        Eigen::MatrixXd covariance_new = Eigen::MatrixXd::Zero(new_rows, new_cols);
        covariance_new.block(0, 0, old_rows, old_cols) = _covariance;
        // _covariance.conservativeResize(new_rows, new_cols);

        int old_loc = variable_to_clone->id();

        covariance_new.block(old_rows, old_cols, variable_size, variable_size) = _covariance.block(old_loc, old_loc, variable_size, variable_size);
        covariance_new.block(0, old_cols, old_rows, variable_size) = _covariance.block(0, old_loc, old_rows, variable_size);
        covariance_new.block(old_rows, 0, variable_size, old_cols) = _covariance.block(old_loc, 0, variable_size, old_cols);

        std::shared_ptr<Type> clone = variable_to_clone->clone();
        clone->set_local_id(old_cols);
        _clone_pose.insert(std::make_pair(clone->ts(), std::dynamic_pointer_cast<Pose>(clone)));
        _variables.push_back(clone);
        _dim += clone->size();
        _covariance = covariance_new;
    }

    std::map<double, CameraPose> AccessClonePoseBuffer() const
    {
        std::map<double, CameraPose> camera_clone_poses;
        for (auto it = _clone_pose.begin(); it != _clone_pose.end(); it++) {
            CameraPose camera_pose;
            camera_pose.Rwi = it->second->quat().normalized().toRotationMatrix();
            camera_pose.pwi = it->second->p();
            Eigen::Matrix3d R_CtoI = _Tic->quat().normalized().toRotationMatrix();
            Eigen::Vector3d p_CinI = _Tic->p();
            camera_pose.Rwc = camera_pose.Rwi * R_CtoI;
            camera_pose.pwc = camera_pose.pwi + camera_pose.Rwi * p_CinI;
            camera_clone_poses.insert(std::make_pair(it->first, camera_pose));
        }
        return camera_clone_poses;
    }

    void marginalize_covariance()
    {
        Eigen::MatrixXd covariance_small = Eigen::MatrixXd::Zero(_dim, _dim);
        int cur_i = 0;
        for (int i = 0; i < _variables.size(); i++) {
            std::shared_ptr<Type> var_i = _variables[i];
            int cur_j = 0;
            for (int j = 0; j < _variables.size(); j++) {
                std::shared_ptr<Type> var_j = _variables[j];
                covariance_small.block(cur_i, cur_j, var_i->size(), var_j->size()) =
                    _covariance.block(var_i->id(), var_j->id(), var_i->size(), var_j->size());
                cur_j += var_j->size();
            }
            cur_i += var_i->size();
        }
        _covariance = Eigen::MatrixXd::Zero(_dim, _dim);
        _covariance = covariance_small;
    }

    void MarginalizeState(std::shared_ptr<Type>& state_to_marg)
    {
        if (state_to_marg == nullptr)
        {
            return;
        }

        if (std::find(_variables.begin(), _variables.end(), state_to_marg) == _variables.end()) {
            LOG(ERROR) << "marginalization failed, no such variable in states";
            return;
        }

        if (state_to_marg == _variables.front()) {
            _variables.erase(_variables.begin());
            for (auto it = _variables.begin(); it != _variables.end(); it++) {
                (*it)->set_local_id((*it)->id() - state_to_marg->size());
            }
        } else if (state_to_marg == _variables.back()) {
            _variables.erase(_variables.end() - 1);
        } else {
            std::vector<std::shared_ptr<Type>>::iterator it_to_marge = std::find(_variables.begin(), _variables.end(), state_to_marg);
            for (auto it = it_to_marge; it != _variables.end(); it++) {
                (*it)->set_local_id((*it)->id() - state_to_marg->size());
            }
            _variables.erase(it_to_marge);
        }

        _dim = _dim - state_to_marg->size();
        double timestamp_to_marge = state_to_marg->ts();
        if (_clone_pose.count(timestamp_to_marge) != 0) {
            _clone_pose.erase(timestamp_to_marge);
        }
        marginalize_covariance();
    }

    int _dim = 0;
    std::shared_ptr<IMU_state> _imu_state;
    std::map<double, std::shared_ptr<Pose>> _clone_pose;
    std::shared_ptr<Pose> _Tic; // R_CtoI, p_CinI
    std::vector<std::shared_ptr<Type>> _variables;
    Eigen::MatrixXd _covariance;
    int32_t _do_calibration_update = 0;

};

#endif