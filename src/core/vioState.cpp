/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-11-23 16:33:14
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
 */
#include "vioState.h"
#include <glog/logging.h>

State::State(const Parameter& param)
{
    _param = param;
    enable_estimate_ric_ = param.estimate_ric;
    enable_estimate_td_visual_ = param.estimate_td_visual;

    _imu_state = std::make_shared<ImuState>();
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

    _vQic.resize(param.camera_num);
    _vPic.resize(param.camera_num);
    for (int i = 0; i < param.camera_num; i++)
    {
        _vQic[i] = std::make_shared<Quat>();
        _vPic[i] = std::make_shared<Vec>();
        InitCamExtrinsic(i, Eigen::Quaterniond(param.Ric[i]), param.tic[i]);
        if (enable_estimate_ric_)
        {
            _vQic[i]->set_local_id(_dim);
            _variables.push_back(_vQic[i]);
            _dim += _vQic[i]->size();

            _vPic[i]->set_local_id(_dim);
            _variables.push_back(_vPic[i]);
            _dim += _vPic[i]->size();
        }
    }

    if (enable_estimate_td_visual_)
    {
        td_visual_ = std::make_shared<Scalar>();
        td_visual_->set_local_id(_dim);
        _variables.push_back(td_visual_);
        _dim += td_visual_->size();
    }

    SetCovariance(Eigen::MatrixXd::Identity(_dim, _dim));  // initialize covariance;
    SetSqrtPt(Eigen::MatrixXd::Identity(_dim, _dim));      // initialize sqrt-root covariance;
}

void State::reset()
{
    // Reset imu state
    _imu_state->reset();

    // Reset td_visual
    if (enable_estimate_td_visual_)
    {
        td_visual_->reset();
    }

    // Clear all clone poses
    for (auto it = _clone_pose.begin(); it != _clone_pose.end();)
    {
        _variables.erase(std::remove(_variables.begin(), _variables.end(), it->second), _variables.end());
        _dim = _dim - it->second->size();
        it = _clone_pose.erase(it);
    }

    // Reset covariance and sqrt_Pt
    SetCovariance(Eigen::MatrixXd::Identity(_dim, _dim));
    SetSqrtPt(Eigen::MatrixXd::Identity(_dim, _dim));
}

void State::insert_after(const std::shared_ptr<Type>& pose_to_insert_after, const std::shared_ptr<Type>& new_variable)
{
    if (!new_variable)
    {
        return;
    }

    if (std::find(_variables.begin(), _variables.end(), new_variable) != _variables.end())
    {
        return;
    }

    const auto target = std::find(_variables.begin(), _variables.end(), pose_to_insert_after);
    const size_t insert_idx = (target == _variables.end()) ? _variables.size() : static_cast<size_t>(std::distance(_variables.begin(), target) + 1);

    const uint32_t new_id = (insert_idx == 0) ? 0 : _variables[insert_idx - 1]->id() + _variables[insert_idx - 1]->size();
    new_variable->set_local_id(new_id);

    const uint32_t delta = new_variable->size();
    for (size_t i = insert_idx; i < _variables.size(); ++i)
    {
        _variables[i]->set_local_id(_variables[i]->id() + delta);
    }

    _variables.insert(_variables.begin() + static_cast<std::ptrdiff_t>(insert_idx), new_variable);
    _dim += delta;
}

void State::AccessClonePoseBuffer(std::map<double, CameraPose>& camera_clone_poses) const
{
    for (auto it = _clone_pose.begin(); it != _clone_pose.end(); it++)
    {
        CameraPose camera_pose(_param.camera_num);
        camera_pose.Rwi = it->second->quat().normalized().toRotationMatrix();
        camera_pose.pwi = it->second->p();
        for (int i_cam = 0; i_cam < _param.camera_num; i_cam++)
        {
            Eigen::Matrix3d R_CtoI = _vQic[i_cam]->q().toRotationMatrix();
            Eigen::Vector3d p_CinI = _vPic[i_cam]->vec();
            camera_pose.Rwc[i_cam] = camera_pose.Rwi * R_CtoI;
            camera_pose.pwc[i_cam] = camera_pose.pwi + camera_pose.Rwi * p_CinI;
        }
        camera_clone_poses.try_emplace(it->first, camera_pose);
    }
}

void State::UpdateSlamFeatureAfterVisualUpdate()
{
    for (auto& [feature_id, slam_feature] : _slam_features)
    {
        Eigen::Vector3d pwf_updated = slam_feature._state_ptr->vec();
        slam_feature._info->_pwf = pwf_updated;
    }
}

bool State::AugumentSlamFeature(Feature* feature,
                                const Eigen::MatrixXd& Hf,
                                const Eigen::MatrixXd& Hx,
                                const std::vector<std::shared_ptr<Type>>& Hx_order,
                                const Eigen::MatrixXd& R,
                                std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx)
{
    // Create SlamFeature
    std::shared_ptr<Vec> feature_state = std::make_shared<Vec>(feature->_pwf);
    SlamFeature new_feature {
        ._id = feature->_id,
        ._state_ptr = feature_state,
        ._info = feature
    };

    Eigen::MatrixXd& old_covariance = _covariance;
    Eigen::MatrixXd Hx_all = Eigen::MatrixXd::Zero(3, old_covariance.rows());
    for (int i = 0; i < Hx_order.size(); i++)
    {
        std::shared_ptr<Type> var = Hx_order[i];
        Hx_all.block(0, var->id(), 3, var->size()) = Hx.block(0, map_hx.at(var), 3, var->size());
    }

    Eigen::MatrixXd Hf_inv = Hf.inverse();
    Eigen::MatrixXd Pff = Hf_inv * (Hx_all * old_covariance * Hx_all.transpose() + R) * Hf_inv.transpose();
    Eigen::MatrixXd Pxf = -old_covariance * Hx_all.transpose() * Hf_inv.transpose();

    // Set local id and augument state vector
    std::shared_ptr<Vec>& landmark_state = new_feature._state_ptr;
    landmark_state->set_local_id(_dim);
    _variables.push_back(landmark_state);
    _dim += landmark_state->size();

    // Augument covariance
    Eigen::MatrixXd new_covariance = Eigen::MatrixXd::Zero(_dim, _dim);
    new_covariance.topLeftCorner(old_covariance.rows(), old_covariance.cols()) = old_covariance;
    new_covariance.bottomRightCorner(new_feature._state_ptr->size(), new_feature._state_ptr->size()) = Pff;
    new_covariance.block(0, new_feature._state_ptr->id(), old_covariance.rows(), new_feature._state_ptr->size()) = Pxf;
    new_covariance.block(new_feature._state_ptr->id(), 0, new_feature._state_ptr->size(), old_covariance.rows()) = Pxf.transpose();

    SetCovariance(new_covariance);

    // Augment SqrtPt directly (avoids numerically fragile full-matrix LLT)
    // Decompose P_aug = S_aug^T * S_aug using block formula:
    //   S_aug = [ S_old | B ]    where B = -S_old * Hx_all^T * Hf_inv^T
    //           [   0   | C ]    where C^T * C = Hf_inv * R * Hf_inv^T (Schur complement)
    if (sqrt_Pt_.size() > 0)
    {
        Eigen::MatrixXd SqrtPt_old = Sqrt_Pt();
        int feat_dim = new_feature._state_ptr->size();
        int old_rows = SqrtPt_old.rows();
        int old_cols = SqrtPt_old.cols();

        Eigen::MatrixXd B = -SqrtPt_old * Hx_all.transpose() * Hf_inv.transpose();
        Eigen::MatrixXd R_schur = Hf_inv * R * Hf_inv.transpose();
        Eigen::MatrixXd C = R_schur.llt().matrixL().transpose();

        Eigen::MatrixXd SqrtPt_new = Eigen::MatrixXd::Zero(old_rows + feat_dim, old_cols + feat_dim);
        SqrtPt_new.topLeftCorner(old_rows, old_cols) = SqrtPt_old;
        SqrtPt_new.block(0, old_cols, old_rows, feat_dim) = B;
        SqrtPt_new.bottomRightCorner(feat_dim, feat_dim) = C;

        SetSqrtPt(SqrtPt_new);
    }
    // Add to slam feature map
    _slam_features.try_emplace(new_feature._id, new_feature);

    return true;
}