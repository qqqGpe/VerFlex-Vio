#include <vector>
#include "visualManager.h"
#include "mathematical_tools.h"
#include "eskf_solver.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace {
    constexpr uint32_t kMinFeatForMapping = 2;
    constexpr double kMaxConditionNum = 20000.f;
    constexpr double kMinTriangDist = 0.2;
    constexpr double kMaxTriangDist = 50;
    constexpr uint32_t kMaxIterationTimes = 5;
    constexpr uint32_t kMinFeatNumToUpdate = 15;
}

bool VisualManager::visual_update()
{
    constexpr uint32_t kMinFeatToUpdate = 15;
    bool visual_updated = false;
    std::map<double, CameraPose> camera_pose_buffer = _state->access_clone_pose_buffer();
    auto feat_origin_input = _feature_tracked;

    std::cout << "_origin_feature_tracked: " << _origin_feature_tracked << std::endl;
    _origin_feature_tracked = _feature_tracked.size();

    feature_triangulation(_feature_tracked, camera_pose_buffer);

    calculate_feature_parallex(_feature_tracked);

    std::vector<Feature*> feat_msckf = select_msckf_features(_feature_tracked);

    if (feat_msckf.size() > kMinFeatToUpdate)
    {
        // pnp_ransac_to_reject_outliers(feat_msckf);
        Eigen::MatrixXd Hx_msckf;
        Eigen::VectorXd res;
        construct_feature_jocabian_full(feat_msckf, Hx_msckf, res);
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(Hx_msckf.rows(), Hx_msckf.rows());

        if (_state->_clone_pose.size() >= 2) {
            eskfSolver::update(_state, Hx_msckf, res, _Hx_order, _map_hx, R);
            visual_updated = true;
        }
        std::cout << "---------------------------" << std::endl;
        LOG(INFO) << cv::format("feature updated, feature mapping success: %d", _feature_mapping_success);

    }
    else
    {
        // LOG(INFO) << cv::format("Not enough features to update, features_tracked: %d, features_msckf: %d, feature mapping success: %d",
        //                        int(_feature_tracked.size()), int(feat_msckf.size()), _feature_mapping_success);
    }

    std::shared_ptr<Type> state_to_marginalize = nullptr;
    // std::cout << "state size: " << _state->_variables.size() << std::endl;
    // for (int i = 0; i < _state->_variables.size(); i++)
    // {
    //     std::cout << _state->_variables[i]->state_name << ", " << std::endl;
    // }
    if (_state->_clone_pose.size() > _max_clone_pose)
    {
        _keyframe = decide_keyframe(_state, feat_origin_input);
        if (_keyframe != keyframe_flag_e::not_keyframe)
        {
            state_to_marginalize = _state->_clone_pose.begin()->second;
            update_feature_base();
            drop_feature_obs(state_to_marginalize->ts());
        }
        else    // drop lastest pose
        {
            auto it = --_state->_clone_pose.end();
            state_to_marginalize = it->second;
            drop_feature_obs(state_to_marginalize->ts());
        }
    }

    _state->marginalize_state(state_to_marginalize);
    return visual_updated;
}

keyframe_flag_e VisualManager::decide_keyframe(std::shared_ptr<State> _state, std::vector<Feature*> feats)
{
    constexpr double kLargeParallexThres = 10.0f; // 大于5个平均像素视差则视为关键帧

    if(_state->_clone_pose.size() < _max_clone_pose) {
        return keyframe_flag_e::not_keyframe;
    }

    uint32_t cnt = 0;
    double pixel_parallex = 0.f;
    double pixel_parallex_avg = 0.f;
    for (auto x : feats) {
        if (x->_visual_obs_buffer.size() < 2) {
            continue;
        }
        auto last_it = x->_visual_obs_buffer.rbegin();
        double last_ts = last_it->first;
        auto sub_last_it = x->_visual_obs_buffer.rbegin();
        sub_last_it++;
        double sub_last_ts = sub_last_it->first;

        assert(sub_last_ts < last_ts);
        cam_obs_t curr_obs = last_it->second;
        cam_obs_t prev_obs = sub_last_it->second;
        double diff_u = curr_obs.u - prev_obs.u;
        double diff_v = curr_obs.v - prev_obs.v;
        pixel_parallex = std::sqrt(std::pow(diff_u, 2) + std::pow(diff_v, 2));
        pixel_parallex_avg += pixel_parallex;
        cnt++;
    }
    pixel_parallex_avg = pixel_parallex_avg / cnt;
    std::cout << "pixel_parallex_avg: " << pixel_parallex_avg << std::endl;
    if (pixel_parallex_avg > kLargeParallexThres) {
        return keyframe_flag_e::large_parallex_flag;
    }

    if (_origin_feature_tracked < 20) {
        return keyframe_flag_e::feat_lost_too_much;
    }

    return keyframe_flag_e::not_keyframe;
}

void VisualManager::drop_feature_obs(const double timestamp_to_drop)
{
    for (auto &feat_base: _feature_base)
    {
        if (feat_base->_visual_obs_buffer.count(timestamp_to_drop) != 0)
        {
            auto it = feat_base->_visual_obs_buffer.find(timestamp_to_drop);
            feat_base->_visual_obs_buffer.erase(it);
        }
    }
}

void VisualManager::update_feature_base()
{
    // std::cout << "feat_lost.size: " << _feature_lost.size() << std::endl;
    for (auto &feat_lost : _feature_lost)
    {
        feat_lost->reset();
    }

    // std::cout << "feat_new.size: " << _feature_obs_new.size() << std::endl;
    for (int i = 0; i < _feature_new_base.size(); i++)
    {
        _feature_new_base[i]->_valid = true;
        _feature_new_base[i]->_id = _feature_obs_new[i].feat_id;
        _feature_new_base[i]->_visual_obs_buffer.insert({_feature_obs_new[i].ts_sec, _feature_obs_new[i]});
    }
}

void VisualManager::update_feature(std::pair<double, std::vector<cam_obs_t>> feature_observes)
{
    double ts_sec = feature_observes.first;
    std::vector<cam_obs_t> feature_obs = feature_observes.second;
    assert(feature_obs.size() == _max_feat_n);
    assert(feature_obs.size() == feature_obs.size());

    _feature_lost.clear();
    _feature_obs_new.clear();
    _feature_tracked.clear();
    _feature_new_base.clear();

    for (int i = 0; i < _feature_base.size(); i++) {
        Feature *feat = _feature_base[i];
        cam_obs_t &feat_obs = feature_obs[i];
        auto feat_norm = _camera_model->back_project(Eigen::Vector2d(feat_obs.u, feat_obs.v));
        feat_obs.u_norm = feat_norm.x();
        feat_obs.v_norm = feat_norm.y();

        if (feat->_valid) {
            if (feat_obs.valid && feat_obs.feat_id == feat->_id) {
                feat->_visual_obs_buffer.insert({ts_sec, feat_obs});
                _feature_tracked.push_back(feat);
            } else {
                _feature_lost.push_back(feat);
                _feature_obs_new.push_back(feat_obs);
                _feature_new_base.push_back(feat);
            }
        }
        else if (feat_obs.valid){
            _feature_obs_new.push_back(feat_obs);
            _feature_new_base.push_back(feat);
        }
    }

    // std::cout << "feature tracked: " << _feature_tracked.size() << std::endl;
    // std::cout << "feature lost: " << _feature_lost.size() << std::endl;
    // std::cout << "feature new: " << _feature_obs_new.size() << std::endl;
}

bool VisualManager::least_square_triangulation(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat)
{
    // std::cout << "feat.size: " << feat->_visual_obs_buffer.size() << std::endl;

    // for (auto x : feat->_visual_obs_buffer)
    // {
    //     std::cout << cv::format("feat ts: %f, feat_id:  %d\n", x.first, x.second.feat_id);
    //     // std::cout << "feat ts: " << x.first << std::endl;
    //     std::cout << cv::format("feat obs: (%f, %f)\n", x.second.u_norm, x.second.v_norm);
    // }
    // std::cout << "--------------------" << std::endl;

    auto first_obs = feat->_visual_obs_buffer.begin();

    Eigen::Matrix3d R_AtoG = clone_pose_buffer[first_obs->first].Rwc;
    Eigen::Vector3d p_AinG = clone_pose_buffer[first_obs->first].pwc;

    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d ATb = Eigen::Vector3d::Zero();

    int feat_index = 0;
    // std::cout << "--------------" << std::endl;
    int cnt = 0;
    for (auto it = feat->_visual_obs_buffer.begin(); it != feat->_visual_obs_buffer.end(); it++) {
        Eigen::Matrix3d R_CitoG = clone_pose_buffer[it->first].Rwc;
        Eigen::Vector3d p_CiinG = clone_pose_buffer[it->first].pwc;

        Eigen::Matrix3d R_CitoA = R_AtoG.transpose() * R_CitoG;
        Eigen::Vector3d p_CiinA = R_AtoG.transpose() * (p_CiinG - p_AinG);

        std::cout << "p_CiinA: " << p_CiinA.transpose() << std::endl;
        // std::cout << "R_CitoA: \n" << R_CitoA << std::endl;

        Eigen::Vector3d b_i;
        // Eigen::Vector2d b(it->second.u, it->second.v);
        b_i << it->second.u_norm, it->second.v_norm, 1;
        // std::cout << "b: " << b.transpose() << std::endl;
        Eigen::Vector3d b_iinA = R_CitoA * b_i;
        ATA += mathematical::skew(b_iinA).transpose() * mathematical::skew(b_iinA);
        ATb += mathematical::skew(b_iinA).transpose() * mathematical::skew(b_iinA) * p_CiinA;
        cnt++;
        if (cnt > 1)
        {
            break;
        }
    }

    Eigen::Vector3d paf = ATA.colPivHouseholderQr().solve(ATb);
    // Check A and paf
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ATA);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);
    feat->_pwf = p_AinG + R_AtoG * paf;
    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    int triang_failed_num = 0;
    if (std::abs(condA) > kMaxConditionNum || paf(2, 0) < kMinTriangDist || paf(2, 0) > kMaxTriangDist || std::isnan(paf.norm())) {
        triang_failed_num++;
        // if (std::abs(condA) > kMaxConditionNum)
        // {
        //     std::cout << cv::format("condition num [%f] > [%f]", condA, kMaxConditionNum) << std::endl;
        // }
        // else if (paf(2, 0) < kMinTriangDist || paf(2, 0) > kMaxTriangDist)
        // {
        //     std::cout << cv::format("paf.z: [%f]", paf(2, 0)) << std::endl;
        // }
        // else if (std::isnan(paf.norm()))
        // {
        //     std::cout << "paf is nan" << std::endl;
        // }
        return false;
    }

    // std::cout << "triangulation failed num: " << triang_failed_num << std::endl;
    return true;
}

bool VisualManager::gaussian_newton_optimization(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat)
{
    auto last_obs = feat->_visual_obs_buffer.end();
    last_obs--;
    Eigen::Matrix3d R_AtoG = clone_pose_buffer[last_obs->first].Rwc;
    Eigen::Vector3d p_AinG = clone_pose_buffer[last_obs->first].pwc;

    Eigen::Vector3d paf = R_AtoG.transpose() * (feat->_pwf - p_AinG);
    if (paf.z() < 1e-6) {
        return false;
    }

    double alpha = paf.x() / paf.z();
    double beta = paf.y() / paf.z();
    double rho = 1 / paf.z();
    int iter_time = 0;
    while (iter_time < kMaxIterationTimes) {
        Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
        Eigen::Vector3d ATb = Eigen::Vector3d::Zero();
        for (auto it = feat->_visual_obs_buffer.begin(); it != feat->_visual_obs_buffer.end(); it++) {
            double tn = (*it).first;
            Eigen::Vector2d z_m;
            z_m << (*it).second.u_norm, (*it).second.v_norm;

            Eigen::Matrix3d R_CitoG = clone_pose_buffer[tn].Rwc;
            Eigen::Vector3d p_CiinG = clone_pose_buffer[tn].pwc;

            Eigen::Matrix3d R_AtoCi = R_CitoG.transpose() * R_AtoG;
            Eigen::Vector3d p_CiinA = R_AtoG.transpose() * (p_CiinG - p_AinG);

            Eigen::Vector3d paf_norm;
            paf_norm << alpha, beta, 1;
            Eigen::Vector3d h = R_AtoCi * (paf_norm - rho * p_CiinA);
            if (h(2, 0) < 1e-6) {
                continue;
            }

            Eigen::Vector2d res;
            res(0, 0) = z_m(0, 0) - h(0, 0) / h(2, 0);
            res(1, 0) = z_m(1, 0) - h(1, 0) / h(2, 0);

            Eigen::MatrixXd Jacobian_1(2, 3);
            Eigen::MatrixXd Jacobian_2(3, 3);
            Jacobian_1 << 1/h(2, 0), 0, -h(0, 0)/std::pow(h(2, 0), 2),
                          0, 1/h(2, 0), -h(1, 0)/std::pow(h(2, 0), 2);
            Jacobian_2.block<3, 1>(0, 0) << 1, 0, 0;
            Jacobian_2.block<3, 1>(0, 1) << 0, 1, 0;
            Jacobian_2.block<3, 1>(0, 2) << -p_CiinA;
            Jacobian_2 = R_AtoCi * Jacobian_2.eval();

            Eigen::MatrixXd Jacobian(2, 3);
            Jacobian = Jacobian_1 * Jacobian_2;
            ATA += Jacobian.transpose() * Jacobian;
            ATb += Jacobian.transpose() * res;
        }

        Eigen::Vector3d delta_x = ATA.colPivHouseholderQr().solve(ATb);
        alpha += delta_x.x();
        beta += delta_x.y();
        rho += delta_x.z();
        iter_time ++;
    }

    Eigen::Vector3d paf_opt;
    paf_opt << alpha/rho, beta/rho, 1/rho;
    feat->_pwf = p_AinG + R_AtoG * paf_opt;

    if (feat->_pwf.norm() > 50 || iter_time == kMaxIterationTimes)
    {
        return false;
    }

    return true;
}

void VisualManager::feed_image(const std::pair<double, std::pair<cv::Mat, cv::Mat>> input)
{
    while (_input_image_buffer.size() > kMaxImageBufferSize) {
        _input_image_buffer.pop();
    }
    _input_image_buffer.push(input);
}

bool VisualManager::pnp_ransac_to_reject_outliers(std::vector<Feature* > feats)
{
    double ts = _state->ts_sec();
    std::vector<cv::Point3d> list_points3d;
    std::vector<cv::Point2d> list_points2d;
    for (auto it = feats.begin(); it != feats.end(); it++) {
        if ((*it)->_valid && (*it)->_is_triangulated) {
            assert((*it)->_visual_obs_buffer.find(ts) != (*it)->_visual_obs_buffer.end());
            list_points3d.emplace_back((*it)->_pwf.x(), (*it)->_pwf.y(), (*it)->_pwf.z());
            cam_obs_t obs_2d = (*it)->_visual_obs_buffer.at(ts);
            list_points2d.emplace_back(obs_2d.u, obs_2d.v);
        }
    }
    if (list_points3d.size() <= 0)
    {
        LOG(INFO) << cv::format("Not enough points for pnpRansac, point triangulated: %d", int(list_points3d.size()));
        return false;
    }
    cv::Mat intrinsic;
    cv::Mat distortion;
    cv::Mat inliers;
    cv::eigen2cv(_camera_model->intrinsic(), intrinsic);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::solvePnPRansac(list_points3d, list_points2d,
                       intrinsic, distortion,
                       rvec, tvec, false, 100, 3.f, 0.8,
                       inliers, cv::SOLVEPNP_ITERATIVE);

    std::vector<int> inliers_id;
    for (int i = 0; i < inliers.rows; i++) {
        int id = feats[inliers.at<int>(i)]->_id;
        inliers_id.push_back(id);
    }

    for (auto it = feats.begin(); it != feats.end();) {
        if (!(*it)->_valid || !(*it)->_is_triangulated) {
            continue;
        }

        if (std::find(inliers_id.begin(), inliers_id.end(), (*it)->_id) == inliers_id.end()) {
            it = feats.erase(it);
            continue;
        } else {
            it++;
        }
    }

    // for debugging
    // for (int i = 0; i < inliers.rows; i++)
    // {
    //     std::cout << inliers.at<int>(i) << std::endl;
    // }
    // Eigen::Vector3d t_, r_;
    // cv::cv2eigen(tvec, t_);
    // std::cout << "translation: " << t_.transpose() << std::endl;
    return true;
}

void VisualManager::feature_triangulation(std::vector<Feature* > &feats, std::map<double, CameraPose> camera_pose_buffer)
{
    // std::map<double, CameraPose> camera_pose_buffer = _state->access_clone_pose_buffer();
    _feature_mapping_success = 0;

    // std::cout << "clone pose size: " << _state->_clone_pose.size() << std::endl;;
    // for (auto it = camera_pose_buffer.begin(); it != camera_pose_buffer.end(); it++)
    // {
    //     std::cout << "ts: " << it->first << std::endl;
    //     // std::cout << "pose q: " << it->second.Rwc << std::endl;
    //     std::cout << "pose p: " << it->second.pwc.transpose() << std::endl;
    //     std::cout << "------------------" << std::endl;
    // }

    // std::cout << "feat.size: " << feats.size() << std::endl;
    // for (auto feat : feats)
    // {
    //     for (auto x : feat->_visual_obs_buffer)
    //     {
    //         std::cout << cv::format("feat ts: %f, feat_id:  %d\n", x.first, x.second.feat_id);
    //         // std::cout << "feat ts: " << x.first << std::endl;
    //         std::cout << cv::format("feat obs: (%f, %f)\n", x.second.u, x.second.v);
    //     }
    //     std::cout << "--------------------" << std::endl;
    // }

    int origin_feats = feats.size();
    int less_obs_delete = 0;
    int triangulate_failed = 0;
    int gaussian_newton_failed = 0;
    for (auto it = feats.begin(); it != feats.end();) {
        if ((*it)->_visual_obs_buffer.size() < kMinFeatForMapping) {
            it = feats.erase(it);
            less_obs_delete ++;
            continue;
        }

        // if (1)
        if ((*it)->_is_triangulated == false)
        {
            if(false == least_square_triangulation(camera_pose_buffer, *it)) {
                (*it)->_pwf.setZero();
                (*it)->parallex = 0;
                it = feats.erase(it);
                triangulate_failed ++;
                continue;
            }
            (*it)->_is_triangulated = true;
        }

        if ((*it)->_is_triangulated && gaussian_newton_optimization(camera_pose_buffer, *it)) {
            _feature_mapping_success ++;
        } else {
            (*it)->_is_triangulated = false;
            it = feats.erase(it);
            gaussian_newton_failed++;
            continue;
        }

        std::cout << "it->is_triagulated: " << (*it)->_is_triangulated << std::endl;

        it++;
    }
    std::cout << "features tracked: " << origin_feats << std::endl;
    // std::cout << "less obs failed: " << less_obs_delete << std::endl;
    std::cout << "triangulated failed: " << triangulate_failed << std::endl;
    std::cout << "gaussian_newton_failed: " << gaussian_newton_failed << std::endl;

    // LOG(INFO) << cv::format("mapping success: %d", _feature_mapping_success);
}

void VisualManager::calculate_feature_parallex(std::vector<Feature *> &feats)
{
    for (auto it = feats.begin(); it != feats.end(); it++)
    {
        auto first_obv = (*it)->_visual_obs_buffer.begin();
        auto last_obv = (*it)->_visual_obs_buffer.end(); last_obv--;
        double dx = first_obv->second.u - last_obv->second.u;
        double dy = first_obv->second.v - last_obv->second.v;
        double cur_parallex = sqrt(dx * dx + dy * dy);
        if (cur_parallex > (*it)->parallex)
        {
            (*it)->parallex = cur_parallex;
        }
    }
}

std::vector<Feature*> VisualManager::select_msckf_features(const std::vector<Feature*> feats)
{
    constexpr double kMinParallexForUse = 2.0;
    std::vector<Feature*> feat_msckf;
    for (int i = 0; i < feats.size(); i++) {
        std::cout << cv::format("is_triangulated: %d, parallex: %f\n", feats[i]->_is_triangulated, feats[i]->parallex);
        if (feats[i]->_is_triangulated && feats[i]->parallex > kMinParallexForUse) {
            feat_msckf.push_back(feats[i]);
        }
    }

    if (feat_msckf.size() > _max_visual_feat_to_use) {
        std::sort(feat_msckf.begin(), feat_msckf.end(), [](Feature* feat_a, Feature* feat_b) { return feat_a->parallex > feat_b->parallex; });
        feat_msckf.resize(_max_visual_feat_to_use);
    }

    return feat_msckf;
}

bool VisualManager::construct_feature_jocabian_full(std::vector<Feature*> feats, Eigen::MatrixXd& Hx_full, Eigen::VectorXd &res)
{
    // constexpr size_t kMinFeatsToUpdate = 15;
    // if (feats.size() < kMinFeatsToUpdate) {
    //     LOG(INFO) << cv::format("Too few features to update, feature size: %d", int(feats.size()));
    //     return false;
    // }

    _map_hx.clear();
    _Hx_order.clear();
    int total_hx = 0;
    if (_state->_do_calibration_update) {
        _map_hx.insert({ _state->_imu_to_cam_extrinsic, total_hx });
        _Hx_order.push_back(_state->_imu_to_cam_extrinsic);
        total_hx += _state->_imu_to_cam_extrinsic->size();
    }

    for (auto x : _state->_clone_pose) {
        _map_hx.insert({ x.second, total_hx });
        _Hx_order.push_back(x.second);
        total_hx += x.second->size();
    }

    Hx_full.resize(2 * feats.size() * _state->_clone_pose.size(), total_hx + 1);
    Hx_full.setZero();

    int rows_id = 0;
    for (int i = 0; i < feats.size(); i++) {
        if (feats[i]->_is_triangulated) {
            Eigen::MatrixXd Hx_single = get_single_feature_jacobian(feats[i], _map_hx, total_hx);
            Hx_full.block(rows_id, 0, Hx_single.rows(), Hx_single.cols()) = Hx_single;
            rows_id += Hx_single.rows();
        }
    }
    Hx_full.conservativeResize(rows_id, Hx_full.cols());

    // measurements compression
    if (Hx_full.rows() > Hx_full.cols()) {
        mathematical::nullspace_project_inplace(Hx_full, Hx_full.cols() - 1);
        res.resize(Hx_full.cols() - 1, 1);
        res = Hx_full.block(0, Hx_full.cols() - 1, Hx_full.cols() - 1, 1);
        Hx_full.conservativeResize(Hx_full.cols() - 1, Hx_full.cols() - 1);
    }
    else {
        res.resize(Hx_full.rows(), 1);
        res = Hx_full.block(0, Hx_full.cols() - 1, Hx_full.rows(), 1);
        Hx_full.conservativeResize(Hx_full.rows(), Hx_full.cols() - 1);
    }

    return true;
}

Eigen::MatrixXd VisualManager::get_single_feature_jacobian(Feature* feat, std::unordered_map<std::shared_ptr<Type>, size_t> map_hx, int total_hx)
{
    constexpr int kPwfDim = 3;
    assert(feat->_valid);
    int obs_size = 2 * feat->_visual_obs_buffer.size();
    Eigen::MatrixXd Hfx = Eigen::MatrixXd::Zero(obs_size, total_hx + kPwfDim + 1); // 3 feature dimension + total_hx + 1 residual
    Eigen::Vector3d p_finG = feat->_pwf;
    std::cout << "pwf: " << feat->_pwf.transpose() << std::endl;
    int c = 0;
    for (auto& obs : feat->_visual_obs_buffer) {
        double obs_ts = obs.first;
        Eigen::Vector2d zm(obs.second.u_norm, obs.second.v_norm);
        std::shared_ptr<Pose> obs_pose = _state->_clone_pose.at(obs_ts);
        Eigen::Matrix3d R_IitoG = obs_pose->quat().toRotationMatrix();
        Eigen::Vector3d p_IiinG = obs_pose->p();

        // Eigen::Matrix3d R_ItoC = _state->_imu_to_cam_extrinsic->quat().toRotationMatrix().transpose();
        // Eigen::Vector3d p_IinC = _state->_imu_to_cam_extrinsic->p();
        Eigen::Matrix3d R_CtoI = _state->_imu_to_cam_extrinsic->quat().toRotationMatrix();
        Eigen::Vector3d p_CinI = _state->_imu_to_cam_extrinsic->p();

        Eigen::Matrix3d R_CitoG = R_IitoG * R_CtoI;
        Eigen::Vector3d p_CiinG = p_IiinG + R_IitoG * p_CinI;

        Eigen::Vector3d p_finCi = R_CitoG.transpose() * (p_finG - p_CiinG);

        // compute visual residual
        Eigen::Vector2d uv_norm;
        uv_norm << p_finCi.x() / p_finCi.z(), p_finCi.y() / p_finCi.z();
        Eigen::Vector2d res = zm - uv_norm;
        Hfx.block<2, 1>(2 * c, Hfx.cols() - 1) = res;
        // std::cout << "res: " << res.transpose() << std::endl;

        // precompute dz_dpcf
        Eigen::MatrixXd dz_dpcf = Eigen::MatrixXd::Zero(2, 3);
        dz_dpcf << 1 / p_finCi(2), 0, -p_finCi(0) / (p_finCi(2) * p_finCi(2)),
                   0, 1 / p_finCi(2), -p_finCi(1) / (p_finCi(2) * p_finCi(2));

        // get jacobian wrt pwf
        Eigen::Matrix3d dpcf_dpwf = R_CitoG.transpose();
        Hfx.block<2, kPwfDim>(2 * c, 0) = dz_dpcf * dpcf_dpwf;

        // get jacobian wrt extrinsic parameters
        if (_state->_do_calibration_update) {
            Eigen::MatrixXd dpcf_dcalib = Eigen::MatrixXd::Zero(3, 6);
            // dpcf_dcalib.block<3, 3>(0, 0) = -R_ItoC * mathematical::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
            dpcf_dcalib.block<3, 3>(0, 0) = mathematical::skew(p_finCi);
            // dpcf_dcalib.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
            dpcf_dcalib.block<3, 3>(0, 3) = -R_CtoI.transpose();
            Hfx.block<2, 6>(2 * c, kPwfDim + map_hx.at(_state->_imu_to_cam_extrinsic)) = dz_dpcf * dpcf_dcalib;
        }

        // get jacobian wrt clone pose
        Eigen::MatrixXd dpcf_dclone = Eigen::MatrixXd::Zero(3, 6);
        dpcf_dclone.block<3, 3>(0, 0) = R_CtoI.transpose() * mathematical::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
        dpcf_dclone.block<3, 3>(0, 3) = -R_CitoG.transpose();
        Hfx.block<2, 6>(2 * c, kPwfDim + map_hx.at(obs_pose)) = dz_dpcf * dpcf_dclone;

        c++;

        // // /*check the correctness of Hx*/
        // // check pwf
        // std::cout << "------------------------\n" << std::endl;
        // Eigen::Vector3d dpwf(0.1, 0.1, 0.1);
        // Eigen::Vector2d Hx_plus_dpwf = Hfx.block<2, 3>(2 * c, 0) * dpwf;
        // Eigen::Vector3d pcf_dpwf = R_CitoG.transpose() * (p_finG + dpwf - p_CiinG);
        // Eigen::Vector2d uv_dpwf(pcf_dpwf(0) / pcf_dpwf(2), pcf_dpwf(1) / pcf_dpwf(2));
        // std::cout << "uv_dpwf: " << (uv_dpwf - uv_norm - Hx_plus_dpwf).transpose() << std::endl;

        // // check R_CtoI
        // Eigen::Vector3d dR_CtoI(0.1, 0.1, 0.1);
        // Eigen::Vector2d Hx_plus_dR_ItoC = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(_state->_imu_to_cam_extrinsic)) * dR_CtoI;
        // Eigen::Matrix3d dR_CtoI_mat = Eigen::Matrix3d::Identity() + mathematical::skew(dR_CtoI.head(3));
        // Eigen::Matrix3d R_CitoG_hat = R_IitoG * dR_CtoI_mat;
        // // Eigen::Vector3d p_CiinG = p_IiinG - R_IitoG * p_CinI;
        // Eigen::Vector3d pcf_dR_ItoC = R_CitoG_hat.transpose() * (p_finG - p_CiinG);
        // Eigen::Vector2d uv_dR_ItoC(pcf_dR_ItoC(0) / pcf_dR_ItoC(2), pcf_dR_ItoC(1) / pcf_dR_ItoC(2));
        // std::cout << "uv_dR_ItoC: " << (uv_dR_ItoC - uv_norm - Hx_plus_dR_ItoC).transpose() << std::endl;

        // // check p_IinC
        // Eigen::Vector3d dp_CinI(0.1, 0.1, 0.1);
        // Eigen::Vector2d Hx_plus_dp_IinC = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(_state->_imu_to_cam_extrinsic) + 3) * dp_CinI;
        // Eigen::Vector3d p_CiinG_hat = p_IiinG + R_IitoG * (p_CinI + dp_CinI);
        // Eigen::Vector3d pcf_dp_ItoC = R_CitoG.transpose() * (p_finG - p_CiinG_hat);
        // Eigen::Vector2d uv_dp_ItoC(pcf_dp_ItoC(0) / pcf_dp_ItoC(2), pcf_dp_ItoC(1) / pcf_dp_ItoC(2));
        // std::cout << "uv_dp_ItoC: " << (uv_dp_ItoC - uv_norm - Hx_plus_dp_IinC).transpose() << std::endl;

        // // check clone_R
        // Eigen::Vector3d dR_ItoG(0.1, 0.1, 0.1);
        // Eigen::Vector2d Hx_plus_dR_ItoG = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(obs_pose)) * dR_ItoG;
        // Eigen::Matrix3d dR_ItoG_mat = Eigen::Matrix3d::Identity() + mathematical::skew(dR_ItoG.head(3));
        // Eigen::Matrix3d R_IitoG_hat = R_IitoG * dR_ItoG_mat;
        // p_CiinG_hat = p_IiinG + R_IitoG_hat * p_CinI;
        // Eigen::Vector3d pcf_dR_ItoG = (R_IitoG * dR_ItoG_mat * R_CtoI).transpose() * (p_finG - p_CiinG_hat);
        // Eigen::Vector2d uv_dR_ItoG(pcf_dR_ItoG(0) / pcf_dR_ItoG(2), pcf_dR_ItoG(1) / pcf_dR_ItoG(2));
        // std::cout << "uv_dR_ItoG: " << (uv_dR_ItoG - uv_norm - Hx_plus_dR_ItoG).transpose() << std::endl;

        // // check clone_p
        // Eigen::Vector3d dp_IinG(0.1, 0.1, 0.1);
        // Eigen::Vector2d Hx_plus_dp_IinG = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(obs_pose) + 3) * dp_IinG;
        // p_CiinG_hat = p_IiinG + dp_IinG + R_IitoG * p_CinI;
        // Eigen::Vector3d pcf_dp_IinG = R_CitoG.transpose() * (p_finG - p_CiinG_hat);
        // Eigen::Vector2d uv_dp_IinG(pcf_dp_IinG(0) / pcf_dp_IinG(2), pcf_dp_IinG(1) / pcf_dp_IinG(2));
        // std::cout << "uv_dp_IinG: " << (uv_dp_IinG - uv_norm - Hx_plus_dp_IinG).transpose() << std::endl;
    }

    // project Hfx to feature left null space
    mathematical::nullspace_project_inplace(Hfx, 3);
    Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(Hfx.rows() - 3, Hfx.cols() - 3);
    Hx.noalias() = Hfx.block(3, 3, Hfx.rows() - 3, Hfx.cols() - 3);

    return Hx;
}