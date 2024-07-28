#include <vector>
#include "visualManager.h"
#include "mathematical_tools.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace {
    constexpr int kMinFeatForMapping = 2;
    constexpr double kMaxConditionNum = 10000.f;
    constexpr double kMinTriangDist = 0.2;
    constexpr double kMaxTriangDist = 20;
    constexpr int kMaxIterationTimes = 5;
}

bool VisualManager::least_square_triangulation(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat)
{
    auto last_obs = feat->_visual_obs_buffer.end();
    last_obs--;

    Eigen::Matrix3d R_AtoG = clone_pose_buffer[last_obs->first].Rwc;
    Eigen::Vector3d p_AinG = clone_pose_buffer[last_obs->first].pwc;

    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d ATb = Eigen::Vector3d::Zero();

    int feat_index = 0;
    for (auto it = feat->_visual_obs_buffer.begin(); it != feat->_visual_obs_buffer.end(); it++) {
        Eigen::Matrix3d R_CitoG = clone_pose_buffer[it->first].Rwc;
        Eigen::Vector3d p_CiinG = clone_pose_buffer[it->first].pwc;

        Eigen::Matrix3d R_CitoA = R_AtoG.transpose() * R_CitoG;
        Eigen::Vector3d p_CiinA = R_AtoG.transpose() * (p_CiinG - p_AinG);

        Eigen::Vector3d b_i;
        b_i << it->second.u_norm, it->second.v_norm, 1;
        Eigen::Vector3d b_iinA = R_CitoA * b_i;
        ATA += mathematical::skew(b_iinA).transpose() * mathematical::skew(b_iinA);
        ATb += mathematical::skew(b_iinA).transpose() * mathematical::skew(b_iinA) * p_CiinA;
    }

    Eigen::Vector3d paf = ATA.colPivHouseholderQr().solve(ATb);
    // Check A and paf
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ATA);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    if (std::abs(condA) > kMaxConditionNum || paf(2, 0) < kMinTriangDist || paf(2, 0) > kMaxTriangDist || std::isnan(paf.norm())) {
        return false;
    }
    feat->_pwf = p_AinG + R_AtoG * paf;

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
    int iter_time = kMaxIterationTimes;
    while (iter_time > 0) {
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
        iter_time--;
    }

    Eigen::Vector3d paf_opt;
    paf_opt << alpha/rho, beta/rho, 1/rho;
    feat->_pwf = p_AinG + R_AtoG * paf_opt;

    return true;
}

void VisualManager::pnp_ransac_to_reject_outliers(std::vector<Feature* > feats)
{
    double ts = _state->ts_sec();
    std::vector<cv::Point3d> list_points3d;
    std::vector<cv::Point2d> list_points2d;
    for (auto it = feats.begin(); it != feats.end(); it++) {
        if ((*it)->_valid && (*it)->_is_triangulated) {
            assert((*it)->_visual_obs_buffer.find(ts) != (*it)->_visual_obs_buffer.end());
            cv::Point3d p3d((*it)->_pwf.x(), (*it)->_pwf.y(), (*it)->_pwf.z());
            cam_obs_t obs_2d = (*it)->_visual_obs_buffer.at(ts);
            cv::Point2d p2d(obs_2d.u, obs_2d.v);
            list_points3d.push_back(p3d);
            list_points2d.push_back(p2d);
        }
    }
    cv::Mat intrinsic;
    cv::Mat distortion;
    cv::Mat inliers;
    cv::eigen2cv(_camera_model->intrinsic(), intrinsic);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::solvePnPRansac(list_points3d, list_points2d,
                       intrinsic, distortion,
                       rvec, tvec, false, 100, 3.f, 0.95,
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
    return;
}

void VisualManager::feature_triangulation(std::vector<Feature* > feats, std::map<double, CameraPose> camera_pose_buffer)
{
    // std::map<double, CameraPose> camera_pose_buffer = _state->access_clone_pose_buffer();

    for (auto it = feats.begin(); it != feats.end(); it++) {
        if ((*it)->_visual_obs_buffer.size() < kMinFeatForMapping) {
            it = feats.erase(it);
            continue;
        }

        if ((*it)->_is_triangulated == false)
        {
            if(false == least_square_triangulation(camera_pose_buffer, *it)) {
                it = feats.erase(it);
                continue;
            }
        }

        if (true == gaussian_newton_optimization(camera_pose_buffer, *it)) {
            (*it)->_is_triangulated = true;
        } else {
            it = feats.erase(it);
            continue;
        }
    }
}

bool VisualManager::construct_feature_jocabian_full(std::vector<Feature*> feats, Eigen::MatrixXd& Hx_full)
{
    constexpr size_t kMinFeatsToUpdate = 15;
    if (feats.size() < kMinFeatsToUpdate) {
        LOG(INFO) << cv::format("Too few features to update, feature size: %d", int(feats.size()));
        return false;
    }

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

    Hx_full.resize(2 * feats.size() * _state->_clone_pose.size(), total_hx);
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
    return true;
}

Eigen::MatrixXd VisualManager::get_single_feature_jacobian(Feature* feat, std::unordered_map<std::shared_ptr<Type>, size_t> map_hx, int total_hx)
{
    constexpr int kPwfDim = 3;
    assert(feat->_valid);
    int obs_size = 2 * feat->_visual_obs_buffer.size();
    Eigen::MatrixXd Hfx = Eigen::MatrixXd::Zero(obs_size, total_hx + kPwfDim + 1); // 3 feature dimension + total_hx + 1 residual
    Eigen::Vector3d p_finG = feat->_pwf;
    int c = 0;
    for (auto& obs : feat->_visual_obs_buffer) {
        double obs_ts = obs.first;
        Eigen::Vector2d zm(obs.second.u_norm, obs.second.v_norm);
        std::shared_ptr<Pose> obs_pose = _state->_clone_pose.at(obs_ts);
        Eigen::Matrix3d R_IitoG = obs_pose->quat().toRotationMatrix();
        Eigen::Vector3d p_IiinG = obs_pose->p();

        Eigen::Matrix3d R_ItoC = _state->_imu_to_cam_extrinsic->quat().toRotationMatrix();
        Eigen::Vector3d p_IinC = _state->_imu_to_cam_extrinsic->p();

        Eigen::Matrix3d R_CitoG = R_IitoG * R_ItoC.transpose();
        Eigen::Vector3d p_CiinG = p_IiinG - R_CitoG * p_IinC;

        Eigen::Vector3d p_finCi = R_CitoG.transpose() * (p_finG - p_CiinG);

        // compute visual residual
        Eigen::Vector2d uv_norm;
        uv_norm << p_finCi.x() / p_finCi.z(), p_finCi.y() / p_finCi.z();
        Eigen::Vector2d res = zm - uv_norm;
        Hfx.block<2, 1>(2 * c, Hfx.cols() - 1) = res;

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
            dpcf_dcalib.block<3, 3>(0, 0) = -R_ItoC * mathematical::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
            dpcf_dcalib.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
            Hfx.block<2, 6>(2 * c, kPwfDim + map_hx.at(_state->_imu_to_cam_extrinsic)) = dz_dpcf * dpcf_dcalib;
        }

        // get jacobian wrt clone pose
        Eigen::MatrixXd dpcf_dclone = Eigen::MatrixXd::Zero(3, 6);
        dpcf_dclone.block<3, 3>(0, 0) = R_ItoC * mathematical::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
        dpcf_dclone.block<3, 3>(0, 3) = -R_CitoG.transpose();
        Hfx.block<2, 6>(2 * c, kPwfDim + map_hx.at(obs_pose)) = dz_dpcf * dpcf_dclone;

        // // /*check the correctness of Hx*/
        // // check pwf
        // std::cout << "------------------------\n" << std::endl;
        // Eigen::Vector3d dpwf(0.01, 0.01, 0.01);
        // Eigen::Vector2d Hx_plus_dpwf = Hfx.block<2, 3>(2 * c, 0) * dpwf;
        // Eigen::Vector3d pcf_dpwf = R_CitoG.transpose() * (p_finG + dpwf - p_CiinG);
        // Eigen::Vector2d uv_dpwf(pcf_dpwf(0) / pcf_dpwf(2), pcf_dpwf(1) / pcf_dpwf(2));
        // std::cout << "uv_dpwf: " << (uv_dpwf - uv_norm - Hx_plus_dpwf).transpose() << std::endl;

        // // check R_ItoC
        // Eigen::Vector3d dR_ItoC(0.01, 0.01, 0.01);
        // Eigen::Vector2d Hx_plus_dR_ItoC = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(_state->_imu_to_cam_extrinsic)) * dR_ItoC;
        // Eigen::Matrix3d dR_ItoC_mat = Eigen::Matrix3d::Identity() + mathematical::skew(dR_ItoC.head(3));
        // Eigen::Matrix3d R_CitoG_hat = R_IitoG * (R_ItoC * dR_ItoC_mat).transpose();
        // Eigen::Vector3d p_CiinG_hat = p_IiinG - R_CitoG_hat * p_IinC;
        // Eigen::Vector3d pcf_dR_ItoC = R_CitoG_hat.transpose() * (p_finG - p_CiinG_hat);
        // Eigen::Vector2d uv_dR_ItoC(pcf_dR_ItoC(0) / pcf_dR_ItoC(2), pcf_dR_ItoC(1) / pcf_dR_ItoC(2));
        // std::cout << "uv_dR_ItoC: " << (uv_dR_ItoC - uv_norm - Hx_plus_dR_ItoC).transpose() << std::endl;

        // // check p_IinC
        // Eigen::Vector3d dp_IinC(0.01, 0.01, 0.01);
        // Eigen::Vector2d Hx_plus_dp_IinC = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(_state->_imu_to_cam_extrinsic) + 3) * dp_IinC;
        // p_CiinG_hat = p_IiinG - R_CitoG * (p_IinC + dp_IinC);
        // Eigen::Vector3d pcf_dp_ItoC = R_CitoG.transpose() * (p_finG - p_CiinG_hat);
        // Eigen::Vector2d uv_dp_ItoC(pcf_dp_ItoC(0) / pcf_dp_ItoC(2), pcf_dp_ItoC(1) / pcf_dp_ItoC(2));
        // std::cout << "uv_dp_ItoC: " << (uv_dp_ItoC - uv_norm - Hx_plus_dp_IinC).transpose() << std::endl;

        // // check clone_R
        // Eigen::Vector3d dR_ItoG(0.01, 0.01, 0.01);
        // Eigen::Vector2d Hx_plus_dR_ItoG = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(obs_pose)) * dR_ItoG;
        // Eigen::Matrix3d dR_ItoG_mat = Eigen::Matrix3d::Identity() + mathematical::skew(dR_ItoG.head(3));
        // R_CitoG_hat = R_IitoG * dR_ItoG_mat * R_ItoC.transpose();
        // p_CiinG_hat = p_IiinG - R_CitoG_hat * p_IinC;
        // Eigen::Vector3d pcf_dR_ItoG = R_CitoG_hat.transpose() * (p_finG - p_CiinG_hat);
        // Eigen::Vector2d uv_dR_ItoG(pcf_dR_ItoG(0) / pcf_dR_ItoG(2), pcf_dR_ItoG(1) / pcf_dR_ItoG(2));
        // std::cout << "uv_dR_ItoG: " << (uv_dR_ItoG - uv_norm - Hx_plus_dR_ItoG).transpose() << std::endl;

        // // check clone_p
        // Eigen::Vector3d dp_IinG(0.01, 0.01, 0.01);
        // Eigen::Vector2d Hx_plus_dp_IinG = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(obs_pose) + 3) * dp_IinG;
        // p_CiinG_hat = p_IiinG + dp_IinG - R_CitoG * p_IinC;
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