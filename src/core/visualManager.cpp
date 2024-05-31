#include <vector>
#include "visualManager.h"
#include "mathematical_tools.h"

namespace {
    constexpr int kMinFeatForMapping = 2;
    constexpr double kMaxConditionNum = 10000.f;
    constexpr double kMinTriangDist = 0.2;
    constexpr double kMaxTriangDist = 20;
    constexpr int kMaxIterationTimes = 10;
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

            Eigen::Vector3d res;
            res(0, 0) = z_m(0, 0) - h(0, 0) / h(2, 0);
            res(1, 0) = z_m(1, 0) - h(1, 0) / h(2, 0);
            res(2, 0) = 0.f;

            Eigen::MatrixXd Jacobian_1(2, 3);
            Eigen::MatrixXd Jacobian_2(3, 3);
            Jacobian_1 << 1 / h(2, 0), 0, -h(0, 0) / std::pow(h(2, 0), 2),
                0, 1 / h(2, 0), -h(1, 0) / std::pow(h(2, 0), 2);
            Jacobian_2.block<3, 1>(0, 0) << 1, 0, 0;
            Jacobian_2.block<3, 1>(0, 1) << 0, 1, 0;
            Jacobian_2.block<3, 1>(0, 2) << -p_CiinA;
            Jacobian_2 = R_AtoCi * Jacobian_2.eval();

            Eigen::Matrix3d Jacobian = Jacobian_1 * Jacobian_2;
            ATA += Jacobian.transpose() * Jacobian;
            ATb += -Jacobian.transpose() * res;
        }

        Eigen::Vector3d delta_x = ATA.colPivHouseholderQr().solve(ATb);
        alpha += delta_x.x();
        beta += delta_x.y();
        rho += delta_x.z();
        iter_time--;
    }

    Eigen::Vector3d paf_opt;
    paf_opt << alpha / rho, beta / rho, 1 / rho;
    feat->_pwf = p_AinG + R_AtoG * paf_opt;

    return true;
}

void VisualManager::feature_triangulation(std::vector<Feature* > feats)
{
    std::map<double, CameraPose> camera_pose_buffer = _state->access_clone_pose_buffer();

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