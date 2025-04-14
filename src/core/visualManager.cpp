#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mathematical_tools.h"
#include "visualManager.h"

#define SHOW_CLONE_POSES 1

namespace
{
constexpr uint32_t kMinFeatForMapping = 1;
constexpr double kMaxConditionNum = 20000.f;
constexpr double kMinTriangDist = 0.05;
constexpr double kMaxTriangDist = 15;
constexpr uint32_t kMaxIterationTimes = 5;
constexpr uint32_t kMinFeatNumToUpdate = 15;
}  // namespace

bool VisualManager::VisualUpdate()
{
    constexpr uint32_t kMinFeatToUpdate = 10;
    bool visual_updated = false;
    std::map<double, CameraPose> camera_pose_buffer = _state->AccessClonePoseBuffer();
    auto feat_origin_input = _feature_tracked;
    FeatureTriangulation(_feature_tracked, camera_pose_buffer);
    CalculateFeatureParallex(_feature_tracked);

    std::vector<Feature*> feat_msckf = SelectMsckfFeatures(_feature_tracked);

    // if (feat_msckf.size() > kMinFeatToUpdate)
    // {
    //     int msckf_feature_num_before_pnp = feat_msckf.size();
    //     PnpRansacToRejectOutliers(feat_msckf);
    //     int msckf_feature_num_after_pnp = feat_msckf.size();
    //     std::cout << "pnp reject outliers: " << msckf_feature_num_before_pnp - msckf_feature_num_after_pnp << std::endl;
    // }

#if SHOW_CLONE_POSES

    constexpr double fr = 11.333;
    constexpr double fb = 22.333;
    constexpr double fg = 33.333;
    std::map<double, cv::Mat> clone_image_map;
    for (auto it = _state->_clone_pose.begin(); it != _state->_clone_pose.end(); it++)
    {
        double timestamp = it->first;
        cv::Mat image = stored_images_.at(timestamp).first.clone();
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        clone_image_map.insert(std::make_pair(timestamp, image));
    }

    for (auto it = _feature_tracked.begin(); it != _feature_tracked.end(); it++)
    {
        if ((*it)->_is_triangulated == false)
        {
            continue;
        }

        for (auto it_feat = (*it)->_visual_obs_buffer.begin(); it_feat != (*it)->_visual_obs_buffer.end(); it_feat++)
        {
            uint32_t feat_id = (*it)->_id;
            double timestamp = it_feat->first;
            cv::Point2f point(it_feat->second.u, it_feat->second.v);

            CameraPose camera_pose = camera_pose_buffer.at(timestamp);
            Eigen::Vector3d pcf = camera_pose.Rwc.transpose() * ((*it)->_pwf - camera_pose.pwc);
            Eigen::Vector2d p_uv = _camera_model->project_left(pcf);
            cv::Point2f point_reproj(p_uv(0), p_uv(1));

            std::ostringstream os;
            os << std::fixed << feat_id;
            std::string depth_text = os.str();
            cv::putText(clone_image_map.at(timestamp), depth_text, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            cv::Scalar color = cv::Scalar(int(fb * feat_id) % 255, int(fg * feat_id) % 255, int(fr * feat_id) % 255);
            cv::circle(clone_image_map.at(timestamp), point, 4, color, -1);
            cv::circle(clone_image_map.at(timestamp), point_reproj, 5, color, 1);
        }
    }

    std::vector<cv::Mat> images_to_show;
    for (auto it = clone_image_map.begin(); it != clone_image_map.end(); it++)
    {
        images_to_show.push_back(it->second);
    }
    Utils::ShowGridImages(images_to_show);

#endif

    _origin_feature_tracked = feat_msckf.size();
    if (feat_msckf.size() > kMinFeatToUpdate)
    {
        Eigen::MatrixXd Hx_msckf;
        Eigen::VectorXd res;
        ConstructFeatureJacobianFull(feat_msckf, Hx_msckf, res);
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(Hx_msckf.rows(), Hx_msckf.rows()) * 4;

        if (_state->_clone_pose.size() >= 2)
        {
            solver_->update(_state, Hx_msckf, res, _Hx_order, _map_hx, R);
            visual_updated = true;
        }
    }
    else
    {
        LOG(INFO) << cv::format("Not enough features to update, features_tracked: %d, features_msckf: %d, feature mapping success: %d",
                                int(_feature_tracked.size()), int(feat_msckf.size()), _feature_mapping_success);
    }

    std::shared_ptr<Type> state_to_marginalize = nullptr;

    if (_state->_clone_pose.size() >= _max_clone_pose)
    {
        _keyframe = KeyFrameStatus::kNone;
        _keyframe = CheckKeyframe(_state, feat_msckf);
        if (_keyframe == KeyFrameStatus::kNone)
        {
            state_to_marginalize = _state->_clone_pose.rbegin()->second;
            DropFeatureObsrvs(state_to_marginalize->ts());
        }
        else  // drop lastest pose
        {
            state_to_marginalize = _state->_clone_pose.begin()->second;
            DropFeatureObsrvs(state_to_marginalize->ts());
            UpdateFeatureBase();
        }
    }

    // _state->MarginalizeState(state_to_marginalize);
    solver_->MarginalizeState(_state, state_to_marginalize);

    return visual_updated;
}

KeyFrameStatus VisualManager::CheckKeyframe(std::shared_ptr<State> _state, std::vector<Feature*> feats)
{
    constexpr double kLargeParallexThres = 15.f;  // 大于5个平均像素视差则视为关键帧

    if (_state->_clone_pose.size() < _max_clone_pose)
    {
        return KeyFrameStatus::kNone;
    }

    uint32_t cnt = 0;
    double pixel_parallex = 0.f;
    double pixel_parallex_avg = 0.f;
    for (auto x : feats)
    {
        if (x->_visual_obs_buffer.size() < 1)
        {
            continue;
        }
        auto last_it = x->_visual_obs_buffer.rbegin();
        double last_ts = last_it->first;
        auto sub_last_it = x->_visual_obs_buffer.rbegin();
        ++sub_last_it;
        double sub_last_ts = sub_last_it->first;

        assert(sub_last_ts < last_ts);
        CameraObs curr_obs = last_it->second;
        CameraObs prev_obs = sub_last_it->second;
        double diff_u = curr_obs.u - prev_obs.u;
        double diff_v = curr_obs.v - prev_obs.v;
        pixel_parallex = std::sqrt(std::pow(diff_u, 2) + std::pow(diff_v, 2));
        pixel_parallex_avg += pixel_parallex;
        cnt++;
    }
    pixel_parallex_avg = pixel_parallex_avg / cnt;
    if (pixel_parallex_avg > kLargeParallexThres)
    {
        return KeyFrameStatus::kLargeParallex;
    }
    else if (_origin_feature_tracked < 20)
    {
        // std::cout << "_origin_feature_tracked: " << _origin_feature_tracked << std::endl;
        return KeyFrameStatus::kFeatureLostTooMuch;
    }

    return KeyFrameStatus::kNone;
}

void VisualManager::DropFeatureObsrvs(const double timestamp_to_drop)
{
    for (auto& feat_base : _feature_base)
    {
        if (feat_base->_visual_obs_buffer.count(timestamp_to_drop) != 0)
        {
            auto it = feat_base->_visual_obs_buffer.find(timestamp_to_drop);
            feat_base->_visual_obs_buffer.erase(it);

            if (feat_base->_visual_obs_buffer.size() == 0)
            {
                feat_base->reset();
            }
        }
    }
}

void VisualManager::UpdateFeatureBase()
{
    for (auto& feat_lost : _feature_lost)
    {
        feat_lost->reset();
    }

    int feature_valid_num = 0;
    for (auto x : _feature_base)
    {
        if (x->_valid)
        {
            feature_valid_num++;
        }
    }

    for (auto& feat : _feature_new)
    {
        for (int i = 0; i < _feature_base.size(); i++)
        {
            if (!_feature_base[i]->_valid)
            {
                _feature_base[i]->reset();
                _feature_base[i]->_id = feat.feat_id;
                _feature_base[i]->_valid = true;
                _feature_base[i]->_visual_obs_buffer.insert({feat.ts_sec, feat});
                break;
            }
        }
    }

    feature_valid_num = 0;
    for (auto x : _feature_base)
    {
        if (x->_valid)
        {
            feature_valid_num++;
        }
    }
}

void VisualManager::UpdateFeature(std::pair<double, std::vector<CameraObs>> feature_observes)
{
    double ts_sec = feature_observes.first;
    std::vector<CameraObs> feature_obs = feature_observes.second;
    std::unordered_map<uint32_t, CameraObs> feature_obsrv_umap;
    std::unordered_set<uint32_t> feature_tracked_id_uset;

    for (auto& feat_obsrv : feature_observes.second)
    {
        _camera_model->back_project_stereo(feat_obsrv);
        feature_obsrv_umap.insert({feat_obsrv.feat_id, feat_obsrv});
    }

    _feature_lost.clear();
    _feature_new.clear();
    _feature_tracked.clear();

    for (int i = 0; i < _feature_base.size(); i++)
    {
        Feature* feature = _feature_base[i];
        if (!feature->_valid)
        {
            continue;
        }

        if (feature_obsrv_umap.find(feature->_id) != feature_obsrv_umap.end())
        {
            _feature_tracked.push_back(feature);
            feature_tracked_id_uset.insert(feature->_id);
            feature->_visual_obs_buffer.insert({ts_sec, feature_obsrv_umap[feature->_id]});
        }
        else
        {
            _feature_lost.push_back(feature);
        }
    }

    for (int i = 0; i < feature_observes.second.size(); i++)
    {
        uint32_t feature_observ_id = feature_observes.second[i].feat_id;
        if (feature_tracked_id_uset.find(feature_observ_id) == feature_tracked_id_uset.end())
        {
            _feature_new.push_back(feature_observes.second[i]);
        }
    }

    // std::cout << "feature observed: " << feature_observes.second.size() << std::endl;
    // std::cout << "feature tracked: " << _feature_tracked.size() << std::endl;
    // std::cout << "feature lost: " << _feature_lost.size() << std::endl;
    // std::cout << "feature new: " << _feature_new.size() << std::endl;
}

void VisualManager::ResetFeatureBase()
{
    // _feature_base.clear();
    _feature_new.clear();
    _feature_lost.clear();
    _feature_tracked.clear();
}

void VisualManager::InitFeatureBase(std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_feature_triangulated)
{
    int cnt = 0;
    for (auto& [feature_id, feature_obs] : stereo_feature_triangulated)
    {
        if (cnt >= _max_feat_n)
        {
            break;
        }

        Feature* feature = new Feature();
        feature->_id = feature_id;
        feature->_pwf = feature_obs.second;
        feature->_valid = true;
        feature->_is_triangulated = true;
        feature->_visual_obs_buffer.insert({feature_obs.first.ts_sec, feature_obs.first});
        _feature_base[cnt] = feature;
        cnt++;
    }
}

bool VisualManager::least_square_triangulation(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat)
{
    auto first_obs = feat->_visual_obs_buffer.begin();
    Eigen::Matrix3d R_AtoG = clone_pose_buffer[first_obs->first].Rwc;
    Eigen::Vector3d p_AinG = clone_pose_buffer[first_obs->first].pwc;

    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d ATb = Eigen::Vector3d::Zero();

    int feat_index = 0;
    for (auto it = feat->_visual_obs_buffer.begin(); it != feat->_visual_obs_buffer.end(); it++)
    {
        _camera_model->back_project_stereo(it->second);
        for (int cam_id = 0; cam_id < MAX_CAM_NUM; cam_id++)
        {
            Eigen::Matrix3d R_CitoG;
            Eigen::Vector3d p_CiinG;
            Eigen::Vector3d b_i;
            if (cam_id == LEFT_CAM)
            {
                R_CitoG = clone_pose_buffer[it->first].Rwc;
                p_CiinG = clone_pose_buffer[it->first].pwc;
                b_i << it->second.u_norm, it->second.v_norm, 1;
            }
            else if (cam_id == RIGHT_CAM)
            {
                R_CitoG = clone_pose_buffer[it->first].Rwc * _camera_model->R_rl();
                p_CiinG = clone_pose_buffer[it->first].pwc + clone_pose_buffer[it->first].Rwc * _camera_model->p_rl();
                b_i << it->second.ur_norm, it->second.vr_norm, 1;
            }

            Eigen::Matrix3d R_CitoA = R_AtoG.transpose() * R_CitoG;
            Eigen::Vector3d p_CiinA = R_AtoG.transpose() * (p_CiinG - p_AinG);
            // std::cout << R_CitoA << std::endl;
            // std::cout << p_CiinA.transpose() << std::endl;

            Eigen::Vector3d b_iinA = R_CitoA * b_i;
            ATA += MathUtils::skew(b_iinA).transpose() * MathUtils::skew(b_iinA);
            ATb += MathUtils::skew(b_iinA).transpose() * MathUtils::skew(b_iinA) * p_CiinA;
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
    // If we have a bad condition number, or it is too close Then set the flag for bad (i.e. set z-axis to nan)
    int triang_failed_num = 0;
    if (std::abs(condA) > kMaxConditionNum || paf(2, 0) < kMinTriangDist || paf(2, 0) > kMaxTriangDist || std::isnan(paf.norm()))
    {
        triang_failed_num++;
        // if (std::abs(condA) > kMaxConditionNum)
        // {
        //     std::cout << cv::format("condition num [%f] > [%f]", condA, kMaxConditionNum) << std::endl;
        // }
        // else if (paf(2, 0) < kMinTriangDist || paf(2, 0) > kMaxTriangDist)
        // {
        //     std::cout << "paf is out of range" <<  paf.transpose() << std::endl;
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

bool VisualManager::GaussianNewtonOptimization(std::map<double, CameraPose>& clone_pose_buffer, Feature* feat)
{
    auto last_obs = feat->_visual_obs_buffer.end();
    last_obs--;
    Eigen::Matrix3d R_AtoG = clone_pose_buffer[last_obs->first].Rwc;
    Eigen::Vector3d p_AinG = clone_pose_buffer[last_obs->first].pwc;

    Eigen::Vector3d paf = R_AtoG.transpose() * (feat->_pwf - p_AinG);
    if (abs(paf.z()) < 1e-2)
    {
        return false;
    }

    double alpha = paf.x() / paf.z();
    double beta = paf.y() / paf.z();
    double rho = 1 / paf.z();
    int iter_time = 0;
    while (iter_time < kMaxIterationTimes)
    {
        Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
        Eigen::Vector3d ATb = Eigen::Vector3d::Zero();
        for (auto it = feat->_visual_obs_buffer.begin(); it != feat->_visual_obs_buffer.end(); it++)
        {
            _camera_model->back_project_stereo((*it).second);
            for (int cam_id = 0; cam_id < MAX_CAM_NUM; cam_id++)
            {
                double feature_timestamp = (*it).first;
                Eigen::Matrix3d R_CitoG;
                Eigen::Vector3d p_CiinG;
                Eigen::Vector2d z_m;

                if (cam_id == LEFT_CAM)
                {
                    z_m << (*it).second.u_norm, (*it).second.v_norm;
                    R_CitoG = clone_pose_buffer[feature_timestamp].Rwc;
                    p_CiinG = clone_pose_buffer[feature_timestamp].pwc;
                }
                else if (cam_id == RIGHT_CAM)
                {
                    z_m << (*it).second.ur_norm, (*it).second.vr_norm;
                    R_CitoG = clone_pose_buffer[feature_timestamp].Rwc * _camera_model->R_rl();
                    p_CiinG = clone_pose_buffer[feature_timestamp].pwc + clone_pose_buffer[feature_timestamp].Rwc * _camera_model->p_rl();
                }

                Eigen::Matrix3d R_AtoCi = R_CitoG.transpose() * R_AtoG;
                Eigen::Vector3d p_CiinA = R_AtoG.transpose() * (p_CiinG - p_AinG);

                Eigen::Vector3d paf_norm;
                paf_norm << alpha, beta, 1;
                Eigen::Vector3d h = R_AtoCi * (paf_norm - rho * p_CiinA);
                if (h(2, 0) < 1e-6)
                {
                    continue;
                }

                Eigen::Vector2d res;
                res(0, 0) = z_m(0, 0) - h(0, 0) / h(2, 0);
                res(1, 0) = z_m(1, 0) - h(1, 0) / h(2, 0);

                Eigen::MatrixXd Jacobian_1(2, 3);
                Eigen::MatrixXd Jacobian_2(3, 3);
                Jacobian_1 << 1 / h(2, 0), 0, -h(0, 0) / std::pow(h(2, 0), 2), 0, 1 / h(2, 0), -h(1, 0) / std::pow(h(2, 0), 2);
                Jacobian_2.block<3, 1>(0, 0) << 1, 0, 0;
                Jacobian_2.block<3, 1>(0, 1) << 0, 1, 0;
                Jacobian_2.block<3, 1>(0, 2) << -p_CiinA;
                Jacobian_2 = R_AtoCi * Jacobian_2.eval();

                Eigen::MatrixXd Jacobian(2, 3);
                Jacobian = Jacobian_1 * Jacobian_2;
                ATA += Jacobian.transpose() * Jacobian;
                ATb += Jacobian.transpose() * res;
            }
        }

        Eigen::Vector3d delta_x = ATA.colPivHouseholderQr().solve(ATb);
        alpha += delta_x.x();
        beta += delta_x.y();
        rho += delta_x.z();

        if (delta_x.norm() < 1e-2)
        {
            break;
        }
        else
        {
            iter_time++;
        }
    }

    Eigen::Vector3d paf_opt;
    paf_opt << alpha / rho, beta / rho, 1 / rho;
    feat->_pwf = p_AinG + R_AtoG * paf_opt;

    if (paf_opt.norm() > 30 || iter_time == kMaxIterationTimes)
    {
        if (paf_opt.norm() > 30)
        {
            std::cout << "paf_opt is too large, paf norm: " << paf_opt.norm() << std::endl;
        }
        else if (iter_time == kMaxIterationTimes)
        {
            std::cout << "iter_time is too large" << std::endl;
        }
        return false;
    }

    return true;
}

bool VisualManager::StereoLeastSqureTriangulation(const std::shared_ptr<CameraModel> camera_model, CameraObs& cam_obs, Eigen::Vector3d& pcf) const
{
    if (camera_model->camera_num() != 2)
    {
        LOG(ERROR) << "Stereo triangulation only support stereo camera model";
        return false;
    }
    camera_model->back_project_stereo(cam_obs);
    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d ATb = Eigen::Vector3d::Zero();

    for (int cam_id = 0; cam_id < MAX_CAM_NUM; cam_id++)
    {
        Eigen::Matrix3d R_CitoA = Eigen::Matrix3d::Identity();
        Eigen::Vector3d p_CiinA = Eigen::Vector3d::Zero();
        Eigen::Vector3d b_i = Eigen::Vector3d::Zero();

        if (cam_id == LEFT_CAM)
        {
            b_i << cam_obs.u_norm, cam_obs.v_norm, 1;
        }
        else if (cam_id == RIGHT_CAM)
        {
            b_i << cam_obs.ur_norm, cam_obs.vr_norm, 1;
            R_CitoA = camera_model->R_rl();
            p_CiinA = camera_model->p_rl();
        }

        Eigen::Vector3d b_iinA = R_CitoA * b_i;
        ATA += MathUtils::skew(b_iinA).transpose() * MathUtils::skew(b_iinA);
        ATb += MathUtils::skew(b_iinA).transpose() * MathUtils::skew(b_iinA) * p_CiinA;
    }

    pcf = ATA.colPivHouseholderQr().solve(ATb);
    if (pcf(2, 0) < kMinTriangDist || pcf(2, 0) > kMaxTriangDist || std::isnan(pcf.norm()))
    {
        return false;
    }

    return true;
}

void VisualManager::reset()
{
    _feature_mapping_success = 0;
    _feature_mapping_in = 0;
    _feature_tracked.clear();
    _feature_lost.clear();
    _feature_new.clear();
    _input_image_buffer = std::queue<std::pair<double, std::pair<cv::Mat, cv::Mat>>>();
    feature_obs_buffer = std::queue<std::pair<double, std::vector<CameraObs>>>();
    stored_images_.clear();

    assert(_feature_base.size() == _max_feat_n);
    for (int i = 0; i < _max_feat_n; i++)
    {
        _feature_base[i]->reset();
    }
}

void VisualManager::FeedImages(const std::pair<double, std::pair<cv::Mat, cv::Mat>> input)
{
    while (_input_image_buffer.size() > kMaxImageBufferSize)
    {
        _input_image_buffer.pop();
    }
    _input_image_buffer.push(input);
    stored_images_.insert(input);
}

bool VisualManager::PnpRansacToRejectOutliers(std::vector<Feature*> feats)
{
    constexpr double kReprijectionErrorThres = 2.0f;
    double timestamp = _state->ts_sec();
    std::vector<cv::Point3d> list_points3d;
    std::vector<cv::Point2d> list_points2d;
    for (auto it = feats.begin(); it != feats.end(); it++)
    {
        if ((*it)->_valid && (*it)->_is_triangulated)
        {
            assert((*it)->_visual_obs_buffer.find(timestamp) != (*it)->_visual_obs_buffer.end());
            list_points3d.emplace_back((*it)->_pwf.x(), (*it)->_pwf.y(), (*it)->_pwf.z());
            CameraObs obs_2d = (*it)->_visual_obs_buffer.at(timestamp);
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
    cv::eigen2cv(_camera_model->K_l(), intrinsic);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::solvePnPRansac(list_points3d, list_points2d, intrinsic, distortion, rvec, tvec, false, 100, kReprijectionErrorThres, 0.8, inliers,
                       cv::SOLVEPNP_UPNP);

    std::vector<int> inliers_id;
    for (int i = 0; i < inliers.rows; i++)
    {
        int id = feats[inliers.at<int>(i)]->_id;
        inliers_id.push_back(id);
    }

    for (auto it = feats.begin(); it != feats.end();)
    {
        if (!(*it)->_valid || !(*it)->_is_triangulated)
        {
            continue;
        }

        if (std::find(inliers_id.begin(), inliers_id.end(), (*it)->_id) == inliers_id.end())
        {
            it = feats.erase(it);
            continue;
        }

        it++;
    }

    return true;
}

bool VisualManager::PnpRansac(const std::shared_ptr<CameraModel> camera_model,
                              std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_obs_triangulated,
                              Eigen::Matrix3d& R_12,
                              Eigen::Vector3d& p_12) const
{
    constexpr int kMinFeaturesForPnp = 15;

    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (auto& [feat_id, obs_pwf] : stereo_obs_triangulated)
    {
        points_3d.push_back(cv::Point3f(obs_pwf.second.x(), obs_pwf.second.y(), obs_pwf.second.z()));
        points_2d.push_back(cv::Point2f(obs_pwf.first.u, obs_pwf.first.v));
    }

    cv::Mat rvec, tvec;
    cv::Mat inliers;
    cv::Mat intrinsic;
    cv::Mat distortion;
    cv::eigen2cv(camera_model->K_l(), intrinsic);
    cv::solvePnPRansac(points_3d, points_2d, intrinsic, distortion, rvec, tvec, false, 100, 3.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

    if (inliers.rows < kMinFeaturesForPnp)
    {
        LOG(ERROR) << cv::format("PnpRansac failed: inliers.rows: %d < %d", inliers.rows, kMinFeaturesForPnp);
        return false;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::cv2eigen(R, R_12);
    cv::cv2eigen(tvec, p_12);
    return true;
}

bool VisualManager::StereoTriangulation(const std::shared_ptr<CameraModel> camera_model, CameraObs& cam_obs, Eigen::Vector3d& pcf) const
{
    constexpr double kMaxStereoEipolarErrorThres = 8.0;
    constexpr double kMinStereoTriangulationParallex = 1.0;
    constexpr double kMaxStereoDepth = 20.0;

    double diff_x = abs(cam_obs.u - cam_obs.ur);
    double diff_y = abs(cam_obs.v - cam_obs.vr);
    // if (diff_x < kMinStereoTriangulationParallex || diff_y > kMaxStereoEipolarErrorThres) {
    //     return false;
    // }

    // camera_model->back_project_stereo(cam_obs);
    const double focal_length = camera_model->K_l()(0, 0);
    const double z_depth = focal_length * camera_model->baseline() / diff_x;
    if (z_depth < 0 || z_depth > kMaxStereoDepth)
    {
        // LOG(ERROR) << cv::format("FATAL ERROR! Stereo triangulation failed: z_depth < 0 or z_depth > %f, z_depth: %f", kMaxStereoDepth, z_depth);
        return false;
    }
    Eigen::Vector3d p3d_norm(cam_obs.u_norm, cam_obs.v_norm, 1.0);
    pcf = z_depth * p3d_norm;
    return true;
}

void VisualManager::FeatureTriangulation(std::vector<Feature*>& feats, std::map<double, CameraPose> camera_pose_buffer)
{
    // std::map<double, CameraPose> camera_pose_buffer = _state->AccessClonePoseBuffer();
    _feature_mapping_success = 0;

    int origin_feats_size = feats.size();
    int less_obs_delete = 0;
    int triangulate_failed = 0;
    int gaussian_newton_failed = 0;
    for (auto it = feats.begin(); it != feats.end();)
    {
        if ((*it)->_visual_obs_buffer.size() < kMinFeatForMapping)
        {
            it = feats.erase(it);
            less_obs_delete++;
            continue;
        }

        if ((*it)->_is_triangulated == false)
        {
            if (false == least_square_triangulation(camera_pose_buffer, *it))
            {
                // (*it)->_pwf.setZero();
                (*it)->parallex = 0;
                it = feats.erase(it);
                triangulate_failed++;
                continue;
            }
            (*it)->_is_triangulated = true;
        }

        if ((*it)->_is_triangulated && GaussianNewtonOptimization(camera_pose_buffer, *it))
        {
            _feature_mapping_success++;
        }
        else
        {
            (*it)->_is_triangulated = false;
            it = feats.erase(it);
            gaussian_newton_failed++;
            continue;
        }

        it++;
    }
    // std::cout << "msckf mapping in: " << origin_feats_size << std::endl;
    // std::cout << "triangulated failed: " << triangulate_failed << std::endl;
    // std::cout << "gaussian_newton_failed: " << gaussian_newton_failed << std::endl;
}

void VisualManager::CalculateFeatureParallex(std::vector<Feature*>& feats)
{
    for (auto it = feats.begin(); it != feats.end(); it++)
    {
        auto first_obv = (*it)->_visual_obs_buffer.begin();
        auto last_obv = (*it)->_visual_obs_buffer.end();
        last_obv--;
        double dx = first_obv->second.u - last_obv->second.u;
        double dy = first_obv->second.v - last_obv->second.v;

        double cur_parallex = sqrt(dx * dx + dy * dy);
        if (cur_parallex > (*it)->parallex)
        {
            (*it)->parallex = cur_parallex;
        }
    }
}

std::vector<Feature*> VisualManager::SelectMsckfFeatures(const std::vector<Feature*> feats)
{
    constexpr double kMinParallexForUse = -1.0;
    std::vector<Feature*> feat_msckf;
    for (int i = 0; i < feats.size(); i++)
    {
        if (feats[i]->_is_triangulated && feats[i]->parallex > kMinParallexForUse)
        {
            feat_msckf.push_back(feats[i]);
        }
    }

    if (feat_msckf.size() > _max_visual_feat_to_use)
    {
        std::sort(feat_msckf.begin(), feat_msckf.end(), [](Feature* feat_a, Feature* feat_b) { return feat_a->parallex > feat_b->parallex; });
        feat_msckf.resize(_max_visual_feat_to_use);
    }

    return feat_msckf;
}

bool VisualManager::ConstructFeatureJacobianFull(std::vector<Feature*> feats, Eigen::MatrixXd& Hx_full, Eigen::VectorXd& res)
{
    // constexpr size_t kMinFeatsToUpdate = 15;
    // if (feats.size() < kMinFeatsToUpdate) {
    //     LOG(INFO) << cv::format("Too few features to update, feature size: %d", int(feats.size()));
    //     return false;
    // }

    _map_hx.clear();
    _Hx_order.clear();
    int total_hx = 0;
    if (_state->_do_calibration_update)
    {
        _map_hx.insert({_state->_Tic, total_hx});
        _Hx_order.push_back(_state->_Tic);
        total_hx += _state->_Tic->size();
    }

    for (auto x : _state->_clone_pose)
    {
        _map_hx.insert({x.second, total_hx});
        _Hx_order.push_back(x.second);
        total_hx += x.second->size();
    }

    Hx_full.resize(4 * feats.size() * _state->_clone_pose.size(), total_hx + 1);
    Hx_full.setZero();

    int Hx_rows = 0;
    for (int i = 0; i < feats.size(); i++)
    {
        Eigen::MatrixXd Hx_single;
        if (feats[i]->_is_triangulated)
        {
            if (SingleFeatureJacobian(feats[i], _map_hx, total_hx, Hx_single) == false)
            {
                continue;
            }
            Hx_full.block(Hx_rows, 0, Hx_single.rows(), Hx_single.cols()) = Hx_single;
            Hx_rows += Hx_single.rows();
        }
    }
    Hx_full.conservativeResize(Hx_rows, Hx_full.cols());

    // measurements compression
    if (Hx_full.rows() > Hx_full.cols())
    {
        Hx_full = MathUtils::GivensRotation(Hx_full, Hx_full.cols() - 1);
        int final_hx_rows = Hx_full.cols() - 1 - 7;
        res.resize(final_hx_rows, 1);
        res = Hx_full.block(0, Hx_full.cols() - 1, final_hx_rows, 1);
        Hx_full.conservativeResize(final_hx_rows, Hx_full.cols() - 1);
        // Utils::show_eigen_matrix(Hx_full, "Hx_full_qr");
    }
    else
    {
        res.resize(Hx_full.rows(), 1);
        res = Hx_full.block(0, Hx_full.cols() - 1, Hx_full.rows(), 1);
        Hx_full.conservativeResize(Hx_full.rows(), Hx_full.cols() - 1);
    }

    return true;
}

bool VisualManager::SingleFeatureJacobian(Feature* feat,
                                          std::unordered_map<std::shared_ptr<Type>, size_t> map_hx,
                                          int total_hx,
                                          Eigen::MatrixXd& Hx_single)
{
    constexpr uint32_t kPwfDim = 3;
    assert(feat->_valid);
    int obs_size = 4 * feat->_visual_obs_buffer.size();
    Eigen::MatrixXd Hfx = Eigen::MatrixXd::Zero(obs_size, total_hx + kPwfDim + 1);  // 3 feature dimension + total_hx + 1 residual
    Eigen::Vector3d p_finG = feat->_pwf;

    Eigen::Vector2d res_total = Eigen::Vector2d::Zero();
    int cnt = 0;
    for (auto& obs : feat->_visual_obs_buffer)
    {
        double obs_ts = obs.first;
        for (int cam_id = 0; cam_id < MAX_CAM_NUM; cam_id++)
        {
            Eigen::Vector2d zm;
            double focal_length;
            Eigen::Matrix3d R_CtoI;
            Eigen::Vector3d p_CinI;

            if (cam_id == LEFT_CAM)
            {
                zm << obs.second.u, obs.second.v;
                focal_length = _camera_model->K_l()(0, 0);
                R_CtoI = _state->_Tic->quat().toRotationMatrix();
                p_CinI = _state->_Tic->p();
            }
            else if (cam_id == RIGHT_CAM)
            {
                zm << obs.second.ur, obs.second.vr;
                focal_length = _camera_model->K_r()(0, 0);
                R_CtoI = _state->_Tic->quat().toRotationMatrix() * _camera_model->R_rl();
                p_CinI = _state->_Tic->p() + _state->_Tic->quat().toRotationMatrix() * _camera_model->p_rl();
            }

            std::shared_ptr<Pose> obs_pose = _state->_clone_pose.at(obs_ts);
            Eigen::Matrix3d R_IitoG = obs_pose->quat().toRotationMatrix();
            Eigen::Vector3d p_IiinG = obs_pose->p();

            Eigen::Matrix3d R_CitoG = R_IitoG * R_CtoI;
            Eigen::Vector3d p_CiinG = p_IiinG + R_IitoG * p_CinI;

            Eigen::Vector3d p_finCi = R_CitoG.transpose() * (p_finG - p_CiinG);

            // compute visual residual
            Eigen::Vector2d uv;
            uv = cam_id == LEFT_CAM ? _camera_model->project_left(p_finCi) : _camera_model->project_right(p_finCi);

            Eigen::Vector2d res = zm - uv;
            Hfx.block<2, 1>(2 * cnt, Hfx.cols() - 1) = res;

            // precompute dz_dpcf
            Eigen::MatrixXd dz_norm_dpcf = Eigen::MatrixXd::Zero(2, 3);
            dz_norm_dpcf << 1 / p_finCi(2), 0, -p_finCi(0) / (p_finCi(2) * p_finCi(2)), 0, 1 / p_finCi(2), -p_finCi(1) / (p_finCi(2) * p_finCi(2));
            Eigen::MatrixXd dz_uv_dz_norm = Eigen::Matrix2d::Identity() * focal_length;
            Eigen::MatrixXd dz_dpcf = dz_uv_dz_norm * dz_norm_dpcf;

            // get jacobian wrt pwf
            Eigen::Matrix3d dpcf_dpwf = R_CitoG.transpose();
            Hfx.block<2, kPwfDim>(2 * cnt, 0) = dz_dpcf * dpcf_dpwf;

            // get jacobian wrt extrinsic parameters
            if (_state->_do_calibration_update)
            {
                Eigen::MatrixXd dpcf_dcalib = Eigen::MatrixXd::Zero(3, 6);
                // dpcf_dcalib.block<3, 3>(0, 0) = -R_ItoC * MathUtils::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
                dpcf_dcalib.block<3, 3>(0, 0) = MathUtils::skew(p_finCi);
                // dpcf_dcalib.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                dpcf_dcalib.block<3, 3>(0, 3) = -R_CtoI.transpose();
                Hfx.block<2, 6>(2 * cnt, kPwfDim + map_hx.at(_state->_Tic)) = dz_dpcf * dpcf_dcalib;
            }

            // get jacobian wrt clone pose
            Eigen::MatrixXd dpcf_dclone = Eigen::MatrixXd::Zero(3, 6);
            dpcf_dclone.block<3, 3>(0, 0) = R_CtoI.transpose() * MathUtils::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
            dpcf_dclone.block<3, 3>(0, 3) = -R_CitoG.transpose();
            Hfx.block<2, 6>(2 * cnt, kPwfDim + map_hx.at(obs_pose)) = dz_dpcf * dpcf_dclone;

            res_total += res;
            cnt++;
        }

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
        // Eigen::Vector2d Hx_plus_dR_ItoC = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(_state->_Tic)) * dR_CtoI;
        // Eigen::Matrix3d dR_CtoI_mat = Eigen::Matrix3d::Identity() + MathUtils::skew(dR_CtoI.head(3));
        // Eigen::Matrix3d R_CitoG_hat = R_IitoG * dR_CtoI_mat;
        // // Eigen::Vector3d p_CiinG = p_IiinG - R_IitoG * p_CinI;
        // Eigen::Vector3d pcf_dR_ItoC = R_CitoG_hat.transpose() * (p_finG - p_CiinG);
        // Eigen::Vector2d uv_dR_ItoC(pcf_dR_ItoC(0) / pcf_dR_ItoC(2), pcf_dR_ItoC(1) / pcf_dR_ItoC(2));
        // std::cout << "uv_dR_ItoC: " << (uv_dR_ItoC - uv_norm - Hx_plus_dR_ItoC).transpose() << std::endl;

        // // check p_IinC
        // Eigen::Vector3d dp_CinI(0.1, 0.1, 0.1);
        // Eigen::Vector2d Hx_plus_dp_IinC = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(_state->_Tic) + 3) * dp_CinI;
        // Eigen::Vector3d p_CiinG_hat = p_IiinG + R_IitoG * (p_CinI + dp_CinI);
        // Eigen::Vector3d pcf_dp_ItoC = R_CitoG.transpose() * (p_finG - p_CiinG_hat);
        // Eigen::Vector2d uv_dp_ItoC(pcf_dp_ItoC(0) / pcf_dp_ItoC(2), pcf_dp_ItoC(1) / pcf_dp_ItoC(2));
        // std::cout << "uv_dp_ItoC: " << (uv_dp_ItoC - uv_norm - Hx_plus_dp_IinC).transpose() << std::endl;

        // // check clone_R
        // Eigen::Vector3d dR_ItoG(0.1, 0.1, 0.1);
        // Eigen::Vector2d Hx_plus_dR_ItoG = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(obs_pose)) * dR_ItoG;
        // Eigen::Matrix3d dR_ItoG_mat = Eigen::Matrix3d::Identity() + MathUtils::skew(dR_ItoG.head(3));
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

    double threshold = 4.0f;
    if (res_total.norm() / cnt > threshold)
    {
        std::cout << "residual is too large: " << res_total.norm() / cnt << ", threashold is: " << threshold << std::endl;
        return false;
    }

    /*show single Hx matrix*/
    // Utils::show_eigen_matrix(Hfx, "Hfx");

    /* project Hfx to feature left null space */
    // MathUtils::NullSpaceProjectInplace(Hfx, 3);
    Hfx = MathUtils::GivensRotation(Hfx, 3);
    Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(Hfx.rows() - 3, Hfx.cols() - 3);
    Hx.noalias() = Hfx.block(3, 3, Hfx.rows() - 3, Hfx.cols() - 3);

    Hx_single = Hx;

    return true;
}