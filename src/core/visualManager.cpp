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

#define SHOW_CLONE_POSES 0

namespace
{
constexpr uint32_t kMinFeatForMapping = 1;
constexpr double kMaxConditionNum = 20000.f;
constexpr double kMinTriangDist = 0.05;
constexpr double kMaxTriangDist = 20.0;
constexpr uint32_t kMaxIterationTimes = 5;
constexpr uint32_t kMinFeatNumToUpdate = 15;
} // namespace

VisualManager::VisualManager(std::shared_ptr<ros::NodeHandle> &nh, const Param &params, std::shared_ptr<State> &state,
                             std::shared_ptr<MsckfSolverBase> solver)
{
    nh_ = nh;
    _state = state;
    param_ = params;
    _keyframe = std::make_shared<KeyFrameStatus>(KeyFrameStatus::kNone);
    vio_frontend = std::make_shared<VioFrontend>(nh, params, _keyframe);
    max_clone_pose_ = params.max_clone_pose;
    solver_ = solver;
    for (int i = 0; i < param_.max_feat_n; i++)
    {
        Feature *feat = new Feature();
        feature_base_.push_back(feat);
    }
}

bool VisualManager::MsckfFeatureUpdate(std::vector<Feature *> feats_msckf)
{
    if (feats_msckf.size() < kMinFeatureForUpdate)
    {
        LOG(INFO) << fmt::format("Not enough features to update, features_tracked: {:d}, features_msckf: {:d}, feature "
                                 "mapping success: {:d}",
                                 static_cast<int>(feature_tracked_.size()), static_cast<int>(feats_msckf.size()),
                                 feature_mapping_success_);
        return false;
    }
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;
    std::vector<std::shared_ptr<Type>> Hx_order;
    Eigen::MatrixXd Hx_msckf;
    Eigen::VectorXd res;
    for (auto &feat : feats_msckf)
    {
        if (feat->_type != FeatureType::kMsckfPoint)
        {
            LOG(WARNING) << fmt::format("Feature {:d} is not msckf point, but type {:d}, skip it for msckf update.",
                                        feat->_id, static_cast<int>(feat->_type));
            exit(1);
        }
    }
    ConstructFeatureJacobianFull(FeatureUpdateType::kMsckfUpdate, feats_msckf, map_hx, Hx_order, Hx_msckf, res);
    Eigen::MatrixXd R =
        Eigen::MatrixXd::Identity(Hx_msckf.rows(), Hx_msckf.rows()) * std::pow(param_.sigma_visual_pix, 2);

    if (_state->_clone_pose.size() >= 2 && Hx_msckf.rows() > 0)
    {
        // std::cout << "Msckf update with " << feats_msckf.size() << " features." << std::endl;
        solver_->update(_state, Hx_msckf, res, Hx_order, map_hx, R);
        return true;
    }
    else
    {
        LOG(WARNING) << fmt::format("Not enough clone poses to update, clone poses: {:d}",
                                    static_cast<int>(_state->_clone_pose.size()));
        return false;
    }
}

/**
 * @brief Slam feature update
 */
bool VisualManager::SlamFeatureUpdate(std::vector<Feature *> feats_slam)
{
    if (feats_slam.empty())
    {
        LOG(INFO) << "No slam features to update.";
        return false;
    }

    if (_state->_clone_pose.size() == param_.max_clone_pose)
    {
        int num_updated = 0;
        for (auto* feat : feats_slam)
        {
            if (!feat->_is_triangulated)
            {
                continue;
            }

            auto& slam_feat = _state->slam_features().at(feat->_id);

            std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
            std::vector<std::shared_ptr<Type>> Hx_order;
            size_t total_hx = 0;

            for (const auto& [ts, pose] : _state->_clone_pose)
            {
                Hx_mapping.emplace(pose, total_hx);
                Hx_order.push_back(pose);
                total_hx += pose->size();
            }

            if (_state->enableEstimateRic())
            {
                for (uint8_t cam_id = 0; cam_id < param_.camera_num; cam_id++)
                {
                    auto qic = _state->mutable_Qic(cam_id);
                    Hx_mapping.emplace(qic, total_hx);
                    Hx_order.push_back(qic);
                    total_hx += qic->size();
                }
            }

            // total_hx_before_feat is the width for Hx output (clone+extrinsic columns only, not feature)
            size_t total_hx_before_feat = total_hx;

            auto feat_ptr = slam_feat._state_ptr;
            Hx_mapping.emplace(feat_ptr, total_hx);
            Hx_order.push_back(feat_ptr);
            total_hx += feat_ptr->size();

            Eigen::MatrixXd Hf, Hx;
            Eigen::VectorXd res;
            if (!SingleFeatureJacobianSlam(feat, Hx_mapping, total_hx_before_feat, Hf, Hx, res))
            {
                LOG(WARNING) << "Failed to compute Jacobian for SLAM feature " << feat->_id;
                continue;
            }

            Eigen::MatrixXd H_combined = Eigen::MatrixXd::Zero(res.rows(), total_hx);
            H_combined.block(0, 0, Hx.rows(), Hx.cols()) = Hx;
            H_combined.block(0, Hx_mapping.at(feat_ptr), Hf.rows(), Hf.cols()) = Hf;

            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(res.rows(), res.rows()) *
                                std::pow(param_.sigma_visual_pix, 2);

            solver_->update(_state, H_combined, res, Hx_order, Hx_mapping, R);
            num_updated++;
        }

        if (num_updated > 0)
        {
            _state->UpdateSlamFeatureAfterVisualUpdate();
        }
        return num_updated > 0;
    }
    else
    {
        LOG(WARNING) << fmt::format("Not enough clone poses to update slam features, clone poses: {:d}, required: {:d}",
                                    static_cast<int>(_state->_clone_pose.size()),
                                    static_cast<int>(param_.max_clone_pose));
        return false;
    }
}

void VisualManager::SelectSlamFeatures(std::vector<Feature *> &feature_tracked, std::vector<Feature *> &feat_slam_old,
                                       std::vector<Feature *> &feat_slam_new)
{
    constexpr double kMinParallexForSlam = 5.0; // In pixel
    for (auto it = feature_tracked.begin(); it != feature_tracked.end();)
    {
        Feature* feature = *it;

        if (_state->IsOldSlamFeature(feature->_id))
        {
            feature->_type = FeatureType::kSlamPoint;
            feat_slam_old.push_back(feature);
            it = feature_tracked.erase(it);
            continue;
        }
        else if (feature->_visual_obs_buffer.size() >= param_.max_clone_pose &&
                 feature->_parallex > kMinParallexForSlam)
        {
            feat_slam_new.push_back(feature);
            it = feature_tracked.erase(it);
            continue;
        }
        it++;
    }

    if (feat_slam_new.size() > 0)
    {
        // auto compare_feat_func = [](Feature* feat_a, Feature* feat_b) { return feat_a->_visual_obs_buffer.size() >
        // feat_b->_visual_obs_buffer.size(); };

        auto compare_feat_func = [](Feature *feat_a, Feature *feat_b)
        { return feat_a->_theta_parallex > feat_b->_theta_parallex; };

        std::sort(feat_slam_new.begin(), feat_slam_new.end(), compare_feat_func);
        feat_slam_new.resize(std::min(static_cast<size_t>(param_.max_slam_feature - _state->slam_features().size()),
                                      feat_slam_new.size()));
    }
}

void VisualManager::SelectMsckfFeatures(const std::vector<Feature*> feats, std::vector<Feature*>& feat_msckf)
{
    constexpr double kMinParallexForUse = 3.0;

    for (int i = 0; i < feats.size(); i++)
    {
        if (feats[i]->_is_triangulated && feats[i]->_parallex > kMinParallexForUse)
        {
            feats[i]->_type = FeatureType::kMsckfPoint;
            feat_msckf.push_back(feats[i]);
        }
    }

    if (feat_msckf.size() > kMaxFeatureForUpdate)
    {
        auto compare_feat_func = [](Feature *feat_a, Feature *feat_b)
        { return feat_a->_visual_obs_buffer.size() > feat_b->_visual_obs_buffer.size(); };

        // auto compare_feat_func = [](Feature* feat_a, Feature* feat_b) { return feat_a->_theta_parallex >
        // feat_b->_theta_parallex; };

        std::sort(feat_msckf.begin(), feat_msckf.end(), compare_feat_func);
        feat_msckf.resize(kMaxFeatureForUpdate);
    }
}

void VisualManager::InitializeNewSlamFeatures(std::vector<Feature*>& feat_slam_new)
{
    if (feat_slam_new.empty())
    {
        return;
    }

    std::map<double, CameraPose> camera_pose_buffer;
    _state->AccessClonePoseBuffer(camera_pose_buffer);
    FeatureTriangulation(camera_pose_buffer, feat_slam_new);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * std::pow(param_.sigma_visual_pix, 2);
    std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;
    std::vector<std::shared_ptr<Type>> Hx_order;
    uint32_t total_hx = 0;
    for (auto it = feat_slam_new.begin(); it != feat_slam_new.end();)
    {
        Feature* feature = *it;
        if (feature->_is_triangulated && feature->_visual_obs_buffer.size() == param_.max_clone_pose)
        {
            for (auto x : _state->_clone_pose)
            {
                map_hx.emplace(x.second, total_hx);
                Hx_order.push_back(x.second);
                total_hx += x.second->size();
            }

            if (_state->enableEstimateRic())
            {
                for (uint8_t i = 0; i < param_.camera_num; i++)
                {
                    map_hx.emplace(_state->mutable_Qic(i), total_hx);
                    Hx_order.push_back(_state->mutable_Qic(i));
                    total_hx += _state->mutable_Qic(i)->size();
                }
            }
        }
        else
        {
            it = feat_slam_new.erase(it);
            continue;
        }

        Eigen::MatrixXd Hfx;
        if (!SingleFeatureJacobian(feature, map_hx, total_hx, Hfx))
        {
            it = feat_slam_new.erase(it);
            continue;
        }

        Eigen::MatrixXd Hfx_qr = utils::math::GivensRotation(Hfx, 3);
        Eigen::VectorXd res = Hfx.rightCols(1);
        Hfx_qr.conservativeResize(Hfx_qr.rows(), Hfx_qr.cols() - 1);

        //        [  Hf(3x3)  |  Hx1(3xn)  |  r1(3x1)  ]
        // Hfx_qr=[-----------+------------+-----------]
        //        [    0      |  Hx2(mxn)  |  r2(mx1)  ]
        Eigen::VectorXd r1 = res.head(3);
        Eigen::VectorXd r2 = res.tail(res.rows() - 3);
        Eigen::MatrixXd Hf = Hfx_qr.topLeftCorner(3, 3);
        Eigen::MatrixXd Hx1 = Hfx_qr.topRightCorner(3, Hfx_qr.cols() - 3);
        Eigen::MatrixXd Hx2 = Hfx_qr.bottomRightCorner(Hfx_qr.rows() - 3, Hfx_qr.cols() - 3);
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3) * std::pow(param_.sigma_visual_pix, 2);

        feature->_pwf += Hf.inverse() * r1;
        feature->_type = FeatureType::kSlamPoint;

        if (!_state->AugumentSlamFeature(feature, Hf, Hx1, Hx_order, R, map_hx))
        {
            LOG(WARNING) << fmt::format("Failed to augment slam feature {:d}", feature->_id);
            it = feat_slam_new.erase(it);
            continue;
        }

        R = Eigen::MatrixXd::Identity(Hx2.rows(), Hx2.rows()) * std::pow(param_.sigma_visual_pix, 2);
        solver_->update(_state, Hx2, r2, Hx_order, map_hx, R);

        it++;
    }
}

bool VisualManager::VisualUpdate()
{
    bool is_msckf_updated = false;
    bool is_slam_updated = false;
    feat_msckf_.clear();
    feat_slam_old_.clear();
    feat_slam_new_.clear();

    std::map<double, CameraPose> camera_pose_buffer;
    _state->AccessClonePoseBuffer(camera_pose_buffer);

    FeatureTriangulation(camera_pose_buffer, feature_tracked_);

    CalculateMaxFeatureParallex(feature_tracked_);

    if (param_.use_slam_feature)
    {
        SelectSlamFeatures(feature_tracked_, feat_slam_old_, feat_slam_new_);
    }

    SelectMsckfFeatures(feature_tracked_, feat_msckf_);

    if (param_.use_pnp_ransac && feat_msckf_.size() > kMinFeatureForUpdate)
    {
        PnpRansacToRejectOutliers(feat_msckf_);
    }

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

    for (auto it = feature_tracked_.begin(); it != feature_tracked_.end(); it++)
    {
        if ((*it)->_is_triangulated == false)
        {
            continue;
        }

        for (auto it_feat = (*it)->_visual_obs_buffer.begin(); it_feat != (*it)->_visual_obs_buffer.end(); it_feat++)
        {
            uint32_t feat_id = (*it)->_id;
            double timestamp = it_feat->first;
            cv::Point2f point(it_feat->second.uv[LEFT_CAM].x(), it_feat->second.uv[LEFT_CAM].y());

            CameraPose camera_pose = camera_pose_buffer.at(timestamp);
            Eigen::Vector3d pcf = camera_pose.Rwc.transpose() * ((*it)->_pwf - camera_pose.pwc);
            Eigen::Vector2d uv = CamModel::getInstance().project(0, pcf);

            std::ostringstream os;
            os << std::fixed << feat_id;
            std::string depth_text = os.str();
            cv::putText(clone_image_map.at(timestamp), depth_text, point, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 255), 1);
            cv::Scalar color = cv::Scalar(int(fb * feat_id) % 255, int(fg * feat_id) % 255, int(fr * feat_id) % 255);
            cv::circle(clone_image_map.at(timestamp), point, 4, color, -1);
            cv::circle(clone_image_map.at(timestamp), cv::Point2f(uv.x(), uv.y()), 5, color, 1);
        }
    }

    std::vector<cv::Mat> images_to_show;
    for (auto it = clone_image_map.begin(); it != clone_image_map.end(); it++)
    {
        images_to_show.push_back(it->second);
    }
    utils::ShowGridImages(images_to_show);
#endif

    if (param_.use_slam_feature)
    {
        if (SlamFeatureUpdate(feat_slam_old_))
        {
            is_slam_updated = true;
        }
        InitializeNewSlamFeatures(feat_slam_new_);
    }

    if (MsckfFeatureUpdate(feat_msckf_))
    {
        is_msckf_updated = true;
    }

    std::shared_ptr<Type> state_to_marginalize = nullptr;

    if (_state->_clone_pose.size() >= max_clone_pose_)
    {
        *_keyframe = KeyFrameStatus::kNone;
        *_keyframe = MaybeSetKeyframe(_state, feat_msckf_);
        if (*_keyframe == KeyFrameStatus::kNone)
        {
            state_to_marginalize = _state->_clone_pose.rbegin()->second;
            ClearOldFeatureObs(state_to_marginalize->ts());
        }
        else  // marginalize lastest pose
        {
            state_to_marginalize = _state->_clone_pose.begin()->second;
            ClearOldFeatureObs(state_to_marginalize->ts());
            MarginalizeSlamFeatureLost();
            UpdateFeatureBase(_state->ts_sec());
        }
    }

    std::cout << "feature slam old: " << feat_slam_old_.size()
              << ", feature slam new: " << feat_slam_new_.size()
              << ", feature msckf num: " << feat_msckf_.size()
              << ", keyframe status: " << static_cast<int>(*_keyframe) << std::endl;

    // TODO: marginalize slam features that are lost
    if (state_to_marginalize != nullptr)
    {
        solver_->MarginalizeState(MarginalizeType::ClonePose, _state, state_to_marginalize);
    }

    return is_msckf_updated || is_slam_updated;
}

KeyFrameStatus VisualManager::MaybeSetKeyframe(std::shared_ptr<State> _state, std::vector<Feature*> feats)
{
    constexpr double kLargeParallexThres = 10.f;

    if (_state->_clone_pose.size() < max_clone_pose_)
    {
        return KeyFrameStatus::kNone;
    }

    // Calculate average parallex w.r.t the last keyframe
    uint32_t cnt = 0;
    double pixel_parallex_avg = 0.f;
    for (auto x : feats)
    {
        if (x->_visual_obs_buffer.size() < 2)
        {
            continue;
        }
        auto lastest_iter = x->_visual_obs_buffer.rbegin();
        auto sub_lastest_iter = std::next(lastest_iter);

        pixel_parallex_avg += RotationCompensatedParallex(
            sub_lastest_iter->first, sub_lastest_iter->second,
            lastest_iter->first, lastest_iter->second);
        cnt++;
    }
    pixel_parallex_avg = pixel_parallex_avg / cnt;

    // Decide keyframe status
    if (pixel_parallex_avg > kLargeParallexThres)
    {
        return KeyFrameStatus::kLargeParallex;  // TODO：这里要优化一下
    }
    else if (feature_lost_.size() > 0.7 * param_.max_feat_n)
    {
        return KeyFrameStatus::kFeatureLostTooMuch;
    }
    else if (feature_tracked_.size() < 20)
    {
        return KeyFrameStatus::kTooFewFeatureTracked;
    }
    else
    {
        return KeyFrameStatus::kNone;
    }
}

void VisualManager::ClearOldFeatureObs(const double timestamp_to_drop)
{
    for (auto& feat_base : feature_base_)
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

void VisualManager::MarginalizeSlamFeatureLost()
{
    if(!param_.use_slam_feature)
    {
        return;
    }

    assert(_state->slam_features().size() <= param_.max_slam_feature);
    // std::cout << "total slam features: " << _state->slam_features().size() << std::endl;

    // Collect features to marginalize first to avoid modifying the map during iteration
    std::vector<std::shared_ptr<Type>> feats_to_marginalize;
    for (auto &[id, feat_slam] : _state->slam_features())
    {
        for (auto& feat_lost : feature_lost_)
        {
            if (feat_lost->_id == id)
            {
                feats_to_marginalize.push_back(feat_slam._state_ptr);
                break;
            }
        }
    }

    for (auto& state_ptr : feats_to_marginalize)
    {
        solver_->MarginalizeState(MarginalizeType::SlamFeature, _state, state_ptr);
    }
}

void VisualManager::UpdateFeatureBase(const double timestamp)
{
    for (auto& feat_lost : feature_lost_)
    {
        feat_lost->reset();
    }

    for (auto& feature : feature_new_)
    {
        for (int i = 0; i < feature_base_.size(); i++)
        {
            if (!feature_base_[i]->_valid)
            {
                feature_base_[i]->reset();
                feature_base_[i]->_id = feature.feat_id;
                feature_base_[i]->_valid = true;
                feature_base_[i]->_type = FeatureType::kUnknown;
                feature_base_[i]->_visual_obs_buffer.insert_or_assign(timestamp, feature);
                break;
            }
        }
    }
}

void VisualManager::UpdateFeatureStatistic(const double timestamp,
                                           std::pair<double, std::vector<CameraObs>> feature_observes)
{
    // double ts_sec = feature_observes.first;
    std::vector<CameraObs> feature_obs = feature_observes.second;
    std::unordered_map<uint32_t, CameraObs> feature_obsrv_umap;
    std::unordered_set<uint32_t> feature_tracked_id_uset;

    for (auto& feat_obsrv : feature_observes.second)
    {
        feature_obsrv_umap.emplace(feat_obsrv.feat_id, feat_obsrv);
    }

    feature_lost_.clear();
    feature_new_.clear();
    feature_tracked_.clear();

    for (int i = 0; i < feature_base_.size(); i++)
    {
        Feature* feature = feature_base_[i];
        if (!feature->_valid)
        {
            continue;
        }

        if (feature_obsrv_umap.find(feature->_id) != feature_obsrv_umap.end())
        {
            feature_tracked_.push_back(feature);
            feature_tracked_id_uset.insert(feature->_id);
            feature->_visual_obs_buffer.insert_or_assign(timestamp, feature_obsrv_umap[feature->_id]);
        }
        else
        {
            feature_lost_.push_back(feature);
        }
    }

    for (int i = 0; i < feature_observes.second.size(); i++)
    {
        uint32_t feature_observ_id = feature_observes.second[i].feat_id;
        if (feature_tracked_id_uset.find(feature_observ_id) == feature_tracked_id_uset.end())
        {
            feature_new_.push_back(feature_observes.second[i]);
        }
    }

    // std::cout << "feature observed: " << feature_observes.second.size() << std::endl;
    // std::cout << "feature tracked: " << feature_tracked_.size() << std::endl;
    // std::cout << "feature lost: " << feature_lost_.size() << std::endl;
    // std::cout << "feature new: " << feature_new_.size() << std::endl;
}

void VisualManager::ResetFeatureBase()
{
    // feature_base_.clear();
    for (auto& feat : feature_base_)
    {
        feat->reset();
    }

    feature_new_.clear();
    feature_lost_.clear();
    feature_tracked_.clear();
}

double VisualManager::calcVisualObsParallex(const std::unordered_map<uint32_t, CameraObs>& visual_obs_a,
                                            const std::unordered_map<uint32_t, CameraObs>& visual_obs_b)
{
    double average_parallex = 0.0;
    auto PixelDistance = [](CameraObs obs_a, CameraObs obs_b) {
        Eigen::Vector2d pix_distance = obs_a.uv[LEFT_CAM] - obs_b.uv[LEFT_CAM];
        return pix_distance.norm();
    };

    int count = 0;
    for (const auto& [feature_id, obs_a] : visual_obs_a)
    {
        if (visual_obs_b.find(feature_id) != visual_obs_b.end())
        {
            CameraObs obs_b = visual_obs_b.at(feature_id);
            double parallex = PixelDistance(obs_a, obs_b);
            average_parallex += parallex;
            count++;
        }
    }
    if (count > 0)
    {
        average_parallex = average_parallex / count;
    }
    return average_parallex;
}

std::map<uint32_t, CameraObs>
VisualManager::covisibleFeatures(const std::unordered_map<uint32_t, CameraObs> &visual_obs_a,
                                 const std::unordered_map<uint32_t, CameraObs> &visual_obs_b)
{
    std::map<uint32_t, CameraObs> covisible_features;
    for (const auto &[feature_id, obs_a] : visual_obs_a)
    {
        if (visual_obs_b.find(feature_id) != visual_obs_b.end())
        {
            covisible_features[feature_id] = obs_a;
        }
    }
    return covisible_features;
}

void VisualManager::InitFeatureBase(
    std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_feature_triangulated)
{
    int cnt = 0;
    for (auto& [feature_id, feature_obs] : stereo_feature_triangulated)
    {
        if (cnt >= param_.max_feat_n)
        {
            break;
        }

        Feature* feature = new Feature();
        feature->_id = feature_id;
        feature->_pwf = feature_obs.second;
        feature->_valid = true;
        feature->_is_triangulated = true;
        feature->_visual_obs_buffer.insert({feature_obs.first.ts_sec, feature_obs.first});
        feature_base_[cnt] = feature;
        cnt++;
    }
}

bool VisualManager::least_square_triangulation(const std::map<double, CameraPose>& clone_pose_buffer, Feature* feat)
{
    auto first_obs = feat->_visual_obs_buffer.begin();
    Eigen::Matrix3d R_AtoG = clone_pose_buffer.at(first_obs->first).Rwc[LEFT_CAM];
    Eigen::Vector3d p_AinG = clone_pose_buffer.at(first_obs->first).pwc[LEFT_CAM];

    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d ATb = Eigen::Vector3d::Zero();

    int feat_index = 0;
    for (auto it = feat->_visual_obs_buffer.begin(); it != feat->_visual_obs_buffer.end(); it++)
    {
        for (int cam_id = 0; cam_id < param_.camera_num; cam_id++)
        {
            Eigen::Matrix3d R_CitoG = clone_pose_buffer.at(it->first).Rwc[cam_id];;
            Eigen::Vector3d p_CiinG = clone_pose_buffer.at(it->first).pwc[cam_id];;
            Eigen::Vector3d b_i{it->second.uv_norm[cam_id].x(), it->second.uv_norm[cam_id].y(), 1.0};

            Eigen::Matrix3d R_CitoA = R_AtoG.transpose() * R_CitoG;
            Eigen::Vector3d p_CiinA = R_AtoG.transpose() * (p_CiinG - p_AinG);

            Eigen::Vector3d b_iinA = R_CitoA * b_i;
            ATA += utils::math::skew(b_iinA).transpose() * utils::math::skew(b_iinA);
            ATb += utils::math::skew(b_iinA).transpose() * utils::math::skew(b_iinA) * p_CiinA;
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
        if (std::abs(condA) > kMaxConditionNum)
        {
            // std::cout << cv::format("condition num [%f] > [%f]", condA, kMaxConditionNum) << std::endl;
        }
        else if (paf(2, 0) < kMinTriangDist || paf(2, 0) > kMaxTriangDist)
        {
            // std::cout << "paf is out of range" <<  paf.transpose() << std::endl;
        }
        else if (std::isnan(paf.norm()))
        {
            std::cout << "paf is nan" << std::endl;
        }
        return false;
    }

    // std::cout << "triangulation failed num: " << triang_failed_num << std::endl;
    return true;
}

bool VisualManager::GaussianNewtonOptimization(const std::map<double, CameraPose>& clone_pose_buffer, Feature* feat)
{
    auto last_obs = feat->_visual_obs_buffer.end();
    last_obs--;
    const Eigen::Matrix3d R_AtoG = clone_pose_buffer.at(last_obs->first).Rwc[LEFT_CAM];
    const Eigen::Vector3d p_AinG = clone_pose_buffer.at(last_obs->first).pwc[LEFT_CAM];

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
            for (int cam_id = 0; cam_id < param_.camera_num; cam_id++)
            {
                double feature_timestamp = (*it).first;
                Eigen::Matrix3d R_CitoG = clone_pose_buffer.at(feature_timestamp).Rwc[cam_id];;
                Eigen::Vector3d p_CiinG = clone_pose_buffer.at(feature_timestamp).pwc[cam_id];;
                Eigen::Vector2d z_m = it->second.uv_norm[cam_id];


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

    if (paf_opt.norm() > 30 || paf_opt.z() < 0 ||iter_time == kMaxIterationTimes)
    {
        if (paf_opt.norm() > 30)
        {
            std::cout << "paf_opt is too large, paf norm: " << paf_opt.norm() << std::endl;
        }
        else if (paf_opt.z() < 0)
        {
            std::cout << "paf_opt z is negative, paf z: " << paf_opt.z() << std::endl;
        }
        else if (iter_time == kMaxIterationTimes)
        {
            std::cout << "iter_time is too large" << std::endl;
        }
        return false;
    }

    // Calcutate the max theta of poseA -> feature3d -> poseB
    for (auto it = feat->_visual_obs_buffer.begin(); it != feat->_visual_obs_buffer.end(); it++)
    {
        for (int cam_id = 0; cam_id < param_.camera_num; cam_id++)
        {
            double feature_timestamp = (*it).first;
            Eigen::Matrix3d R_CitoG = clone_pose_buffer.at(feature_timestamp).Rwc[cam_id];;
            Eigen::Vector3d p_CiinG = clone_pose_buffer.at(feature_timestamp).pwc[cam_id];;

            Eigen::Vector3d vec_a = (feat->_pwf - p_AinG).normalized();
            Eigen::Vector3d vec_b = (feat->_pwf - p_CiinG).normalized();
            double cos_theta = vec_a.dot(vec_b);
            double theta = std::acos(cos_theta) * 180.0 / M_PI;
            if (theta > feat->_theta_parallex)
            {
                feat->_theta_parallex = theta;
            }
        }
    }

    return true;
}

bool VisualManager::StereoLeastSqureTriangulation(CameraObs& cam_obs, Eigen::Vector3d& pcf) const
{
    const int32_t camera_num = CamModel::getInstance().camera_num();
    if (camera_num != 2)
    {
        LOG(ERROR) << "Stereo triangulation only support stereo camera model";
        return false;
    }
    Eigen::Matrix3d ATA = Eigen::Matrix3d::Zero();
    Eigen::Vector3d ATb = Eigen::Vector3d::Zero();

    for (int cam_id = 0; cam_id < camera_num; cam_id++)
    {
        Eigen::Matrix3d R_CitoA = Eigen::Matrix3d::Identity();
        Eigen::Vector3d p_CiinA = Eigen::Vector3d::Zero();
        Eigen::Vector3d b_i = Eigen::Vector3d::Zero();

        if (cam_id == LEFT_CAM)
        {
            b_i << cam_obs.uv_norm[cam_id], 1;
        }
        else if (cam_id == RIGHT_CAM)
        {
            b_i << cam_obs.uv_norm[cam_id], 1;
            R_CitoA = CamModel::getInstance().Rlr();
            p_CiinA = CamModel::getInstance().plr();
        }

        Eigen::Vector3d b_iinA = R_CitoA * b_i;
        ATA += utils::math::skew(b_iinA).transpose() * utils::math::skew(b_iinA);
        ATb += utils::math::skew(b_iinA).transpose() * utils::math::skew(b_iinA) * p_CiinA;
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
    feature_mapping_success_ = 0;
    feature_mapping_in_ = 0;
    feature_tracked_.clear();
    feature_lost_.clear();
    feature_new_.clear();

    // Clear buffers with mutex protection
    {
        std::lock_guard<std::mutex> lock(input_image_buffer_mutex_);
        std::queue<std::pair<double, std::vector<cv::Mat>>>().swap(_input_image_buffer);
    }

    std::queue<std::pair<double, std::vector<CameraObs>>>().swap(feature_obs_buffer);
    std::map<double, std::vector<cv::Mat>>().swap(stored_images_);
    for (int i = 0; i < param_.max_feat_n; i++)
    {
        feature_base_[i]->reset();
    }
}

void VisualManager::ClearExpiredMeasurements(const double timestamp)
{
    for (auto it = stored_images_.begin(); it != stored_images_.end();)
    {
        if (it->first < timestamp)  // Clear images older than 100ms
        {
            it = stored_images_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void VisualManager::FeedImages(const std::pair<double, std::vector<cv::Mat>> input)
{
    // Protect image buffer with mutex
    {
        std::lock_guard<std::mutex> lock(input_image_buffer_mutex_);
        while (_input_image_buffer.size() > kMaxImageBufferSize)
        {
            _input_image_buffer.pop();
        }
        _input_image_buffer.push(input);
    }

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
            list_points2d.emplace_back(obs_2d.uv[LEFT_CAM].x(), obs_2d.uv[LEFT_CAM].y());
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
    Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
    cv::eigen2cv(K, intrinsic);
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

bool VisualManager::PnpRansac(Eigen::Matrix3d& R_12,
                              Eigen::Vector3d& p_12,
                              std::unordered_map<int32_t, std::pair<CameraObs, Eigen::Vector3d>> stereo_obs_triangulated) const
{
    constexpr int kMinFeaturesForPnp = 15;

    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    for (auto& [feat_id, obs_pwf] : stereo_obs_triangulated)
    {
        points_3d.push_back(cv::Point3f(obs_pwf.second.x(), obs_pwf.second.y(), obs_pwf.second.z()));
        points_2d.push_back(cv::Point2f(obs_pwf.first.uv[LEFT_CAM].x(), obs_pwf.first.uv[LEFT_CAM].y()));
    }

    cv::Mat rvec, tvec;
    cv::Mat inliers;
    cv::Mat intrinsic;
    cv::Mat distortion;
    Eigen::Matrix3d K = CamModel::getInstance().K(LEFT_CAM);
    cv::eigen2cv(K, intrinsic);
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

bool VisualManager::StereoTriangulation(CameraObs& cam_obs, Eigen::Vector3d& pcf) const
{
    constexpr double kMaxStereoEipolarErrorThres = 8.0;
    constexpr double kMinStereoTriangulationParallex = 1.0;
    constexpr double kMaxStereoDepth = 20.0;

    const double diff_x = abs(cam_obs.uv[LEFT_CAM].x() - cam_obs.uv[RIGHT_CAM].x());
    const double diff_y = abs(cam_obs.uv[LEFT_CAM].y() - cam_obs.uv[RIGHT_CAM].y());

    const double focal_length = CamModel::getInstance().K(LEFT_CAM)(0, 0);
    const double baseline = CamModel::getInstance().getBaseline();
    const double z_depth = focal_length * baseline / diff_x;
    if (z_depth < 0 || z_depth > kMaxStereoDepth)
    {
        return false;
    }
    Eigen::Vector3d p3d_norm = Eigen::Vector3d(cam_obs.uv_norm[0].x(), cam_obs.uv_norm[0].y(), 1.0);
    pcf = z_depth * p3d_norm;
    return true;
}

void VisualManager::FeatureTriangulation(const std::map<double, CameraPose> camera_pose_buffer, std::vector<Feature*>& feats)
{
    feature_mapping_success_ = 0;
    int origin_feats_size = feats.size();
    int less_obs_delete = 0;
    int triangulate_failed = 0;
    int gaussian_newton_failed = 0;
    for (auto it = feats.begin(); it != feats.end();)
    {
        if ((*it)->_type == FeatureType::kSlamPoint)
        {
            it++;
            continue;
        }

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
                (*it)->_parallex = 0;
                it = feats.erase(it);
                triangulate_failed++;
                continue;
            }
            (*it)->_is_triangulated = true;
        }

        if ((*it)->_is_triangulated && GaussianNewtonOptimization(camera_pose_buffer, *it))
        {
            feature_mapping_success_++;
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

    // LOG(INFO) << fmt::format("msckf mapping in: {:d}, mapping_success: {:d}, triangulated failed: {:d}, gaussian_newton_failed: {:d}",
    //                          feature_mapping_success_, feature_mapping_success_, triangulate_failed, gaussian_newton_failed);
}

double VisualManager::RotationCompensatedParallex(double ts_a, const CameraObs& obs_a,
                                                   double ts_b, const CameraObs& obs_b)
{
    // Get rotation matrices for both poses
    Eigen::Matrix3d R_a = _state->_clone_pose.at(ts_a)->quat().toRotationMatrix();
    Eigen::Matrix3d R_b = _state->_clone_pose.at(ts_b)->quat().toRotationMatrix();

    // Compute relative rotation from frame a to frame b
    Eigen::Matrix3d R_a_to_b = R_b.transpose() * R_a;

    // Get normalized coordinates of observation a
    Eigen::Vector3d ray_a(obs_a.uv_norm.at(LEFT_CAM).x(), obs_a.uv_norm.at(LEFT_CAM).y(), 1.0);

    // Rotate ray a to frame b (removing rotation effect)
    Eigen::Vector3d ray_a_rotated = R_a_to_b * ray_a;
    ray_a_rotated /= ray_a_rotated.z();

    // Project rotated ray to pixel coordinates
    Eigen::Vector2d uv_a_rotated = CamModel::getInstance().project_distort(LEFT_CAM, ray_a_rotated);

    // Calculate pixel distance after rotation compensation
    return (obs_b.uv.at(LEFT_CAM) - uv_a_rotated).norm();
}

void VisualManager::CalculateMaxFeatureParallex(std::vector<Feature *> &feats)
{
    for (auto *feat : feats)
    {
        const auto &obs_buf = feat->_visual_obs_buffer;
        if (obs_buf.size() < 2)
        {
            feat->_parallex = 0.0;
            continue;
        }

        double max_parallex = 0.0;
        for (auto it_i = obs_buf.begin(); it_i != obs_buf.end(); ++it_i)
        {
            auto it_j = it_i;
            ++it_j;
            for (; it_j != obs_buf.end(); ++it_j)
            {
                const double cur_parallex =
                    RotationCompensatedParallex(it_i->first, it_i->second, it_j->first, it_j->second);
                if (cur_parallex > max_parallex)
                {
                    max_parallex = cur_parallex;
                }
            }
        }

        feat->_parallex = max_parallex;
    }
}

bool VisualManager::ConstructFeatureJacobianFull(FeatureUpdateType update_type,
                                                 std::vector<Feature*> feats,
                                                 std::unordered_map<std::shared_ptr<Type>, size_t>& map_hx,
                                                 std::vector<std::shared_ptr<Type>>& Hx_order,
                                                 Eigen::MatrixXd& Hx_full,
                                                 Eigen::VectorXd& res)
{
    map_hx.clear();
    Hx_order.clear();
    int total_hx = 0;

    // Add slam features to Hx map
    if (update_type == FeatureUpdateType::kSlamUpdate)
    {
        for (auto x : feats)
        {
            assert(x->_type == FeatureType::kSlamPoint);
            std::shared_ptr<Type> pwf_state = _state->slam_features().at(x->_id)._state_ptr;
            map_hx.emplace(pwf_state, total_hx);
            Hx_order.push_back(pwf_state);
            total_hx += pwf_state->size();
        }
    }

    // Add clone poses to Hx map
    for (auto x : _state->_clone_pose)
    {
        map_hx.emplace(x.second, total_hx);
        Hx_order.push_back(x.second);
        total_hx += x.second->size();
    }

    // Add extrinsic to Hx map
    if (_state->enableEstimateRic())
    {
        for (uint8_t i = 0; i < param_.camera_num; i++)
        {
            map_hx.emplace(_state->mutable_Qic(i), total_hx);
            Hx_order.push_back(_state->mutable_Qic(i));
            total_hx += _state->mutable_Qic(i)->size();
        }
    }

    Hx_full.resize(4 * feats.size() * _state->_clone_pose.size(), total_hx + 1);
    Hx_full.setZero();

    int Hx_rows = 0;
    for (int i = 0; i < feats.size(); i++)
    {
        Eigen::MatrixXd Hfx_single;
        Feature* feat = feats[i];
        if (feat->_is_triangulated)
        {
            if (!SingleFeatureJacobian(feat, map_hx, total_hx, Hfx_single))
            {
                LOG(WARNING) << fmt::format("{:d} type feature single feature failed", static_cast<int>(update_type));
                continue;
            }

            if (feat->_type == FeatureType::kMsckfPoint)
            {
                Eigen::MatrixXd Hfx_qr = utils::math::GivensRotation(Hfx_single, 3);
                Eigen::MatrixXd Hx = Hfx_qr.block(3, 3, Hfx_qr.rows() - 3, Hfx_qr.cols() - 3);
                Hx_full.block(Hx_rows, 0, Hx.rows(), Hx.cols()) = Hx;
                Hx_rows += Hx.rows();
            }
            else if (feat->_type == FeatureType::kSlamPoint)
            {
                Hx_full.block(Hx_rows, 0, Hfx_single.rows(), Hfx_single.cols()) = Hfx_single;
                Hx_rows += Hfx_single.rows();
            }
        }
    }
    Hx_full.conservativeResize(Hx_rows, Hx_full.cols());

    if(Hx_rows == 0)
    {
        LOG(WARNING) << "No valid feature jacobian constructed.";
        return false;
    }

    // measurements compression
    if (Hx_full.rows() > Hx_full.cols())
    {
        Hx_full = utils::math::GivensRotation(Hx_full, Hx_full.cols() - 1);
        int final_hx_rows = Hx_full.cols() - 1;
        res.resize(final_hx_rows, 1);
        res = Hx_full.block(0, Hx_full.cols() - 1, final_hx_rows, 1);
        Hx_full.conservativeResize(final_hx_rows, Hx_full.cols() - 1);
        // utils::show_eigen_matrix(Hx_full, "Hx_full_qr");
    }
    else
    {
        res.resize(Hx_full.rows(), 1);
        res = Hx_full.block(0, Hx_full.cols() - 1, Hx_full.rows(), 1);
        Hx_full.conservativeResize(Hx_full.rows(), Hx_full.cols() - 1);
    }

    return true;
}

bool VisualManager::SingleFeatureJacobian(const Feature* feat,
                                          const std::unordered_map<std::shared_ptr<Type>, size_t> map_hx,
                                          const int total_hx,
                                          Eigen::MatrixXd& Hfx_single)
{
    // constexpr uint32_t kPwfCols = 3;
    constexpr uint32_t kResidualCols = 1;

    assert(feat->_valid);
    // assert(feat->_type != FeatureType::kUnknown);

    const uint32_t obs_size = 4 * feat->_visual_obs_buffer.size();
    const uint32_t reserve_cols = (feat->_type == FeatureType::kSlamPoint) ? 0 : 3;
    const uint32_t Hfx_cols = reserve_cols + total_hx + kResidualCols;
    Eigen::MatrixXd Hfx = Eigen::MatrixXd::Zero(obs_size, Hfx_cols);
    Eigen::Vector3d p_finG = feat->_pwf;

    Eigen::Vector2d res_total = Eigen::Vector2d::Zero();
    int cnt = 0;
    for (auto& obs : feat->_visual_obs_buffer)
    {
        double obs_ts = obs.first;
        for (int cam_id = 0; cam_id < param_.camera_num; cam_id++)
        {
            double focal_length;
            Eigen::Matrix3d R_CtoI;
            Eigen::Vector3d p_CinI;

            focal_length = CamModel::getInstance().K(LEFT_CAM)(0, 0);
            R_CtoI = _state->Qic(cam_id).toRotationMatrix();
            p_CinI = _state->Pic(cam_id);

            std::shared_ptr<Pose> obs_pose = _state->_clone_pose.at(obs_ts);
            Eigen::Matrix3d R_IitoG = obs_pose->quat().toRotationMatrix();
            Eigen::Vector3d p_IiinG = obs_pose->p();

            Eigen::Matrix3d R_CitoG = R_IitoG * R_CtoI;
            Eigen::Vector3d p_CiinG = p_IiinG + R_IitoG * p_CinI;

            Eigen::Vector3d p_finCi = R_CitoG.transpose() * (p_finG - p_CiinG);

            // // Compute visual observations and residuals
            Eigen::Vector2d zm = obs.second.uv.at(cam_id);
            Eigen::Vector2d uv_dist = CamModel::getInstance().project_distort(cam_id, p_finCi);
            Eigen::Vector2d res = zm - uv_dist;
            Hfx.block<2, 1>(2 * cnt, Hfx.cols() - kResidualCols) = res;

            if (param_.use_fej)
            {
                R_IitoG = obs_pose->quat_fej().toRotationMatrix();
                p_IiinG = obs_pose->p_fej();
                R_CitoG = R_IitoG * R_CtoI;
                p_CiinG = p_IiinG + R_IitoG * p_CinI;
                p_finCi = R_CitoG.transpose() * (p_finG - p_CiinG);
            }

            // Pre-compute dz_dpcf
            Eigen::MatrixXd dzn_dpcf = Eigen::MatrixXd::Zero(2, 3);
            dzn_dpcf << 1 / p_finCi(2), 0, -p_finCi(0) / (p_finCi(2) * p_finCi(2)),
                        0, 1 / p_finCi(2), -p_finCi(1) / (p_finCi(2) * p_finCi(2));

            Eigen::MatrixXd dz_dzn;
            Eigen::Vector2d uv_norm(p_finCi(0) / p_finCi(2), p_finCi(1) / p_finCi(2));
            CamModel::getInstance().compute_distort_jacobian(cam_id, uv_norm, dz_dzn);
            Eigen::MatrixXd dz_dpcf = dz_dzn * dzn_dpcf;

            // Get jacobian wrt pwf
            Eigen::Matrix3d dpcf_dpwf = R_CitoG.transpose();
            uint32_t pwf_id = 0;
            if (feat->_type == FeatureType::kSlamPoint)
            {
                pwf_id = map_hx.at(_state->slam_features().at(feat->_id)._state_ptr);
            }
            Hfx.block<2, 3>(2 * cnt, pwf_id) = dz_dpcf * dpcf_dpwf;

            // Get jacobian wrt extrinsic parameters
            if (_state->enableEstimateRic())
            {
                Eigen::Matrix3d dpcf_dqic = utils::math::skew(p_finCi);
                Hfx.block<2, 3>(2 * cnt, reserve_cols + map_hx.at(_state->mutable_Qic(cam_id))) = dz_dpcf * dpcf_dqic;
            }

            // Get jacobian wrt clone pose
            Eigen::MatrixXd dpcf_dclone = Eigen::MatrixXd::Zero(3, 6);
            dpcf_dclone.block<3, 3>(0, 0) = R_CtoI.transpose() * utils::math::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
            dpcf_dclone.block<3, 3>(0, 3) = -R_CitoG.transpose();
            Hfx.block<2, 6>(2 * cnt, reserve_cols + map_hx.at(obs_pose)) = dz_dpcf * dpcf_dclone;

            res_total += res;

            // // Debug: Check the correctness of Hx
            // // ------------------------ Check pwf ------------------------
            // std::cout << "------------------------\n" << std::endl;
            // Eigen::Vector3d dpwf(0.1, 0.1, 0.1);
            // Eigen::Vector2d Hx_plus_dpwf = Hfx.block<2, 3>(2 * c, 0) * dpwf;
            // Eigen::Vector3d pcf_dpwf = R_CitoG.transpose() * (p_finG + dpwf - p_CiinG);
            // Eigen::Vector2d uv_dpwf(pcf_dpwf(0) / pcf_dpwf(2), pcf_dpwf(1) / pcf_dpwf(2));
            // std::cout << "uv_dpwf: " << (uv_dpwf - uv_norm - Hx_plus_dpwf).transpose() << std::endl;

            // // ------------------------ Check R_CtoI ------------------------
            // Eigen::Vector3d dtheta_CtoI(0.01, 0.01, 0.01);
            // Eigen::Vector2d Hx_plus_dR_ItoC = Hfx.block<2, 3>(2 * cnt, kPwfCols + map_hx.at(_state->qic_)) * dtheta_CtoI;
            // Eigen::Matrix3d delta_R_CtoI = Eigen::Matrix3d::Identity() + utils::math::skew(dtheta_CtoI);
            // Eigen::Matrix3d R_CtoI_hat, R_CitoG_hat;
            // Eigen::Vector3d p_CinI_hat, p_CiinG_hat, p_finCi_hat;

            // if (cam_id == LEFT_CAM)
            // {
            //     R_CtoI_hat = _state->qic_->q().toRotationMatrix() * delta_R_CtoI;
            //     p_CinI_hat = _state->tic_->vec();
            // }
            // else if (cam_id == RIGHT_CAM)
            // {
            //     R_CtoI_hat = _state->qic_->q().toRotationMatrix() * delta_R_CtoI * CamModel::getInstance().Rlr();
            //     p_CinI_hat = _state->tic_->vec() + _state->qic_->q().toRotationMatrix() * delta_R_CtoI * CamModel::getInstance().plr();
            // }
            // R_CitoG_hat = R_IitoG * R_CtoI_hat;
            // p_CiinG_hat = p_IiinG + R_IitoG * p_CinI_hat;
            // p_finCi_hat = R_CitoG_hat.transpose() * (p_finG - p_CiinG_hat);
            // Eigen::Vector2d uv_hat = CamModel::getInstance().project_distort(cam_id, p_finCi_hat);
            // Eigen::Vector2d dis = uv_hat - uv - Hx_plus_dR_ItoC;
            // std::cout << cv::format("cam_id: %d, uv_hat: [%f, %f], uv: [%f, %f], Hx_plus_dR_ItoC: [%f, %f], distance: [%f, %f]",
            //                          cam_id, uv_hat(0),uv_hat(1), uv(0), uv(1), Hx_plus_dR_ItoC(0), Hx_plus_dR_ItoC(1), dis(0), dis(1))
            //           << std::endl;

            // // ------------------------ Check p_IinC ------------------------
            // Eigen::Vector3d dp_CinI(0.1, 0.1, 0.1);
            // Eigen::Vector2d Hx_plus_dp_IinC = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(_state->_Tic) + 3) * dp_CinI;
            // Eigen::Vector3d p_CiinG_hat = p_IiinG + R_IitoG * (p_CinI + dp_CinI);
            // Eigen::Vector3d pcf_dp_ItoC = R_CitoG.transpose() * (p_finG - p_CiinG_hat);
            // Eigen::Vector2d uv_dp_ItoC(pcf_dp_ItoC(0) / pcf_dp_ItoC(2), pcf_dp_ItoC(1) / pcf_dp_ItoC(2));
            // std::cout << "uv_dp_ItoC: " << (uv_dp_ItoC - uv_norm - Hx_plus_dp_IinC).transpose() << std::endl;

            // // ------------------------ Check clone pose R ------------------------
            // Eigen::Vector3d dR_ItoG(0.1, 0.1, 0.1);
            // Eigen::Vector2d Hx_plus_dR_ItoG = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(obs_pose)) * dR_ItoG;
            // Eigen::Matrix3d dR_ItoG_mat = Eigen::Matrix3d::Identity() + utils::math::skew(dR_ItoG.head(3));
            // Eigen::Matrix3d R_IitoG_hat = R_IitoG * dR_ItoG_mat;
            // p_CiinG_hat = p_IiinG + R_IitoG_hat * p_CinI;
            // Eigen::Vector3d pcf_dR_ItoG = (R_IitoG * dR_ItoG_mat * R_CtoI).transpose() * (p_finG - p_CiinG_hat);
            // Eigen::Vector2d uv_dR_ItoG(pcf_dR_ItoG(0) / pcf_dR_ItoG(2), pcf_dR_ItoG(1) / pcf_dR_ItoG(2));
            // std::cout << "uv_dR_ItoG: " << (uv_dR_ItoG - uv_norm - Hx_plus_dR_ItoG).transpose() << std::endl;

            // // ------------------------ Check clone pose p ------------------------
            // Eigen::Vector3d dp_IinG(0.1, 0.1, 0.1);
            // Eigen::Vector2d Hx_plus_dp_IinG = Hfx.block<2, 3>(2 * c, kPwfDim + map_hx.at(obs_pose) + 3) * dp_IinG;
            // p_CiinG_hat = p_IiinG + dp_IinG + R_IitoG * p_CinI;
            // Eigen::Vector3d pcf_dp_IinG = R_CitoG.transpose() * (p_finG - p_CiinG_hat);
            // Eigen::Vector2d uv_dp_IinG(pcf_dp_IinG(0) / pcf_dp_IinG(2), pcf_dp_IinG(1) / pcf_dp_IinG(2));
            // std::cout << "uv_dp_IinG: " << (uv_dp_IinG - uv_norm - Hx_plus_dp_IinG).transpose() << std::endl;

            cnt++;
        }
    }

    double threshold = 4.0f;
    if (res_total.norm() / cnt > threshold)
    {
        std::cout << "residual is too large: " << res_total.norm() / cnt << ", threashold is: " << threshold << std::endl;
        return false;
    }

    /*show single Hx matrix*/
    // utils::show_eigen_matrix(Hfx, "Hfx");

    /* project Hfx to feature left null space */
    // utils::math::NullSpaceProjectInplace(Hfx, 3);
    Hfx = utils::math::GivensRotation(Hfx, 3);
    Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(Hfx.rows() - 3, Hfx.cols() - 3);
    Hx.noalias() = Hfx.block(3, 3, Hfx.rows() - 3, Hfx.cols() - 3);

    Hfx_single = Hfx;

    return true;
}

bool VisualManager::SingleFeatureJacobianSlam(const Feature* feat,
                                              const std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping,
                                              const int total_hx,
                                              Eigen::MatrixXd& Hf,
                                              Eigen::MatrixXd& Hx,
                                              Eigen::VectorXd& residual)
{
    uint32_t total_meas = 0;
    for (auto& obs : feat->_visual_obs_buffer)
    {
        // Mono and Stereo have different measurement dimension, but we can simply use camera_num() to represent it
        // since the right camera will not contribute to the residual if it's mono
        total_meas += CamModel::getInstance().camera_num();
    }
    if (total_meas == 0) {
        LOG(WARNING) << "Feature " << feat->_id << " has no observations";
        return false;
    }
    Hf = Eigen::MatrixXd::Zero(2 * total_meas, 3);
    Hx = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);
    residual = Eigen::VectorXd::Zero(2 * total_meas);

    Eigen::Vector3d p_finG = feat->_pwf;
    Eigen::Vector2d res_total = Eigen::Vector2d::Zero();
    uint32_t cnt = 0;

    for (auto& obs : feat->_visual_obs_buffer)
    {
        double obs_ts = obs.first;
        for (int cam_id = 0; cam_id < param_.camera_num; cam_id++)
        {
            double focal_length;
            Eigen::Matrix3d R_CtoI;
            Eigen::Vector3d p_CinI;

            focal_length = CamModel::getInstance().K(LEFT_CAM)(0, 0);
            R_CtoI = _state->Qic(cam_id).toRotationMatrix();
            p_CinI = _state->Pic(cam_id);

            auto pose_it = _state->_clone_pose.find(obs_ts);
            if (pose_it == _state->_clone_pose.end()) {
                LOG(WARNING) << "Observation timestamp not in clone poses: " << obs_ts;
                continue;
            }
            std::shared_ptr<Pose> obs_pose = pose_it->second;
            Eigen::Matrix3d R_IitoG = obs_pose->quat().toRotationMatrix();
            Eigen::Vector3d p_IiinG = obs_pose->p();

            Eigen::Matrix3d R_CitoG = R_IitoG * R_CtoI;
            Eigen::Vector3d p_CiinG = p_IiinG + R_IitoG * p_CinI;

            Eigen::Vector3d p_finCi = R_CitoG.transpose() * (p_finG - p_CiinG);

            if (p_finCi(2) <= 0) {
                LOG(WARNING) << "Feature " << feat->_id << " is behind camera (z=" << p_finCi(2) << ")";
                return false;
            }

            // // Compute visual observations and residuals
            Eigen::Vector2d zm = obs.second.uv.at(cam_id);
            Eigen::Vector2d uv_dist = CamModel::getInstance().project_distort(cam_id, p_finCi);
            Eigen::Vector2d res = zm - uv_dist;
            residual.segment<2>(2 * cnt) = res;

            if (param_.use_fej)
            {
                R_IitoG = obs_pose->quat_fej().toRotationMatrix();
                p_IiinG = obs_pose->p_fej();
                R_CitoG = R_IitoG * R_CtoI;
                p_CiinG = p_IiinG + R_IitoG * p_CinI;
                p_finCi = R_CitoG.transpose() * (p_finG - p_CiinG);
            }

            // Pre-compute dz_dpcf
            Eigen::MatrixXd dzn_dpcf = Eigen::MatrixXd::Zero(2, 3);
            dzn_dpcf << 1 / p_finCi(2), 0, -p_finCi(0) / (p_finCi(2) * p_finCi(2)),
                        0, 1 / p_finCi(2), -p_finCi(1) / (p_finCi(2) * p_finCi(2));

            Eigen::MatrixXd dz_dzn;
            Eigen::Vector2d uv_norm(p_finCi(0) / p_finCi(2), p_finCi(1) / p_finCi(2));
            CamModel::getInstance().compute_distort_jacobian(cam_id, uv_norm, dz_dzn);
            Eigen::MatrixXd dz_dpcf = dz_dzn * dzn_dpcf;

            // Get jacobian wrt pwf
            Eigen::Matrix3d dpcf_dpwf = R_CitoG.transpose();
            Hf.block<2, 3>(2 * cnt, 0) = dz_dpcf * dpcf_dpwf;

            // Get jacobian wrt extrinsic parameters
            if (_state->enableEstimateRic())
            {
                Eigen::Matrix3d dpcf_dqic = utils::math::skew(p_finCi);
                Hx.block<2, 3>(2 * cnt, Hx_mapping.at(_state->mutable_Qic(cam_id))) = dz_dpcf * dpcf_dqic;
            }

            // Get jacobian wrt clone pose
            Eigen::MatrixXd dpcf_dclone = Eigen::MatrixXd::Zero(3, 6);
            dpcf_dclone.block<3, 3>(0, 0) = R_CtoI.transpose() * utils::math::skew(R_IitoG.transpose() * (p_finG - p_IiinG));
            dpcf_dclone.block<3, 3>(0, 3) = -R_CitoG.transpose();
            Hx.block<2, 6>(2 * cnt, Hx_mapping.at(obs_pose)) = dz_dpcf * dpcf_dclone;

            res_total += res;
            cnt++;
        }
    }

    double threshold = 4.0f;
    if (res_total.norm() / cnt > threshold)
    {
        std::cout << "residual is too large: " << res_total.norm() / cnt << ", threashold is: " << threshold << std::endl;
        return false;
    }

    // utils::math::NullSpaceProjectInplace(Hfx, 3);
    // // Hfx = utils::math::GivensRotation(Hfx, 3);
    // Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(Hfx.rows() - 3, Hfx.cols() - 3);
    // Hx.noalias() = Hfx.block(3, 3, Hfx.rows() - 3, Hfx.cols() - 3);

    return true;
}

