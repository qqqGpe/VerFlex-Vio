#include "sfm.h"

bool Sfm::MaybeAddSfmKeyframes(const std::pair<double, std::vector<CameraObs>>& current_feature_observe)
{
    if (feature_observes.second.size() < kMinRequiredFeaturesPerFrame)
    {
        return false;
    }

    if (all_feature_observes_.empty())
    {
        all_feature_observes_.insert(current_feature_observe);
        latest_keyframe_observe_ = current_feature_observe;
        return true;
    }
    else
    {
        std::unordered_map<uint32_t, CameraObs> latest_keyframe_observe_umap;
        std::unordered_map<uint32_t, CameraObs> current_keyframe_observe_umap;
        for (auto& obs : latest_keyframe_observe_.second)
        {
            latest_keyframe_observe_umap.insert({obs.feat_id, obs});
        }
        for (auto& obs : current_feature_observe.second)
        {
            current_keyframe_observe_umap.insert({obs.feat_id, obs});
        }

        double pixel_parallex = VisualManager::calcVisualObsParallex(latest_keyframe_observe_umap, current_keyframe_observe_umap);
        if (pixel_parallex >= kMinPixelParallexBetweenKeyframes)
        {
            all_feature_observes_.insert(current_feature_observe);
            latest_keyframe_observe_ = current_feature_observe;
            return true;
        }
        else if (abs(current_feature_observe.first - latest_keyframe_observe_.first) > kMaxTimeIntervalBetweenKeyframes)
        {
            LOG(INFO) << "Platform moves too slow, can not perform dynamic-initialization";
            ResetSfm();
            return false;
        }
    }

    return false;
}