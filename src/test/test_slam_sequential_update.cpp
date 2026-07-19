/**
 * @file test_slam_sequential_update.cpp
 * @brief Unit tests for VisualManager::SlamFeatureUpdate sequential update behavior
 */

#include <gtest/gtest.h>

#include "eskf_solver.h"
#include "visualManager.h"
#include "vioState.h"
#include "camera_model.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace
{
constexpr double kCameraTs[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
constexpr double kIntrinsic[] = {458.654, 457.296, 367.215, 248.375};
constexpr int kMaxClonePose = 6;
constexpr int kCameraNum = 1;
}

class TestableVisualManager : public VisualManager
{
public:
    using VisualManager::_state;
    using VisualManager::param_;
    using VisualManager::solver_;
    
    void setParam(const Parameter& params) { param_ = params; }
    void setSolver(std::shared_ptr<MsckfSolverBase> solver) { solver_ = solver; }
};

class SlamSequentialUpdateTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        initParams();
        
        static bool cam_model_initialized = false;
        if (!cam_model_initialized)
        {
            CamModel::Init(params_);
            cam_model_initialized = true;
        }
        
        state_ = std::make_shared<State>(params_);
        solver_ = std::make_shared<eskfSolver>(params_.use_fej, params_.enable_schmidt_eskf);
        visual_manager_ = createVisualManager();
    }

    void TearDown() override
    {
        for (auto* f : features_) delete f;
        features_.clear();
    }

    void initParams()
    {
        params_.camera_num = kCameraNum;
        params_.max_clone_pose = kMaxClonePose;
        params_.max_feat_n = 225;
        params_.sigma_visual_pix = 1.0;
        params_.use_fej = false;
        params_.estimate_ric = false;
        params_.estimate_td_visual = false;
        
        params_.intrinsics.resize(kCameraNum);
        params_.intrinsics[0] << kIntrinsic[0], 0, kIntrinsic[2],
                                  0, kIntrinsic[1], kIntrinsic[3],
                                  0, 0, 1;
        params_.distortion.resize(kCameraNum);
        params_.distortion[0] = Eigen::VectorXd::Zero(4);
        
        params_.Ric.resize(kCameraNum, Eigen::Matrix3d::Identity());
        params_.tic.resize(kCameraNum, Eigen::Vector3d::Zero());
    }

    std::shared_ptr<TestableVisualManager> createVisualManager()
    {
        auto vm = std::make_shared<TestableVisualManager>();
        vm->set_state(state_);
        vm->param_ = params_;
        vm->setSolver(solver_);
        return vm;
    }

    void addClonePoses(int count)
    {
        for (int i = 0; i < count; i++)
        {
            auto pose = std::make_shared<Pose>();
            pose->set_ts(kCameraTs[i]);
            pose->set_pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d(i * 0.1, 0, 0));
            
            std::shared_ptr<Type> last_var = state_->_variables.back();
            state_->insert_after(last_var, pose);
            state_->mutable_clone_poses().emplace(kCameraTs[i], pose);
        }
        // Resize covariance to match state dimension after adding clone poses
        state_->SetCovariance(Eigen::MatrixXd::Identity(state_->_dim, state_->_dim));
    }

    Feature* createFeature(uint32_t id)
    {
        Feature* feat = new Feature();
        feat->_id = id;
        feat->_type = FeatureType::kSlamPoint;
        feat->_is_triangulated = true;
        feat->_pwf = Eigen::Vector3d(1.0, 0.5, 2.0);
        feat->_valid = true;
        features_.push_back(feat);
        return feat;
    }

    Feature* createFeatureWithObservations(uint32_t id, const Eigen::Vector3d& pwf)
    {
        Feature* feat = new Feature();
        feat->_id = id;
        feat->_type = FeatureType::kSlamPoint;
        feat->_is_triangulated = true;
        feat->_pwf = pwf;
        feat->_valid = true;
        
        const double fu = kIntrinsic[0];
        const double fv = kIntrinsic[1];
        const double cu = kIntrinsic[2];
        const double cv = kIntrinsic[3];
        
        for (int i = 0; i < kMaxClonePose; i++)
        {
            Eigen::Matrix3d R_ItoG = Eigen::Matrix3d::Identity();
            Eigen::Vector3d p_IinG(i * 0.1, 0, 0);
            
            Eigen::Matrix3d R_CtoG = R_ItoG * params_.Ric[0];
            Eigen::Vector3d p_CinG = p_IinG + R_ItoG * params_.tic[0];
            
            Eigen::Vector3d p_in_C = R_CtoG.transpose() * (pwf - p_CinG);
            
            double x_norm = p_in_C.x() / p_in_C.z();
            double y_norm = p_in_C.y() / p_in_C.z();
            
            CameraObs obs;
            obs.ts_sec = kCameraTs[i];
            obs.feat_id = id;
            obs.valid = true;
            obs.uv[0] = Eigen::Vector2d(fu * x_norm + cu, fv * y_norm + cv);
            obs.uv_norm[0] = Eigen::Vector2d(x_norm, y_norm);
            feat->_visual_obs_buffer.emplace(kCameraTs[i], obs);
        }
        
        features_.push_back(feat);
        return feat;
    }

    void addSlamFeatureToState(Feature* feat)
    {
        auto feat_state = std::make_shared<Vec>(feat->_pwf);
        SlamFeature slam_feat{
            ._id = feat->_id,
            ._state_ptr = feat_state,
            ._info = feat
        };
        
        int old_dim = state_->_dim;
        std::shared_ptr<Type> last_var = state_->_variables.back();
        state_->insert_after(last_var, feat_state);
        state_->mutable_slam_features().emplace(feat->_id, slam_feat);
        
        // Augment covariance: copy old covariance and add diagonal for feature
        Eigen::MatrixXd old_cov = state_->covariance();
        int new_dim = state_->_dim;
        Eigen::MatrixXd new_cov = Eigen::MatrixXd::Zero(new_dim, new_dim);
        new_cov.topLeftCorner(old_dim, old_dim) = old_cov;
        new_cov.bottomRightCorner(3, 3) = Eigen::Matrix3d::Identity() * 0.1;
        state_->SetCovariance(new_cov);
    }

    Parameter params_;
    std::shared_ptr<State> state_;
    std::shared_ptr<eskfSolver> solver_;
    std::shared_ptr<TestableVisualManager> visual_manager_;
    std::vector<Feature*> features_;
};

TEST_F(SlamSequentialUpdateTest, EmptyFeatureList_ReturnsFalse)
{
    addClonePoses(kMaxClonePose);
    std::vector<Feature*> empty_feats;
    
    bool result = visual_manager_->SlamFeatureUpdate(empty_feats);
    
    EXPECT_FALSE(result);
}

TEST_F(SlamSequentialUpdateTest, NotEnoughClonePoses_ReturnsFalse)
{
    addClonePoses(kMaxClonePose - 1);
    
    Feature* feat = createFeature(0);
    addSlamFeatureToState(feat);
    
    std::vector<Feature*> feats = {feat};
    
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_FALSE(result);
}

TEST_F(SlamSequentialUpdateTest, NonTriangulatedFeature_Skipped)
{
    addClonePoses(kMaxClonePose);
    
    Feature* feat = createFeature(0);
    feat->_is_triangulated = false;
    addSlamFeatureToState(feat);
    
    std::vector<Feature*> feats = {feat};
    
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_FALSE(result);
}

TEST_F(SlamSequentialUpdateTest, SingleFeatureUpdate_ReturnsTrue)
{
    addClonePoses(kMaxClonePose);
    
    Feature* feat = createFeatureWithObservations(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    addSlamFeatureToState(feat);
    
    std::vector<Feature*> feats = {feat};
    
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_TRUE(result);
}

TEST_F(SlamSequentialUpdateTest, MultipleFeatures_UpdatesAll)
{
    addClonePoses(kMaxClonePose);
    
    Feature* feat1 = createFeatureWithObservations(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    Feature* feat2 = createFeatureWithObservations(1, Eigen::Vector3d(1.5, 0.3, 2.5));
    Feature* feat3 = createFeatureWithObservations(2, Eigen::Vector3d(0.8, 0.7, 1.8));
    
    addSlamFeatureToState(feat1);
    addSlamFeatureToState(feat2);
    addSlamFeatureToState(feat3);
    
    std::vector<Feature*> feats = {feat1, feat2, feat3};
    
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(state_->slam_features().size(), 3);
}

TEST_F(SlamSequentialUpdateTest, MixedValidInvalidFeatures_ProcessesValid)
{
    addClonePoses(kMaxClonePose);
    
    Feature* valid_feat = createFeatureWithObservations(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    Feature* invalid_feat = createFeatureWithObservations(1, Eigen::Vector3d(1.5, 0.3, 2.5));
    invalid_feat->_is_triangulated = false;
    
    addSlamFeatureToState(valid_feat);
    addSlamFeatureToState(invalid_feat);
    
    std::vector<Feature*> feats = {valid_feat, invalid_feat};
    
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_TRUE(result);
}

TEST_F(SlamSequentialUpdateTest, CovarianceChangesAfterUpdate)
{
    addClonePoses(kMaxClonePose);
    
    // Create feature with perturbed position to generate non-zero residuals
    Feature* feat = createFeatureWithObservations(0, Eigen::Vector3d(1.05, 0.52, 2.03));
    addSlamFeatureToState(feat);
    
    // Store covariance diagonal before update
    Eigen::VectorXd cov_diag_before = state_->covariance().diagonal();
    
    std::vector<Feature*> feats = {feat};
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_TRUE(result);
    
    // Get covariance after update
    const Eigen::MatrixXd& cov_after = state_->covariance();
    Eigen::VectorXd cov_diag_after = cov_after.diagonal();
    
    // Verify covariance diagonal has changed
    bool diagonal_changed = !cov_diag_before.isApprox(cov_diag_after);
    EXPECT_TRUE(diagonal_changed) << "Covariance diagonal should change after update";
    
    // Verify covariance remains symmetric
    EXPECT_TRUE(cov_after.isApprox(cov_after.transpose()))
        << "Covariance should remain symmetric after update";
    
    // Verify all diagonal elements remain positive
    for (int i = 0; i < cov_diag_after.size(); i++)
    {
        EXPECT_GT(cov_diag_after(i), 0.0)
            << "Diagonal element " << i << " should be positive, got " << cov_diag_after(i);
    }
}

TEST_F(SlamSequentialUpdateTest, FeaturePositionUpdatedAfterUpdate)
{
    addClonePoses(kMaxClonePose);
    
    // Create feature with perturbed position (observations are for exact position)
    Eigen::Vector3d perturbed_pwf(1.05, 0.52, 2.03);
    Feature* feat = createFeatureWithObservations(0, perturbed_pwf);
    addSlamFeatureToState(feat);
    
    // Store original position
    Eigen::Vector3d pwf_before = feat->_pwf;
    
    std::vector<Feature*> feats = {feat};
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_TRUE(result);
    
    // Sync state back to feature
    state_->UpdateSlamFeatureAfterVisualUpdate();
    
    // Verify _pwf has changed after update
    Eigen::Vector3d pwf_after = feat->_pwf;
    bool position_changed = !pwf_before.isApprox(pwf_after);
    EXPECT_TRUE(position_changed)
        << "Feature position should change after update. Before: " << pwf_before.transpose()
        << ", After: " << pwf_after.transpose();
}

TEST_F(SlamSequentialUpdateTest, SlamFeatureStateMatchesPwf)
{
    addClonePoses(kMaxClonePose);
    
    // Create feature with perturbed position
    Eigen::Vector3d perturbed_pwf(1.05, 0.52, 2.03);
    Feature* feat = createFeatureWithObservations(0, perturbed_pwf);
    addSlamFeatureToState(feat);
    
    std::vector<Feature*> feats = {feat};
    bool result = visual_manager_->SlamFeatureUpdate(feats);
    
    EXPECT_TRUE(result);
    
    // Sync state back to feature
    state_->UpdateSlamFeatureAfterVisualUpdate();
    
    // Verify slam feature state matches _pwf
    const auto& slam_features = state_->slam_features();
    ASSERT_EQ(slam_features.count(0), 1) << "SLAM feature 0 should exist";
    
    const auto& slam_feat = slam_features.at(0);
    Eigen::Vector3d state_vec = slam_feat._state_ptr->vec();
    Eigen::Vector3d pwf = feat->_pwf;
    
    EXPECT_TRUE(state_vec.isApprox(pwf))
        << "SLAM feature state should match _pwf. State: " << state_vec.transpose()
        << ", _pwf: " << pwf.transpose();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
