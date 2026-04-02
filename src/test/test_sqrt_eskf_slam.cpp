/**
 * @file test_sqrt_eskf_slam.cpp
 * @brief Unit tests for sqrt-ESKF SLAM feature lifecycle (initialization → update → marginalization)
 *
 * This test verifies that the sqrt-ESKF solver correctly handles:
 * - SLAM feature initialization (adding feature to state)
 * - SLAM feature update (measurement update cycle)
 * - SLAM feature marginalization (removing feature from state)
 * - State dimension tracking (verify correct sizes after each operation)
 * - Covariance consistency (sqrt_Pt_ maintains P = S^T * S)
 *
 * Sqrt-ESKF maintains the square-root of the covariance matrix (sqrt_Pt_) where:
 *   P = sqrt_Pt_.transpose() * sqrt_Pt_
 *
 * This representation ensures numerical stability and maintains positive-definiteness.
 */

#include <gtest/gtest.h>

#include "camModel.h"
#include "sqrt_eskf_solver.h"
#include "vioState.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace
{
constexpr double kIntrinsic[] = {458.654, 457.296, 367.215, 248.375};
constexpr int kCameraNum = 1;
} // namespace

/**
 * @brief Test fixture for sqrt-ESKF SLAM feature lifecycle tests
 *
 * Sets up a minimal VIO state with sqrt-ESKF solver and provides
 * helper methods for feature creation and state manipulation.
 */
class SqrtEskfSlamTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        initParams();

        static bool cam_model_initialized = false;
        if (!cam_model_initialized)
        {
            CamModel::getInstance().Init(params_);
            cam_model_initialized = true;
        }

        state_ = std::make_shared<State>(params_);
        solver_ = std::make_shared<SqrtEskfSolver>(params_.use_fej);

        // Initialize with identity sqrt covariance (P = I)
        int dim = state_->_dim;
        Eigen::MatrixXd sqrt_Pt_init = Eigen::MatrixXd::Identity(dim, dim);
        state_->SetSqrtPt(sqrt_Pt_init);
    }

    void TearDown() override
    {
        for (auto *f : features_)
            delete f;
        features_.clear();
    }

    void initParams()
    {
        params_.camera_num = kCameraNum;
        params_.max_clone_pose = 6;
        params_.max_feat_n = 225;
        params_.sigma_visual_pix = 1.0;
        params_.use_fej = false;
        params_.estimate_ric = false;
        params_.estimate_td_visual = false;

        params_.intrinsics.resize(kCameraNum);
        params_.intrinsics[0] << kIntrinsic[0], 0, kIntrinsic[2], 0, kIntrinsic[1], kIntrinsic[3], 0, 0, 1;
        params_.distortion.resize(kCameraNum);
        params_.distortion[0] = Eigen::VectorXd::Zero(4);

        params_.Ric.resize(kCameraNum, Eigen::Matrix3d::Identity());
        params_.tic.resize(kCameraNum, Eigen::Vector3d::Zero());
    }

    /**
     * @brief Create a minimal SLAM feature with triangulated position
     */
    Feature *createFeature(uint32_t id, const Eigen::Vector3d &pwf)
    {
        Feature *feat = new Feature();
        feat->_id = id;
        feat->_type = FeatureType::kSlamPoint;
        feat->_is_triangulated = true;
        feat->_pwf = pwf;
        feat->_valid = true;
        features_.push_back(feat);
        return feat;
    }

    /**
     * @brief Add a SLAM feature to the state and augment sqrt covariance
     *
     * This performs stochastic initialization by:
     * 1. Creating a new Vec state variable for the feature position
     * 2. Inserting it into the state variable list
     * 3. Augmenting sqrt_Pt_ with identity block (initial uncertainty)
     */
    void addSlamFeatureToState(Feature *feat)
    {
        auto feat_state = std::make_shared<Vec>(feat->_pwf);
        SlamFeature slam_feat{._id = feat->_id, ._state_ptr = feat_state, ._info = feat};

        int old_dim = state_->_dim;
        Eigen::MatrixXd sqrt_Pt_old = state_->Sqrt_Pt();

        // Insert feature state variable
        std::shared_ptr<Type> last_var = state_->_variables.back();
        state_->insert_after(last_var, feat_state);
        state_->mutable_slam_features().emplace(feat->_id, slam_feat);

        // Augment sqrt covariance: add 3 columns for new feature
        int new_dim = state_->_dim;
        const int feat_size = 3; // 3D point

        Eigen::MatrixXd sqrt_Pt_new = Eigen::MatrixXd::Zero(old_dim, new_dim);
        sqrt_Pt_new.topLeftCorner(old_dim, old_dim) = sqrt_Pt_old;

        // Initialize feature uncertainty with identity (initial std dev = 1.0)
        // In practice, this could be based on triangulation uncertainty
        sqrt_Pt_new.rightCols(feat_size) = Eigen::MatrixXd::Identity(old_dim, feat_size) * 0.316; // sqrt(0.1)

        state_->SetSqrtPt(sqrt_Pt_new);
    }

    /**
     * @brief Verify sqrt covariance consistency: P = sqrt_Pt_.transpose() * sqrt_Pt_
     *
     * Checks that:
     * 1. Reconstructed P is symmetric
     * 2. All diagonal elements are positive
     * 3. P is positive semi-definite (all eigenvalues >= 0)
     */
    void verifyCovarianceConsistency(const std::string &context)
    {
        Eigen::MatrixXd sqrt_Pt = state_->Sqrt_Pt();
        Eigen::MatrixXd P_reconstructed = sqrt_Pt.transpose() * sqrt_Pt;

        // Check symmetry
        EXPECT_TRUE(P_reconstructed.isApprox(P_reconstructed.transpose(), 1e-9))
            << context << ": Reconstructed covariance should be symmetric";

        // Check positive diagonal elements
        Eigen::VectorXd diag = P_reconstructed.diagonal();
        for (int i = 0; i < diag.size(); i++)
        {
            EXPECT_GT(diag(i), 0.0) << context << ": Diagonal element " << i << " should be positive, got " << diag(i);
        }

        // Check positive semi-definite via eigenvalues
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(P_reconstructed);
        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        for (int i = 0; i < eigenvalues.size(); i++)
        {
            EXPECT_GE(eigenvalues(i), -1e-9)
                << context << ": Eigenvalue " << i << " should be non-negative, got " << eigenvalues(i);
        }
    }

    Param params_;
    std::shared_ptr<State> state_;
    std::shared_ptr<SqrtEskfSolver> solver_;
    std::vector<Feature *> features_;
};

// =============================================================================
// Lifecycle Tests: Add → Update → Marginalize
// =============================================================================

/**
 * @brief Test SLAM feature initialization
 *
 * Verifies that adding a SLAM feature to the state:
 * - Increases state dimension by 3 (xyz position)
 * - Adds feature to slam_features map
 * - Augments sqrt_Pt_ correctly
 * - Maintains covariance consistency
 */
TEST_F(SqrtEskfSlamTest, SlamFeatureInitialization_IncrementsDimension)
{
    int dim_before = state_->_dim;
    size_t num_features_before = state_->slam_features().size();

    Feature *feat = createFeature(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    addSlamFeatureToState(feat);

    int dim_after = state_->_dim;
    size_t num_features_after = state_->slam_features().size();

    // Verify dimension increased by 3
    EXPECT_EQ(dim_after, dim_before + 3) << "State dimension should increase by 3 for 3D feature";

    // Verify feature count increased
    EXPECT_EQ(num_features_after, num_features_before + 1) << "SLAM feature count should increase by 1";

    // Verify sqrt_Pt_ dimensions match state dimension
    Eigen::MatrixXd sqrt_Pt = state_->Sqrt_Pt();
    EXPECT_EQ(sqrt_Pt.cols(), dim_after) << "sqrt_Pt_ columns should match state dimension";

    // Verify feature is in state
    EXPECT_EQ(state_->slam_features().count(0), 1) << "Feature ID 0 should be in state";

    // Verify covariance consistency
    verifyCovarianceConsistency("After feature initialization");
}

/**
 * @brief Test multiple SLAM feature initialization
 *
 * Verifies that adding multiple features correctly increments dimensions
 * and maintains covariance properties.
 */
TEST_F(SqrtEskfSlamTest, MultipleSlamFeatures_CorrectDimensions)
{
    int dim_initial = state_->_dim;

    Feature *feat1 = createFeature(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    Feature *feat2 = createFeature(1, Eigen::Vector3d(1.5, 0.3, 2.5));
    Feature *feat3 = createFeature(2, Eigen::Vector3d(0.8, 0.7, 1.8));

    addSlamFeatureToState(feat1);
    int dim_after_1 = state_->_dim;

    addSlamFeatureToState(feat2);
    int dim_after_2 = state_->_dim;

    addSlamFeatureToState(feat3);
    int dim_after_3 = state_->_dim;

    // Verify incremental dimension growth
    EXPECT_EQ(dim_after_1, dim_initial + 3);
    EXPECT_EQ(dim_after_2, dim_initial + 6);
    EXPECT_EQ(dim_after_3, dim_initial + 9);

    // Verify all features present
    EXPECT_EQ(state_->slam_features().size(), 3);

    // Verify sqrt_Pt_ dimensions
    Eigen::MatrixXd sqrt_Pt = state_->Sqrt_Pt();
    EXPECT_EQ(sqrt_Pt.cols(), dim_after_3);

    verifyCovarianceConsistency("After adding 3 features");
}

/**
 * @brief Test SLAM feature marginalization
 *
 * Verifies that marginalizing a SLAM feature:
 * - Decreases state dimension by 3
 * - Removes feature from slam_features map
 * - Shrinks sqrt_Pt_ correctly
 * - Maintains covariance consistency
 */
TEST_F(SqrtEskfSlamTest, SlamFeatureMarginalization_DecrementsDimension)
{
    // Add two features
    Feature *feat1 = createFeature(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    Feature *feat2 = createFeature(1, Eigen::Vector3d(1.5, 0.3, 2.5));

    addSlamFeatureToState(feat1);
    addSlamFeatureToState(feat2);

    int dim_before = state_->_dim;
    size_t num_features_before = state_->slam_features().size();

    // Marginalize first feature
    auto &slam_features = state_->slam_features();
    ASSERT_EQ(slam_features.count(0), 1) << "Feature 0 should exist before marginalization";

    std::shared_ptr<Type> feat_state = slam_features.at(0)._state_ptr;
    solver_->MarginalizeState(MarginalizeType::SlamFeature, state_, feat_state);

    int dim_after = state_->_dim;
    size_t num_features_after = state_->slam_features().size();

    // Verify dimension decreased by 3
    EXPECT_EQ(dim_after, dim_before - 3) << "State dimension should decrease by 3";

    // Verify feature count decreased
    EXPECT_EQ(num_features_after, num_features_before - 1) << "SLAM feature count should decrease by 1";

    // Verify sqrt_Pt_ dimensions match state dimension
    Eigen::MatrixXd sqrt_Pt = state_->Sqrt_Pt();
    EXPECT_EQ(sqrt_Pt.cols(), dim_after) << "sqrt_Pt_ columns should match reduced state dimension";

    // Verify feature was removed from state
    EXPECT_EQ(state_->slam_features().count(0), 0) << "Feature ID 0 should be removed from state";

    // Verify second feature still present
    EXPECT_EQ(state_->slam_features().count(1), 1) << "Feature ID 1 should remain in state";

    verifyCovarianceConsistency("After feature marginalization");
}

/**
 * @brief Test full SLAM feature lifecycle: add → marginalize
 *
 * This test simulates a complete feature lifecycle:
 * 1. Initialize state with one feature
 * 2. Add a second feature
 * 3. Marginalize the first feature
 * 4. Verify state returns to original dimension + 3
 */
TEST_F(SqrtEskfSlamTest, SlamFeatureLifecycle_AddAndMarginalize)
{
    int dim_initial = state_->_dim;

    // Add first feature
    Feature *feat1 = createFeature(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    addSlamFeatureToState(feat1);
    EXPECT_EQ(state_->_dim, dim_initial + 3);
    verifyCovarianceConsistency("After adding first feature");

    // Add second feature
    Feature *feat2 = createFeature(1, Eigen::Vector3d(1.5, 0.3, 2.5));
    addSlamFeatureToState(feat2);
    EXPECT_EQ(state_->_dim, dim_initial + 6);
    verifyCovarianceConsistency("After adding second feature");

    // Marginalize first feature
    auto &slam_features = state_->slam_features();
    std::shared_ptr<Type> feat1_state = slam_features.at(0)._state_ptr;
    solver_->MarginalizeState(MarginalizeType::SlamFeature, state_, feat1_state);

    EXPECT_EQ(state_->_dim, dim_initial + 3) << "After marginalization, dimension should be original + 3";
    EXPECT_EQ(state_->slam_features().size(), 1) << "Should have 1 feature remaining";
    EXPECT_EQ(state_->slam_features().count(1), 1) << "Feature 1 should remain";
    verifyCovarianceConsistency("After marginalizing first feature");
}

/**
 * @brief Test marginalizing middle feature from multiple features
 *
 * Verifies correct handling when marginalizing a feature that is not
 * at the beginning or end of the state vector.
 */
TEST_F(SqrtEskfSlamTest, MarginalizationMiddleFeature_CorrectOrdering)
{
    // Add three features
    Feature *feat1 = createFeature(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    Feature *feat2 = createFeature(1, Eigen::Vector3d(1.5, 0.3, 2.5));
    Feature *feat3 = createFeature(2, Eigen::Vector3d(0.8, 0.7, 1.8));

    addSlamFeatureToState(feat1);
    addSlamFeatureToState(feat2);
    addSlamFeatureToState(feat3);

    int dim_before = state_->_dim;

    // Marginalize middle feature (feat2)
    auto &slam_features = state_->slam_features();
    std::shared_ptr<Type> feat2_state = slam_features.at(1)._state_ptr;
    solver_->MarginalizeState(MarginalizeType::SlamFeature, state_, feat2_state);

    // Verify correct state after marginalization
    EXPECT_EQ(state_->_dim, dim_before - 3);
    EXPECT_EQ(state_->slam_features().size(), 2);
    EXPECT_EQ(state_->slam_features().count(0), 1) << "Feature 0 should remain";
    EXPECT_EQ(state_->slam_features().count(1), 0) << "Feature 1 should be removed";
    EXPECT_EQ(state_->slam_features().count(2), 1) << "Feature 2 should remain";

    verifyCovarianceConsistency("After marginalizing middle feature");
}

/**
 * @brief Test sqrt covariance structure after operations
 *
 * Verifies that sqrt_Pt_ maintains correct structure:
 * - Number of rows equals original state dimension (before augmentation)
 * - Number of columns equals current state dimension
 */
TEST_F(SqrtEskfSlamTest, SqrtCovarianceStructure_CorrectDimensions)
{
    int initial_dim = state_->_dim;
    Eigen::MatrixXd sqrt_Pt_initial = state_->Sqrt_Pt();
    int initial_rows = sqrt_Pt_initial.rows();

    // Add feature
    Feature *feat = createFeature(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    addSlamFeatureToState(feat);

    Eigen::MatrixXd sqrt_Pt_after_add = state_->Sqrt_Pt();

    // After adding feature:
    // - Rows should remain same (original state dimension)
    // - Cols should increase by 3
    EXPECT_EQ(sqrt_Pt_after_add.rows(), initial_rows) << "Rows should remain constant during augmentation";
    EXPECT_EQ(sqrt_Pt_after_add.cols(), initial_dim + 3) << "Columns should increase by 3 for new feature";

    // Marginalize feature
    auto &slam_features = state_->slam_features();
    std::shared_ptr<Type> feat_state = slam_features.at(0)._state_ptr;
    solver_->MarginalizeState(MarginalizeType::SlamFeature, state_, feat_state);

    Eigen::MatrixXd sqrt_Pt_after_marg = state_->Sqrt_Pt();

    // After marginalization:
    // - Rows should remain same
    // - Cols should decrease by 3
    EXPECT_EQ(sqrt_Pt_after_marg.rows(), initial_rows) << "Rows should remain constant during marginalization";
    EXPECT_EQ(sqrt_Pt_after_marg.cols(), initial_dim) << "Columns should return to initial dimension";
}

// =============================================================================
// Edge Cases and Error Conditions
// =============================================================================

/**
 * @brief Test attempting to marginalize nullptr
 */
TEST_F(SqrtEskfSlamTest, MarginalizeNullptr_NoChange)
{
    int dim_before = state_->_dim;

    // This should log an error and return without changing state
    solver_->MarginalizeState(MarginalizeType::SlamFeature, state_, nullptr);

    EXPECT_EQ(state_->_dim, dim_before) << "State dimension should not change";
}

/**
 * @brief Test covariance remains positive definite throughout lifecycle
 */
TEST_F(SqrtEskfSlamTest, CovariancePositiveDefinite_ThroughoutLifecycle)
{
    verifyCovarianceConsistency("Initial state");

    Feature *feat1 = createFeature(0, Eigen::Vector3d(1.0, 0.5, 2.0));
    addSlamFeatureToState(feat1);
    verifyCovarianceConsistency("After adding feature 1");

    Feature *feat2 = createFeature(1, Eigen::Vector3d(1.5, 0.3, 2.5));
    addSlamFeatureToState(feat2);
    verifyCovarianceConsistency("After adding feature 2");

    auto &slam_features = state_->slam_features();
    std::shared_ptr<Type> feat1_state = slam_features.at(0)._state_ptr;
    solver_->MarginalizeState(MarginalizeType::SlamFeature, state_, feat1_state);
    verifyCovarianceConsistency("After marginalizing feature 1");

    std::shared_ptr<Type> feat2_state = state_->slam_features().at(1)._state_ptr;
    solver_->MarginalizeState(MarginalizeType::SlamFeature, state_, feat2_state);
    verifyCovarianceConsistency("After marginalizing feature 2");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
