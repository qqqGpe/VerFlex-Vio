#include "cameraModel.h"

void CameraModel::CalculateUndistortRectifyMap(const Eigen::Matrix3d& K,
                                               const Eigen::VectorXd& D,
                                               Eigen::Matrix3d& K_undistort,
                                               cv::Mat& map1, cv::Mat& map2) {
  cv::Mat K_cv, D_cv;
  cv::eigen2cv(K, K_cv);
  cv::eigen2cv(D, D_cv);

  cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat new_K = cv::getOptimalNewCameraMatrix(K_cv, D_cv, image_size_, 0, image_size_, 0);
  cv::initUndistortRectifyMap(K_cv, D_cv, R, new_K, image_size_, CV_32FC1, map1, map2);
  cv::cv2eigen(new_K, K_undistort);
}

void CameraModel::set_camera_intrin_matrix(const std::vector<double>& intrinsic_coeff, Eigen::Matrix3d& K) {
  K.setIdentity();
  K(0, 0) = intrinsic_coeff[0];
  K(1, 1) = intrinsic_coeff[1];
  K(0, 2) = intrinsic_coeff[2];
  K(1, 2) = intrinsic_coeff[3];
}

void CameraModel::set_camera_distort_coeff(const std::vector<double>& distort_coeff, Eigen::VectorXd& param) {
  if (distort_coeff.empty()) {
    return;
  }
  param = Eigen::VectorXd::Zero(distort_coeff.size());
  for (size_t i = 0; i < distort_coeff.size(); i++) {
    param(i) = distort_coeff[i];
  }
}

Eigen::Vector2d CameraModel::project_left(Eigen::Vector3d p3d_norm) {
  Eigen::Vector2d feature_norm(
      Kl_undistort_(0, 0) * p3d_norm.x() / p3d_norm.z() + Kl_undistort_(0, 2),
      Kl_undistort_(1, 1) * p3d_norm.y() / p3d_norm.z() + Kl_undistort_(1, 2));
  return feature_norm;
}

Eigen::Vector2d CameraModel::project_right(Eigen::Vector3d p3d_norm) {
  Eigen::Vector2d feature_norm(
      Kr_undistort_(0, 0) * p3d_norm.x() / p3d_norm.z() + Kr_undistort_(0, 2),
      Kr_undistort_(1, 1) * p3d_norm.y() / p3d_norm.z() + Kr_undistort_(1, 2));
  return feature_norm;
}

Eigen::Vector3d CameraModel::back_project(Eigen::Vector2d uv_2d) {
  Eigen::Vector3d feat_norm(
      (uv_2d.x() - Kr_origin_(0, 2)) / Kr_origin_(0, 0),
      (uv_2d.y() - Kr_origin_(1, 2)) / Kr_origin_(1, 1), 1.0);
  return feat_norm;
}

void CameraModel::back_project_stereo(CameraObs& obs) {

  Eigen::Vector3d feat_norm_left(
      (obs.u - Kl_undistort_(0, 2)) / Kl_undistort_(0, 0),
      (obs.v - Kl_undistort_(1, 2)) / Kl_undistort_(1, 1), 1.0);

  Eigen::Vector3d feat_norm_right(
      (obs.ur - Kr_undistort_(0, 2)) / Kr_undistort_(0, 0),
      (obs.vr - Kr_undistort_(1, 2)) / Kr_undistort_(1, 1), 1.0);

  obs.u_norm = feat_norm_left.x();
  obs.v_norm = feat_norm_left.y();
  obs.ur_norm = feat_norm_right.x();
  obs.vr_norm = feat_norm_right.y();
}

void CameraModel::RectifyStereoImages(const cv::Mat& img_left,
                                      const cv::Mat& img_right,
                                      cv::Mat& rectified_left,
                                      cv::Mat& rectified_right) {
  // Apply rectification
  cv::remap(img_left, rectified_left, rectify_map1_left, rectify_map2_left, cv::INTER_LINEAR);
  cv::remap(img_right, rectified_right, rectify_map1_right, rectify_map2_right, cv::INTER_LINEAR);

  // // Visualize the origin and rectified images
  // {
  //   cv::Mat top, bottom;
  //   cv::hconcat(img_left, img_right, top);
  //   cv::hconcat(rectified_left, rectified_right, bottom);

  //   // Concatenate the top and bottom images vertically
  //   cv::Mat stitched_img;
  //   cv::vconcat(top, bottom, stitched_img);
  //   cv::imshow("hconcat", stitched_img);
  //   cv::waitKey(0);
  // }
}
