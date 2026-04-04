/*
 * @Author: pengen.gao gaope.hb@gmail.com
 * @Date: 2025-02-03 20:11:13
 * Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
 */
#ifndef __UTILS__
#define __UTILS__

#include <Eigen/Core>
#include <glog/logging.h>
#include <fmt/format.h>
#include <opencv2/opencv.hpp>

#define RAD2DEG 180 / M_PI
#define DEG2RAD M_PI / 180

// Terminal color macros
#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"
#define BOLD "\033[1m"

#define LOG_INFO(...)  LOG(INFO)    << fmt::format("[@{}:{}] ", __FUNCTION__, __LINE__) << fmt::format(__VA_ARGS__)
#define LOG_WARN(...)  LOG(WARNING) << YELLOW << fmt::format("[@{}:{}] ", __FUNCTION__, __LINE__) << fmt::format(__VA_ARGS__) << RESET
#define LOG_ERROR(...) LOG(ERROR)   << RED    << fmt::format("[@{}:{}] ", __FUNCTION__, __LINE__) << fmt::format(__VA_ARGS__) << RESET
#define LOG_FATAL(...) LOG(FATAL)   << RED    << fmt::format("[@{}:{}] ", __FUNCTION__, __LINE__) << fmt::format(__VA_ARGS__) << RESET

namespace utils
{
void visualize_feature_tracking_results(const cv::Mat &image,
                                        const std::vector<std::pair<int32_t, cv::Point2f>> &points,
                                        const int32_t wait_key_time_ms, cv::Mat *out_image);

void show_eigen_matrix(const Eigen::MatrixXd matrix, const std::string win_name);

void ShowGridImages(const std::vector<cv::Mat> &images, const std::string win_name = "default");

// Function to draw matches between two images in one image
void visualizeStereoMatches(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &points1,
                            const std::vector<cv::Point2f> &points2);

void DisplayFeaturePoints(const cv::Mat &img, const std::vector<cv::Point2d> &points, const std::vector<double> &depths,
                          const std::vector<int> &ids);
}; // namespace utils

#endif