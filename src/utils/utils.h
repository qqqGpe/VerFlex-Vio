#ifndef __UTILS__
#define __UTILS__

#include "camModel.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "sensor_data.h"

#define RAD2DEG 180 / M_PI
#define DEG2RAD M_PI / 180

class Utils
{
   public:
    static bool transfer_image(const sensor_msgs::ImageConstPtr& msg, cv::Mat& output)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception& e)
        {
            std::cerr << e.what() << std::endl;
            return false;
        }
        cv::Mat image = cv_ptr->image.clone();
        output = image;
        return true;
    }

    static void visualize_feature_tracking_results(const cv::Mat& image, const std::pair<double, std::vector<CameraObs>>& frame_output)
    {
        double fr = 11.333;
        double fg = 22.333;
        double fb = 33.333;

        cv::Mat image_to_show;
        cv::cvtColor(image, image_to_show, cv::COLOR_GRAY2BGR);
        for (auto& feat : frame_output.second)
        {
            if (feat.valid == false)
            {
                continue;
            }
            cv::Point2f point(feat.uv.at(LEFT_CAM).x(), feat.uv.at(LEFT_CAM).y());
            std::ostringstream os;
            os << std::fixed << feat.feat_id;
            std::string depth_text = os.str();
            cv::putText(image_to_show, depth_text, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            cv::Scalar color = cv::Scalar(int(fb * feat.feat_id) % 255, int(fg * feat.feat_id) % 255, int(fr * feat.feat_id) % 255);
            cv::circle(image_to_show, point, 3, color, cv::FILLED);
        }

        const uint32_t image_width = image_to_show.cols;
        const uint32_t image_height = image_to_show.rows;
        const double grid_w_step = image_width / 15.0;  // 每个格子宽度
        const double grid_h_step = image_height / 15.0; // 每个格
        for (int i = 0; i <= 15; ++i)
        {
            int x = i * grid_w_step;
            cv::line(image_to_show, cv::Point(x, 0), cv::Point(x, image_height), cv::Scalar(128, 128, 128), 1);
        }

        for (int j = 0; j <= 15; ++j)
        {
            int y = j * grid_h_step;
            cv::line(image_to_show, cv::Point(0, y), cv::Point(image_width, y), cv::Scalar(128, 128, 128), 1);
        }

        cv::imshow("feat_to_track", image_to_show);
        // cv::imwrite("/home/gao/ws/catkin_vio_ws/src/vio/figure/feature_to_track.png", image_to_show);
        cv::waitKey(0);
    }

    static void show_eigen_matrix(const Eigen::MatrixXd matrix, const std::string win_name)
    {
        constexpr int kBlockSize = 20;
        int rows = matrix.rows();
        int cols = matrix.cols();

        int image_width = cols * kBlockSize;
        int image_height = rows * kBlockSize;

        cv::Mat image(image_height, image_width, CV_8UC1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int32_t mag = 20 * log10(abs(matrix(i, j)) / 1e-8);
                mag = (mag > 255) ? 255 : mag;
                mag = (mag < 0) ? 0 : mag;
                mag = 255 - mag;
                for (int w = 0; w < kBlockSize; w++)
                {
                    for (int h = 0; h < kBlockSize; h++)
                    {
                        image.at<uchar>(i * kBlockSize + w, j * kBlockSize + h) = mag;
                    }
                }
            }
        }

        static int show_eigen_wait_sec = 1;
        if (cv::waitKey(show_eigen_wait_sec) == 'w')
        {
            if (show_eigen_wait_sec == 1)
            {
                show_eigen_wait_sec = 0;
            }
            else
            {
                show_eigen_wait_sec = 1;
            }
        }
        cv::imshow(win_name, image);
        cv::waitKey(show_eigen_wait_sec);
    }

    static void ShowGridImages(const std::vector<cv::Mat>& images, const std::string win_name = "default")
    {
        if (images.size() != 6)
        {
            return;
        }
        constexpr uint8_t scale = 1;
        int height = 2 * images[0].rows / scale;
        int width = 3 * images[0].cols / scale;
        cv::Mat gridImage = cv::Mat::zeros(height, width, CV_8UC3);

        int rows = 2;
        int cols = 3;

        int cellWidth = width / cols;
        int cellHeight = height / rows;

        for (int i = 0; i < images.size(); ++i)
        {
            cv::Mat img = images[i];
            cv::resize(img, img, cv::Size(cellWidth, cellHeight));

            int row = i / cols;
            int col = i % cols;
            cv::Rect roi(col * cellWidth, row * cellHeight, cellWidth, cellHeight);

            img.copyTo(gridImage(roi));
        }

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                cv::Point topLeft(j * cellWidth, i * cellHeight);
                cv::Point bottomRight((j + 1) * cellWidth, (i + 1) * cellHeight);
                cv::rectangle(gridImage, topLeft, bottomRight, cv::Scalar(255, 255, 255), 1);
            }
        }

        static int wait_sec = 1;
        cv::imshow(win_name.c_str(), gridImage);
        if (cv::waitKey(wait_sec) == 'w')
        {
            if (wait_sec == 1)
            {
                wait_sec = 0;
            }
            else
            {
                wait_sec = 1;
            }
        }
    }

    // Function to draw matches between two images in one image
    static void visualizeStereoMatches(const cv::Mat& img1,
                                       const cv::Mat& img2,
                                       const std::vector<cv::Point2f>& points1,
                                       const std::vector<cv::Point2f>& points2)
    {
        // Create an output image to display matches
        cv::Mat outImg;
        cv::hconcat(img1, img2, outImg);
        cv::cvtColor(outImg, outImg, cv::COLOR_GRAY2BGR);

        // Draw lines between matching points
        for (size_t i = 0; i < points1.size(); ++i)
        {
            cv::Point2f pt1 = points1[i];
            cv::Point2f pt2 = points2[i];
            pt2.x += img1.cols;  // Offset the x-coordinate for the second image

            cv::line(outImg, pt1, pt2, cv::Scalar(0, 255, 0), 2);
            cv::circle(outImg, pt1, 5, cv::Scalar(0, 0, 255), -1);
            cv::circle(outImg, pt2, 5, cv::Scalar(0, 0, 255), -1);
        }

        // Display the image with matches
        cv::imshow("Stereo Matches", outImg);
        cv::waitKey(0);
    }

    // Function to draw matches between two images in one image
    static void VisualizeStereoMatchesAndDepth(const cv::Mat& img1, const std::vector<std::pair<CameraObs, Eigen::Vector3d>>& point_obs)
    {
        // Create an output image to display matches
        cv::Mat outImg;
        cv::cvtColor(img1, outImg, cv::COLOR_GRAY2BGR);

        for (const auto& obs : point_obs)
        {
            cv::Point2f pt_l(obs.first.uv.at(LEFT_CAM).x(), obs.first.uv.at(LEFT_CAM).y());
            cv::Point2f pt_r(obs.first.uv.at(RIGHT_CAM).x(), obs.first.uv.at(RIGHT_CAM).y());
            std::ostringstream depth;
            depth << std::fixed << std::setprecision(2) << obs.second.z();
            std::string depth_text = depth.str();
            cv::circle(outImg, pt_l, 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(outImg, pt_r, 3, cv::Scalar(255, 0, 0), -1);
            cv::line(outImg, pt_l, pt_r, cv::Scalar(0, 255, 0), 2);
            cv::putText(outImg, depth_text, pt_l, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }

        // Display the image with matches
        cv::imshow("Stereo Matches", outImg);
        cv::waitKey(0);
    }

    static void DisplayFeaturePoints(const cv::Mat& img,
                                     const std::vector<cv::Point2d>& points,
                                     const std::vector<double>& depths,
                                     const std::vector<int>& ids)
    {
        // Create a copy of the input image to draw points on
        double fr = 13.3;
        double fg = 53.3;
        double fb = 73.3;
        cv::Mat img_with_points;
        cv::cvtColor(img, img_with_points, cv::COLOR_GRAY2BGR);

        // Draw each point on the image
        for (int i = 0; i < points.size(); ++i)
        {
            cv::Point2f point = points[i];
            std::ostringstream os;
            os << std::fixed << std::setprecision(2) << depths[i];
            std::string depth_text = os.str();
            cv::Scalar color = cv::Scalar(int(fb * ids[i]) % 255, int(fg * ids[i]) % 255, int(fr * ids[i]) % 255);
            cv::circle(img_with_points, point, 5, color, -1);  // Red color for points
            cv::putText(img_with_points, depth_text, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        // Display the image with points
        cv::imshow("Feature Points", img_with_points);
        cv::waitKey(1);
    }
};

#endif