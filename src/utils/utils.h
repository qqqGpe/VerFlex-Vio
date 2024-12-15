#ifndef __UTILS__
#define __UTILS__

#include "sensor_data.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


#define RAD2DEG 180 / M_PI
#define DEG2RAD M_PI / 180

class Utils {
public:
    static bool transfer_image(const sensor_msgs::ImageConstPtr& msg, cv::Mat &output)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception& e) {
            std::cerr << e.what() << std::endl;
            return false;
        }
        cv::Mat image = cv_ptr->image.clone();
        output = image;
        return true;
    }

    static void visualize_feature_tracking_results(const cv::Mat &image, const std::pair<double, std::vector<cam_obs_t>> &frame_output) {
        double fr = 1.333;
        double fg = 2.333;
        double fb = 3.333;

        cv::Mat image_to_show;
        cv::cvtColor(image, image_to_show, cv::COLOR_GRAY2BGR);
        for(auto &feat : frame_output.second) {
            if(feat.valid == false) {
                continue;
            }
            cv::Point2f point(feat.u, feat.v);
            cv::Scalar color = cv::Scalar(int(fb * feat.feat_id) % 255, int(fg * feat.feat_id) % 255, int(fr * feat.feat_id) % 255);
            cv::circle(image_to_show, point, 3, color, cv::FILLED);
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

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uint32_t mag = 20 * log10(abs(matrix(i, j)) / 1e-8);
                mag = (mag > 255) ? 254 : mag;
                mag = (mag < 0) ? 0 : mag;
                for (int w = 0; w < kBlockSize; w++){
                    for (int h = 0; h < kBlockSize; h++) {
                        image.at<uchar>(i * kBlockSize + w, j * kBlockSize + h) = mag;
                    }
                }
            }
        }
        cv::imshow(win_name, image);
        cv::waitKey(0);
    }

    static void draw_grid_with_images(const std::vector<cv::Mat>& images) {
        constexpr uint8_t scale = 2;
        int height = 2 * images[0].rows / scale;
        int width = 3 * images[0].cols / scale;
        cv::Mat gridImage = cv::Mat::zeros(height, width, CV_8UC1);

        int rows = 2;
        int cols = 3;

        int cellWidth = width / cols;
        int cellHeight = height / rows;

        for (int i = 0; i < images.size(); ++i) {
            cv::Mat img = images[i];

            cv::resize(img, img, cv::Size(cellWidth, cellHeight));

            int row = i / cols;
            int col = i % cols;
            cv::Rect roi(col * cellWidth, row * cellHeight, cellWidth, cellHeight);

            img.copyTo(gridImage(roi));
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cv::Point topLeft(j * cellWidth, i * cellHeight);
                cv::Point bottomRight((j + 1) * cellWidth, (i + 1) * cellHeight);
                cv::rectangle(gridImage, topLeft, bottomRight, cv::Scalar(255, 255, 255), 1);
            }
        }

        cv::imshow("keyframes", gridImage);
        cv::waitKey(1);
    }
};


#endif