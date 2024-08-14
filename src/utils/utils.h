#ifndef __UTILS__
#define __UTILS__

#include "sensor_data.h"
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

class Utils {
public:
    static bool transfer_image(const sensor_msgs::ImageConstPtr& msg, cv::Mat &output)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception& e) {
            LOG(ERROR) << "cv_bridge exception: " << e.what();
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
        cv::waitKey(1);
    }
};


#endif