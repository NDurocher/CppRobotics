#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using keypoint_descriptor = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

class OrbFeatureDetector {
public:
    void get_matches(keypoint_descriptor &kp_descript1, keypoint_descriptor &kp_descript2,
                     std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

    keypoint_descriptor compute_features(cv::Mat &image);

private:
    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(300);
    cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
};
