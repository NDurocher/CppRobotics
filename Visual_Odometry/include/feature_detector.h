#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using keypoint_descriptor = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

class OrbFeatureDetector {
public:
    std::vector<cv::DMatch> get_matches(const keypoint_descriptor &kp_descript1,
                                        const keypoint_descriptor &kp_descript2,
                                        std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2) const;

    keypoint_descriptor compute_features(const cv::Mat &image) const;

private:
    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(300);
    cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    float m_ratio_thresh = 0.85;
};
