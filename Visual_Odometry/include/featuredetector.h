#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


class OrbFeatureDetector {
 
public: 
 void get_matches(std::pair<cv::Mat, cv::Mat> &images, std::vector<cv::Point2f> &q1, std::vector<cv::Point2f> &q2);	

private:
	cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(300);
    cv::Ptr<cv::DescriptorMatcher> flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
};