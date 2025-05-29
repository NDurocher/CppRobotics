#include "feature_detector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

std::vector<cv::DMatch> OrbFeatureDetector::get_matches(const keypoint_descriptor &kp_descript1,
                                                        const keypoint_descriptor &kp_descript2,
                                                        std::vector<cv::Point2f> &points1,
                                                        std::vector<cv::Point2f> &points2) const {
    std::vector<std::vector<cv::DMatch> > knn_matches;

    flann->knnMatch(kp_descript1.second, kp_descript2.second, knn_matches, 4);

    //-- Filter matches using the Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    for (auto &knn_match: knn_matches) {
        if (knn_match[0].distance < m_ratio_thresh * knn_match[1].distance) {
            good_matches.push_back(knn_match[0]);
        }
    }
    for (const auto match: good_matches) {
        points1.push_back(kp_descript1.first[match.queryIdx].pt);
        points2.push_back(kp_descript2.first[match.trainIdx].pt);
    }
    return good_matches;
}

keypoint_descriptor OrbFeatureDetector::compute_features(const cv::Mat &image) const {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);


    if (descriptors.type() != CV_32F) {
        descriptors.convertTo(descriptors, CV_32F);
    }
    return keypoint_descriptor{keypoints, descriptors};
}
