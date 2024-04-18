#include "featuredetector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

void OrbFeatureDetector::get_matches(std::pair<cv::Mat, cv::Mat> &images, std::vector<cv::Point2f> &q1, std::vector<cv::Point2f> &q2) {
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(images.first, cv::noArray(), keypoints1, descriptors1);
    std::cout << "here?" << std::endl;
    orb->detectAndCompute(images.second, cv::noArray(), keypoints2, descriptors2);
    std::vector<std::vector<cv::DMatch> > knn_matches;

    if (descriptors1.type() != CV_32F) {
        descriptors1.convertTo(descriptors1, CV_32F);
        descriptors2.convertTo(descriptors2, CV_32F);
    }

    flann->knnMatch(descriptors1, descriptors2, knn_matches, 4);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.8f;
    std::vector<cv::DMatch> good_matches;
    for (auto &knn_match: knn_matches) {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
            good_matches.push_back(knn_match[0]);
        }
    }
    for (auto match: good_matches) {
        q1.push_back(keypoints1[match.queryIdx].pt);
        q2.push_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat outImg;
    cv::drawMatches(images.first, keypoints1, images.second, keypoints2, good_matches, outImg);
    cv::imshow("matches", outImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

}