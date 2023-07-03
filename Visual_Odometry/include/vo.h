#pragma once

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

class VO
{
public:
	VO(std::string& dataDir, std::string& sequence);
	 	
	std::vector<cv::Mat> _calibs; // Contains calin for both Left and Right Camera
	std::vector<cv::Mat> _poses; // Contains ground truth poses for camera
	std::vector<std::string> _strPoses;
	std::vector<cv::Mat> _images;
	// cv::Ptr<cv::xfeatures2d::SURF> _orb = cv::xfeatures2d::SURF::create(400, 4, 2, true, false);
	cv::Ptr<cv::FeatureDetector> _orb = cv::ORB::create(3000);
	cv::Ptr<cv::DescriptorMatcher> _flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

	cv::Mat _K; // Camera Intrinsic Parameters (3x3)
	cv::Mat _P; // Camera Projection Matrix (3x4)

	void load_images(std::string filePath, std::string& sequence);

	void load_calib(std::string filePath, std::string& sequence);

	void load_poses(std::string filePath, std::string& sequence);

	void get_matches(unsigned& i, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2);

	void get_poses(std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2, cv::Mat& Trans_Mat);

	void decomp_essential_mat(cv::Mat& Emat, cv::Mat& R, cv::Mat& t, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2);

	void get_relativeScale(Eigen::MatrixXf& HQ1, Eigen::MatrixXf& HQ2, std::vector<float>& relative_scale);

	void formTransformMat(cv::Mat& R, cv::Mat& t, cv::Mat& T);

	static void stringLine2Matrix(cv::Mat& tempMat, int& rows, int& cols, std::string& line);

	void showvideo();
};