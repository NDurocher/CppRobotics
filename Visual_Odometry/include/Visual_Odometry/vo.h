// #ifndef VO_H_
// #define VO_H_

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

class VO
{
public:
	// VO();
	VO(std::string& dataDir);
	// ~VO();
	 	
	std::vector<Eigen::MatrixXd> _calibs; // Contains calin for both Left and Right Camera
	std::vector<Eigen::MatrixXd> _poses; // Contains ground truth poses for camera
	std::vector<std::string> _strPoses;
	std::vector<cv::Mat> _images;
	cv::Ptr<cv::FeatureDetector> _orb = cv::ORB::create(3000);
	cv::Ptr<cv::DescriptorMatcher> _flann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

	Eigen::MatrixXd _K; // Camera Intrinsic Parameters (3x3)
	Eigen::MatrixXd _P; // Camera Projection Matrix (3x4)

	void load_images(std::string filePath);

	void load_calib(std::string filePath);

	void load_poses(std::string filePath);

	void get_matches(unsigned& i, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2);

	static void stringLine2Matrix(Eigen::MatrixXd& tempMat, int& rows, int& cols, std::string& line);
};