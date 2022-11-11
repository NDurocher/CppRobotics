#include "vo.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdio.h>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>


namespace fs = std::filesystem;

// VO::VO(){}

VO::VO(std::string& dataDir){
	_K.resize(3,3);
	_P.resize(3,4);

	load_images(dataDir);
	load_poses(dataDir);
	load_calib(dataDir);
}

void VO::load_images(std::string filePath){

	// Import left camera images (can change to image_r and check diff in output)
	filePath.append("image_l");

	for (auto & image : fs::directory_iterator(filePath)) {
		_images.push_back(cv::imread(image.path(), cv::IMREAD_GRAYSCALE));
	}
}

void VO::load_calib(std::string filePath){
	filePath.append("calib.txt");

	std::ifstream calibFile(filePath);

	int rows = 3;
	int cols = 4;
	Eigen::MatrixXd tempMat(rows,cols);

	std::string line;
	while (getline(calibFile, line)){
		
		stringLine2Matrix(tempMat, rows, cols, line);	
		_calibs.push_back(tempMat);
	}

	calibFile.close();

	_K = _calibs[0].block(0,0,3,3);
	_P = _calibs[0].block(0,0,3,4);
}

void VO::load_poses(std::string filePath){
	
	filePath.append("poses.txt");

	std::ifstream poseFile(filePath);

	int rows = 3;
	int cols = 4;
	Eigen::MatrixXd tempMat(rows,cols);

	std::string line;
	while (getline(poseFile, line)){
		
		stringLine2Matrix(tempMat, rows, cols, line);	
		_poses.push_back(tempMat);
	}

	poseFile.close();
}

void VO::get_matches(unsigned& i, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2){
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
    _orb->detectAndCompute(_images[i-1], cv::noArray(), keypoints1, descriptors1 );
    _orb->detectAndCompute(_images[i], cv::noArray(), keypoints2, descriptors2 );
    std::vector< std::vector<cv::DMatch> > knn_matches;

    _flann->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

        //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    for (auto match : good_matches){
    	q1.push_back(keypoints1[match.queryIdx].pt);
    	q2.push_back(keypoints2[match.trainIdx].pt);
    }

}

void VO::stringLine2Matrix(Eigen::MatrixXd& tempMat, int& rows, int& cols, std::string& line){
	int r = 0;
	int c = 0;
	std::string strNum;
	const std::string delim= " ";

	for (std::string::const_iterator i =line.begin(); i != line.end(); i++){
		
		// If i is not a delim, then append it to strnum
		if (delim.find(*i) == std::string::npos){ 
			strNum += *i;
			if (i+1 != line.end()){
				continue;
			}
		}
		if (strNum.empty()){
			continue;
		}

		double number;
		std::istringstream(strNum) >> number;
		tempMat(r,c) = number;
		if ((c+1) == cols){
			if((r+1) == rows){
				strNum.clear();
				break;
			}
			c = 0;
			r++;
			strNum.clear();
			continue;
		}
		c++;
		strNum.clear();
	}
}







