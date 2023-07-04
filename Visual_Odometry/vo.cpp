#include "vo.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>


namespace fs = std::__fs::filesystem;

VO::VO(std::string dataDir, std::string sequence, std::string& camera){

	load_images(dataDir, sequence, camera);
	std::cout << "Images Loaded" << std::endl;
	load_poses(dataDir, sequence);
	std::cout << "GT Poses Loaded" << std::endl;
	load_calib(dataDir, sequence);
	std::cout << "Camera Calibs Loaded" << std::endl;
}

void VO::load_images(std::string filePath, std::string& sequence, std::string& camera){
	// Import left camera images (can change to image_r and check diff in output)
	filePath.append("/" + sequence + camera);
	// std::sort(files_in_directory.begin(), files_in_directory.end())
	const fs::path fs_path{filePath};
	std::vector<std::string> sorted_paths;
	for (auto & f : fs::directory_iterator(fs_path)){
		sorted_paths.push_back(f.path());
	}
	std::sort(sorted_paths.begin(), sorted_paths.end());
	for (auto & image : sorted_paths) {
		_images.push_back(cv::imread(image, cv::IMREAD_GRAYSCALE));
	}
}

void VO::load_calib(std::string filePath, std::string& sequence){
	filePath.append("/" + sequence + "/calib.txt");

	std::ifstream calibFile(filePath);

	int rows = 3;
	int cols = 4;
	cv::Mat tempMat = cv::Mat::zeros(rows, cols, CV_32F);
	std::string line;
	while (getline(calibFile, line)){
		stringLine2Matrix(tempMat, rows, cols, line);	
		_calibs.push_back(tempMat.clone());
	}

	calibFile.close();

	_K = _calibs[1].rowRange(0,3).colRange(0,3);
	_P = _calibs[1].rowRange(0,3).colRange(0,4);;
}

void VO::load_poses(std::string filePath, std::string& sequence){
	filePath.append("/poses/" + sequence + ".txt");

	std::ifstream poseFile(filePath);

	int rows = 3;
	int cols = 4;
	cv::Mat tempMat = cv::Mat::zeros(rows, cols, CV_32F);

	std::string line;
	while (getline(poseFile, line)){
		stringLine2Matrix(tempMat, rows, cols, line);
		_poses.push_back(tempMat.clone());
	}
	poseFile.close();
}

void VO::get_matches(unsigned& i, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2){
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
    _orb->detectAndCompute(_images[i-1], cv::noArray(), keypoints1, descriptors1 );
    _orb->detectAndCompute(_images[i], cv::noArray(), keypoints2, descriptors2 );
    std::vector< std::vector<cv::DMatch> > knn_matches;

    if(descriptors1.type()!=CV_32F) {
    	descriptors1.convertTo(descriptors1, CV_32F);
    	descriptors2.convertTo(descriptors2, CV_32F);
	}

    _flann->knnMatch( descriptors1, descriptors2, knn_matches, 2);

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

void VO::get_poses(std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2, cv::Mat& Trans_Mat){
	cv::Mat mask_ransac;
	cv::findHomography(q1, q2, cv::RANSAC, 3, mask_ransac, 1000, 0.995);
	
	cv::Mat Emat = cv::findEssentialMat(q1, q2, _K, cv::RANSAC, 0.999, 2.0, mask_ransac);
	// Decompose essitial matrix into R and t
	cv::Mat R,t;
	decomp_essential_mat(Emat, R, t, q1, q2);
	formTransformMat(R, t, Trans_Mat);
}

void VO::decomp_essential_mat(cv::Mat& Emat, cv::Mat& R, cv::Mat& t, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2){
	cv::Mat R1, R2;
	cv::decomposeEssentialMat(Emat, R1, R2, t);
	if(R1.type()!=CV_32F) {
    	R1.convertTo(R1, CV_32F);
    	R2.convertTo(R2, CV_32F);
    	t.convertTo(t, CV_32F);
	}
	std::vector<std::vector<cv::Mat>> T_list = {{R1,t},{R1, -t},{R2, t}, {R2, -t}};

	std::vector<int> pos_count = {0,0,0,0};
	std::vector<float> relative_scale;
	for (unsigned i=0; i < T_list.size(); i++){
		cv::Mat T = cv::Mat::zeros(4,4, CV_32F);
		formTransformMat(T_list[i][0], T_list[i][1], T);
		cv::Mat Ptemp = _P*T;
		cv::Mat HQ1;
		cv::triangulatePoints(_P, Ptemp, q1, q2, HQ1);
		cv::Mat HQ2 = T*HQ1;

		Eigen::MatrixXf EHQ1, EHQ2;
		cv::cv2eigen(HQ1, EHQ1);
		cv::cv2eigen(HQ2, EHQ2);
		for (unsigned c=0; c < EHQ1.cols(); c++ ){
			EHQ1.block(0,c,4,1) /= EHQ1(3,c);
			EHQ2.block(0,c,4,1) /= EHQ2(3,c);
			if (EHQ1(2,c) > 0){
				pos_count[i]++;
			}
			if (EHQ2(2,c) > 0){
				pos_count[i]++;
			}
		}
		get_relativeScale(EHQ1, EHQ2, relative_scale);
	}

	int idx_max = std::distance(pos_count.begin(), std::max_element(pos_count.begin(), pos_count.end()));
	R = T_list[idx_max][0];
	t = T_list[idx_max][1]*relative_scale[idx_max];
}	

void VO::get_relativeScale(Eigen::MatrixXf& HQ1, Eigen::MatrixXf& HQ2, std::vector<float>& relative_scale){
	Eigen::MatrixXf Temp1 = HQ1.transpose().block(0,0,HQ1.cols()-1,HQ1.rows()-1) - HQ1.transpose().block(1,0, HQ1.cols()-1, HQ1.rows()-1);
	Eigen::MatrixXf Temp2 = HQ2.transpose().block(0,0,HQ2.cols()-1,HQ2.rows()-1) - HQ2.transpose().block(1,0, HQ2.cols()-1, HQ2.rows()-1);
	Temp1.colwise().normalize();
	Temp2.colwise().normalize();
	
	relative_scale.push_back((Temp1.array()/Temp2.array()).mean());
}


void VO::formTransformMat(cv::Mat& R, cv::Mat& t, cv::Mat& T){
	cv::hconcat(R,t,T);
	T.push_back(cv::Mat::zeros(1,4, CV_32F));
	Eigen::MatrixXf tempT;
	cv::cv2eigen(T,tempT);
	tempT(3,3) = 1.0;
	cv::eigen2cv(tempT, T);
}

void VO::stringLine2Matrix(cv::Mat& tempMat, int& rows, int& cols, std::string& line){
	int r = 0;
	int c = 0;
	std::string strNum;
	const std::string delim= " ";

	// if the line contains the sensor prefix, erase it
	if (line.find("P") != std::string::npos){
		line.erase(0,4);
	}

	for (std::string::const_iterator i = line.begin(); i != line.end(); i++){
	
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

		float number;
		std::istringstream(strNum) >> number;
		tempMat.at<float>(r,c) = number;
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

void VO::showvideo(){
	for (auto img : _images){
		cv::imshow("KITTI_Data", img);
	    int k = cv::waitKey(0); // Wait for a keystroke in the window
    }
    cv::destroyAllWindows();
}





