#include "include/pose_loader.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <filesystem>


PoseLoader::PoseLoader(std::string filePath, std::string &sequence) {
    filePath.append("poses/" + sequence + ".txt");

    std::ifstream poseFile(filePath);

    int rows = 3;
    int cols = 4;
    cv::Mat tempMat = cv::Mat::zeros(rows, cols, CV_32F);

    std::string line;
    while (getline(poseFile, line)) {
        vo_utils::stringLine2Matrix(tempMat, rows, cols, line);
        m_poses.push_back(tempMat.clone());
    }
    poseFile.close();
    std::cout << "GT Poses Loaded" << std::endl;
    m_num_frames = m_poses.size();
}


std::vector<cv::Mat> PoseLoader::ground_truth_poses() {
    return m_poses;
}
