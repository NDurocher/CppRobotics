#pragma once

#include <vector>
#include <string>

#include "opencv2/core.hpp"

class PoseLoader {
    public:
    PoseLoader(std::string filePath, std::string &sequence);

    std::vector<cv::Mat> ground_truth_poses();

private:
    std::vector<cv::Mat> m_poses; // Contains ground truth poses for camera
    size_t m_num_frames{};
};

