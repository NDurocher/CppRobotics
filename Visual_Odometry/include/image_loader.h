#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ImageLoader {
public:
    ImageLoader(const std::string &dataDir, const std::string &sequence);

    std::pair<cv::Mat, cv::Mat> get_image_pair(int index) const;

    void show_video(bool use_image_0 = true) const;

private:
    std::string m_data_dir;
    std::string m_sequence;
};
