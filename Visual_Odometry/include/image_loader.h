#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using ImageList = std::vector<std::pair<cv::Mat, cv::Mat>>;

class ImageLoader {
public:
    ImageLoader(const std::string &dataDir, const std::string &sequence);

    std::pair<cv::Mat, cv::Mat> get_image_pair(int index);

    void show_video() const;

private:
    void load_images(const std::string &filePath, const std::string &sequence);

    ImageList images;
};
