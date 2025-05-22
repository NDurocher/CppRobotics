#include "image_loader.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::__fs::filesystem;

ImageLoader::ImageLoader(std::string &dataDir, std::string &sequence) {
    load_images(dataDir, sequence);
    std::cout << "Images Loaded" << std::endl;
}

void ImageLoader::load_images(std::string &filePath, std::string &sequence) {
    auto image_0 = filePath + sequence + "/image_0/";
    auto image_1 = filePath + sequence + "/image_1/";

    const fs::path fs_path_0{image_0};
    std::vector<std::string> sorted_paths;
    for (auto &f: fs::directory_iterator(fs_path_0)) {
        sorted_paths.push_back(f.path());
    }
    std::sort(sorted_paths.begin(), sorted_paths.end());
    for (auto &image: sorted_paths) {
        images.first.push_back(cv::imread(image, cv::IMREAD_GRAYSCALE));
    }

    const fs::path fs_path_1{image_1};
    sorted_paths.clear();
    for (auto &f: fs::directory_iterator(fs_path_1)) {
        sorted_paths.push_back(f.path());
    }
    std::sort(sorted_paths.begin(), sorted_paths.end());
    for (auto &image: sorted_paths) {
        images.second.push_back(cv::imread(image, cv::IMREAD_GRAYSCALE));
    }
}

std::pair<cv::Mat, cv::Mat> ImageLoader::get_image_pair(int index) {
    return {images.first[index], images.second[index]};
}

void ImageLoader::showvideo() {
    for (const auto &img: images.first) {
        cv::imshow("KITTI_Data", img);
        cv::waitKey(0); // Wait for a keystroke in the window
    }
    cv::destroyAllWindows();
}
