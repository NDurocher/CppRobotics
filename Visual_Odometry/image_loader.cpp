#include "image_loader.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::__fs::filesystem;

static std::vector<std::string> get_sorted_paths(const std::string &path) {
    const fs::path fs_path{path};
    std::vector<std::string> sorted_paths;
    for (auto &f: fs::directory_iterator(fs_path)) {
        sorted_paths.push_back(f.path());
    }
    std::sort(sorted_paths.begin(), sorted_paths.end());
    return sorted_paths;
}

ImageLoader::ImageLoader(const std::string &dataDir, const std::string &sequence) {
    load_images(dataDir, sequence);
    std::cout << "Images Loaded" << std::endl;
}

void ImageLoader::load_images(const std::string &filePath, const std::string &sequence) {
    const auto image_0_dir = filePath + sequence + "/image_0/";
    const auto image_1_dir = filePath + sequence + "/image_1/";

    const auto image_0_paths = get_sorted_paths(image_0_dir);
    const auto image_1_paths = get_sorted_paths(image_1_dir);
    for (size_t i = 0; i < std::min(image_0_paths.size(), image_1_paths.size()); ++i) {
        auto& image_0 = image_0_paths[i];
        auto& image_1 = image_1_paths[i];
        images.push_back(std::make_pair(cv::imread(image_0, cv::IMREAD_GRAYSCALE),
                                        cv::imread(image_1, cv::IMREAD_GRAYSCALE)));
    }
}


std::pair<cv::Mat, cv::Mat> ImageLoader::get_image_pair(const int index) {
    return {images[index].first, images[index].second};
}

void ImageLoader::show_video() const {
    for (const auto &img: images) {
        cv::imshow("KITTI_Data", img.first);
        cv::waitKey(0); // Wait for a keystroke in the window
    }
    cv::destroyAllWindows();
}
