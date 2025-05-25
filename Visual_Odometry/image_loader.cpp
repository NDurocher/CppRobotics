#include "image_loader.h"

#include <fstream>
#include <filesystem>

namespace fs = std::__fs::filesystem;

std::string format_image_file_name(const int index) {
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(6) << index << ".png";
    return oss.str();
}

ImageLoader::ImageLoader(const std::string &dataDir, const std::string &sequence) : m_data_dir{dataDir},
    m_sequence{sequence} {
}

std::pair<cv::Mat, cv::Mat> ImageLoader::get_image_pair(const int index) const {
    const auto image_0_path = m_data_dir + m_sequence + "/image_0/" + format_image_file_name(index);
    const auto image_1_path = m_data_dir + m_sequence + "/image_1/" + format_image_file_name(index);
    return {
        cv::imread(image_0_path, cv::IMREAD_GRAYSCALE),
        cv::imread(image_1_path, cv::IMREAD_GRAYSCALE)
    };
}

void ImageLoader::show_video(const bool use_image_0) const {
    const std::string image_dir = use_image_0 ? "/image_0" : "/image_1";
    const fs::path fs_path{m_data_dir + m_sequence + image_dir};
    for (const auto &img: fs::directory_iterator(fs_path)) {
        cv::imshow("KITTI_Data", cv::imread(img.path(), cv::IMREAD_GRAYSCALE));

        const int key = cv::waitKey(40) & 0xFF;

        // Exit on 'q' key
        if (key == 'q') {
            break;
        }
    }
    cv::destroyAllWindows();
}
