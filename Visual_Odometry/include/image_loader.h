#pragma one

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


using ImageList = std::pair<std::vector<cv::Mat>, std::vector<cv::Mat> >;

class ImageLoader {
public:
    ImageLoader(std::string &dataDir, std::string &sequence);

    std::pair<cv::Mat, cv::Mat> get_image_pair(int index);

    void showvideo();

private:
    void load_images(std::string &filePath, std::string &sequence);

    ImageList images;
};
