#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class TimeSeriesVisualizer {
public:
    TimeSeriesVisualizer(int w = 800, int h = 400, cv::Scalar line_col = cv::Scalar(0, 255, 0),
                         cv::Scalar bg_col = cv::Scalar(255, 255, 255), int m = 50);

    cv::Mat visualize(const std::vector<float> &data);

private:
    void drawAxes(cv::Mat &img, float min_val, float max_val, size_t data_size);

    int width, height;
    cv::Scalar line_color;
    cv::Scalar bg_color;
    int margin;
};
