#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class PlotVisualizer {
public:
    explicit PlotVisualizer(int w = 800, int h = 600, int m = 60);

    // Plot single x,y series as scatter plot
    cv::Mat plotScatter(const std::vector<float> &x, const std::vector<float> &y,
                        int point_size = 3, const cv::Scalar &color = cv::Scalar(0, 255, 0)) const;

    // Plot single x,y series as line plot
    cv::Mat plotLine(const std::vector<float> &x, const std::vector<float> &y,
                     int line_width = 2, const cv::Scalar &color = cv::Scalar(0, 255, 0)) const;

    // Plot multiple x,y series
    cv::Mat plotMultiple(const std::vector<std::vector<float> > &x_data,
                         const std::vector<std::vector<float> > &y_data,
                         const std::vector<std::string> &labels = {},
                         bool as_lines = true) const;

private:
    void drawAxes(cv::Mat &img, float x_min, float x_max, float y_min, float y_max) const;

    void drawLegend(cv::Mat &img, size_t num_series, const std::vector<std::string> &labels) const;

    int width, height, margin;
    cv::Scalar bg_color;
    std::vector<cv::Scalar> colors;
};
