#include "include/utils/time_series_visualizer.h"

#include <algorithm>

TimeSeriesVisualizer::TimeSeriesVisualizer(int w, int h, cv::Scalar line_col,
                        cv::Scalar bg_col, int m)
        : width(w), height(h), line_color(line_col), bg_color(bg_col), margin(m) {}

cv::Mat TimeSeriesVisualizer::visualize(const std::vector<float>& data) {
    if (data.empty()) {
        std::cerr << "Empty data vector!" << std::endl;
        return cv::Mat::zeros(height, width, CV_8UC3);
    }

    // Create canvas
    cv::Mat img(height, width, CV_8UC3, bg_color);

    // Find min/max for scaling
    auto minmax = std::minmax_element(data.begin(), data.end());
    float min_val = *minmax.first;
    float max_val = *minmax.second;

    // Avoid division by zero
    if (max_val == min_val) {
        max_val = min_val + 1.0f;
    }

    // Calculate drawing area
    int draw_width = width - 2 * margin;
    int draw_height = height - 2 * margin;

    // Scale and draw the time series
    std::vector<cv::Point> points;
    for (size_t i = 0; i < data.size(); ++i) {
        int x = margin + (i * draw_width) / (data.size() - 1);
        int y = margin + draw_height - ((data[i] - min_val) / (max_val - min_val)) * draw_height;
        points.push_back(cv::Point(x, y));
    }

    // Draw lines between consecutive points
    for (size_t i = 1; i < points.size(); ++i) {
        cv::line(img, points[i-1], points[i], line_color, 2);
    }

    // Draw axes
    drawAxes(img, min_val, max_val, data.size());

    return img;
}

void TimeSeriesVisualizer::drawAxes(cv::Mat& img, float min_val, float max_val, size_t data_size) {
    cv::Scalar axis_color(0, 0, 0);

    // Y-axis
    cv::line(img, cv::Point(margin, margin),
             cv::Point(margin, height - margin), axis_color, 1);

    // X-axis
    cv::line(img, cv::Point(margin, height - margin),
             cv::Point(width - margin, height - margin), axis_color, 1);

    // Add labels
    cv::putText(img, std::to_string(max_val), cv::Point(5, margin + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
    cv::putText(img, std::to_string(min_val), cv::Point(5, height - margin - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
    cv::putText(img, "0", cv::Point(margin - 10, height - margin + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
    cv::putText(img, std::to_string(data_size - 1),
                cv::Point(width - margin - 20, height - margin + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
}