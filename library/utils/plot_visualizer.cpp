#include "include/utils/plot_visualizer.h"

#include <algorithm>
#include <iostream>

PlotVisualizer::PlotVisualizer(const int w, const int h, const int m)
    : width(w), height(h), margin(m), bg_color(255, 255, 255) {
    colors = {
        cv::Scalar(0, 255, 0), // Green
        cv::Scalar(255, 0, 0), // Blue
        cv::Scalar(0, 0, 255), // Red
        cv::Scalar(255, 255, 0), // Cyan
        cv::Scalar(255, 0, 255), // Magenta
        cv::Scalar(0, 255, 255), // Yellow
        cv::Scalar(128, 0, 128), // Purple
        cv::Scalar(255, 165, 0) // Orange
    };
}

cv::Mat PlotVisualizer::plotScatter(const std::vector<float> &x, const std::vector<float> &y,
                                    const int point_size, const cv::Scalar &color) const {
    if (x.size() != y.size() || x.empty()) {
        std::cerr << "Invalid data: x and y must have same non-zero size!" << std::endl;
        return cv::Mat::zeros(height, width, CV_8UC3);
    }

    cv::Mat img(height, width, CV_8UC3, bg_color);

    auto [x_min_it, x_max_it] = std::minmax_element(x.begin(), x.end());
    auto [y_min_it, y_max_it] = std::minmax_element(y.begin(), y.end());

    const float x_min = *x_min_it;
    float x_max = *x_max_it;
    const float y_min = *y_min_it;
    float y_max = *y_max_it;

    if (x_max == x_min) x_max = x_min + 1.0f;
    if (y_max == y_min) y_max = y_min + 1.0f;

    const int draw_width = width - 2 * margin;
    const int draw_height = height - 2 * margin;

    // Draw points
    for (size_t i = 0; i < x.size(); ++i) {
        const int px = margin + (x[i] - x_min) / (x_max - x_min) * draw_width;
        const int py = margin + draw_height - (y[i] - y_min) / (y_max - y_min) * draw_height;
        cv::circle(img, cv::Point(px, py), point_size, color, -1);
    }

    drawAxes(img, x_min, x_max, y_min, y_max);
    return img;
}

cv::Mat PlotVisualizer::plotLine(const std::vector<float> &x, const std::vector<float> &y,
                                 const int line_width, const cv::Scalar &color) const {
    if (x.size() != y.size() || x.empty()) {
        std::cerr << "Invalid data: x and y must have same non-zero size!" << std::endl;
        return cv::Mat::zeros(height, width, CV_8UC3);
    }

    cv::Mat img(height, width, CV_8UC3, bg_color);

    auto [x_min_it, x_max_it] = std::minmax_element(x.begin(), x.end());
    auto [y_min_it, y_max_it] = std::minmax_element(y.begin(), y.end());

    const float x_min = *x_min_it;
    float x_max = *x_max_it;
    const float y_min = *y_min_it;
    float y_max = *y_max_it;

    if (x_max == x_min) x_max = x_min + 1.0f;
    if (y_max == y_min) y_max = y_min + 1.0f;

    const int draw_width = width - 2 * margin;
    const int draw_height = height - 2 * margin;

    std::vector<cv::Point> points;
    for (size_t i = 0; i < x.size(); ++i) {
        const int px = margin + ((x[i] - x_min) / (x_max - x_min)) * draw_width;
        const int py = margin + draw_height - (y[i] - y_min) / (y_max - y_min) * draw_height;
        points.push_back(cv::Point(px, py));
    }

    // Draw lines
    for (size_t i = 1; i < points.size(); ++i) {
        cv::line(img, points[i - 1], points[i], color, line_width);
    }

    drawAxes(img, x_min, x_max, y_min, y_max);
    return img;
}

cv::Mat PlotVisualizer::plotMultiple(const std::vector<std::vector<float> > &x_data,
                                     const std::vector<std::vector<float> > &y_data,
                                     const std::vector<std::string> &labels,
                                     const bool as_lines) const {
    if (x_data.size() != y_data.size() || x_data.empty()) {
        std::cerr << "Invalid data!" << std::endl;
        return cv::Mat::zeros(height, width, CV_8UC3);
    }

    cv::Mat img(height, width, CV_8UC3, bg_color);

    // Find global bounds
    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();

    for (size_t s = 0; s < x_data.size(); ++s) {
        auto [xmin_it, xmax_it] = std::minmax_element(x_data[s].begin(), x_data[s].end());
        auto [ymin_it, ymax_it] = std::minmax_element(y_data[s].begin(), y_data[s].end());
        x_min = std::min(x_min, *xmin_it);
        x_max = std::max(x_max, *xmax_it);
        y_min = std::min(y_min, *ymin_it);
        y_max = std::max(y_max, *ymax_it);
    }

    if (x_max == x_min) x_max = x_min + 1.0f;
    if (y_max == y_min) y_max = y_min + 1.0f;

    const int draw_width = width - 2 * margin;
    const int draw_height = height - 2 * margin;

    // Draw each series
    for (size_t s = 0; s < x_data.size(); ++s) {
        cv::Scalar color = colors[s % colors.size()];

        std::vector<cv::Point> points;
        for (size_t i = 0; i < x_data[s].size(); ++i) {
            const int px = margin + ((x_data[s][i] - x_min) / (x_max - x_min)) * draw_width;
            const int py = margin + draw_height - (y_data[s][i] - y_min) / (y_max - y_min) * draw_height;
            points.push_back(cv::Point(px, py));
        }

        if (as_lines) {
            for (size_t i = 1; i < points.size(); ++i) {
                cv::line(img, points[i - 1], points[i], color, 2);
            }
        } else {
            for (const auto &pt: points) {
                cv::circle(img, pt, 3, color, -1);
            }
        }
    }

    drawAxes(img, x_min, x_max, y_min, y_max);
    drawLegend(img, x_data.size(), labels);

    return img;
}

void PlotVisualizer::drawAxes(cv::Mat &img, const float x_min, const float x_max, const float y_min,
                              const float y_max) const {
    const cv::Scalar axis_color(0, 0, 0);

    // Y-axis
    cv::line(img, cv::Point(margin, margin),
             cv::Point(margin, height - margin), axis_color, 1);

    // X-axis
    cv::line(img, cv::Point(margin, height - margin),
             cv::Point(width - margin, height - margin), axis_color, 1);

    // Labels
    cv::putText(img, cv::format("%.1f", y_max), cv::Point(5, margin + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
    cv::putText(img, cv::format("%.1f", y_min), cv::Point(5, height - margin - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
    cv::putText(img, cv::format("%.1f", x_min), cv::Point(margin - 15, height - margin + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
    cv::putText(img, cv::format("%.1f", x_max), cv::Point(width - margin - 30, height - margin + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1);
}

void PlotVisualizer::drawLegend(cv::Mat &img, const size_t num_series, const std::vector<std::string> &labels) const {
    const int legend_x = width - margin - 100;
    const int legend_y = margin + 20;

    for (size_t i = 0; i < num_series; ++i) {
        cv::Scalar color = colors[i % colors.size()];
        const int y_pos = legend_y + i * 20;

        cv::line(img, cv::Point(legend_x, y_pos),
                 cv::Point(legend_x + 20, y_pos), color, 2);

        std::string label = i < labels.size() ? labels[i] : "Series " + std::to_string(i);
        cv::putText(img, label, cv::Point(legend_x + 25, y_pos + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
}
