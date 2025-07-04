#pragma once
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

namespace vo_utils {
    inline std::vector<cv::Mat> vectorToMatVector2xN_Fast(const std::vector<std::vector<cv::Point2f> > &pointVectors) {
        std::vector<cv::Mat> matVector;
        matVector.reserve(pointVectors.size());

        for (const auto &points: pointVectors) {
            if (!points.empty()) {
                // Create 2×N matrix by reshaping the Point2f data
                auto data = points.data();
                cv::Mat pointsMat(points.size(), 1, CV_32FC2, &data);
                cv::Mat mat = pointsMat.reshape(1, 2).clone(); // Reshape to 2×N and clone for safety

                matVector.push_back(mat);
            } else {
                matVector.push_back(cv::Mat()); // Empty Mat for empty vectors
            }
        }

        return matVector;
    }

    inline void stringLine2Matrix(cv::Mat &tempMat, const int &rows, const int &cols, std::string &line) {
        int r = 0;
        int c = 0;
        std::string strNum;
        const std::string delim = " ";

        // if the line contains the sensor prefix, erase it
        if (line.find('P') != std::string::npos) {
            line.erase(0, 4);
        }

        for (std::string::const_iterator i = line.begin(); i != line.end(); ++i) {
            // If i is not a delim, then append it to strnum
            if (delim.find(*i) == std::string::npos) {
                strNum += *i;
                if (i + 1 != line.end()) {
                    continue;
                }
            }
            if (strNum.empty()) {
                continue;
            }

            float number;
            std::istringstream(strNum) >> number;
            tempMat.at<float>(r, c) = number;
            if (c + 1 == cols) {
                if (r + 1 == rows) {
                    strNum.clear();
                    break;
                }
                c = 0;
                r++;
                strNum.clear();
                continue;
            }
            c++;
            strNum.clear();
        }
    }

    inline void formTransformMat(const cv::Mat &R, const cv::Mat &t, cv::Mat &T) {
        cv::hconcat(R, t, T);
        T.push_back(cv::Mat::zeros(1, 4, T.type()));
        Eigen::MatrixXf tempT;
        cv::cv2eigen(T, tempT);
        tempT(3, 3) = 1.0;
        cv::eigen2cv(tempT, T);
    }
} // namespace vo_utils
