#include "vo.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/sfm//triangulation.hpp>
#include <Eigen/Dense>

std::vector<cv::Mat> vectorToMatVector2xN_Fast(const std::vector<std::vector<cv::Point2f> > &pointVectors) {
    std::vector<cv::Mat> matVector;
    matVector.reserve(pointVectors.size());

    for (const auto &points: pointVectors) {
        if (!points.empty()) {
            // Create 2×N matrix by reshaping the Point2f data
            cv::Mat pointsMat(points.size(), 1, CV_32FC2, (void *) points.data());
            cv::Mat mat = pointsMat.reshape(1, 2).clone(); // Reshape to 2×N and clone for safety

            matVector.push_back(mat);
        } else {
            matVector.push_back(cv::Mat()); // Empty Mat for empty vectors
        }
    }

    return matVector;
}

// Helper function to check if 3D points are degenerate
bool isPointSetDegenerate(const std::vector<cv::Point3f> &points3d) {
    if (points3d.size() < 4) return true;

    // Check if all points are coplanar by computing volume of tetrahedron
    cv::Point3f p1 = points3d[0];
    cv::Point3f p2 = points3d[1];
    cv::Point3f p3 = points3d[2];
    cv::Point3f p4 = points3d[3];

    // Vectors from p1 to other points
    cv::Point3f v1 = p2 - p1;
    cv::Point3f v2 = p3 - p1;
    cv::Point3f v3 = p4 - p1;

    // Cross product v1 × v2
    cv::Point3f cross = cv::Point3f(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );

    // Dot product with v3 gives volume (×6)
    float volume = std::abs(cross.x * v3.x + cross.y * v3.y + cross.z * v3.z);

    // If volume is too small, points are nearly coplanar
    return volume < 1e-6;
}


VO::VO(const std::string &dataDir, std::string sequence, const bool use_left_image) {
    load_poses(dataDir, sequence);
    std::cout << "GT Poses Loaded" << std::endl;
    load_calib(dataDir, sequence, use_left_image);
    std::cout << "Camera Calibrations Loaded" << std::endl;
}

void VO::load_calib(std::string filePath, std::string &sequence, bool use_left_image) {
    filePath.append(sequence + "/calib.txt");

    std::ifstream calibFile(filePath);

    int rows = 3;
    int cols = 4;
    cv::Mat tempMat = cv::Mat::zeros(rows, cols, CV_32F);
    std::string line;
    while (getline(calibFile, line)) {
        stringLine2Matrix(tempMat, rows, cols, line);
        camera_calibrations.push_back(tempMat.clone());
    }

    calibFile.close();

    // calibration file has P0/P1 for left/right grey image cameras
    auto camera_matrix = use_left_image ? camera_calibrations[0] : camera_calibrations[1];

    // where P = K[R|t] and K is the intrinsic matrix
    m_camera_intrinsics = camera_matrix.rowRange(0, 3).colRange(0, 3);
    m_camera_projection = camera_matrix.rowRange(0, 3).colRange(0, 4);
}

void VO::load_poses(std::string filePath, std::string &sequence) {
    filePath.append("poses/" + sequence + ".txt");

    std::ifstream poseFile(filePath);

    int rows = 3;
    int cols = 4;
    cv::Mat tempMat = cv::Mat::zeros(rows, cols, CV_32F);

    std::string line;
    while (getline(poseFile, line)) {
        stringLine2Matrix(tempMat, rows, cols, line);
        poses.push_back(tempMat.clone());
    }
    poseFile.close();
    m_num_frames = poses.size();
}

void VO::get_poses(const std::vector<cv::Point2f> &points2d, const std::vector<cv::Point3f> &points3d,
                   cv::Mat &Trans_Mat) const {
    cv::Mat R, r, t;
    cv::solvePnPRansac(points3d, points2d, m_camera_intrinsics, {}, r, t);
    cv::Rodrigues(r, R);
    formTransformMat(R, t, Trans_Mat);
}

void VO::get_poses(std::vector<cv::Point2f> &points2d_1, std::vector<cv::Point2f> &points2d_2,
                   cv::Mat &Trans_Mat) const {
    cv::Mat R, r, t;

    cv::Mat Emat = cv::findEssentialMat(points2d_1, points2d_2, m_camera_intrinsics,
                                        cv::RANSAC, 0.999, 1.0, {});

    if (Emat.empty()) {
        std::cerr << "Failed to compute Essential Matrix" << std::endl;
        Trans_Mat = cv::Mat::eye(4, 4, CV_32F);
        return;
    }

    decomp_essential_mat(Emat, r, t, points2d_1, points2d_2);
    // cv::Rodrigues(r, R);
    formTransformMat(r, t, Trans_Mat);

    // cv::solvePnPRansac(points3d_cv, points2d_1, m_camera_intrinsics, {}, r, t);
    // formTransformMat(R, t, Trans_Mat);
}

std::vector<cv::Point3f> VO::point2d23d(const std::vector<cv::Point2f> &points2d, cv::Mat &depth_map) {
    std::vector<cv::Point3f> points3d;

    for (const auto &point: points2d) {
        cv::Point3f point3;
        const auto int_point_x = static_cast<int>(point.x);
        const auto int_point_y = static_cast<int>(point.y);
        const float depth = depth_map.at<uint16_t>(int_point_y, int_point_x);

        point3.x = (point.x - m_camera_intrinsics.at<float>(0, 2)) * depth / m_camera_intrinsics.at<float>(0, 0);
        // (x - cx) * depth / fx
        point3.y = (point.y - m_camera_intrinsics.at<float>(1, 2)) * depth / m_camera_intrinsics.at<float>(1, 1);
        // (y - cy) * depth / fy

        const auto tx0 = camera_calibrations[0].at<float>(0, 3) / 1000;
        const auto tx1 = camera_calibrations[1].at<float>(0, 3) / 1000;
        const auto fx = camera_calibrations[0].at<float>(0, 0);
        point3.z = std::abs(tx1 - tx0) * fx / depth;

        points3d.push_back(point3);
    }

    return points3d;
}

void VO::decomp_essential_mat(cv::Mat &Emat, cv::Mat &R, cv::Mat &t, std::vector<cv::Point2f> &q1,
                              std::vector<cv::Point2f> &q2) const {
    cv::Mat R1, R2;
    cv::decomposeEssentialMat(Emat, R1, R2, t);

    // Ensure correct data type
    if (R1.type() != CV_32F) {
        R1.convertTo(R1, CV_32F);
        R2.convertTo(R2, CV_32F);
        t.convertTo(t, CV_32F);
    }

    // Four possible [R|t] combinations from Essential Matrix decomposition
    std::vector<std::vector<cv::Mat> > T_list = {
        {R1, t},
        {R1, -t},
        {R2, t},
        {R2, -t}
    };

    std::vector<int> pos_count = {0, 0, 0, 0};
    std::vector<std::vector<cv::Point3f> > triangulated_points(4);

    // Test each of the 4 possible solutions
    for (size_t i = 0; i < T_list.size(); i++) {
        // Create projection matrices
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32F);
        cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32F);

        // P1 = K[I|0] (first camera at origin)
        m_camera_intrinsics.copyTo(P1(cv::Rect(0, 0, 3, 3)));

        // P2 = K[R|t] (second camera with test pose)
        cv::Mat RT = cv::Mat::zeros(3, 4, CV_32F);
        T_list[i][0].copyTo(RT(cv::Rect(0, 0, 3, 3)));
        T_list[i][1].copyTo(RT(cv::Rect(3, 0, 1, 3)));
        P2 = m_camera_intrinsics * RT;

        // Triangulate points
        cv::Mat points4d;
        cv::triangulatePoints(P1, P2, q1, q2, points4d);

        // Convert to 3D points and check cheirality (positive depth)
        triangulated_points[i].clear();
        pos_count[i] = 0;

        for (int j = 0; j < points4d.cols; j++) {
            float w = points4d.at<float>(3, j);
            if (std::abs(w) < 1e-6) continue; // Skip points at infinity

            // Convert from homogeneous coordinates
            float x = points4d.at<float>(0, j) / w;
            float y = points4d.at<float>(1, j) / w;
            float z = points4d.at<float>(2, j) / w;

            // Check depth in first camera (should be positive)
            if (z > 0) {
                // Transform to second camera coordinate system
                cv::Mat point3d_cam1 = (cv::Mat_<float>(3, 1) << x, y, z);
                cv::Mat point3d_cam2 = T_list[i][0] * point3d_cam1 + T_list[i][1];

                // Check depth in second camera (should also be positive)
                if (point3d_cam2.at<float>(2, 0) > 0) {
                    pos_count[i]++;
                    triangulated_points[i].emplace_back(x, y, z);
                }
            }
        }

        std::cout << "Solution " << i << ": " << pos_count[i]
                << " points with positive depth" << std::endl;
    }

    // Choose solution with most points having positive depth in both cameras
    int idx_max = std::distance(pos_count.begin(),
                                std::max_element(pos_count.begin(), pos_count.end()));

    if (pos_count[idx_max] < 10) {
        // Minimum threshold for reliable solution
        std::cerr << "Warning: Best solution only has " << pos_count[idx_max]
                << " valid points" << std::endl;
    }

    R = T_list[idx_max][0].clone();
    t = T_list[idx_max][1].clone();

    std::cout << "Selected solution " << idx_max << " with "
            << pos_count[idx_max] << " valid points" << std::endl;
    // cv::Mat R1, R2;
    // cv::decomposeEssentialMat(Emat, R1, R2, t);
    // if (R1.type() != CV_32F) {
    //     R1.convertTo(R1, CV_32F);
    //     R2.convertTo(R2, CV_32F);
    //     t.convertTo(t, CV_32F);
    // }
    // std::vector<std::vector<cv::Mat> > T_list = {
    //     {R1, t},
    //     {R1, -t},
    //     {R2, t},
    //     {R2, -t}
    // };
    //
    // std::vector<int> pos_count = {0, 0, 0, 0};
    // std::vector<float> relative_scale;
    // for (unsigned i = 0; i < T_list.size(); i++) {
    //     cv::Mat T = cv::Mat::zeros(4, 4, CV_32F);
    //     formTransformMat(T_list[i][0], T_list[i][1], T);
    //     cv::Mat Ptemp = m_camera_projection * T;
    //     cv::Mat HQ1;
    //     cv::triangulatePoints(m_camera_projection, Ptemp, q1, q2, HQ1);
    //     cv::Mat HQ2 = T * HQ1;
    //
    //     Eigen::MatrixXf EHQ1, EHQ2;
    //     cv::cv2eigen(HQ1, EHQ1);
    //     cv::cv2eigen(HQ2, EHQ2);
    //     for (unsigned c = 0; c < EHQ1.cols(); c++) {
    //         EHQ1.block(0, c, 4, 1) /= EHQ1(3, c);
    //         EHQ2.block(0, c, 4, 1) /= EHQ2(3, c);
    //         if (EHQ1(2, c) > 0) {
    //             pos_count[i]++;
    //         }
    //         if (EHQ2(2, c) > 0) {
    //             pos_count[i]++;
    //         }
    //     }
    //     get_relativeScale(EHQ1, EHQ2, relative_scale);
    // }
    //
    // int idx_max = std::distance(pos_count.begin(), std::max_element(pos_count.begin(), pos_count.end()));
    // R = T_list[idx_max][0];
    // t = T_list[idx_max][1] * relative_scale[idx_max];
}

void VO::get_relativeScale(Eigen::MatrixXf &HQ1, Eigen::MatrixXf &HQ2, std::vector<float> &relative_scale) {
    Eigen::MatrixXf Temp1 = HQ1.transpose().block(0, 0, HQ1.cols() - 1, HQ1.rows() - 1) -
                            HQ1.transpose().block(1, 0, HQ1.cols() - 1, HQ1.rows() - 1);
    Eigen::MatrixXf Temp2 = HQ2.transpose().block(0, 0, HQ2.cols() - 1, HQ2.rows() - 1) -
                            HQ2.transpose().block(1, 0, HQ2.cols() - 1, HQ2.rows() - 1);
    Temp1.colwise().normalize();
    Temp2.colwise().normalize();

    relative_scale.push_back((Temp1.array() / Temp2.array()).mean());
}


void VO::formTransformMat(const cv::Mat &R, const cv::Mat &t, cv::Mat &T) {
    cv::hconcat(R, t, T);
    T.push_back(cv::Mat::zeros(1, 4, T.type()));
    Eigen::MatrixXf tempT;
    cv::cv2eigen(T, tempT);
    tempT(3, 3) = 1.0;
    cv::eigen2cv(tempT, T);
}

void VO::stringLine2Matrix(cv::Mat &tempMat, const int &rows, const int &cols, std::string &line) {
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
        if ((c + 1) == cols) {
            if ((r + 1) == rows) {
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

std::vector<cv::Mat> VO::ground_truth_poses() {
    return poses;
}
