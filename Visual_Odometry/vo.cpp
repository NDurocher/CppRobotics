#include "vo.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>

VO::VO(const std::string &dataDir, std::string sequence, const bool use_left_image) {
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
        vo_utils::stringLine2Matrix(tempMat, rows, cols, line);
        camera_calibrations.push_back(tempMat.clone());
    }

    calibFile.close();

    // calibration file has P0/P1 for left/right grey image cameras
    auto camera_matrix = use_left_image ? camera_calibrations[0] : camera_calibrations[1];

    // where P = K[R|t] and K is the intrinsic matrix
    m_camera_intrinsics = camera_matrix.rowRange(0, 3).colRange(0, 3);
    m_camera_projection = camera_matrix.rowRange(0, 3).colRange(0, 4);
}

void VO::get_poses(const std::vector<cv::Point2f> &points2d, const std::vector<cv::Point3f> &points3d,
                   cv::Mat &Trans_Mat) const {
    cv::Mat R, r, t;
    cv::solvePnPRansac(points3d, points2d, m_camera_intrinsics, {}, r, t);
    cv::Rodrigues(r, R);
    vo_utils::formTransformMat(R, t, Trans_Mat);
}

void VO::get_poses(std::vector<cv::Point2f> &points2d_1, std::vector<cv::Point2f> &points2d_2,
                   cv::Mat &Trans_Mat) const {
    cv::Mat R, r, t;

    cv::Mat Emat = cv::findEssentialMat(points2d_1, points2d_2, m_camera_intrinsics,
                                        cv::RANSAC);

    if (Emat.empty()) {
        std::cerr << "Failed to compute Essential Matrix" << std::endl;
        Trans_Mat = cv::Mat::eye(4, 4, CV_32F);
        return;
    }

    cv::Mat homography_mat = cv::findHomography(points2d_1, points2d_2);
    const cv::Point2d camera_centre(m_camera_intrinsics.at<float>(0, 2), m_camera_intrinsics.at<float>(1, 2));
    cv::recoverPose(Emat, points2d_1, points2d_2, R, t, m_camera_intrinsics.at<float>(0, 0), camera_centre);
    vo_utils::formTransformMat(R, t, Trans_Mat);
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

// void VO::decomp_essential_mat(cv::Mat &Emat, cv::Mat &R, cv::Mat &t, std::vector<cv::Point2f> &q1,
//                               std::vector<cv::Point2f> &q2) const {
//     cv::Mat R1, R2;
//     cv::decomposeEssentialMat(Emat, R1, R2, t);
//
//     // Ensure correct data type
//     if (R1.type() != CV_32F) {
//         R1.convertTo(R1, CV_32F);
//         R2.convertTo(R2, CV_32F);
//         t.convertTo(t, CV_32F);
//     }
//
//     // Four possible [R|t] combinations from Essential Matrix decomposition
//     std::vector<std::vector<cv::Mat> > T_list = {
//         {R1, t},
//         {R1, -t},
//         {R2, t},
//         {R2, -t}
//     };
//
//
//     std::vector<std::vector<cv::Point3f> > triangulated_points(4);
//
//     // Test each of the 4 possible solutions
//     int best_idx = 0;
//     int max_count = 0;
//     for (int i = 0; i < T_list.size(); i++) {
//         // Create projection matrices
//         // cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32F);
//         cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32F);
//
//         // P1 = K[I|0] (first camera at origin)
//         auto P1 = m_camera_projection.clone();
//
//         // P2 = K[R|t] (second camera with test pose)
//         cv::Mat RT = cv::Mat::zeros(3, 4, CV_32F);
//         T_list[i][0].copyTo(RT(cv::Rect(0, 0, 3, 3)));
//         T_list[i][1].copyTo(RT(cv::Rect(3, 0, 1, 3)));
//         P2 = m_camera_intrinsics * RT;
//
//         // Triangulate points
//         cv::Mat points4d;
//         cv::triangulatePoints(P1, P2, q1, q2, points4d);
//
//         // Convert to 3D points and check cheirality (positive depth)
//         triangulated_points[i].clear();
//         int pos_count = 0;
//
//         for (int j = 0; j < points4d.cols; j++) {
//             float w = points4d.at<float>(3, j);
//             if (std::abs(w) < 1e-6) continue; // Skip points at infinity
//
//             // Convert from homogeneous coordinates
//             float x = points4d.at<float>(0, j) / w;
//             float y = points4d.at<float>(1, j) / w;
//             float z = points4d.at<float>(2, j) / w;
//
//             // Check depth in first camera (should be positive)
//             if (z > 0) {
//                 // Transform to second camera coordinate system
//                 cv::Mat point3d_cam1 = (cv::Mat_<float>(3, 1) << x, y, z);
//                 cv::Mat point3d_cam2 = T_list[i][0] * point3d_cam1 + T_list[i][1];
//
//                 // Check depth in second camera (should also be positive)
//                 if (point3d_cam2.at<float>(2, 0) > 0) {
//                     pos_count++;
//                     triangulated_points[i].emplace_back(x, y, z);
//                 }
//             }
//         }
//         if (pos_count > max_count) {
//             max_count = pos_count;
//             best_idx = i;
//         }
//     }
//
//     if (max_count < 10) {
//         // Minimum threshold for reliable solution
//         std::cerr << "Warning: Best solution only has " << max_count
//                 << " valid points" << std::endl;
//     }
//
//     R = T_list[best_idx][0].clone();
//     t = T_list[best_idx][1].clone();
//     computeScale(triangulated_points[best_idx], R, t);
// }

void VO::computeScale(const std::vector<cv::Point3f> &triangulated_points,
                      const cv::Mat &R, cv::Mat &t,
                      float camera_height, // meters above ground
                      float camera_pitch, // radians (0 = looking straight ahead)
                      float motion_threshold) {
    if (triangulated_points.empty()) {
        std::cerr << "No triangulated points for scale recovery" << std::endl;
        return;
    }

    // Step 1: Project 3D points to ground plane (Y-Z coordinates in camera frame)
    std::vector<cv::Point2f> plane_points;
    for (const auto &pt: triangulated_points) {
        // Assuming camera coordinate system: X=forward, Y=left, Z=up
        plane_points.emplace_back(pt.y, pt.z); // Y-Z plane projection
    }

    if (plane_points.empty()) {
        return;
    }

    // Step 2: Create ground plane normal vector based on camera pitch
    cv::Point2f normal(sin(camera_pitch), -cos(camera_pitch)); // Normal to expected ground plane

    // Step 3: Compute distance from each point to the expected ground plane
    std::vector<float> distances;
    for (const auto &pt: plane_points) {
        // Distance = dot product with normal vector
        float dist = pt.x * normal.x + pt.y * normal.y;
        distances.push_back(dist);
    }

    // Step 4: Find median distance for robust estimation
    std::vector<float> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    float median_dist = sorted_distances[sorted_distances.size() / 2];

    // Step 5: Robust consensus - find the distance with most support
    float sigma = median_dist / 50.0f; // Adaptive bandwidth
    float weight = 1.0f / (2.0f * sigma * sigma);
    float best_sum = 0.0f;
    float best_distance = median_dist;

    for (size_t i = 0; i < distances.size(); i++) {
        // Only consider points that are reasonable candidates for ground plane
        if (distances[i] > median_dist / motion_threshold) {
            float sum = 0.0f;

            // Count support from other points using Gaussian weighting
            for (size_t j = 0; j < distances.size(); j++) {
                float diff = distances[j] - distances[i];
                sum += exp(-diff * diff * weight);
            }

            // Keep track of distance with maximum support
            if (sum > best_sum) {
                best_sum = sum;
                best_distance = distances[i];
            }
        }
    }

    // Step 6: Compute scale factor
    if (std::abs(best_distance) < 1e-6) {
        std::cerr << "Ground plane distance too small for reliable scaling" << std::endl;
        return;
    }

    // Scale factor = known_height / estimated_height
    float scale_factor = camera_height / std::abs(best_distance);

    // Apply scale to translation
    t = t * scale_factor;
}
