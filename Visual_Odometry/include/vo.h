#pragma once

#include <vector>
#include <string>

#include <opencv2/core.hpp>

class VO {
public:
    VO(const std::string &dataDir, std::string sequence, bool use_left_image = true);

    void load_calib(std::string filePath, std::string &sequence, bool use_left_image);

    void get_poses(const std::vector<cv::Point2f> &points2d, const std::vector<cv::Point3f> &points3d,
                   cv::Mat &Trans_Mat) const;

    void get_poses(std::vector<cv::Point2f> &points2d_1, std::vector<cv::Point2f> &points2d_2,
                   cv::Mat &Trans_Mat) const;

    std::vector<cv::Point3f> point2d23d(const std::vector<cv::Point2f> &points2d, cv::Mat &depth_map);

    // void decomp_essential_mat(cv::Mat &Emat, cv::Mat &R, cv::Mat &t, std::vector<cv::Point2f> &q1,
    //                           std::vector<cv::Point2f> &q2) const;

    static void computeScale(const std::vector<cv::Point3f> &triangulated_points,
                             const cv::Mat &R, cv::Mat &t,
                             float camera_height = 1.6f, // meters above ground
                             float camera_pitch = 0.0f, // radians (0 = looking straight ahead)
                             float motion_threshold = 5.0f);

private:
    std::vector<cv::Mat> camera_calibrations; // Contains calibrations for both Left and Right Camera
    cv::Mat m_camera_intrinsics; // Camera Intrinsic Parameters K (3x3)
    cv::Mat m_camera_projection; // Camera Projection Matrix P (3x4)
};
