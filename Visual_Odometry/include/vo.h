#pragma once

#include <vector>
#include <string>

#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

class VO {
public:
    VO(const std::string &dataDir, std::string sequence, bool use_left_image = true);

    void load_calib(std::string filePath, std::string &sequence, bool use_left_image);

    void load_poses(std::string filePath, std::string &sequence);

    void get_poses(const std::vector<cv::Point2f> &points2d, const std::vector<cv::Point3f> &points3d,
                   cv::Mat &Trans_Mat) const;

    void get_poses(std::vector<cv::Point2f> &points2d_1, std::vector<cv::Point2f> &points2d_2,
                   cv::Mat &Trans_Mat) const;

    std::vector<cv::Point3f> point2d23d(const std::vector<cv::Point2f> &points2d, cv::Mat &depth_map);

    void decomp_essential_mat(cv::Mat &Emat, cv::Mat &R, cv::Mat &t, std::vector<cv::Point2f> &q1,
                              std::vector<cv::Point2f> &q2) const;

    static void get_relativeScale(Eigen::MatrixXf &HQ1, Eigen::MatrixXf &HQ2, std::vector<float> &relative_scale);

    static void formTransformMat(const cv::Mat &R, const cv::Mat &t, cv::Mat &T);

    static void stringLine2Matrix(cv::Mat &tempMat, const int &rows, const int &cols, std::string &line);

    std::vector<cv::Mat> ground_truth_poses();

private:
    std::vector<cv::Mat> camera_calibrations; // Contains calibrations for both Left and Right Camera
    std::vector<cv::Mat> poses; // Contains ground truth poses for camera
    size_t m_num_frames{};

    cv::Mat m_camera_intrinsics; // Camera Intrinsic Parameters K (3x3)
    cv::Mat m_camera_projection; // Camera Projection Matrix P (3x4)
};
