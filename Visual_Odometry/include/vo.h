#pragma once

#include <vector>
#include <string>

#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

class VO {
public:
    VO(const std::string &dataDir, std::string sequence);

    void load_calib(std::string filePath, std::string &sequence);

    void load_poses(std::string filePath, std::string &sequence);

    void get_poses(std::vector<cv::Point2f> &points2d, std::vector<cv::Point3f> &points3d, cv::Mat &Trans_Mat);

    std::vector<cv::Point3f> point2d23d(std::vector<cv::Point2f> points2d, cv::Mat &depth_map);

    void decomp_essential_mat(cv::Mat &Emat, cv::Mat &R, cv::Mat &t, std::vector<cv::Point3f> &q1,
                              std::vector<cv::Point3f> &q2);

    static void get_relativeScale(Eigen::MatrixXf &HQ1, Eigen::MatrixXf &HQ2, std::vector<float> &relative_scale);

    static void formTransformMat(cv::Mat &R, cv::Mat &t, cv::Mat &T);

    static void stringLine2Matrix(cv::Mat &tempMat, int &rows, int &cols, std::string &line);

    std::vector<cv::Mat> ground_truth_poses();

private:
    std::vector<cv::Mat> calibs; // Contains calibrations for both Left and Right Camera
    std::vector<cv::Mat> poses; // Contains ground truth poses for camera
    std::vector<std::string> strPoses;

    cv::Mat K; // Camera Intrinsic Parameters (3x3)
    cv::Mat P; // Camera Projection Matrix (3x4)
};
