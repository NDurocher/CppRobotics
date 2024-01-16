#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#define SHOW_PLOT

#include "vo.h"

#ifdef SHOW_PLOT
    #include "matplotlibcpp.h"
    namespace plt = matplotlibcpp;
#endif

using namespace std;

int main() {
    std::string dataDir = "/Users/NathanDurocher/cppyourself/CppRobotics/Visual_Odometry/data/dataset/sequences";
    std::string sequence = "03"; // # 00-21, include leading zero
    std::string camera_0 = "/image_0";
    std::string camera_1 = "/image_1";
    VO vo(dataDir, sequence, camera_0);
    VO vo_2(dataDir, sequence, camera_1);

    std::vector<Eigen::MatrixXf> gt_pose;
    cv::Mat current_pose, Transform;
    std::vector<cv::Point2f> q1, q2;

    cv::Mat current_pose_2, Transform_2;
    std::vector<cv::Point2f> q1_2, q2_2;

    std::vector<float> est_x;
    std::vector<float> est_y;
    std::vector<float> true_x;
    std::vector<float> true_y;
    std::vector<float> error;
    std::vector<float> framenum;

    // vo.showvideo();

    for (unsigned p = 0; p < 100; p++) { //vo._poses.size()
//        std::cout << "Proccesing pose #: " << p + 1 << std::endl;
        if (p == 0) {
            current_pose = vo.ground_truth_poses()[p];
            current_pose_2 = vo_2.ground_truth_poses()[p];
        } else {
            q1.clear();
            q2.clear();
            vo.get_matches(p, q1, q2);
            vo.get_poses(q1, q2, Transform);
            Eigen::MatrixXf Ecp, Etrans;
            cv::cv2eigen(current_pose, Ecp);
            cv::cv2eigen(Transform.inv(), Etrans);
            Ecp = Ecp * Etrans;
            cv::eigen2cv(Ecp, current_pose);

            q1_2.clear();
            q2_2.clear();
            vo_2.get_matches(p, q1_2, q2_2);
            vo_2.get_poses(q1_2, q2_2, Transform_2);
            Ecp = Eigen::MatrixXf{};
            Etrans = Eigen::MatrixXf{};
            cv::cv2eigen(current_pose_2, Ecp);
            cv::cv2eigen(Transform_2.inv(), Etrans);
            Ecp = Ecp * Etrans;
            cv::eigen2cv(Ecp, current_pose_2);

        }
        cv::Mat avg_pose = (current_pose + current_pose_2) / 2;

        est_x.push_back(avg_pose.clone().at<float>(0, 3));
        est_y.push_back(avg_pose.clone().at<float>(2, 3));

        true_x.push_back(vo.ground_truth_poses()[p].at<float>(0, 3));
        true_y.push_back(vo.ground_truth_poses()[p].at<float>(2, 3));

        float E = sqrt(pow(est_x[p] - true_x[p], 2) + pow(est_y[p] - true_y[p], 2));
        error.push_back(E);
        framenum.push_back(static_cast<float>(p));
    }


    // Plot paths here:
    #ifdef SHOW_PLOT
        plt::named_plot("Estimated Position", est_x, est_y);
        plt::named_plot("True Position", true_x, true_y);
        plt::legend();
        plt::show();
    #endif

    return 0;
}