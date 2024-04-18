#include "vo.h"
#include "featuredetector.h"
#include "image_loader.h"
#include "stereo_depth.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#define SHOW_PLOT

#ifdef SHOW_PLOT
    #include "matplotlibcpp.h"
    namespace plt = matplotlibcpp;
#endif

using namespace std;

int main() {
    std::string dataDir = "/Users/NathanDurocher/cppyourself/CppRobotics/Visual_Odometry/data/dataset/sequences/";
    std::string sequence = "03"; // # 00-21, include leading zero
    VO vo(dataDir, sequence);
    ImageLoader loader(dataDir, sequence);
    OrbFeatureDetector fd;
    auto depth_stereo = cv::StereoBM::create(0,7);

    std::vector<Eigen::MatrixXf> gt_pose;
    cv::Mat current_pose, Transform;
    std::vector<cv::Point2f> q1, q2;

    std::pair<cv::Mat, cv::Mat> image_pair;
    cv::Mat depth_image_1, depth_image_2;

    std::vector<float> est_x;
    std::vector<float> est_y;
    std::vector<float> true_x;
    std::vector<float> true_y;
    std::vector<float> error;
    std::vector<float> framenum;

    // vo.showvideo();

    for (unsigned p = 0; p < 25; p++) { //vo._poses.size()
//        std::cout << "Proccesing pose #: " << p + 1 << std::endl;
        if (p == 0) {
            current_pose = vo.ground_truth_poses()[p];
            image_pair = loader.get_image_pair(p);
            depth_stereo->compute(image_pair.first, image_pair.second, depth_image_1);
        } else {
            q1.clear();
            q2.clear();

            image_pair = std::pair<cv::Mat, cv::Mat>{depth_image_1, depth_image_2};
            fd.get_matches(image_pair, q1, q2);
            vo.get_poses(q1, q2, Transform);
            Eigen::MatrixXf Ecp, Etrans;
            cv::cv2eigen(current_pose, Ecp);
            cv::cv2eigen(Transform.inv(), Etrans);
            Ecp = Ecp * Etrans;
            cv::eigen2cv(Ecp, current_pose);

        }
        depth_image_2 = depth_image_1;

        est_x.push_back(current_pose.clone().at<float>(0, 3));
        est_y.push_back(current_pose.clone().at<float>(2, 3));

        true_x.push_back(vo.ground_truth_poses()[p].at<float>(0, 3));
        true_y.push_back(vo.ground_truth_poses()[p].at<float>(2, 3));

        // float E = sqrt(pow(est_x[p] - true_x[p], 2) + pow(est_y[p] - true_y[p], 2));
        // error.push_back(E);
        // framenum.push_back(static_cast<float>(p));
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