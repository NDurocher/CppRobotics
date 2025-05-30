#include "vo.h"
#include "feature_detector.h"
#include "image_loader.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "utils/plot_visualizer.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Required argument for image data directory and sequence number (01-23)" << std::endl;
        return EXIT_FAILURE;
    }
    std::string dataDir = argv[1];
    std::string sequence = argv[2];
    VO vo(dataDir, sequence);
    ImageLoader loader(dataDir, sequence);
    OrbFeatureDetector fd;
    // auto disparity_stereo = cv::StereoBM::create(0, 7);

    std::vector<Eigen::MatrixXf> gt_pose;
    cv::Mat current_pose, Transform;

    keypoint_descriptor kpd1, kpd2;
    std::vector<cv::Point2f> q1, q2;
    std::vector<cv::Point3f> q3d_1, q3d_2;

    std::pair<cv::Mat, cv::Mat> image_pair;
    // cv::Mat disparity_image;

    std::vector<float> est_x;
    std::vector<float> est_y;
    std::vector<float> true_x;
    std::vector<float> true_y;

    // loader.show_video();
    auto frame_nums = vo.ground_truth_poses().size();
    for (int p = 0; p < frame_nums; p++) {
        image_pair = loader.get_image_pair(p);
        // get features from this image and give points z point from depth map
        kpd2 = fd.compute_features(image_pair.first);

        if (p == 0) {
            current_pose = vo.ground_truth_poses()[p];
        } else {
            q1.clear();
            q2.clear();

            auto _matches = fd.get_matches(kpd1, kpd2, q1, q2);

            // disparity_stereo->compute(image_pair.first, image_pair.second, disparity_image);

            vo.get_poses(q1, q2, Transform);

            Eigen::MatrixXf Ecp, Etrans;
            cv::cv2eigen(current_pose, Ecp);
            cv::cv2eigen(Transform.inv(), Etrans);
            Ecp = Ecp * Etrans;
            cv::eigen2cv(Ecp, current_pose);
        }
        kpd1 = kpd2;

        est_x.push_back(current_pose.clone().at<float>(0, 3));
        est_y.push_back(current_pose.clone().at<float>(2, 3));

        true_x.push_back(vo.ground_truth_poses()[p].at<float>(0, 3));
        true_y.push_back(vo.ground_truth_poses()[p].at<float>(2, 3));
    }

    // Create visualizer and generate plot
    PlotVisualizer viz(800, 400);
    cv::Mat plot = viz.plotMultiple({est_x, true_x}, {est_y, true_y},
                                    {"estimated", "ground truth"});

    // Display the plot
    cv::imshow("Estimated vs Ground Truth", plot);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // Optionally save the plot
    // cv::imwrite("timeseries.png", plot);

    return 0;
}
