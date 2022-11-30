#include <stdio.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "../include/Visual_Odometry/vo.h"
#include "gnuplot-iostream.h"

int main(int argc, char** argv )
{   
    Gnuplot gp;
    Gnuplot gp2;
    // gp << "set xrange [-10:10]\n";
    bool SHOW_PLOT = true;
    std::string dataDir = "../data/KITTI_sequence_2/";
    VO vo(dataDir);

    std::vector<Eigen::MatrixXf> gt_pose;
    std::vector<Eigen::MatrixXf> estimated_pose;
    cv::Mat current_pose, Transform;
    std::vector<cv::Point2f> q1, q2;

    std::vector<float> est_x;
    std::vector<float> est_y;
    std::vector<float> true_x;
    std::vector<float> true_y;
    std::vector<float> error;
    std::vector<float> framenum;

    // vo.showvideo();

    for (unsigned p=0; p < vo._poses.size(); p++){ //vo._poses.size()
        std::cout << "Proccesing pose #: " << p+1 << std::endl;
        if (p == 0){
            current_pose = vo._poses[p];
        }
        else{
            q1.clear();
            q2.clear();
            vo.get_matches(p, q1, q2);
            vo.get_poses(q1,q2, Transform);
            Eigen::MatrixXf Ecp, Etrans;
            cv::cv2eigen(current_pose, Ecp);
            cv::cv2eigen(Transform, Etrans);
            Ecp = Ecp * Etrans.inverse();
            cv::eigen2cv(Ecp, current_pose);
            
        }
        est_x.push_back(current_pose.clone().at<float>(0,3));
        est_y.push_back(current_pose.clone().at<float>(2,3));

        true_x.push_back(vo._poses[p].at<float>(0,3));
        true_y.push_back(vo._poses[p].at<float>(2,3));

        float E = sqrt(pow(est_x[p]-true_x[p],2)+pow(est_y[p]-true_y[p],2));
        error.push_back(E);
        framenum.push_back(p);
    }
 
    // Plot paths here:
    if (SHOW_PLOT){
        gp << "plot" << gp.file1d(boost::make_tuple(est_x, est_y)) \
            <<  "with lines title 'Estimated Position'," \
            << gp.file1d(boost::make_tuple(true_x, true_y)) \
            << "with lines title 'Ground Truth Position'\n";
        gp2 << "plot" << gp.file1d(boost::make_tuple(framenum, error)) \
            << "with points title 'Error'\n";
    }
    return 0;
}