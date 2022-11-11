#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "../include/Visual_Odometry/vo.h"
#include "gnuplot-iostream.h"

int main(int argc, char** argv )
{   
    Gnuplot gp;
    bool SHOW_PLOT = false;
    std::string dataDir = "../data/KITTI_sequence_1/";
    VO vo(dataDir);

    std::vector<Eigen::MatrixXd> gt_pose;
    std::vector<Eigen::MatrixXd> estimated_pose;
    Eigen::MatrixXd current_pose;
    std::vector<cv::Point2f> q1, q2;

    for (unsigned p=0; p < vo._poses.size(); p++){
        if (p == 0){
            current_pose = vo._poses[p];
        }
        else{
            vo.get_matches(p, q1, q2);
            std::cout << "matching and shit goes here" << std::endl;
        }
    }
    // Plot paths here:
    // if (SHOW_PLOT){
    //     gp << "plot" << gp.file1d(boost::make_tuple(Est_x, Est_y)) <<  "with lines title 'Estimated Position'\n" << gp.file1d(boost::make_tuple(True_x, True_y)) \
    //      << "with lines title 'Ground Truth Position'\n";
    // }
    return 0;
}