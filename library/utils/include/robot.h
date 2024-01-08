#pragma once

#include <Eigen/Dense>
#include <random>

namespace diff_drive {

    Eigen::MatrixXd motion_model(Eigen::MatrixXd &X, Eigen::MatrixXd &U, double timestep);

    Eigen::MatrixXd sample_gps(Eigen::MatrixXd &state, double pos_variance);

    Eigen::MatrixXd corrupt_input(Eigen::MatrixXd &input, double velocity_variance, double heading_variance);

}







