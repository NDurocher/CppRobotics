#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include "robot.h"
#include "my_time.h"


struct SlamVariables {
    Eigen::MatrixXd P_t{{1, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1}}; // Covariance matrix for x
    Eigen::MatrixXd G{{1, 0, 0},
                      {0, 1, 0},
                      {0, 0, 1}};
    Eigen::MatrixXd Cx{{std::pow(0.5, 2), 0,                0},
                       {0,                std::pow(0.5, 2), 0},
                       {0,                0,                std::pow(30.0 * M_PI / 180.0,
                                                                     2)}}; // Covariance mat for EKF state
    Eigen::MatrixXd Q{{std::pow(0.2, 2), 0},
                      {0,                std::pow(1.0 * M_PI / 180.0, 2)}}; // Covariance mat for sensor noise
    Eigen::MatrixXd R{{std::pow(1.0, 2), 0},
                      {0,                std::pow(10.0 * M_PI / 180.0, 2)}}; // Covariance mat of observation noise //
    double MAX_RANGE{20.}; // meters
    double M_DIST_TH{2.0}; // mahalanobis distance threshold for new landmark
};

struct SimVariables {
    int state_size{3};
    int LM_size{2};
    double dt{0.1};
};

double pi2pi(double angle);

int CountLMs(Eigen::MatrixXd &Xest, const SimVariables &sim_vars);

Eigen::MatrixXd Observation(Eigen::MatrixXd &U, Eigen::MatrixXd &Xtrue, Eigen::MatrixXd &LM_pos,
                            const SlamVariables &slam_vars, const SimVariables &sim_vars);

std::vector<Eigen::MatrixXd> Motion_jacobian(Eigen::MatrixXd X, Eigen::MatrixXd U, const SimVariables &sim_vars);

void
Predict(Eigen::MatrixXd &Xest, Eigen::MatrixXd &U, SlamVariables &slam_vars, const SimVariables &sim_vars);

void Update(Eigen::MatrixXd &Xest, Eigen::MatrixXd &z_obs, SlamVariables &slam_vars,
            const SimVariables &sim_vars);

Eigen::MatrixXd Calc_LM_pos(Eigen::MatrixXd &xest, Eigen::MatrixXd zi, const SimVariables &sim_vars);

std::vector<Eigen::MatrixXd>
Innovation(Eigen::MatrixXd &est_lm_pos, Eigen::MatrixXd &xest, Eigen::MatrixXd zi_obs, int idx,
           SlamVariables &slam_vars, const SimVariables &sim_vars);

Eigen::MatrixXd get_lm_pos_from_state(Eigen::MatrixXd Xest, int i, const SimVariables &sim_vars);

int search_lm_id(const Eigen::MatrixXd &zi_obs, Eigen::MatrixXd &xest, SlamVariables &slam_vars,
                 const SimVariables &sim_vars);

Eigen::MatrixXd H_jacob(double q, Eigen::MatrixXd &delta, Eigen::MatrixXd &xest, int i, const SimVariables &sim_vars);


