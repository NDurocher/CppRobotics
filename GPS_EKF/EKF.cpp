#include "EKF.h"
#include <cmath>
#include <iostream>

using namespace std;

EKF::EKF(double dt) : robot{0.0, 0.0, 0.0, dt} {

    J_g.resize(2, 3);
    J_g << 1, 0, 0,
            0, 1, 0;

    /* Covariance matrix for x */
    P_t.resize(3, 3);
    P_t << 1, 0, 0,
            0, 1, 0,
            0, 0, 1;

    /* Covariance mat for process noise */
    Q.resize(3, 3);
    Q << 0.1, 0, 0,
            0, 0.1, 0,
            0, 0, 1 * M_PI / 180.0;
    Q = Q * Q.transpose();

    /* Covariance mat of observation noise */
    R.resize(2, 2);
    R << 1, 0,
            0, 1;
}

void EKF::predict_update(Eigen::MatrixXd &Xest, Eigen::MatrixXd &U, Eigen::MatrixXd &z) {
    Eigen::MatrixXd J_f(4, 4);

    J_f << 1, 0, -U(0, 0) * sin(Xest(2, 0)) * timestep(), cos(Xest(2, 0)) * timestep(),
            0, 1, U(0, 0) * cos(Xest(2, 0)) * timestep(), sin(Xest(2, 0)) * timestep(),
            0, 0, 1, 0,
            0, 0, 0, 1;

    step(U);
    Xest = eigen_state();


    Eigen::MatrixXd P_pred = J_f * P_t * J_f.transpose() + Q;


    Eigen::MatrixXd H(2, 4);
    H << 1, 0, 0, 0,
            0, 1, 0, 0;

    Eigen::MatrixXd z_pred = H * Xest;

    Eigen::MatrixXd y = z - z_pred;

    Eigen::MatrixXd S = J_g * P_pred * J_g.transpose() + R;

    Eigen::MatrixXd K = P_pred * J_g.transpose() * S.inverse();

    Xest = Xest + K * y;
    set_state(Xest);

    P_t = (Eigen::Matrix<double, 4, 4>::Identity() - K * J_g) * P_pred;
}

void EKF::RTS(Eigen::MatrixXd &Xest, Eigen::MatrixXd &XestPrev, Eigen::MatrixXd &Pprev) {
    Eigen::MatrixXd F(4, 4);
    F << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 0;

    Eigen::MatrixXd Cx = Pprev * F.transpose() * Pprev.transpose();

    XestPrev = XestPrev + Cx * (Xest - XestPrev);
    Pprev = Pprev + Cx * (P_t - Pprev) * Cx.transpose();
}















