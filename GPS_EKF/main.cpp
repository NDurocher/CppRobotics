#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

#include "gnuplot-iostream.h"
#include "EKF.h"
#include "utils/robot.h"
#include "utils/my_time.h"

using namespace std;

int main() {

    srand(get_time());
    Gnuplot gp;
    // gp << "set xrange [-10:10]\nset yrange [0:20]\n";
    double SIM_TIME = 40;
    double dt = 0.1;
    double positional_variance{0.25};
    double velocity_variance{1.};
    double heading_variance{0.174533};
    int state_size{4};
    EKF ekf;

    std::vector<double> Est_x;
    std::vector<double> Est_y;
    std::vector<double> SM_Est_x;
    std::vector<double> SM_Est_y;
    std::vector<double> Noisy_x;
    std::vector<double> Noisy_y;
    std::vector<double> True_x;
    std::vector<double> True_y;

    Eigen::MatrixXd U(2, 1), Un(2, 1), z(2, 1);
    Eigen::MatrixXd Xest = Eigen::MatrixXd::Zero(state_size, 1);
    Eigen::MatrixXd Xtrue = Eigen::MatrixXd::Zero(state_size, 1);
    Eigen::MatrixXd XestPrev = Eigen::MatrixXd::Zero(state_size, 1);
    Eigen::MatrixXd Pprev = Eigen::MatrixXd::Identity(4, 4);

    U << 1.0,
            0.1;

    double time = 0;
    while (time < SIM_TIME) {
        if (time >= SIM_TIME / 2) {
            U(1, 0) = 0.5;
        }
        Xtrue = diff_drive::motion_model(Xtrue, U, dt);

        Un = diff_drive::corrupt_input(U, velocity_variance, heading_variance);

        Eigen::MatrixXd state_position = Xtrue.block(0,0,2,1);
        z = diff_drive::sample_gps(state_position, positional_variance);

        ekf.predict_update(Xest, Un, z, dt);

        Est_x.push_back(Xest(0, 0));
        Est_y.push_back(Xest(1, 0));

        Noisy_x.push_back(z(0, 0));
        Noisy_y.push_back(z(1, 0));

        True_x.push_back(Xtrue(0, 0));
        True_y.push_back(Xtrue(1, 0));

        ekf.RTS(Xest, XestPrev, Pprev);

        SM_Est_x.push_back(XestPrev(0, 0));
        SM_Est_y.push_back(XestPrev(1, 0));

        gp << "plot '-' with lines title 'Estimated Position', \
					'-' with lines title 'True Position', \
					'-' with points title 'GPS Readings', \
					'-' with lines title 'RTS Position'\n";

        gp.send1d(boost::make_tuple(Est_x, Est_y));
        gp.send1d(boost::make_tuple(True_x, True_y));
        gp.send1d(boost::make_tuple(Noisy_x, Noisy_y));
        gp.send1d(boost::make_tuple(SM_Est_x, SM_Est_y));

        time += dt;
    }

    return 0;
}