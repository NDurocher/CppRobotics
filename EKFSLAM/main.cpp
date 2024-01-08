#include <iostream>

#include "gnuplot-iostream.h"
#include "ekf_slam.h"
#include "robot.h"
#include "my_time.h"

using namespace std;

int main() {
    srand(get_time());

    Gnuplot gp;
    Gnuplot gp2;
    gp << "set xrange [-15:15]\nset yrange [-5:25]\n";
    gp2 << "set xrange [-15:15]\nset yrange [-5:25]\n";
    double SIM_TIME = 61.;
    double velocity_variance{1.};
    double heading_variance{0.174533};
    bool SHOW_PLOT = true;
    bool SHOW_EST_LM_POS_PLOT = false;

    SlamVariables slam_vars;
    SimVariables sim_vars;

    std::vector<double> Est_x;
    std::vector<double> Est_y;
    std::vector<double> Noisy_x;
    std::vector<double> Noisy_y;
    std::vector<double> True_x;
    std::vector<double> True_y;
    std::vector<double> LM_pos_vec_x;
    std::vector<double> LM_pos_vec_y;
    std::vector<double> Est_LM_pos_vec_x;
    std::vector<double> Est_LM_pos_vec_y;

    Eigen::MatrixXd LM_pos(4, 2);
    LM_pos << 10.0, -2.0,
            15.0, 10.0,
            3.0, 15.0,
            -5.0, 20.0;

    Eigen::MatrixXd U(2, 1), Un(2, 1);
    Eigen::MatrixXd Xest = Eigen::MatrixXd::Zero(sim_vars.state_size, 1);
    Eigen::MatrixXd Xtrue = Eigen::MatrixXd::Zero(sim_vars.state_size, 1);


    U << 1.0,
            0.1;

    for (int i = 0; i < LM_pos.rows(); i++) {
        LM_pos_vec_x.push_back(LM_pos(i, 0));
        LM_pos_vec_y.push_back(LM_pos(i, 1));
    }

    double time = 0.;
    while (time < SIM_TIME) {

        Eigen::MatrixXd Z_obs = Observation(U, Xtrue, LM_pos, slam_vars, sim_vars);
        Un = diff_drive::corrupt_input(U, velocity_variance, heading_variance);

        Predict(Xest, Un, slam_vars, sim_vars);
        Update(Xest, Z_obs, slam_vars, sim_vars);

        Est_x.push_back(Xest(0, 0));
        Est_y.push_back(Xest(1, 0));

        True_x.push_back(Xtrue(0, 0));
        True_y.push_back(Xtrue(1, 0));

        if (SHOW_PLOT) {

            gp << "plot" << gp.file1d(boost::make_tuple(Est_x, Est_y)) << "with lines title 'Estimated Position',"
               << gp.file1d(boost::make_tuple(True_x, True_y)) \
 << "with lines title 'True Position'," << gp.file1d(boost::make_tuple(LM_pos_vec_x, LM_pos_vec_y))
               << "with points title 'Landmark positions'\n";

            if (SHOW_EST_LM_POS_PLOT) {
                for (int ii = 0; ii < CountLMs(Xest, sim_vars); ii++) {
                    Est_LM_pos_vec_x.push_back(Xest(sim_vars.state_size + ii * 2, 0));
                    Est_LM_pos_vec_y.push_back(Xest(sim_vars.state_size + ii * 2 + 1, 0));
                }

                gp2 << "plot" << gp.file1d(boost::make_tuple(LM_pos_vec_x, LM_pos_vec_y))
                    << "with points title 'Landmark positions'," \
 << gp.file1d(boost::make_tuple(Est_LM_pos_vec_x, Est_LM_pos_vec_y))
                    << "with points title 'Estimated Landmark positions'\n";
            }
        }

        time += sim_vars.dt;
    }

    return 0;
}