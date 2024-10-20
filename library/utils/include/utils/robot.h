#pragma once

#include "my_time.h"

#include <Eigen/Dense>
#include <random>
#include <cmath>

using namespace std;

namespace diff_drive
{

    class IRobot
    {
    public:
        virtual void move(double v, double omega, double timestep) = 0;
        virtual std::vector<double> getPosition() = 0;
    };

    class DifferentialRobot : public IRobot
    {
    public:
        DifferentialRobot(double init_x, double init_y, double init_w);

        void move(double v, double omega, double timestep) final;
        std::vector<double> getPosition() final;

    private: 
        Eigen::MatrixXd m_state;
    };

    Eigen::MatrixXd motion_model(Eigen::MatrixXd &X, Eigen::MatrixXd &U, double timestep);

    Eigen::MatrixXd sample_gps(Eigen::MatrixXd &state, double pos_variance);

    Eigen::MatrixXd corrupt_input(Eigen::MatrixXd &input, double velocity_variance, double heading_variance);

} // namespace diff_drive
