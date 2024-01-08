#include "robot.h"

#include <Eigen/Dense>
#include <cmath>

#include "my_time.h"

using namespace std;

namespace diff_drive {

    Eigen::MatrixXd motion_model(Eigen::MatrixXd &X, Eigen::MatrixXd &U, double timestep) {

        Eigen::MatrixXd F(4, 4), B(4, 2);
        F << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 0;

        B << cos(X(2, 0)) * timestep, 0,
                sin(X(2, 0)) * timestep, 0,
                0, timestep,
                1, 0;

        X = F * X + B * U;

        return X;
    }


    Eigen::MatrixXd sample_gps(Eigen::MatrixXd &state, double pos_variance) {

        default_random_engine m_generator(get_time());
        normal_distribution<double> m_distribution{0.0, 1.0};

        // Function to get noisy GPS sample in a Matrix
        double Nx = m_distribution(m_generator);
        double Ny = m_distribution(m_generator);

        Eigen::MatrixXd noisy_state(2, 1);

        noisy_state << state(0, 0) + Nx * pos_variance,
                state(1, 0) + Ny * pos_variance;

        return noisy_state;
    }

    Eigen::MatrixXd corrupt_input(Eigen::MatrixXd &input, double velocity_variance, double heading_variance) {

        default_random_engine m_generator(get_time());
        normal_distribution<double> m_distribution{0.0, 1.0};

        // Function to get noisy input signals
        double Nv = m_distribution(m_generator);
        double Nphi = m_distribution(m_generator);

        Eigen::MatrixXd noisy_input(2, 1);

        noisy_input << input(0, 0) + Nv * velocity_variance,
                input(1, 0) + Nphi * heading_variance;

        return noisy_input;
    }

}












