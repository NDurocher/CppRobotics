#include "include/utils/robot.h"
#include "include/utils/my_time.h"

#include <Eigen/Dense>
#include <cmath>

using namespace std;

namespace diff_drive
{

    DifferentialRobot::DifferentialRobot(double init_x, double init_y, double init_w)
    {
        m_state.resize(3, 1);
        m_state << init_x,
            init_y,
            init_w;
    }

    void DifferentialRobot::move(double v, double omega, double timestep)
    {
        Eigen::MatrixXd F(3, 3), B(3, 2), U(2, 1);

        F << 1, 0, 0,
            0, 1, 0,
            0, 0, 1;

        U << v,
            omega;

        B << cos(m_state(2, 0)) * timestep, 0,
            sin(m_state(2, 0)) * timestep, 0,
            0, timestep;

        m_state = F * m_state + B * U;
    }

    std::vector<double> DifferentialRobot::getPosition()
    {
        return std::vector<double>{m_state(0, 0), m_state(1, 0), m_state(2, 0)};
    }

    Eigen::MatrixXd motion_model(Eigen::MatrixXd &X, Eigen::MatrixXd &U, double timestep)
    {
        Eigen::MatrixXd F(X.rows(), X.rows()), B(X.rows(), 2);

        if (X.rows() == 4)
        {
            F << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 0;

            B << cos(X(2, 0)) * timestep, 0,
                sin(X(2, 0)) * timestep, 0,
                0, timestep,
                1, 0;
        }
        else
        {
            F << 1, 0, 0,
                0, 1, 0,
                0, 0, 1;

            B << cos(X(2, 0)) * timestep, 0,
                sin(X(2, 0)) * timestep, 0,
                0, timestep;
        }

        X = F * X + B * U;

        return X;
    }

    Eigen::MatrixXd sample_gps(Eigen::MatrixXd &state, double pos_variance)
    {

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

    Eigen::MatrixXd corrupt_input(Eigen::MatrixXd &input, double velocity_variance, double heading_variance)
    {

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
