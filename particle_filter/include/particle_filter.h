#pragma once

#include <vector>
#include <utility>
#include "utils/robot.h"
#include "utils/obstacle.h"

namespace cpp_rob {
    class Particle {
    public:
        Particle(double x_init, double y_init, double theta_init);

        void noisy_move(double v, double omega, double delta_t);

        double x, y, theta; // State variables
        double input_std{0.25};
    };

    class ParticleFilter {
    public:
        // Constructor
        ParticleFilter(int num_particles, double noise_x, double noise_y, double noise_theta,
                       diff_drive::IRobot *robot);

        void initialize(double x_range[], double y_range[], double theta_range[]);

        void predict(double delta_t, double v, double omega);

        std::vector<double> updateWeights(std::pair<double, double> measurement, double measurement_std);

        void resample(std::vector<double> weights);

        void run(double delta_t, double v, double omega, Obstacle landmark, double measurement_std);

        std::vector<Particle> get_particles();

    private:
        std::vector<Particle> particles; // The particles
        std::vector<double> weights; // Weights for each particle
        int num_particles;
        double m_std_dev{0.5};

        // Parameters
        double noise_std_x, noise_std_y, noise_std_theta;
        diff_drive::IRobot *m_robot;
    };
} // namespace ccp_rob
