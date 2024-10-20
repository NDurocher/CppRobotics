#include "particle_filter.h"
#include "utils/my_time.h"
#include <numeric>

namespace cpp_rob
{
    double GaussianLikelihood(double diff, double noise)
    {
        return exp(-(diff * diff) / (2 * noise * noise)) / (sqrt(2 * M_PI) * noise);
    }

    Particle::Particle(double x_init, double y_init, double theta_init)
        : x(x_init), y(y_init), theta(theta_init) {}

    void Particle::noisy_move(double v, double omega, double delta_t)
    {
        v = GaussianNoise(v, input_std);
        omega = GaussianNoise(omega, input_std);

        x = v * cos(theta) * delta_t;
        y = v * sin(theta) * delta_t;
        theta = omega * delta_t;
    }

    ParticleFilter::ParticleFilter(int num_particles, double noise_x, double noise_y, double noise_theta, diff_drive::IRobot *robot)
        : num_particles(num_particles), noise_std_x(noise_x),
          noise_std_y(noise_y), noise_std_theta(noise_theta), m_robot(robot)
    {
        particles.reserve(num_particles);
        weights.resize(num_particles, 1.0 / num_particles); // Initialize uniform weights
    }

    void ParticleFilter::initialize(double x_range[], double y_range[], double theta_range[])
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> x_dist(x_range[0], x_range[1]);
        std::uniform_real_distribution<double> y_dist(y_range[0], y_range[1]);
        std::uniform_real_distribution<double> theta_dist(theta_range[0], theta_range[1]);

        for (int i = 0; i < num_particles; ++i)
        {
            double x = x_dist(generator);
            double y = y_dist(generator);
            double theta = theta_dist(generator);
            particles.emplace_back(x, y, theta);
        }
    }

    // Implement other methods (predict, updateWeights, resample)
    void ParticleFilter::predict(double delta_t, double v, double omega)
    {
        m_robot->move(v, omega, delta_t);
        for (auto particle : particles)
        {
            particle.noisy_move(v, omega, delta_t);
        }
    }

    std::vector<double> ParticleFilter::updateWeights(std::pair<double, double> measurement, double measurement_std)
    {
        std::vector<double> weights(particles.size(), 1); // Initialize weight
        double expected_measurement = hypot(measurement.first, measurement.second);
        auto robot_position = m_robot->getPosition();

        for (int ii = 0; ii < particles.size(); ii++)
        {
            std::vector<double> delta_position{robot_position[0] - particles[ii].x, robot_position[1] - particles[ii].y};

            double particle_measurement = hypot(measurement.first + delta_position[0], measurement.second + delta_position[1]);

            double diff = particle_measurement - expected_measurement;
            double measurement_likelihood = GaussianLikelihood(diff, measurement_std);

            weights[ii] *= measurement_likelihood; // Update weight
        }

        auto weigth_total = std::accumulate(weights.begin(), weights.end(), 0);
        for (auto w : weights)
        {
            w = w / weigth_total;
        }
        return weights;
    }

    void ParticleFilter::resample(std::vector<double> weights)
    {
        int N = particles.size();
        std::vector<Particle> resampledParticles;
        resampledParticles.reserve(N);

        // Create a cumulative distribution
        std::vector<double> cumulativeWeights(N);
        cumulativeWeights[0] = weights[0];
        for (int i = 1; i < N; ++i)
        {
            cumulativeWeights[i] = cumulativeWeights[i - 1] + weights[i];
        }

        // Systematic resampling
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (int i = 0; i < N; ++i)
        {
            double u = distribution(generator); // Sample a random number in [0, 1)
            // Find the index of the first cumulative weight greater than or equal to u
            auto it = std::lower_bound(cumulativeWeights.begin(), cumulativeWeights.end(), u);
            int index = std::distance(cumulativeWeights.begin(), it);
            
            // Generate new (x, y) values based on the selected particle
            std::normal_distribution<double> x_dist(particles[index].x, m_std_dev);
            std::normal_distribution<double> y_dist(particles[index].y, m_std_dev);
            std::normal_distribution<double> theta_dist(particles[index].theta, m_std_dev);

            double newX = x_dist(generator);
            double newY = y_dist(generator);
            double newTheta = theta_dist(generator);
            
            resampledParticles.push_back(Particle(newX, newY, newTheta));
        }

        particles = resampledParticles;
    }

    void ParticleFilter::run(double delta_t, double v, double omega, Obstacle landmark, double measurement_std)
    {
        predict(delta_t, v, omega);
        auto landmark_pos = landmark.get_position();
        auto robot_pos = m_robot->getPosition();
        std::pair<double, double> measurement(robot_pos[0] - landmark_pos.first, robot_pos[1] - landmark_pos.second);
        auto weights = updateWeights(measurement, measurement_std);
        resample(weights);
    }

    std::vector<Particle> ParticleFilter::get_particles()
    {
        return particles;
    }

} // namespace ccp_rob