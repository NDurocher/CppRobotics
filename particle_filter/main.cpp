#include "particle_filter.h"
#include "utils/robot.h"
#include "utils/my_time.h"
#include "utils/obstacle.h"

#include <opencv2/opencv.hpp>

#include <vector>

void plotParticles(const std::vector<cpp_rob::Particle> &particles, cv::Mat &image, diff_drive::IRobot &robot) {
    // Clear the image
    image.setTo(cv::Scalar(255, 255, 255)); // White background

    cv::circle(image, cv::Point(static_cast<int>(robot.getPosition()[0] * 10),
                                static_cast<int>(robot.getPosition()[1] * 10)), 10, cv::Scalar(255, 0, 255),
               -1); // Blue

    for (const auto &particle: particles) {
        // Draw each particle as a circle
        cv::circle(image, cv::Point(static_cast<int>(particle.x * 10), static_cast<int>(particle.y * 10)), 5,
                   cv::Scalar(0, 0, 255), -1); // Red
    }

    // Show the image in a window
    cv::imshow("Particle Filter Simulation", image);
    cv::waitKey(1); // Wait for a short time (non-blocking)
}


int main() {
    srand(get_time());
    cv::Mat image(1000, 1000, CV_8UC3);
    double SIM_TIME = 40;
    double delta_t = 0.1;
    double positional_variance{0.25};
    double velocity{1.};
    double heading_variance{0.174533};
    double heading{0.125};

    double x_range[] = {0, 10};
    double y_range[] = {0, 10};
    double theta_range[] = {0, 2 * M_PI};

    double measurement_std = 0.2; // measurement noise
    double measurement = 5; // TODO some measurement

    diff_drive::DifferentialRobot robot(0, 0, heading);
    cpp_rob::ParticleFilter pf(100, positional_variance, positional_variance, heading_variance, &robot);
    pf.initialize(x_range, y_range, theta_range);

    Obstacle landmark(6.9, 3.2);

    for (int i = 0; i < 100; ++i) {
        pf.run(delta_t, velocity, heading, landmark, measurement_std);
        // Output results or perform additional tasks
        plotParticles(pf.get_particles(), image, robot);
    }
}
