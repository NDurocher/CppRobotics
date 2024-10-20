#include "include/utils/my_time.h"
#include <random>
#include <cmath>

uint64_t get_time() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}


// Function to generate Gaussian (normal) distributed noise
double GaussianNoise(double mean, double std_deviation) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, std_deviation);
    return dist(gen);
}