#pragma once

#include <chrono>

uint64_t get_time();

// Function to generate Gaussian (normal) distributed noise
double GaussianNoise(double mean, double std_deviation);