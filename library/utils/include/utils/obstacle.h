#pragma once
#include <utility>

class Obstacle
{
public:
    Obstacle(double x, double y);

    std::pair<double, double> get_position();

private:
    std::pair<double, double> m_position{};
};