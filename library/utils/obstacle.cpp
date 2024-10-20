#include "include/utils/obstacle.h"

Obstacle::Obstacle(double x, double y) : m_position{std::pair<double,double>{x, y}} {}

std::pair<double, double> Obstacle::get_position()
{
    return m_position;
}