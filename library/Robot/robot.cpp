#include "robot.h"
#include <Eigen/Dense>
#include <cmath>

using namespace std;

robot::robot(double position_x, double position_y, double yaw, double velocity, double timestep) : m_position_x{position_x}, \
m_position_y{position_y}, m_yaw{yaw}, m_velocity{velocity}, m_timestep{timestep} {}

double robot::yaw() const{
	return m_yaw;
}

double robot::velocity() const{
	return m_velocity;
}

double robot::position_x() const{
	return m_position_x;
}

double robot::position_y() const{
	return m_position_y;
}

double robot::timestep() const {
	return m_timestep;
}

int robot::state_size() const {
	return m_state_size;
}

void robot::step(Eigen::MatrixXd& U){
	Eigen::MatrixXd F(4,4), B(4,2), X(4,4);
	F << 1,0,0,0,
		 0,1,0,0,
		 0,0,1,0,
		 0,0,0,0;

	B << cos(X(2,0))*m_timestep,0,
		 sin(X(2,0))*m_timestep,0,
		 0,m_timestep,
		 1,0;

	X << m_position_x,0,0,0,
		 0,m_position_y,0,0,
		 0,0,m_yaw,0,0,
		 0,0,0,m_velocity;

	X = F*X + B*U;

	m_position_x = X(1,1);
	m_position_y = X(2,2);
	m_yaw = X(3,3);
	m_velocity = X(4,4);
}

///////////////////////////////

Eigen::MatrixXd GPS::sample(robot r){
	
	// Function to get noisy GPS sample in a Matrix
	double Nx = m_distribution(m_generator);
	double Ny = m_distribution(m_generator);

	Eigen::MatrixXd sample_position(2,1);

	sample_position << r.position_x() + Nx*m_x_variance,
					   r.position_y() + Ny*m_y_variance;

    return sample_position;
}

void GPS::corrupt_input(Eigen::MatrixXd& U, Eigen::MatrixXd& Un){
	
	// Function to get noisy input signals
	double Nv = m_distribution(m_generator);
	double Nphi = m_distribution(m_generator);

	Un << U(0,0) + Nv*m_velocity_variance,
		  U(1,0) + Nphi*m_yaw_variance;
}















