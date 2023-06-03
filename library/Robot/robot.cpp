#include "robot.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

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
	Eigen::MatrixXd F(4,4), B(4,2), X(4,1);
	F << 1,0,0,0,
		 0,1,0,0,
		 0,0,1,0,
		 0,0,0,0;

	X = eigen_state();

	B << cos(X(2,0))*m_timestep,0,
		 sin(X(2,0))*m_timestep,0,
		 0,m_timestep,
		 1,0;
	

	X = F*X + B*U;

	m_position_x = X(0,0);
	m_position_y = X(1,0);
	m_yaw = X(2,0);
	m_velocity = X(3,0);
}

Eigen::MatrixXd robot::eigen_state() {
	Eigen::MatrixXd state(4,1);
	state << m_position_x,
			 m_position_y,
			 m_yaw,
			 m_velocity;

	return state;
}

void robot::set_state(Eigen::MatrixXd& state_est) {
	m_position_x = state_est(0,0);
	m_position_y = state_est(1,0);
	m_yaw = state_est(2,0);
	m_velocity = state_est(3,0);
}

///////////////////////////////

Eigen::MatrixXd GPS::sample(robot& r){
	
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















