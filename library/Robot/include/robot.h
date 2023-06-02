#pragma once

#include <Eigen/Dense>
#include <random>

using namespace std;

class robot {
public:
	robot(double position_x, double position_y, double yaw, double velocity, double timestep);

	double yaw() const;
	double velocity() const;
	double position_x() const;
	double position_y() const;
	double timestep() const;
	int state_size() const;
	void step(Eigen::MatrixXd& U);	

private:
  	double m_position_x{0};
	double m_position_y{0};
	double m_yaw{0};
	double m_velocity{0};
	double m_timestep{0.5};
	int m_state_size{4};

};


class GPS{
public: 
	Eigen::MatrixXd sample(robot r);
	void corrupt_input(Eigen::MatrixXd& U, Eigen::MatrixXd& Un); 	

private:
	default_random_engine m_generator;
  	normal_distribution<double> m_distribution {0.0,1.0};

	double m_x_variance{0.25};
	double m_y_variance{0.25};
	double m_velocity_variance{1};
	double m_yaw_variance{pow(30/180*M_PI,2)};
};






