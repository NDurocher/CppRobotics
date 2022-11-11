#include <cstdio>
#include <iostream>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>

#include "gnuplot-iostream.h"
#include "../include/GPS_EKF/robot.h"

using namespace std;

int main() {
	
	srand(time(0));
	Gnuplot gp;
	// gp << "set xrange [-10:10]\nset yrange [0:20]\n";
	double SIM_TIME = 63;
	double dt = 0.1;
	robot rob(0.0, 0.0, 0.0, 0.0, dt);
	
	std::vector<double> Est_x;
	std::vector<double> Est_y;
	std::vector<double> Noisy_x;
	std::vector<double> Noisy_y;
	std::vector<double> True_x;
	std::vector<double> True_y;
	
	Eigen::MatrixXd U(2,1), Un(2,1), z(2,1);;
	Eigen::MatrixXd Xest = Eigen::MatrixXd::Zero(rob.state_size,1);
	Eigen::MatrixXd Xtrue = Eigen::MatrixXd::Zero(rob.state_size,1);

	 U << 1.0,
	 	  0.1;

	for (double i = 0; i < SIM_TIME; i=i+dt)
	{
		rob.Kinematics(Xtrue, U);

		rob.Corrupt_input(U, Un);
		rob.GPS_reading(Xtrue, z);

		rob.predict_update(Xest, Un, z);

		Est_x.push_back(Xest(0,0));
		Est_y.push_back(Xest(1,0));

		Noisy_x.push_back(z(0,0));
		Noisy_y.push_back(z(1,0));

		True_x.push_back(Xtrue(0,0));
		True_y.push_back(Xtrue(1,0));
		
		gp << "plot '-' with lines title 'Estimated Position', '-' with lines title 'True Position', '-' with points title 'GPS Readings'\n";

		gp.send1d(boost::make_tuple(Est_x, Est_y));
		gp.send1d(boost::make_tuple(True_x, True_y));
		gp.send1d(boost::make_tuple(Noisy_x, Noisy_y));
		
	}

	return 0;
}