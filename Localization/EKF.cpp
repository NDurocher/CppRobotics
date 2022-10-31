#include <cstdio>
#include <iostream>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>

#include "gnuplot-iostream/gnuplot-iostream.h"
// #include "EKF.h"
#include "robot.h"

using namespace std;

int main()
{	
	srand(time(0));
	robot rob;
	Gnuplot gp;
	// gp << "set xrange [-10:10]\nset yrange [0:20]\n";
	double SIM_TIME = 63;
	double dt = 0.1;
	rob.init_robot(0.0, 0.0, 0.0, 0.0, dt);
	
	std::vector<double> Est_x;
	std::vector<double> Est_y;
	std::vector<double> Noisy_x;
	std::vector<double> Noisy_y;
	std::vector<double> True_x;
	std::vector<double> True_y;
	std::vector<double> heading_t;
	std::vector<double> heading_y;
	double time = 1.;
	
	Eigen::MatrixXd Xest(4,1), Xtrue(4,1), U(2,1), Un(2,1);

	Xtrue << 0,
			 0,
			 0,
			 0;

	 U << 1.0,
	 	  0.1;

	for (double i = 0; i < SIM_TIME; i=i+dt)
	{
		Xtrue = rob.Kinematics(Xtrue, U);

		Un = rob.Corrupt_input(U);
		Eigen::MatrixXd z = rob.GPS_reading(Xtrue);

		Xest = rob.predict_update(Un, z);

		Est_x.push_back(Xest(0,0));
		Est_y.push_back(Xest(1,0));

		Noisy_x.push_back(z(0,0));
		Noisy_y.push_back(z(1,0));

		True_x.push_back(Xtrue(0,0));
		True_y.push_back(Xtrue(1,0));
		
		heading_y.push_back(rob.get_heading());
		heading_t.push_back(time);
		
		gp << "plot '-' with lines title 'Estimated Position', '-' with lines title 'True Position', '-' with points title 'GPS Readings'\n";
		// gp << "plot '-' with steps title 'Estimated Heading'\n";

		gp.send1d(boost::make_tuple(Est_x, Est_y));
		gp.send1d(boost::make_tuple(True_x, True_y));
		gp.send1d(boost::make_tuple(Noisy_x, Noisy_y));
		// gp.send1d(boost::make_tuple(heading_t, heading_y));
		
		time = time + 1.;
		
	}

	return 0;
}