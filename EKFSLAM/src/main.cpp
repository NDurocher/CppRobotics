#include <cstdio>
#include <iostream>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>

#include "gnuplot-iostream.h"
#include "../include/EKFSLAM/robot.h"

using namespace std;

int main()
{	
	srand(time(0));
	
	Gnuplot gp;
	Gnuplot gp2;
	gp << "set xrange [-15:15]\nset yrange [-5:25]\n";
	gp2 << "set xrange [-15:15]\nset yrange [-5:25]\n";
	double SIM_TIME = 61.;
	double dt = 0.1;
	bool SHOW_PLOT = true;
	bool SHOW_EST_LM_POS_PLOT = false;

	robot rob(0.0, 0.0, 0.0, 0.0, dt);

	std::vector<double> Est_x;
	std::vector<double> Est_y;
	std::vector<double> Noisy_x;
	std::vector<double> Noisy_y;
	std::vector<double> True_x;
	std::vector<double> True_y;
	std::vector<double> LM_pos_vec_x;
	std::vector<double> LM_pos_vec_y;
	std::vector<double> Est_LM_pos_vec_x;
	std::vector<double> Est_LM_pos_vec_y;

	Eigen::MatrixXd LM_pos(4,2);
	LM_pos << 4,3,
			  3,15,
			  -12,10,
		   	  -5,20; 
	
	Eigen::MatrixXd U(2,1), Un(2,1);
	Eigen::MatrixXd Xest = Eigen::MatrixXd::Zero(rob.state_size,1);
	Eigen::MatrixXd Xtrue = Eigen::MatrixXd::Zero(rob.state_size,1);
	

	 U << 1.0,
	 	  0.1;

  	for (int i=0; i < LM_pos.rows(); i++){
  		LM_pos_vec_x.push_back(LM_pos(i,0));
  		LM_pos_vec_y.push_back(LM_pos(i,1));
  	}
  	
	for (double i = 0; i < SIM_TIME; i=i+dt)
	{	
		Eigen::MatrixXd Z_obs = rob.Observation(Xtrue, LM_pos);
		
		Xtrue = rob.Kinematics(Xtrue, U);
		Un = rob.Corrupt_input(U);

		rob.Predict(Xest, Un, Z_obs);
		rob.Update(Xest, Un, Z_obs);

		Est_x.push_back(Xest(0,0));
		Est_y.push_back(Xest(1,0));
	
		True_x.push_back(Xtrue(0,0));
		True_y.push_back(Xtrue(1,0));
		
		if (SHOW_PLOT) {

		gp << "plot" << gp.file1d(boost::make_tuple(Est_x, Est_y)) <<  "with lines title 'Estimated Position'," << gp.file1d(boost::make_tuple(True_x, True_y)) \
		 << "with lines title 'True Position'," << gp.file1d(boost::make_tuple(LM_pos_vec_x, LM_pos_vec_y)) << "with points title 'Landmark positions'\n";

	 	if (SHOW_EST_LM_POS_PLOT) {
			for (int ii = 0; ii < rob.CountLMs(Xest); ii++){
				Est_LM_pos_vec_x.push_back(Xest(rob.state_size + ii * 2,0));
				Est_LM_pos_vec_y.push_back(Xest(rob.state_size + ii * 2 + 1,0));
			}

			gp2 << "plot" << gp.file1d(boost::make_tuple(LM_pos_vec_x, LM_pos_vec_y)) << "with points title 'Landmark positions'," \
			 << gp.file1d(boost::make_tuple(Est_LM_pos_vec_x, Est_LM_pos_vec_y)) <<  "with points title 'Estimated Landmark positions'\n";
		}
		}
		
	}

	return 0;
}