#include <cstdio>
#include <iostream>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>

#include "gnuplot-iostream.h"
// #include "EKF.h"
#include "robot.h"

using namespace std;

int main()
{	
	srand(time(0));
	robot rob;
	Gnuplot gp;
	Gnuplot gp2;
	gp << "set xrange [-15:15]\nset yrange [-5:25]\n";
	gp2 << "set xrange [-15:15]\nset yrange [-5:25]\n";
	double SIM_TIME = 40.;
	double dt = 0.1;
	bool SHOW_PLOT = true;

	rob.init_robot(0.0, 0.0, 0.0, 0.0, dt);
	
	std::vector<double> Est_x;
	std::vector<double> Est_y;
	std::vector<double> Noisy_x;
	std::vector<double> Noisy_y;
	std::vector<double> True_x;
	std::vector<double> True_y;
	std::vector<double> heading_t;
	std::vector<double> heading_y;
	std::vector<double> LM_pos_vec_x;
	std::vector<double> LM_pos_vec_y;
	std::vector<double> est_LM_pos_vec_x;
	std::vector<double> est_LM_pos_vec_y;
	// double time = 1.;


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
		Un = rob.Corrupt_input(U);
		
		Eigen::MatrixXd Z_obs = rob.Observation(Xtrue, LM_pos);
		
		Xtrue = rob.Kinematics(Xtrue, U);
		
		Xest = rob.Predict(Xest, Un, Z_obs);
		Xest = rob.Update(Un, Xest, Z_obs);

		Est_x.push_back(Xest(0,0));
		Est_y.push_back(Xest(1,0));
		
		// Noisy_x.push_back(Z_obs(0,0));
		// Noisy_y.push_back(Z_obs(0,1));
	
		True_x.push_back(Xtrue(0,0));
		True_y.push_back(Xtrue(1,0));
		
		// heading_y.push_back(rob.get_heading());
		// heading_t.push_back(time);
		if (SHOW_PLOT) {
		
		for (int ii = 0; ii < rob.CountLMs(Xest); ii++){
			est_LM_pos_vec_x.push_back(Xest(rob.state_size + ii * 2,0));
			est_LM_pos_vec_y.push_back(Xest(rob.state_size + ii * 2 + 1,0));
		}

		gp << "plot" << gp.file1d(boost::make_tuple(Est_x, Est_y)) <<  "with lines title 'Estimated Position'," << gp.file1d(boost::make_tuple(True_x, True_y)) \
		 << "with lines title 'True Position'," << gp.file1d(boost::make_tuple(LM_pos_vec_x, LM_pos_vec_y)) << "with points title 'Landmark positions'\n";

		gp2 << "plot" << gp.file1d(boost::make_tuple(LM_pos_vec_x, LM_pos_vec_y)) << "with points title 'Landmark positions'," \
		 << gp.file1d(boost::make_tuple(est_LM_pos_vec_x,est_LM_pos_vec_y)) <<  "with points title 'Estimated Landmark positions'\n";
	
		}
		// time = time + 1.;
		
	}

	return 0;
}