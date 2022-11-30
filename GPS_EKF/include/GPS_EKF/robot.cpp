#include "robot.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using namespace std;

robot::robot(){
	position_x = 0.0;
	position_y = 0.0;
	heading = 0.0;
	velocity = 0.0;
	dt = 0.5;

	J_g.resize(2,4);
	J_g << 1,0,0,0,
		   0,1,0,0;

	/* Covariance matrix for x */
	P_t.resize(4,4);
	P_t << 1,0,0,0,
		   0,1,0,0,
		   0,0,1,0,
		   0,0,0,1;

	/* Covariance mat for process noise */
    Q.resize(4,4);
	Q << 0.1,0,0,0,
		 0,0.1,0,0,
		 0,0,1*M_PI/180.0,0,
		 0,0,0,1;
	Q = Q*Q.transpose();

	/* Covariance mat of observation noise */
	R.resize(2,2);
	R << 1,0,
		 0,1;
}

robot::robot(double x, double y, double phi, double v, double deltat) : robot(){
	position_x = x;
	position_y = y;
	heading = phi;
	velocity = v;
	dt = deltat;
}

double robot::get_heading() const{
	return heading;
}

double robot::get_velocity() const{
	return velocity;
}

double robot::get_position_x() const{
	return position_x;
}

double robot::get_position_y() const{
	return position_y;
}

double robot::get_delta_time() const {
	return dt;
}

void robot::GPS_reading(Eigen::MatrixXd& Xtrue, Eigen::MatrixXd& z) {
	// Function to get noisy GPS signal (z)
	double Nx = distribution(generator);
	double Ny = distribution(generator);

	z << Xtrue(0,0) + Nx*0.25,
		 Xtrue(1,0) + Ny*0.25;
}

void robot::Corrupt_input(Eigen::MatrixXd& U, Eigen::MatrixXd& Un){
	// Function to get noisy input signals
	double Nv = distribution(generator);
	double Nphi = distribution(generator);

	Un << U(0,0) + Nv*1,
		  U(1,0) + Nphi*pow(30/180*M_PI,2);
}

void robot::Kinematics(Eigen::MatrixXd& X, Eigen::MatrixXd& U){
	Eigen::MatrixXd F(4,4), B(4,2);
	F << 1,0,0,0,
		 0,1,0,0,
		 0,0,1,0,
		 0,0,0,0;

	B << cos(X(2,0))*dt,0,
		 sin(X(2,0))*dt,0,
		 0,dt,
		 1,0;

	X = F*X + B*U;

}

void robot::predict_update(Eigen::MatrixXd& Xest, Eigen::MatrixXd& U, Eigen::MatrixXd& z) {
	Eigen::MatrixXd J_f(4,4);

	// X << position_x,
	// 	 position_y,
	// 	 heading,
	// 	 velocity;

	J_f << 1,0,-U(0,0)*sin(Xest(2,0))*dt,cos(Xest(2,0))*dt,
		   0,1,U(0,0)*cos(Xest(2,0))*dt,sin(Xest(2,0))*dt,
		   0,0,1,0,
		   0,0,0,1;
	
	robot::Kinematics(Xest,U);


	Eigen::MatrixXd P_pred = J_f*P_t*J_f.transpose() + Q;


	Eigen::MatrixXd H(2,4);
	H << 1,0,0,0,
		 0,1,0,0;
	
	Eigen::MatrixXd z_pred = H*Xest;

	Eigen::MatrixXd y = z - z_pred;

	Eigen::MatrixXd S = J_g*P_pred*J_g.transpose() + R;

	Eigen::MatrixXd K = P_pred*J_g.transpose()*S.inverse();

	Xest = Xest + K*y;

	P_t = (Eigen::Matrix<double, 4, 4>::Identity()-K*J_g)*P_pred;

	position_x = Xest(0,0);
	position_y = Xest(1,0);
	heading = Xest(2,0);
	velocity = Xest(3,0);

	// return X;
}

void robot::RTS(Eigen::MatrixXd& Xest, Eigen::MatrixXd& XestPrev, Eigen::MatrixXd& Pprev){
	Eigen::MatrixXd F(4,4);
	F << 1,0,0,0,
		 0,1,0,0,
		 0,0,1,0,
		 0,0,0,0;

 	Eigen::MatrixXd Cx = Pprev*F.transpose()*Pprev.transpose();

	XestPrev = XestPrev + Cx*(Xest - XestPrev);
	Pprev = Pprev + Cx*(P_t - Pprev)*Cx.transpose();
}















