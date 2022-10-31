#include "robot.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>


using namespace std;


void robot::init_robot(double x, double y, double phi, double v, double deltat){
	position_x = x;
	position_y = y;
	heading = phi;
	velocity = v;
	dt = deltat;

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

Eigen::MatrixXd robot::GPS_reading(Eigen::MatrixXd Xtrue) {
	// Function to get noisy GPS signal (z)
	double Nx = distribution(generator);
	double Ny = distribution(generator);

	Eigen::MatrixXd z(2,1);
	z << Xtrue(0,0) + Nx*0.25,
		 Xtrue(1,0) + Ny*0.25;
	return z;
}

Eigen::MatrixXd robot::Corrupt_input(Eigen::MatrixXd U){
	// Function to get noisy input signals
	double Nv = distribution(generator);
	double Nphi = distribution(generator);
	Eigen::MatrixXd Un(2,1);
	Un << U(0,0) + Nv*1,
		  U(1,0) + Nphi*pow(30/180*M_PI,2);
	return Un;
}

Eigen::MatrixXd robot::Kinematics(Eigen::MatrixXd X, Eigen::MatrixXd U){
	Eigen::MatrixXd F(4,4), B(4,2), state_pred(4,1);
	F << 1,0,0,0,
		 0,1,0,0,
		 0,0,1,0,
		 0,0,0,0;

	B << cos(X(2,0))*dt,0,
		 sin(X(2,0))*dt,0,
		 0,dt,
		 1,0;

	state_pred = F*X + B*U;

	return state_pred;
}

Eigen::MatrixXd robot::predict_update(Eigen::MatrixXd U, Eigen::MatrixXd z) {
	Eigen::MatrixXd X(4,1), X_pred(4,1), J_f(4,4);

	X << position_x,
		 position_y,
		 heading,
		 velocity;

	J_f << 1,0,-U(0,0)*sin(X(2,0))*dt,cos(X(2,0))*dt,
		   0,1,U(0,0)*cos(X(2,0))*dt,sin(X(2,0))*dt,
		   0,0,1,0,
		   0,0,0,1;
	
	X_pred = robot::Kinematics(X,U);


	Eigen::MatrixXd P_pred = J_f*P_t*J_f.transpose() + Q;


	Eigen::MatrixXd H(2,4);
	H << 1,0,0,0,
		 0,1,0,0;
	
	Eigen::MatrixXd z_pred = H*X_pred;

	Eigen::MatrixXd y = z - z_pred;

	Eigen::MatrixXd S = J_g*P_pred*J_g.transpose() + R;

	Eigen::MatrixXd K = P_pred*J_g.transpose()*S.inverse();

	X = X_pred + K*y;

	P_t = (Eigen::Matrix<double, 4, 4>::Identity()-K*J_g)*P_pred;

	position_x = X(0,0);
	position_y = X(1,0);
	heading = X(2,0);
	velocity = X(3,0);

	return X;
}
















