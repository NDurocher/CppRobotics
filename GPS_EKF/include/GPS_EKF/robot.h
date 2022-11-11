#ifndef ROBOT_H_
#define ROBOT_H_

#include <Eigen/Dense>
#include <random>

using namespace std;

class robot {
public:
	robot();
	robot(double x, double y, double phi, double v, double deltat);
	// ~robot();

	int state_size = 4;

	double get_heading() const;

	double get_velocity() const;

	double get_position_x() const;

	double get_position_y() const;

	double get_delta_time() const;

	default_random_engine generator;
  	normal_distribution<double> distribution {0.0,1.0};

	void predict_update(Eigen::MatrixXd& Xest, Eigen::MatrixXd& U, Eigen::MatrixXd& z);

	void GPS_reading(Eigen::MatrixXd& Xtrue, Eigen::MatrixXd& z); 

	void Corrupt_input(Eigen::MatrixXd& U, Eigen::MatrixXd& Un); 	

	void Kinematics(Eigen::MatrixXd& X, Eigen::MatrixXd& U);	


private:
	double position_y;

	double position_x;

	double heading;

	double velocity;

	double dt;

	Eigen::MatrixXd J_g;

	/* Covariance matrix for x */
	Eigen::MatrixXd P_t;

	/* Covariance mat for process noise */
	Eigen::MatrixXd Q;

	/* Covariance mat of observation noise */
 	Eigen::MatrixXd R;

};

#endif







