#ifndef ROBOT_H_
#define ROBOT_H_

#include <Eigen/Dense>
#include <random>

using namespace std;

double pi2pi(double angle);

class robot
{
public:
	// robot();
	// ~robot();

	double MAX_RANGE = 20.; // meters

	int state_size = 3;

	int LM_size = 2;

	double M_DIST_TH = 2.0; // mahalanobis distance threshold for new landmark
	
	void init_robot(double x, double y, double phi, double v, double deltat);

	double get_heading() const;

	double get_velocity() const;

	double get_position_x() const;

	double get_position_y() const;

	double get_delta_time() const;

	default_random_engine generator;
  	normal_distribution<double> distribution {0.0,1.0};

  	int CountLMs(Eigen::MatrixXd Xest);

	Eigen::MatrixXd Predict(Eigen::MatrixXd Xest, Eigen::MatrixXd U, Eigen::MatrixXd z);

	Eigen::MatrixXd Update(Eigen::MatrixXd U, Eigen::MatrixXd Xest, Eigen::MatrixXd z_obs);

	Eigen::MatrixXd Observation(Eigen::MatrixXd Xtrue, Eigen::MatrixXd LM_pos); 

	Eigen::MatrixXd Corrupt_input(Eigen::MatrixXd U); 	

	Eigen::MatrixXd Kinematics(Eigen::MatrixXd X, Eigen::MatrixXd U);	

	std::vector<Eigen::MatrixXd> Motion_jacobian(Eigen::MatrixXd X, Eigen::MatrixXd U);	

	int search_lm_id(Eigen::MatrixXd zi_obs, Eigen::MatrixXd xest);

	Eigen::MatrixXd get_lm_pos_from_state(Eigen::MatrixXd Xest, int i);

	Eigen::MatrixXd Calc_LM_pos(Eigen::MatrixXd xest, Eigen::MatrixXd zi);
	
	std::vector<Eigen::MatrixXd> Innovation(Eigen::MatrixXd est_lm_pos, Eigen::MatrixXd xest, Eigen::MatrixXd zi_obs, int idx);

	Eigen::MatrixXd H_jacob(double q, Eigen::MatrixXd delta, Eigen::MatrixXd xest, int i);

	// Eigen::MatrixXd GetLandmarks(Eigen::MatrixXd X);

private:
	double position_y;

	double position_x;

	double heading;

	double velocity;

	double dt;

	/* Covariance matrix for x */
	Eigen::MatrixXd P_t;

	Eigen::MatrixXd G;

	// Eigen::MatrixXd Fx;

	/* Covariance mat for EKF state */
	Eigen::MatrixXd Cx;

	/* Covariance mat for sensor noise */
	Eigen::MatrixXd Q;

	/* Covariance mat of observation noise */
 	Eigen::MatrixXd R;

	// Eigen::MatrixXd(state_size,state_size) P_pred;

};

#endif







