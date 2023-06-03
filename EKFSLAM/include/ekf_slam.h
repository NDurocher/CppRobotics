#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "robot.h"

using namespace std;

double pi2pi(double angle);

// Update to inherit robot from library
class EKFSLAM
{
public:
	EKFSLAM();

	double MAX_RANGE = 15.; // meters

	int state_size = 3;

	int LM_size = 2;

	double M_DIST_TH = 2.0; // mahalanobis distance threshold for new landmark

  	int CountLMs(Eigen::MatrixXd& Xest);

	void Predict(Eigen::MatrixXd& Xest, Eigen::MatrixXd& U, Eigen::MatrixXd& z);

	void Update(Eigen::MatrixXd& Xest, Eigen::MatrixXd& U, Eigen::MatrixXd& z_obs);

	Eigen::MatrixXd Observation(Eigen::MatrixXd& Xtrue, Eigen::MatrixXd& LM_pos); 

	std::vector<Eigen::MatrixXd> Motion_jacobian(Eigen::MatrixXd X, Eigen::MatrixXd U);	

	int search_lm_id(Eigen::MatrixXd zi_obs, Eigen::MatrixXd& xest);

	Eigen::MatrixXd get_lm_pos_from_state(Eigen::MatrixXd Xest, int i);

	Eigen::MatrixXd Calc_LM_pos(Eigen::MatrixXd& xest, Eigen::MatrixXd zi);
	
	std::vector<Eigen::MatrixXd> Innovation(Eigen::MatrixXd& est_lm_pos, Eigen::MatrixXd& xest, Eigen::MatrixXd zi_obs, int idx);

	Eigen::MatrixXd H_jacob(double q, Eigen::MatrixXd& delta, Eigen::MatrixXd& xest, int i);

private:
	Eigen::MatrixXd P_t; // Covariance matrix for x
	Eigen::MatrixXd G;
	Eigen::MatrixXd Cx; // Covariance mat for EKF state
	Eigen::MatrixXd Q; // Covariance mat for sensor noise
 	Eigen::MatrixXd R; /* Covariance mat of observation noise */
};






