#pragma once

#include <Eigen/Dense>

#include "robot.h"

class EKF {
public:
	explicit EKF(robot robot);

	void predict_update(Eigen::MatrixXd& Xest, Eigen::MatrixXd& U, Eigen::MatrixXd& z);
	void RTS(Eigen::MatrixXd& Xest, Eigen::MatrixXd& XestPrev, Eigen::MatrixXd& Pprev);

private:
    robot m_robot;

	Eigen::MatrixXd J_g; // Jacobian for something TODO: figure that out
	Eigen::MatrixXd P_t; // Covariance matrix for X state
	Eigen::MatrixXd Q; // Covariance mat for process noise

 	Eigen::MatrixXd R; // Covariance mat of observation noise

};







