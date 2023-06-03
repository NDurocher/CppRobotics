#include "ekf_slam.h"
#include <iostream>
#include <vector>

using namespace std;

EKFSLAM::EKFSLAM(double dt) : robot{0.0, 0.0, 0.0, 0.0, dt} {

		G.resize(state_size,state_size);

		/* Covariance matrix for x */
		P_t.resize(3,3);
		P_t << 1,0,0,
			   0,1,0,
			   0,0,1;

		/* Covariance mat for process noise */
	    Q.resize(2,2);
		Q << 0.1, 0,
			 0, 0.5*M_PI/180.0;
		Q = Q*Q.transpose();

		/* Covariance mat of observation noise */
		R.resize(2,2);
		R << 1.0, 0,
			 0, 10.0*M_PI/180.0;
		R = R*R.transpose();

		/* Covariance mat of relative landmark state */
		Cx.resize(3,3);
		Cx << 0.5, 0, 0,
			  0, 0.5, 0,
			  0, 0, 30.0*M_PI/180.0;
		Cx = Cx * Cx.transpose();
}

int EKFSLAM::CountLMs(Eigen::MatrixXd& Xest){
	double con = (Xest.rows() - state_size) / 2;
	return con > 0 ? int(con) : 0;
}

Eigen::MatrixXd EKFSLAM::Observation(Eigen::MatrixXd& Xtrue, Eigen::MatrixXd& LM_pos) {
	double Nd_LM = m_distribution(m_generator);
	double Na_LM = m_distribution(m_generator);

	// Z is the read measurements from range finder for LMs
	Eigen::MatrixXd Z(0,3);
	
	for (int i = 0; i < LM_pos.rows(); i++){
		// find LM's within max range, find distance to and angle from heading
		double dx = LM_pos(i,0) - Xtrue(0,0);
		double dy = LM_pos(i,1) - Xtrue(1,0);
		double distance = sqrt(pow(dx,2)+pow(dy,2));
		double angle = pi2pi(atan2(dy,dx) - Xtrue(2,0));
		// cout << angle << endl << endl;

		if (distance <= MAX_RANGE){
			int s = Z.rows()+1;
			Z.conservativeResize(s,state_size);
			Z(s-1,0) = distance + Nd_LM*Q(0,0);
			Z(s-1,1) = angle + Na_LM*Q(1,1);
			Z(s-1,2) = i;
		}
	}

	return Z;
}

std::vector<Eigen::MatrixXd> EKFSLAM::Motion_jacobian(Eigen::MatrixXd X, Eigen::MatrixXd U) {
	// count number of LMs
	int num_LM = CountLMs(X);
	
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_size, state_size);
	Eigen::MatrixXd matlm = Eigen::MatrixXd::Zero(state_size, num_LM*LM_size);
	Eigen::MatrixXd Fx(state_size, state_size + num_LM*LM_size);
	Fx << I, matlm;
	
	Eigen::MatrixXd J_f(state_size,state_size), G(state_size,state_size);
	J_f << 0,0,-U(0,0)*sin(X(2,0))*timestep(),
		   0,0,U(0,0)*cos(X(2,0))*timestep(),
		   0,0,0;

    G = Eigen::MatrixXd::Identity(state_size, state_size) + Fx.transpose() * J_f * Fx;
    std::vector<Eigen::MatrixXd> ret_vec;
	ret_vec.push_back(G);
    ret_vec.push_back(Fx);

	return ret_vec;
}

void EKFSLAM::Predict(Eigen::MatrixXd& Xest, Eigen::MatrixXd& U, Eigen::MatrixXd& z) {
	// G in example code (need to update to be size of state plus landmarks)
	std::vector<Eigen::MatrixXd> G_fx = Motion_jacobian(Xest.block(0,0,state_size,1), U);
	Eigen::MatrixXd Fx = G_fx.back(); 

	step(U);
	Xest.block(0,0,state_size,1) = eigen_state().block(0,0,state_size,1);
	
	P_t.block(0,0,state_size,state_size) = G*P_t.block(0,0,state_size,state_size)*G.transpose() + Fx.transpose() * Cx * Fx;
}

void EKFSLAM::Update(Eigen::MatrixXd& Xest, Eigen::MatrixXd& U, Eigen::MatrixXd& z_obs){
	Eigen::MatrixXd P_init = Eigen::MatrixXd::Identity(LM_size, LM_size);	
	
	// Determind which landmarks are being referenced, are they new?
	for (int ii = 0; ii < z_obs.rows(); ii++){
		int minid = search_lm_id(z_obs.block(ii,0,1,state_size), Xest);

		int nLM = CountLMs(Xest);

		if (minid == nLM) {
			cout << "New LM!" <<  endl;
			// Extend state and covariance matrix
			Eigen::MatrixXd xAug(Xest.rows() + LM_size, Xest.cols());
			Eigen::MatrixXd lm_pos = Calc_LM_pos(Xest, z_obs.block(ii,0,1,state_size));
			xAug << Xest,
					lm_pos;
			
			Eigen::MatrixXd PAug(P_t.rows() + LM_size, P_t.cols()+LM_size);
			Eigen::MatrixXd PAug_top(P_t.rows(), P_t.cols()+LM_size), PAug_bot(LM_size, P_t.cols() + LM_size);
			PAug_top << P_t, Eigen::MatrixXd::Zero(Xest.rows(), LM_size);
			PAug_bot << Eigen::MatrixXd::Zero(LM_size, Xest.rows()), P_init;
			PAug << PAug_top,
					PAug_bot;
			

			Xest = xAug;	
			P_t = PAug;	
		}
		else {
			// cout << "Min id: " << minid << " # of LMs: " << nLM << endl;
		}
		Eigen::MatrixXd lm = get_lm_pos_from_state(Xest, minid);
		
		std::vector<Eigen::MatrixXd> y_s_H = Innovation(lm, Xest, z_obs.block(ii,0,1,state_size), minid);
		Eigen::MatrixXd K = (P_t * y_s_H[2].transpose()) * y_s_H[1].inverse(); // Kalman gain
		Xest = Xest + (K * y_s_H[0]);
		P_t = (Eigen::MatrixXd::Identity(Xest.rows(),Xest.rows()) - (K * y_s_H[2])) * P_t;
	}

	Xest(2,0) = pi2pi(Xest(2,0));
}

Eigen::MatrixXd EKFSLAM::Calc_LM_pos(Eigen::MatrixXd& xest, Eigen::MatrixXd zi){
	
	Eigen::MatrixXd zp(LM_size, 1);
	zp << xest(0,0) + zi(0,0) * cos(-(xest(2,0) + zi(0,1))),
		  xest(1,0) + zi(0,0) * sin(xest(2,0) + zi(0,1));

	return zp;
}

std::vector<Eigen::MatrixXd> EKFSLAM::Innovation(Eigen::MatrixXd& est_lm_pos, Eigen::MatrixXd& xest, Eigen::MatrixXd zi_obs, int idx){

	Eigen::MatrixXd delta(2,1), zp(1,2);

	delta = est_lm_pos - xest.block(0,0,2,1);
	
	double q = (delta.transpose() * delta)(0,0);
	double zangle = atan2(delta(1,0), delta(0,0)) - xest(2,0);
	zp << sqrt(q), pi2pi(zangle);
	
	Eigen::MatrixXd y = (zi_obs.block(0,0,1,LM_size) - zp).transpose();
	
	y(1,0) = pi2pi(y(1,0));

	Eigen::MatrixXd H = H_jacob(q, delta, xest, idx+1);
	
	Eigen::MatrixXd S = H * P_t * H.transpose() + Cx.block(0,0,2,2);
	
	std::vector<Eigen::MatrixXd> y_s_H;
	y_s_H.push_back(y);
	y_s_H.push_back(S);
	y_s_H.push_back(H);

	return y_s_H;
}

Eigen::MatrixXd EKFSLAM::get_lm_pos_from_state(Eigen::MatrixXd Xest, int i){
	
	Eigen::MatrixXd lm(2,1);

	lm = Xest.block(state_size + LM_size *i,0, LM_size,1);

	return lm;
}

int EKFSLAM::search_lm_id(Eigen::MatrixXd zi_obs, Eigen::MatrixXd& xest){

	int num_LM = CountLMs(xest);
	std::vector<double> mdist;

	for (int ii = 0; ii < num_LM; ii++){
		// get lm position
		Eigen::MatrixXd est_lm_pos = get_lm_pos_from_state(xest, ii);
		std::vector<Eigen::MatrixXd> y_s_H = Innovation(est_lm_pos, xest, zi_obs, ii);

		Eigen::MatrixXd temp = y_s_H[0].transpose() * y_s_H[1].inverse() * y_s_H[0];
		// if (temp(0,0) > M_DIST_TH){
		// 	cout << "temp: " << temp(0,0) << endl;
		// }
		mdist.push_back(temp(0,0));
	}

	mdist.push_back(M_DIST_TH);

	auto minitr = min_element(begin(mdist),end(mdist));
	int minid = distance(mdist.begin(), minitr);

	return minid;
}

Eigen::MatrixXd EKFSLAM::H_jacob(double q, Eigen::MatrixXd& delta, Eigen::MatrixXd& xest, int i){
	double root_q = sqrt(q);
	Eigen::MatrixXd G_H(2,5);
	
	G_H << -root_q * delta(0,0), -root_q * delta(1,0), 0, root_q * delta(0,0), root_q * delta(1,0),
		 delta(1,0), -delta(0,0), -q, -delta(1,0), delta(0,0);
	
	G_H = G_H / q;
	int num_LM = CountLMs(xest);

	Eigen::MatrixXd F1(state_size, state_size + 2 * num_LM), F2(LM_size, state_size + 2 * num_LM);

	F1 << Eigen::MatrixXd::Identity(state_size, state_size), Eigen::MatrixXd::Zero(state_size, LM_size * num_LM);
	F2 << Eigen::MatrixXd::Zero(LM_size, state_size), Eigen::MatrixXd::Zero(LM_size, LM_size * (i-1)), Eigen::MatrixXd::Identity(LM_size, LM_size), Eigen::MatrixXd::Zero(LM_size, LM_size * num_LM - LM_size * i);
  	
  	Eigen::MatrixXd F(F1.rows()+F2.rows(), F1.cols());
  	F << F1,
  		 F2;

	Eigen::MatrixXd H;
	H = G_H * F;

	return H;
}

double pi2pi(double angle){
	double remain = fmod(angle + 3 * M_PI, 2 * M_PI);
	return remain - M_PI;
}









