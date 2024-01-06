#include "ekf_slam.h"
#include <iostream>
#include <vector>
#include <random>


int CountLMs(Eigen::MatrixXd &Xest, const SimVariables &sim_vars) {
    double con = (Xest.rows() - sim_vars.state_size) / 2;
    return con > 0 ? int(con) : 0;
}

Eigen::MatrixXd Observation(robot rob, Eigen::MatrixXd &U, Eigen::MatrixXd &Xtrue, Eigen::MatrixXd &LM_pos,
                            const SlamVariables &slam_vars, const SimVariables &sim_vars) {
    Xtrue = rob.motion_model(Xtrue, U);

    default_random_engine m_generator{static_cast<unsigned int>(get_time())};
    normal_distribution<double> m_distribution{0.0, 1.0};

    double Nd_LM = m_distribution(m_generator);
    double Na_LM = m_distribution(m_generator);

    // Z is the read measurements from range finder for LMs
    Eigen::MatrixXd Z(0, 3);

    for (int i = 0; i < LM_pos.rows(); i++) {
        // find LM's within max range, find distance to and angle from heading
        double dx = LM_pos(i, 0) - Xtrue(0, 0);
        double dy = LM_pos(i, 1) - Xtrue(1, 0);
        double distance = sqrt(pow(dx, 2) + pow(dy, 2));
        double angle = pi2pi(atan2(dy, dx) - Xtrue(2, 0));

        if (distance <= slam_vars.MAX_RANGE) {
            int s = Z.rows() + 1;
            Z.conservativeResize(s, sim_vars.state_size);
            Z(s - 1, 0) = distance + Nd_LM * slam_vars.Q(0, 0);
            Z(s - 1, 1) = angle + Na_LM * slam_vars.Q(1, 1);
            Z(s - 1, 2) = i;
        }
    }

    return Z;
}

std::vector<Eigen::MatrixXd> Motion_jacobian(Eigen::MatrixXd X, Eigen::MatrixXd U, const SimVariables &sim_vars) {
    // count number of LMs
    int num_LM = CountLMs(X, sim_vars);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(sim_vars.state_size, sim_vars.state_size);
    Eigen::MatrixXd matlm = Eigen::MatrixXd::Zero(sim_vars.state_size, num_LM * sim_vars.LM_size);
    Eigen::MatrixXd Fx(sim_vars.state_size, sim_vars.state_size + num_LM * sim_vars.LM_size);
    Fx << I, matlm;

    Eigen::MatrixXd J_f(sim_vars.state_size, sim_vars.state_size), G(sim_vars.state_size, sim_vars.state_size);
    J_f << 0, 0, -U(0, 0) * sin(X(2, 0)) * sim_vars.dt,
            0, 0, U(0, 0) * cos(X(2, 0)) * sim_vars.dt,
            0, 0, 0;

    G = Eigen::MatrixXd::Identity(sim_vars.state_size, sim_vars.state_size) + Fx.transpose() * J_f * Fx;
    std::vector<Eigen::MatrixXd> ret_vec;
    ret_vec.push_back(G);
    ret_vec.push_back(Fx);

    return ret_vec;
}

void
Predict(robot rob, Eigen::MatrixXd &Xest, Eigen::MatrixXd &U, SlamVariables &slam_vars, const SimVariables &sim_vars) {
    std::vector<Eigen::MatrixXd> G_fx = Motion_jacobian(Xest.block(0, 0, sim_vars.state_size, 1), U, sim_vars);
    Eigen::MatrixXd Fx = G_fx.back();
    slam_vars.G = G_fx.front();
    Eigen::MatrixXd block_Xest = Xest.block(0, 0, 3, 1);
    Xest.block(0, 0, 3, 1) = rob.motion_model(block_Xest, U);
    slam_vars.P_t.block(0, 0, sim_vars.state_size, sim_vars.state_size) =
            slam_vars.G * slam_vars.P_t.block(0, 0, sim_vars.state_size, sim_vars.state_size) *
            slam_vars.G.transpose() + Fx.transpose() * slam_vars.Cx * Fx;
}

void Update(Eigen::MatrixXd &Xest, Eigen::MatrixXd &U, Eigen::MatrixXd &z_obs, SlamVariables &slam_vars,
            const SimVariables &sim_vars) {
    Eigen::MatrixXd P_init = Eigen::MatrixXd::Identity(sim_vars.LM_size, sim_vars.LM_size);

    // Determind which landmarks are being referenced, are they new?
    for (int ii = 0; ii < z_obs.rows(); ii++) {
        int minid = search_lm_id(z_obs.block(ii, 0, 1, sim_vars.state_size), Xest, slam_vars, sim_vars);

        int nLM = CountLMs(Xest, sim_vars);

        if (minid == nLM) {
            cout << "New LM!" << endl;
            // Extend state and covariance matrix
            Eigen::MatrixXd xAug(Xest.rows() + sim_vars.LM_size, Xest.cols());
            Eigen::MatrixXd lm_pos = Calc_LM_pos(Xest, z_obs.block(ii, 0, 1, sim_vars.state_size), sim_vars);
            xAug << Xest,
                    lm_pos;

            Eigen::MatrixXd PAug(slam_vars.P_t.rows() + sim_vars.LM_size, slam_vars.P_t.cols() + sim_vars.LM_size);
            Eigen::MatrixXd PAug_top(slam_vars.P_t.rows(), slam_vars.P_t.cols() + sim_vars.LM_size), PAug_bot(
                    sim_vars.LM_size, slam_vars.P_t.cols() + sim_vars.LM_size);
            PAug_top << slam_vars.P_t, Eigen::MatrixXd::Zero(Xest.rows(), sim_vars.LM_size);
            PAug_bot << Eigen::MatrixXd::Zero(sim_vars.LM_size, Xest.rows()), P_init;
            PAug << PAug_top,
                    PAug_bot;


            Xest = xAug;
            slam_vars.P_t = PAug;
        } else {
            // cout << "Min id: " << minid << " # of LMs: " << nLM << endl;
        }
        Eigen::MatrixXd lm = get_lm_pos_from_state(Xest, minid, sim_vars);

        std::vector<Eigen::MatrixXd> y_s_H = Innovation(lm, Xest, z_obs.block(ii, 0, 1, sim_vars.state_size), minid,
                                                        slam_vars, sim_vars);
        Eigen::MatrixXd K = (slam_vars.P_t * y_s_H[2].transpose()) * y_s_H[1].inverse(); // Kalman gain
        Xest = Xest + (K * y_s_H[0]);
        slam_vars.P_t = (Eigen::MatrixXd::Identity(Xest.rows(), Xest.rows()) - (K * y_s_H[2])) * slam_vars.P_t;
    }

    Xest(2, 0) = pi2pi(Xest(2, 0));
}

Eigen::MatrixXd Calc_LM_pos(Eigen::MatrixXd &xest, Eigen::MatrixXd zi, const SimVariables &sim_vars) {

    Eigen::MatrixXd zp(sim_vars.LM_size, 1);
    zp << xest(0, 0) + zi(0, 0) * cos(-(xest(2, 0) + zi(0, 1))),
            xest(1, 0) + zi(0, 0) * sin(xest(2, 0) + zi(0, 1));

    return zp;
}

std::vector<Eigen::MatrixXd>
Innovation(Eigen::MatrixXd &est_lm_pos, Eigen::MatrixXd &xest, Eigen::MatrixXd zi_obs, int idx,
           SlamVariables &slam_vars, const SimVariables &sim_vars) {

    Eigen::MatrixXd delta(2, 1), zp(1, 2);

    delta = est_lm_pos - xest.block(0, 0, 2, 1);

    double q = (delta.transpose() * delta)(0, 0);
    double zangle = atan2(delta(1, 0), delta(0, 0)) - xest(2, 0);
    zp << sqrt(q), pi2pi(zangle);

    Eigen::MatrixXd y = (zi_obs.block(0, 0, 1, sim_vars.LM_size) - zp).transpose();

    y(1, 0) = pi2pi(y(1, 0));

    Eigen::MatrixXd H = H_jacob(q, delta, xest, idx + 1, sim_vars);

    Eigen::MatrixXd S = H * slam_vars.P_t * H.transpose() + slam_vars.Cx.block(0, 0, 2, 2);

    std::vector<Eigen::MatrixXd> y_s_H;
    y_s_H.push_back(y);
    y_s_H.push_back(S);
    y_s_H.push_back(H);

    return y_s_H;
}

Eigen::MatrixXd get_lm_pos_from_state(Eigen::MatrixXd Xest, int i, const SimVariables &sim_vars) {

    Eigen::MatrixXd lm(2, 1);

    lm = Xest.block(sim_vars.state_size + sim_vars.LM_size * i, 0, sim_vars.LM_size, 1);

    return lm;
}

int
search_lm_id(Eigen::MatrixXd zi_obs, Eigen::MatrixXd &xest, SlamVariables &slam_vars, const SimVariables &sim_vars) {

    int num_LM = CountLMs(xest, sim_vars);
    std::vector<double> mdist;

    for (int ii = 0; ii < num_LM; ii++) {
        // get lm position
        Eigen::MatrixXd est_lm_pos = get_lm_pos_from_state(xest, ii, sim_vars);
        std::vector<Eigen::MatrixXd> y_s_H = Innovation(est_lm_pos, xest, zi_obs, ii, slam_vars, sim_vars);

        Eigen::MatrixXd temp = y_s_H[0].transpose() * y_s_H[1].inverse() * y_s_H[0];
        mdist.push_back(temp(0, 0));
    }

    mdist.push_back(slam_vars.M_DIST_TH);

    auto minitr = min_element(begin(mdist), end(mdist));
    int minid = distance(mdist.begin(), minitr);

    return minid;
}

Eigen::MatrixXd H_jacob(double q, Eigen::MatrixXd &delta, Eigen::MatrixXd &xest, int i, const SimVariables &sim_vars) {
    double root_q = sqrt(q);
    Eigen::MatrixXd G_H(2, 5);

    G_H << -root_q * delta(0, 0), -root_q * delta(1, 0), 0, root_q * delta(0, 0), root_q * delta(1, 0),
            delta(1, 0), -delta(0, 0), -q, -delta(1, 0), delta(0, 0);

    G_H = G_H / q;
    int num_LM = CountLMs(xest, sim_vars);

    Eigen::MatrixXd F1(sim_vars.state_size, sim_vars.state_size + 2 * num_LM), F2(sim_vars.LM_size,
                                                                                  sim_vars.state_size + 2 * num_LM);

    F1 << Eigen::MatrixXd::Identity(sim_vars.state_size, sim_vars.state_size), Eigen::MatrixXd::Zero(
            sim_vars.state_size, sim_vars.LM_size * num_LM);
    F2 << Eigen::MatrixXd::Zero(sim_vars.LM_size, sim_vars.state_size), Eigen::MatrixXd::Zero(sim_vars.LM_size,
                                                                                              sim_vars.LM_size * (i -
                                                                                                                  1)), Eigen::MatrixXd::Identity(
            sim_vars.LM_size, sim_vars.LM_size), Eigen::MatrixXd::Zero(sim_vars.LM_size, sim_vars.LM_size * num_LM -
                                                                                         sim_vars.LM_size * i);

    Eigen::MatrixXd F(F1.rows() + F2.rows(), F1.cols());
    F << F1,
            F2;

    Eigen::MatrixXd H;
    H = G_H * F;

    return H;
}

double pi2pi(double angle) {
    double remain = fmod(angle + 3 * M_PI, 2 * M_PI);
    return remain - M_PI;
}









